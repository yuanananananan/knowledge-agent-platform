import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
# from vit_pytorch import ViT
import matplotlib.pyplot as plt
from .gcn_lib import Grapher, act_layer


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class CrossAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super(CrossAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output


class FeedForward(nn.Module):

    def __init__(self, d_model, intermediate_size, drop_prob):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class cross_encoder(nn.Module):

    def __init__(self, d_model, n_heads, intermediate_size, drop_prob):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(512, d_model)
        self.attention = CrossAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size, drop_prob)

    def forward(self, x1, x2, mask=None):
        # len_seq = x.shape[1]
        # x = self.layer_norm_1(x)
        # i = 0
        # rest_indices = torch.cat([torch.arange(0, i), torch.arange(i + 1, len_seq)])
        q_hidden_state = x1
        kv_hidden_state = self.layer_norm_1(self.w_q(x2))

        x = q_hidden_state + self.attention(q_hidden_state, kv_hidden_state, kv_hidden_state, mask=mask)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class Knowledge_emb(nn.Module):

    def __init__(self, d_model=128):
        super(Knowledge_emb, self).__init__()
        self.inner_channel = 17
        self.otra_feat_encoder = nn.Sequential(
            # nn.Conv2d(15, d_model // 2, 1, 1),  # 扩大中间维度
            # nn.BatchNorm2d(d_model // 2),  # 添加正则化
            # nn.ReLU(),
            # nn.Conv2d(d_model // 2, d_model, 1, 1),
            # nn.BatchNorm2d(d_model),  # 添加正则化
            # nn.ReLU(),
            nn.Conv2d(17, d_model, 3, 2, 1),  # 扩大中间维度
            nn.BatchNorm2d(d_model),  # 添加正则化
            nn.ReLU(),
        )
        self.ae_conv = nn.Conv2d(d_model, self.inner_channel, 1, 1)

    def forward(self, _feature, **kwargs):
        feature = self.otra_feat_encoder(_feature)
        ae_feat = self.ae_conv(feature)
        encoder_loss = torch.abs(ae_feat - nn.MaxPool2d(2, 2)(_feature)).mean()
        return feature, encoder_loss


class Unit_KED(nn.Module):

    def __init__(self, in_dim, out_dim, layer_nums, knn_nums):
        super(Unit_KED, self).__init__()

        # self.conv_in1 = nn.Sequential(nn.Conv2d(in_dim, d_model // 2, 3, 2, 1), nn.BatchNorm2d(d_model // 2))
        self.conv_in2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.ReLU())
        self.conv_out = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, 1, 1, 0), nn.BatchNorm2d(out_dim))
        # self.cbam1 = CBAM(out_dim1)
        # self.conv_out2 = nn.Sequential(nn.Conv2d(d_model, out_dim2, 1, 1, 0), nn.BatchNorm2d(out_dim2))
        # self.cbam2 = CBAM(out_dim2)
        self.gnn_fuse = Base_GNN_block(k=knn_nums, n_blocks=layer_nums, d_model=out_dim * 2)

    def forward(self, feature, deep_feat, **kwargs):
        # feat_emb_fuse1 = nn.MaxPool2d(kernel_size=2, stride=2)(feature)
        feat_emb_fuse1 = feature
        feat_emb_fuse2 = self.conv_in2(deep_feat)
        feat_emb_fuse = torch.cat([feat_emb_fuse1, feat_emb_fuse2], dim=1)
        feat_emb_fuse = self.gnn_fuse((nn.MaxPool2d(2, 2)(feat_emb_fuse)))
        return feat_emb_fuse


class Base_GNN_block(nn.Module):

    def __init__(self, k, n_blocks, d_model=128):
        super(Base_GNN_block, self).__init__()
        self.k = 4
        self.conv = 'mr'
        self.act = 'gelu'
        self.channels = d_model
        self.norm = 'batch'  # batch or instance normalization {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.n_blocks = 2  # number of basic blocks in the backbone
        self.dropout = 0.1  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = 0.
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(self.k, 2 * self.k, self.n_blocks)]
        self.backbone = nn.Sequential(*[
            nn.Sequential(
                Grapher(self.channels,
                        num_knn[i],
                        1,
                        self.conv,
                        self.act,
                        self.norm,
                        self.bias,
                        self.use_stochastic,
                        self.epsilon,
                        1,
                        drop_path=dpr[i]),
                FFN(
                    self.channels,
                    self.channels * 4,
                    act=self.act,
                    drop_path=dpr[i],
                ),
            ) for i in range(self.n_blocks)
        ])

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
        return x


class ResidualBlock1D(nn.Module):
    """1D残差块"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    1D ResNet模型，将[B,1,512]编码为[B,C,1]
    
    Args:
        in_channels: 输入通道数，默认为1
        out_channels: 输出通道数C（特征维度）
        layers: 每个阶段的残差块数量
    """

    def __init__(self, in_channels=1, out_channels=256, layers=[2, 2, 2]):
        super(ResNet1D, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(16, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, out_channels, layers[2], stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride, downsample))

        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # [B, 64, 256]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, 128]

        x = self.layer1(x)  # [B, 64, 128]
        x = self.layer2(x)  # [B, 128, 64]
        x = self.layer3(x)  # [B, 256, 64]

        x = self.adaptive_pool(x)  # [B, num_classes, 1]
        x.view(x.size(0), -1)

        return x
