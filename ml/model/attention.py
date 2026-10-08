import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn_lib import Grapher, act_layer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):

    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


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


class GNN(nn.Module):

    def __init__(self, d_model, n_node, n_blocks=4, k=4):
        super(GNN, self).__init__()

        self.k = 4
        self.conv = 'mr'
        self.act = 'gelu'
        self.channels = d_model
        self.n_node = n_node
        self.norm = 'batch'  # batch or instance normalization {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.n_blocks = 4  # number of basic blocks in the backbone
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
        # self.n_node_sqrt = int(self.n_node**0.5)
        # self.pos_embed = nn.Parameter(torch.zeros(1, d_model, self.n_node_sqrt, self.n_node_sqrt))
        # out_dim=256
        # self.convdown_2d = nn.Sequential(nn.Conv2d(d_model, out_dim, 1, 1), nn.BatchNorm2d(out_dim))

    def forward(self, x):
        if len(x.shape) == 3:
            self.n_node = x.shape[-1]
            x = x.permute(0, 2, 1).reshape(x.shape[0], self.channels, self.n_node_sqrt, self.n_node_sqrt)

        # x = x + self.pos_embed
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        # x = self.convdown_2d(x)
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
        self.attention = CrossAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size, drop_prob)

    def forward(self, x1, x2, mask=None):
        B, _, h1, w1 = x1.shape
        B, _, h2, w2 = x2.shape
        x1 = x1.reshape(B, -1, h1 * w1).permute(0, 2, 1)
        x2 = x2.reshape(B, -1, h2 * w2).permute(0, 2, 1)
        q_hidden_state = x1
        kv_hidden_state = self.layer_norm_1(x2)

        x = q_hidden_state + self.attention(q_hidden_state, kv_hidden_state, kv_hidden_state, mask=mask)
        x = x + self.feed_forward(self.layer_norm_2(x))

        x = x.permute(0, 2, 1).reshape(B, -1, h1, w1)
        return x


class GateSigmoid(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 将拼接后的特征映射
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()  # 输出0-1的注意力权重
        )

    def forward(self, x1, x2):
        context = torch.cat((x1, x2), dim=-1)
        attention_weights = self.attention(context)
        fused = attention_weights * x1
        return fused


class GatedFusion(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.fc_z = nn.Linear(dim * 2, dim)  # 更新门
        self.fc_r = nn.Linear(dim * 2, dim)  # 重置门
        self.fc_h = nn.Linear(dim * 2, dim)  # 候选状态

    def forward(self, feat_a, feat_b):
        # 将feat_a作为主特征，feat_b作为新输入（可互换）
        combined = torch.cat((feat_a, feat_b), dim=-1)
        z = torch.sigmoid(self.fc_z(combined))  # 更新门：决定保留多少旧状态
        r = torch.sigmoid(self.fc_r(combined))  # 重置门：决定如何组合新输入与旧状态
        combined_reset = torch.cat((r * feat_a, feat_b), dim=-1)
        h_hat = torch.tanh(self.fc_h(combined_reset))  # 候选新状态
        fused = (1 - z) * feat_a + z * h_hat  # 融合结果
        return fused


class MatweightAttention_light(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, feat_a, feat_b):
        Q = feat_a
        K = feat_b
        V = feat_b

        attn_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)).squeeze()
        attn_weights = torch.sigmoid(attn_scores)  # 用Sigmoid得到门控值 [b]
        fused = attn_weights.unsqueeze(1) * V  # [b, 512]
        return fused


class LowRankFusion(nn.Module):

    def __init__(self, input_dims, output_dim, rank):
        super().__init__()
        # 简化思路：分别投影后，使用外积+低秩投影
        self.proj_a = nn.Linear(input_dims, rank)
        self.proj_b = nn.Linear(input_dims, rank)
        self.fc_out = nn.Linear(rank, output_dim)

    def forward(self, feat_a, feat_b):
        za = self.proj_a(feat_a)  # [32, rank]
        zb = self.proj_b(feat_b)  # [32, rank]
        # 计算外积（近似）
        # fused = za * zb  # 元素乘是外积的简化
        outer_product = torch.bmm(za.unsqueeze(2), zb.unsqueeze(1))
        fused = torch.mean(outer_product, dim=2)  # [batch_size, rank]
        output = self.fc_out(fused)
        return output
