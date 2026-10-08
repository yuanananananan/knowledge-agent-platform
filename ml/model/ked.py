import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .attention import CBAM, GNN, cross_encoder
from .unit_knowledge import Unit_KED, Knowledge_emb, ResNet1D


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class KED(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], cfg=None):
        print(cfg)
        num_classes = cfg['data']['num_classes']
        d_model = cfg['model']['d_model']
        n_feat = len(cfg['feature_combinations'])

        self.inplanes = 64
        super(KED, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_sub = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)  # 对应layer2的输出通道数
        self.cbam3 = CBAM(256)  # 对应layer3的输出通道数
        self.cbam4 = CBAM(512)  # 对应layer4的输出通道数
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(12, 64, kernel_size=4, stride=2, padding=1),  # 上采样到 32x32
            nn.BatchNorm2d(64),  # 归一化层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(64, 64, kernel_size=1),  # 1x1 卷积调整通道
            nn.BatchNorm2d(64),  # 可选的额外归一化
            nn.ReLU(inplace=True)  # 可选的额外激活
        )
        self.downsample = nn.Sequential(
            # 3x3 卷积 (保持空间尺寸不变，仅提取特征)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 1x1 卷积 (通道下采样: 128 -> 64)
            nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.res1d = ResNet1D(in_channels=1, out_channels=128, layers=[2, 2, 2])

        self.kemb = Knowledge_emb(64)
        self.unit_kd = Unit_KED(64, 64, layer_nums=4, knn_nums=8)
        self.space_kernel = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),  # [B, 16, 512]
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),  # [B, 3, 256]
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
        )
        # self.cross_at = CrossAttention(args.d_model, 512)
        self.gnn_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.kemb_convdown = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            # bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            # bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            print("ResNet unknown block error !!!")

        return [bn2, bn3, bn4]

    def get_stage_channels(self):
        return [256, 512, 1024, 2048]

    def forward(self, x, otra_feat):
        feature = otra_feat.unsqueeze(1)  # [B, 1, n]
        gnn_x = self.res1d(feature)

        x = self.conv1(x)
        x = self.bn1(x)
        stem = x
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)

        # gnn_x = self.unit_kd(feature, feat1)
        # gnn_x = self.gnn_conv(gnn_x)
        # feat1 = feat1 + nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)(encoded_list[0])

        feat2 = self.layer2(feat1) + gnn_x.unsqueeze(-1)
        feat3 = self.layer3(feat2)
        # feat3 = (deep_x) + _feat3
        feat4 = self.layer4(feat3)

        x = self.avgpool(F.relu(feat4))
        x = x.view(x.size(0), -1)
        avg = x

        out = self.fc(x)

        feats = {}
        # feats["feat_emb_fuse"] = feat_emb_fuse
        feats["pooled_feat"] = avg
        feats["feats"] = [
            F.relu(stem),
            F.relu(feat1),
            F.relu(feat2),
            F.relu(feat3),
            F.relu(feat4),
        ]
        feats["preact_feats"] = [stem, feat1, feat2, feat3, feat4]

        return out, feats
