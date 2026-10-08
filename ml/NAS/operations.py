# import torch
# import torch.nn as nn

# OPS = {
#   'none' : lambda C, stride, affine: Zero(stride),
#   'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#   'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
#   'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#   'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
#   'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
#   'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
#   'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
#   'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
#   'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
#     nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#     ),
# }

# class ReLUConvBN(nn.Module):

#   def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#     super(ReLUConvBN, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine)
#     )

#   def forward(self, x):
#     return self.op(x)

# class DilConv(nn.Module):
    
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#     super(DilConv, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine),
#       )

#   def forward(self, x):
#     return self.op(x)


# class SepConv(nn.Module):
    
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#     super(SepConv, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_in, affine=affine),
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine),
#       )

#   def forward(self, x):
#     return self.op(x)


# class Identity(nn.Module):

#   def __init__(self):
#     super(Identity, self).__init__()

#   def forward(self, x):
#     return x


# class Zero(nn.Module):

#   def __init__(self, stride):
#     super(Zero, self).__init__()
#     self.stride = stride

#   def forward(self, x):
#     if self.stride == 1:
#       return x.mul(0.)
#     return x[:,:,::self.stride,::self.stride].mul(0.)


# class FactorizedReduce(nn.Module):

#   def __init__(self, C_in, C_out, affine=True):
#     super(FactorizedReduce, self).__init__()
#     assert C_out % 2 == 0
#     self.relu = nn.ReLU(inplace=False)
#     self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#     self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
#     self.bn = nn.BatchNorm2d(C_out, affine=affine)

#   def forward(self, x):
#     x = self.relu(x)
#     out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
#     out = self.bn(out)
#     return out

import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),



  'daa_conv_5x5': lambda C, stride, affine: DAAConv(C, C, kernel_size=5, stride=stride, padding=2, affine=affine),
  'daa_conv_3x3': lambda C, stride, affine: DAAConv(C, C, kernel_size=3, stride=stride, padding=1, affine=affine),

  'lvg_conv_5x5': lambda C, stride, affine: LVGConv(C, C, kernel_size=5, stride=stride, padding=2, affine=affine),
  'lvg_conv_3x3': lambda C, stride, affine: LVGConv(C, C, kernel_size=3, stride=stride, padding=1, affine=affine),

  # 3x3 的稀疏池化 (适合捕捉近距离的离散强散射点)
  'ss_max_pool_3x3': lambda C, stride, affine: SSMaxPool(C, kernel_size=3, stride=stride, dilation=2, affine=affine),
  # 5x5 的稀疏池化 (感受野更大，适合捕捉大型舰船或大块目标的离散特征)
  'ss_max_pool_5x5': lambda C, stride, affine: SSMaxPool(C, kernel_size=5, stride=stride, dilation=2, affine=affine),

}

"1. 动态轴向感知卷积 (Dynamic Axial-Aware Conv, DAAConv)创新点： "
"由于你的 NAS 算法要同时适应 SAR（空间特征为主）和 RD（轴向特征为主），"
"这个操作层引入了自适应的轴向注意力。它内部包含水平（1*5）和垂直（5*1）两个并行的卷积核，"
"并通过一个极其轻量级的通道注意力机制动态分配权重。在 RD 数据上： 网络会自动学会在不同的层级侧重多普勒轴或距离轴。"
"在 SAR/MON 数据上： 网络会将其等效为一个感受野更大的十字形空间滤波器，有助于提取舰船或车辆的几何交叉特征。"



class DAAConv(nn.Module):
    """
    修正版：动态轴向感知卷积，修复 Reduction Cell 中的非对称降采样维度崩溃问题
    """
    def __init__(self, C_in, C_out, kernel_size=5, stride=1, padding=2, affine=True):
        super(DAAConv, self).__init__()
        
        # 垂直方向 (Range/Height): (kernel_size, 1)
        # 修复点: stride 必须全局统一，以适应 DARTS 的 Reduction 要求
        self.conv_v = nn.Conv2d(C_in, C_in, (kernel_size, 1), stride=stride, 
                                padding=(padding, 0), groups=C_in, bias=False)
        
        # 水平方向 (Doppler/Width): (1, kernel_size)
        # 修复点: 同上
        self.conv_h = nn.Conv2d(C_in, C_in, (1, kernel_size), stride=stride, 
                                padding=(0, padding), groups=C_in, bias=False)
                                
        self.pointwise = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.relu = nn.ReLU(inplace=False)
        
        # 轻量级动态权重生成器
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 把全局特征压缩到一个像素大小（感受野覆盖全局）
            nn.Conv2d(C_in, 2, 1, bias=False),   # 用一个 1x1 卷积将通道数压缩到 2（分别代表垂直和水平的权重）
            nn.Softmax(dim=1)  # 最后用 Softmax(dim=1) 归一化，使得这两个权重之和等于 1
        )

    def forward(self, x):
        x_in = self.relu(x)  # 预激活 (Pre-activation)
        
        out_v = self.conv_v(x_in)  # 垂直方向卷积 负责提取 Y 轴（高度）上的空间特征
        out_h = self.conv_h(x_in)  # 水平方向卷积 负责提取 X 轴（宽度）上的空间特征
        
        # 计算动态权重 (B, 2, 1, 1)
        att_weights = self.attention(x_in)   # 动态权重生成
        w_v = att_weights[:, 0:1, :, :]
        w_h = att_weights[:, 1:2, :, :]
        
        # 特征融合 (此时 out_v 和 out_h 的 shape 绝对一致)
        out_fused = out_v * w_v + out_h * w_h  # 网络会根据当前输入的图像内容，动态决定是更看重垂直特征（比如建筑物的边缘）还是水平特征（比如地平线）
        return self.bn(self.pointwise(out_fused))


"2. 局部方差门控卷积 (Local Variance-Gated Conv, LVGConv)创新点："
"针对 SAR 图像特有的乘性相干斑噪声设计的操作。传统的深度学习层倾向于放大高亮度的散斑噪声。"
"这个层借鉴了雷达信号处理中 Lee 滤波器的思想：计算局部的空间方差。"
"方差大的区域（通常是目标边缘或强散射中心）保留特征，方差小的区域（通常是同质背景噪声）抑制特征。"

class LVGConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(LVGConv, self).__init__()
    # 用于计算局部均值的平滑池化层
    self.local_mean_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

    self.conv = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )
    self.relu = nn.ReLU(inplace=False)

  def forward(self, x):
    # 1. 估算局部统计量 (模拟 Lee 滤波器原理)
    local_mean = self.local_mean_pool(x)
    # 近似局部方差: E[x^2] - (E[x])^2
    local_var = self.local_mean_pool(x ** 2) - local_mean ** 2

    # 2. 生成门控注意力图 (归一化方差)
    # 加上极小值防止除零，通过 Sigmoid 将其映射为 0~1 的软门控
    gate = torch.sigmoid(local_var / (local_mean ** 2 + 1e-5))

    # 3. 施加门控：抑制同质区域噪声，保留强散射中心
    x_gated = x * gate

    # 4. 常规特征提取
    # return self.relu(self.conv(x_gated))
    return self.conv(self.relu(x_gated))

"3. 稀疏散射中心保持池化 (Sparse Scattering Max-Pool, SSMaxPool)创新点："
"在雷达图像（尤其是 SAR）中，目标的形状往往由几个离散的高亮像素（强散射中心）构成。"
"常规的 Average Pooling 会把这些高亮像素模糊掉，而普通的 Max Pooling 可能会丢失相对较弱但重要的次级散射中心。)"
"这个操作通过一个膨胀的池化窗口，在降采样的同时最大程度保留离散的亮斑网络，非常适合作为 NAS 中的 skip_connect 的替代或并列选项。"

class SSMaxPool(nn.Module):
    """
    支持自定义核大小的稀疏散射中心保持池化层 (Sparse Scattering Max-Pool)
    通过外挂 ZeroPad2d 绕过 PyTorch 对 MaxPool padding <= kernel_size/2 的限制
    """
    def __init__(self, C_in, kernel_size=3, stride=1, dilation=2, affine=True):
        super(SSMaxPool, self).__init__()
        
        # 动态计算所需的 padding 大小
        pad = dilation * (kernel_size - 1) // 2
        
        # 1. 显式地在四周进行零填充
        self.pad = nn.ZeroPad2d(pad)
        
        # 2. 将 MaxPool2d 内部的 padding 设为 0
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, 
                                 padding=0, dilation=dilation)
        
        # 保持通道数并进行归一化
        self.bn = nn.BatchNorm2d(C_in, affine=affine)

    def forward(self, x):
        # 先手动 padding，再进行不带 padding 的空洞池化
        x = self.pad(x)
        x = self.pool(x)
        return self.bn(x)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

