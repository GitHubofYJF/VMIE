import torch.nn as nn
import torch.nn.functional as F
# 空间-通道变换工具函数
def space_to_channel(x, p=2):
    """ [B,C,H,W] -> [B,p²C,H/p,W/p] """
    B, C, H, W = x.shape
    x = x.view(B, C, H//p, p, W//p, p)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    return x.view(B, p*p*C, H//p, W//p)

def channel_to_space(x, p=2):
    """ [B,p²C,H/p,W/p] -> [B,C,H,W] """
    B, Cp2, H, W = x.shape
    C = Cp2 // (p*p)
    x = x.view(B, p, p, C, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    return x.view(B, C, H*p, W*p)

class new_Downsample(nn.Module):    
    def __init__(self, in_ch, out_ch, p=2):
        super().__init__()
        self.p = p  # 下采样因子（与PixelUnshuffle的downscale_factor一致）
        
        # 主分支：先下采样再卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        
        # 残差路径：用PixelUnshuffle实现空间→通道变换，再调整通道数
        self.shortcut = nn.PixelUnshuffle(downscale_factor=p)  # 替代自定义space_to_channel
        self.res_conv = nn.Conv2d(in_ch * (p**2), out_ch, kernel_size=1)  # 匹配主分支通道数

    def forward(self, x):
        # 主分支：先平均池化下采样（保持与原逻辑一致），再卷积
        main = F.avg_pool2d(x, self.p)
        main = self.conv(main)
        
        # 残差路径：PixelUnshuffle变换后调整通道
        residual = self.shortcut(x)  # [B, in_ch, H, W] → [B, in_ch*p², H/p, W/p]
        residual = self.res_conv(residual)
        
        return main + residual  # 残差连接

# 残差上采样模块（改用PixelShuffle）
class new_UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=2):
        super().__init__()
        self.p = p  # 上采样因子（与PixelShuffle的upscale_factor一致）
        
        # 主分支：先上采样再卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        
        # 残差路径：先调整通道数，再用PixelShuffle实现通道→空间变换
        self.res_conv = nn.Conv2d(in_ch, out_ch * (p**2), kernel_size=1)  # 扩展通道数为p²倍
        self.shortcut = nn.PixelShuffle(upscale_factor=p)  # 替代自定义channel_to_space

    def forward(self, x):
        # 主分支：先最近邻插值上采样（保持与原逻辑一致），再卷积
        main = F.interpolate(x, scale_factor=self.p, mode='nearest')
        main = self.conv(main)
        
        # 残差路径：先扩通道，再PixelShuffle上采样
        residual = self.res_conv(x)  # [B, in_ch, H, W] → [B, out_ch*p², H, W]
        residual = self.shortcut(residual)  # [B, out_ch*p², H, W] → [B, out_ch, H*p, W*p]
        
        return main + residual  # 残差连接

# 残差下采样模块
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=2):
        super().__init__()
        self.p = p
        
        # 主分支
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        
        # 残差路径
        self.res_conv = nn.Conv2d(in_ch*(p**2), out_ch, 1)

    def forward(self, x):
        # 主分支处理
        main = F.avg_pool2d(x, self.p)
        main = self.conv(main)
        
        # 残差路径
        residual = space_to_channel(x, self.p)
        residual = self.res_conv(residual)
        
        return main + residual  # 残差相加

# 残差上采样模块
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=2):
        super().__init__()
        self.p = p
        
        # 主分支
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        
        # 残差路径
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch*(p**2), 1),
            nn.PixelShuffle(p)
        )

    def forward(self, x):
        # 主分支处理
        main = F.interpolate(x, scale_factor=self.p, mode='nearest')
        main = self.conv(main)
        
        # 残差路径
        residual = self.res_conv(x)
        
        return main + residual  # 残差相加

# 中间层增强模块
class MiddleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU()
            ) for _ in range(3)]
        )
        
    def forward(self, x):
        return x + self.blocks(x)  # 残差连接

# DC_AE模型
class DCAE_64to16(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (64x64x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),     # 32x32x64
            DownBlock(64, 128, p=2),   # 16x16x128
            nn.Conv2d(128, 40, 1),     # 16x16x40
            MiddleBlock(40),            # 特征增强
        )
        
        # 解码器 (16x16x40 -> 64x64x3)
        self.decoder = nn.Sequential(
            MiddleBlock(40),           # 特征增强
            UpBlock(40, 128, p=2),     # 32x32x128
            UpBlock(128, 64, p=2),     # 64x64x64
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()               # 输出[0,1]
        )

    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出64x64x3
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    
class DCAE_128to16(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),     # 64x64x64
            DownBlock(64, 128, p=2),   # 32x32x128
            DownBlock(128, 256, p=2),  # 16x16x256
            nn.Conv2d(256, 40, 1),     # 通道压缩到40
            MiddleBlock(40)            # 16x16x40
        )
        
        # 解码器 (16x16x40 -> 128x128x3)
        self.decoder = nn.Sequential(
            MiddleBlock(40),           # 特征增强
            UpBlock(40, 256, p=2),     # 32x32x256
            UpBlock(256, 128, p=2),    # 64x64x128
            UpBlock(128, 64, p=2),     # 128x128x64
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()               # 输出归一化
        )

    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40 (10,240元素)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        return self.decode(self.encode(x))


# DC-AE 256×256→16×16*160→256×256模型
class DCAE_256to128_Middle(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (256x256x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 128x128x32
            DownBlock(32, 64, p=2),     # 64x64x64
            DownBlock(64, 128, p=2),    # 32x32x128
            DownBlock(128, 256, p=2),   # 16x16x256
            nn.Conv2d(256, 160, 1),      # 16x16x160
            MiddleBlock(160)
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            MiddleBlock(160),           # 特征增强
            UpBlock(160, 256, p=2),       # 32x32x256
            UpBlock(256, 128, p=2),      # 64x64x128
            UpBlock(128, 64, p=2),       # 128x128x64
            UpBlock(64, 32, p=2),        # 256x256x32
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))
    # DC-AE 256×256→16×16*160→256×256模型
class DCAE_256to128(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (256x256x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 128x128x32
            DownBlock(32, 64, p=2),     # 64x64x64
            DownBlock(64, 128, p=2),    # 32x32x128
            DownBlock(128, 256, p=2),   # 16x16x256
            nn.Conv2d(256, 160, 1),      # 16x16x160
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(160, 256, p=2),       # 32x32x256
            UpBlock(256, 128, p=2),      # 64x64x128
            UpBlock(128, 64, p=2),       # 128x128x64
            UpBlock(64, 32, p=2),        # 256x256x32
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
class DCAE_128to128(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x64
            DownBlock(64, 128, p=2),     # 32x32x128
            DownBlock(128, 256, p=2),    # 16x16x256
            nn.Conv2d(256, 160, 1),      # 16x16x160
            MiddleBlock(160)         # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            MiddleBlock(160),
            UpBlock(160, 256, p=2),       # 32x32x256
            UpBlock(256, 128, p=2),      # 64x64x128
            UpBlock(128, 64, p=2),       # 128x128x64
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )


    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
class DCAE_128to128_new(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),    # 16x16x128
            DownBlock(128, 256, p=2),   # 8x8x256
            nn.Conv2d(256, 160, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(160, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64, p=2),      # 32x32x64
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    
class DCAE_512to128(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (256x256x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 256x256x32
            DownBlock(32, 64, p=2),     # 128x128x64
            DownBlock(64, 128, p=2),    # 64x64x128
            DownBlock(128, 256, p=2),   # 32x32x256
            DownBlock(256, 512, p=2),   # 16x16x512
            nn.Conv2d(512, 160, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(160, 512, p=2),       # 32x32x256
            UpBlock(512, 256, p=2),  
            UpBlock(256, 128, p=2),      # 64x64x128
            UpBlock(128, 64, p=2),       # 128x128x64
            UpBlock(64, 32, p=2),        # 256x256x32
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))





class L16_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 16, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(16, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))
    
class L32_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 32, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(32, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))
class L48_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 48, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(48, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))
class L64_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 64, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(64, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))
class L80_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 80, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(80, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))
class L96_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 96, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(96, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))


class L112_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 112, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(112, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))

class L128_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 8x8x256
            DownBlock(128, 256, p=2),   # 16x16x256
            nn.Conv2d(256, 128, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(128, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),
            UpBlock(128, 64, p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))
    
class L144_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 144, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(144, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))
    
class L176_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 176, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(176, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))
    
class L160_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),   # 16*16*128
            DownBlock(128, 256, p=2),   # 16*16*128
            nn.Conv2d(256, 160, 1),      # 16x16x144
        )
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(160, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64,p=2),     # 16x16×128
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))
    

class L144(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),    # 16x16x128
            DownBlock(128, 256, p=2),   # 8x8x256
            nn.Conv2d(256, 144, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(144, 256, p=2),     # 8x8x256
            UpBlock(256, 128, p=2),     # 16x16×128
            UpBlock(128, 64, p=2),      # 32x32x64
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    def forward(self, x):
        return self.decode(self.encode(x))    

class L180(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),    # 16x16x128
            nn.Conv2d(128, 180, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(180, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),      # 32x32x64
            UpBlock(64, 32, p=2),        # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))

class L188(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),    # 16x16x128
            nn.Conv2d(128, 188, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(188, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),      # 32x32x64
            UpBlock(64, 32, p=2),        # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))

class L16(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),    # 16x16x128
            DownBlock(64, 128, p=2),   # 8x8x256
            nn.Conv2d(128, 16, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(16, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),     # 16x16×128
            UpBlock(64, 32, p=2),      # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))

class L32(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),    # 16x16x128
            DownBlock(64, 128, p=2),   # 8x8x256
            nn.Conv2d(128, 32, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(32, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),     # 16x16×128
            UpBlock(64, 32, p=2),      # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))

class L48(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),    # 16x16x128
            DownBlock(64, 128, p=2),   # 8x8x256
            nn.Conv2d(128, 48, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(48, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),     # 16x16×128
            UpBlock(64, 32, p=2),      # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))

class L64(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),    # 16x16x128
            DownBlock(64, 128, p=2),   # 8x8x256
            nn.Conv2d(128, 64, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(64, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),     # 16x16×128
            UpBlock(64, 32, p=2),      # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))

class L80(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),    # 16x16x128
            DownBlock(64, 128, p=2),   # 8x8x256
            nn.Conv2d(128, 80, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(80, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),     # 16x16×128
            UpBlock(64, 32, p=2),      # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))

class L96(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),    # 32*32*64
            DownBlock(64, 128, p=2),   # 16*16*128
            nn.Conv2d(128, 96, 1),      # 16x16x144
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(96, 128, p=2),     # 8x8x256
            UpBlock(128, 64,p=2),     # 16x16×128
            UpBlock( 64,32, p=2),      # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
class L112(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),    # 16x16x128
            DownBlock(64, 128, p=2),   # 8x8x256
            nn.Conv2d(128, 112, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(112, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),     # 16x16×128
            UpBlock(64, 32, p=2),      # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))  

class L128(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),    # 16x16x128
            DownBlock(64, 128, p=2),   # 8x8x256
            nn.Conv2d(128, 128, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(128, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),     # 16x16×128
            UpBlock(64, 32, p=2),      # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )


    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
class DCAE_128to128_144(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),    # 16x16x128
            DownBlock(64, 128, p=2),   # 8x8x256
            nn.Conv2d(128, 144, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(144, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),     # 16x16×128
            UpBlock(64, 32, p=2),      # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )


    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))    

class DCAE_128to128_180(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (128x128x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 32, p=2),      # 64x64x32
            DownBlock(32, 64, p=2),      # 64x64x32
            DownBlock(64, 128, p=2),    # 16x16x128
            nn.Conv2d(128, 180, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(180, 128, p=2),     # 8x8x256
            UpBlock(128, 64, p=2),      # 32x32x64
            UpBlock(64, 32, p=2),        # 32x32x64
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))

class L256(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (256x256x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),      # 128x128x64
            DownBlock(64, 128, p=2),     # 64x64x128
            DownBlock(128, 256, p=2),    # 32x32x256
            DownBlock(256, 512, p=2),   # 16x16x512
            nn.Conv2d(512, 160, 1),      # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(160, 512, p=2),       # 32x32x512
            UpBlock(512, 256, p=2),       # 64x64x256
            UpBlock(256, 128, p=2),       # 128x128x128
            UpBlock(128, 64, p=2),        # 256x256x64
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )

    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))

class L512(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器 (256x256x3 -> 16x16x40)
        self.encoder = nn.Sequential(
            DownBlock(3, 64, p=2),       # 256x256x64
            DownBlock(64, 128, p=2),     # 128x128x128 
            DownBlock(128, 256, p=2),    # 64x64x256
            DownBlock(256, 512, p=2),    # 32x32x512 
            DownBlock(512, 1024, p=2),   # 16x16x1024
            nn.Conv2d(1024, 160, 1),     # 16x16x160
        )
        
        # 解码器 (16x16x40 -> 256x256x3)
        self.decoder = nn.Sequential(
            UpBlock(160, 1024, p=2),     # 32x32x1024
            UpBlock(1024, 512, p=2),     # 64x64x512
            UpBlock(512, 256, p=2),      # 128x128x256
            UpBlock(256, 128, p=2),      # 256x256x128
            UpBlock(128, 64, p=2),       # 512x512x64
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()                 # 输出[0,1]
        )
    def encode(self, x):
        return self.encoder(x)  # 输出16x16x40
    
    def decode(self, z):
        return self.decoder(z)  # 输出256x256x3
    
    def forward(self, x):
        return self.decode(self.encode(x))