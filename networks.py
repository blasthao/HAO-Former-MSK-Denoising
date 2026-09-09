import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 核心组件：转置自注意力机制 (MDTA)
# (极大地节省了显存，专门针对高分辨率图像设计)
# ==========================================
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out

# ==========================================
# 2. 核心组件：门控前馈神经网络 (GDFN)
# ==========================================
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(GDFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# ==========================================
# 3. Restormer 基础块
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # LayerNorm 需要通道在最后一位，算完再换回来
        x_norm1 = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm1)
        x_norm2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm2)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
    def forward(self, x):
        return self.body(x)

# ==========================================
# 4. 主网络：医学版 Restormer (单通道 MRI)
# ==========================================
class Restormer(nn.Module):
    def __init__(self, inp_channels=1, out_channels=1, dim=48, num_blocks=[2,2,2,2], num_heads=[1,2,4,4], ffn_expansion_factor=2.66):
        super(Restormer, self).__init__()
        
        # 初始特征提取
        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)

        # 编码器 (Encoder)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim, dim*2)

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim*2, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim*2, dim*4)

        # 瓶颈层 (Latent)
        self.latent = nn.Sequential(*[TransformerBlock(dim=dim*4, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor) for _ in range(num_blocks[2])])

        # 解码器 (Decoder)
        self.up3_2 = Upsample(dim*4, dim*2)
        self.reduce_chan_level2 = nn.Conv2d(dim*4, dim*2, kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim*2, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor) for _ in range(num_blocks[3])])

        self.up2_1 = Upsample(dim*2, dim)
        self.reduce_chan_level1 = nn.Conv2d(dim*2, dim, kernel_size=1, bias=False)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor) for _ in range(num_blocks[3])])

        # 输出层
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        latent = self.latent(inp_enc_level3)

        # U-Net 跳跃连接 (Skip Connections)
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out = self.output(out_dec_level1)
        
        # 全局残差学习：预测出来的噪声图 + 原图 = 清晰降噪图
        return out + inp_img