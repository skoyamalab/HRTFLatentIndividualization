import torch
import torch.nn as nn
import math
from icecream import ic

# class FrequencyAwareConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()
        
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
#         # Frequency attention module - Modified to match output dimensions
#         self.freq_attention = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1),  # 1x1 conv to match channel dimensions
#             nn.BatchNorm2d(out_channels),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         # Compute FFT
#         x_freq = torch.fft.fft2(x, dim=(-2, -1))
#         x_freq = torch.abs(x_freq)  # Get magnitude spectrum
        
#         # Apply frequency attention and handle strided convolutions
#         freq_weights = self.freq_attention(x_freq.real)
#         if self.conv.stride[0] > 1:
#             freq_weights = F.interpolate(
#                 freq_weights, 
#                 size=(x.shape[2] // self.conv.stride[0], 
#                       x.shape[3] // self.conv.stride[1]),
#                 mode='bilinear',
#                 align_corners=False
#             )
        
#         # Standard convolution
#         x_spatial = self.conv(x)
        
#         # Ensure dimensions match before multiplication
#         assert x_spatial.shape == freq_weights.shape, f"Shape mismatch: {x_spatial.shape} vs {freq_weights.shape}"
        
#         # Combine spatial and frequency information
#         out = x_spatial * freq_weights
        
#         return out

# class FrequencyAwareConvTranspose2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
#         super().__init__()
        
#         self.convt = nn.ConvTranspose2d(
#             in_channels, out_channels, kernel_size, 
#             stride=stride, padding=padding, 
#             output_padding=output_padding
#         )
        
#         # Input frequency attention
#         self.freq_attention = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 1),
#             nn.BatchNorm2d(in_channels),
#             nn.Sigmoid()
#         )
        
#         # Output frequency refinement
#         self.freq_refinement = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         # Analyze input frequencies
#         x_freq = torch.fft.fft2(x, dim=(-2, -1))
#         x_freq_mag = torch.abs(x_freq)
        
#         # Apply frequency attention to input
#         freq_weights_in = self.freq_attention(x_freq_mag.real)
#         x_weighted = x * freq_weights_in
        
#         # Perform transposed convolution
#         x_upscaled = self.convt(x_weighted)
        
#         # Analyze output frequencies
#         x_up_freq = torch.fft.fft2(x_upscaled, dim=(-2, -1))
#         x_up_freq_mag = torch.abs(x_up_freq)
        
#         # Apply frequency refinement to maintain frequency characteristics
#         freq_weights_out = self.freq_refinement(x_up_freq_mag.real)
        
#         # Ensure dimensions match
#         assert x_upscaled.shape == freq_weights_out.shape, \
#             f"Shape mismatch: {x_upscaled.shape} vs {freq_weights_out.shape}"
        
#         out = x_upscaled * freq_weights_out
        
#         return out

# Keep this frozen with already computed values for mean and std
class FeatureNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        # Normalize along the feature dimension (dim=1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + self.eps)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
    
    def forward(self, x):
        return self.up(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
    
    def forward(self, x):
        return self.down(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Use BatchNorm2d if in_channels is 1, otherwise use GroupNorm
        if in_channels == 1:
            self.norm1 = nn.BatchNorm2d(in_channels)
        else:
            self.norm1 = nn.GroupNorm(32, in_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t):
        residual = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        t = self.act(t)
        t = self.time_emb(t).type(x.dtype)
        x = x + t[:, :, None, None]

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        return x + self.shortcut(residual)

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, cond_emb_dim=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.norm = nn.GroupNorm(32, channels)
        self.cond_proj = nn.Linear(cond_emb_dim, channels)  # Project conditional embedding to match channels

    def forward(self, x, c=None):
        residual = x
        x = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2)  # (batch, seq_len, channels)
        
        if c is not None:
            # Project conditional embedding and expand dimensions to match x
            c = self.cond_proj(c).unsqueeze(1).expand(-1, x.shape[1], -1)  # (batch, seq_len, channels)
            # Cross-attention: query is x, key and value are c
            x = self.attn(x, c, c)[0]
        else:
            # Self-attention: query, key, and value are all x
            x = self.attn(x, x, x)[0]
        
        x = x.transpose(1, 2).view(residual.shape)  # (batch, channels, height, width)
        return self.norm(x + residual)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_step_dim=256, cond_dim=12):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_step_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        ).to('cuda')
        
        
        # Conditional embedding (could be FFT mapping)
        self.cond_mlp = nn.Sequential(
            # FeatureNorm(),
            # nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 32), # Hyperparameter
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 32),
        ).to('cuda')
        
        # Down blocks
        self.down1 = nn.ModuleList([
            ResnetBlock(in_channels, 32), # 64 -> 32
            ResnetBlock(32, 32), # 64 -> 32
            CrossAttentionBlock(32), # 64 -> 32
            Down(32, 32),  # Downsampling, 64 -> 32
        ]).to('cuda')
        
        self.down2 = nn.ModuleList([
            ResnetBlock(32, 64), # 64 -> 32, 128 -> 64
            ResnetBlock(64, 64), # 128 -> 64
            CrossAttentionBlock(64), # 128 -> 64
            Down(64, 64),  # Downsampling, 128 -> 64
        ]).to('cuda')
        
        # Middle blocks
        self.mid = nn.ModuleList([
            ResnetBlock(64, 128), # 128 -> 64, 256 -> 128
            CrossAttentionBlock(128), # 256 -> 128
            ResnetBlock(128, 128), # 256 -> 128
        ]).to('cuda')
        
        # Up blocks
        self.up1 = nn.ModuleList([
            ResnetBlock(192, 64), # 384 -> 192, 128 -> 64
            ResnetBlock(64, 64), # 128 -> 64
            CrossAttentionBlock(64), # 128 -> 64
            Up(64, 64),  # Upsampling, 128 -> 64
        ]).to('cuda')
        
        self.up2 = nn.ModuleList([
            ResnetBlock(128, 32), # 256 -> 128, 64 -> 32
            ResnetBlock(32, 32), # 64 -> 32
            CrossAttentionBlock(32), # 64 -> 32
            Up(32, 32),  # Upsampling, 64 -> 32
        ]).to('cuda')
        
        # Output
        self.out = nn.Conv2d(32, out_channels, 3, padding=1).to('cuda') # 64 -> 32
        # ic(self)
    
    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = 256 // 2 # time_step_dim // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward_layer(self, layer, x, t, c=None):
        if isinstance(layer, ResnetBlock):
            x = layer(x, t)
        elif isinstance(layer, CrossAttentionBlock):
            x = layer(x, c)
        else:
            x = layer(x)
        return x

    def forward(self, x, time_step, cond):
        # Time and conditional embeddings
        time_emb = self.time_mlp(time_step)
        cond_emb = self.cond_mlp(cond) if cond is not None else None
        
        # Downsampling
        skips = []
        for layer in self.down1:
            x = self.forward_layer(layer, x, time_emb, cond_emb)
            skips.append(x)
        
        for layer in self.down2:
            x = self.forward_layer(layer, x, time_emb, cond_emb)
            skips.append(x)

        # Middle
        for layer in self.mid:
            x = self.forward_layer(layer, x, time_emb, cond_emb)
        
        # Upsampling
        for i, layer in enumerate(self.up1):
            x = self.forward_layer(layer, torch.cat([x, skips.pop()], dim=1) if i == 0 else x, time_emb, cond_emb)
        
        for i, layer in enumerate(self.up2):
            x = self.forward_layer(layer, torch.cat([x, skips.pop()], dim=1) if i == 0 else x, time_emb, cond_emb)
        
        return self.out(x)
    
if __name__ == "__main__":
    # Usage
    latent_channels = 1
    time_step_dim = 256
    cond_dim = 30
    unet = ConditionalUNet(in_channels=latent_channels, out_channels=latent_channels, 
                           time_step_dim=time_step_dim, cond_dim=cond_dim)
    
    # Example forward pass
    batch_size = 1
    latent_height, latent_width = 64, 256
    x = torch.randn(batch_size, latent_channels, latent_width, latent_height)
    time_step = torch.randn(batch_size, time_step_dim)
    cond = torch.randn(batch_size, cond_dim)
    # cond_emb = None
    
    ic(x)
    output = unet(x, time_step, cond)
    ic(output)