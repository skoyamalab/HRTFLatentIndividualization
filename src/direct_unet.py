import torch
import torch.nn as nn
import math
from models import FourierFeatureMapping


class Up1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.up = nn.ConvTranspose1d(in_channels, out_channels, 3, 2, 1, 1)

        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="linear", align_corners=False
            ),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.up(x)


class Down1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv1d(in_channels, out_channels, 3, 2, 1, 1)

    def forward(self, x):
        return self.down(x)


class ResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim=128,
        comb_emb_dim=192,
        dropout=0.15,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(in_channels, in_channels)
        self.norm2 = nn.GroupNorm(out_channels, out_channels)

        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.act = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        residual = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        t = self.act(t)
        t = self.time_emb(t).type(x.dtype)
        x = x + t[:, :, None]

        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x + self.shortcut(residual)


class CrossAttentionBlock1D(nn.Module):
    def __init__(self, channels, cond_emb_dim=64, num_heads=8, dropout=0.15):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True, dropout=dropout
        )

        self.norm1 = nn.GroupNorm(channels, channels)
        self.norm2 = nn.GroupNorm(channels, channels)
        self.cond_proj = nn.Linear(cond_emb_dim, channels)

    def forward(self, x, c=None):
        residual = x
        B, C, L = x.shape

        x = self.norm1(x)
        x = x.view(B, C, L).transpose(1, 2)  # (batch, length, channels)
        # ic(x.shape, residual.shape, c.shape if c is not None else None)
        if c is None or torch.all(c == 0):
            # Self-attention: query, key, and value are all x
            x = self.attn(x, x, x)[0]
        else:
            # Project conditional embedding and permute to match x
            # ic(c.shape)
            c = self.cond_proj(c.permute(0, 2, 1))  # (batch, length, channels)
            x = self.attn(x, c, c)[0]

        x = x.transpose(1, 2).reshape(B, C, L)  # (batch, channels, length)
        # x = self.norm2(x)
        return x + residual


class SelfAttentionBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        # --- Self-Attention Part ---
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # --- Feed-Forward Part ---
        self.norm2 = nn.LayerNorm(channels)
        hidden_features = int(channels * ffn_expansion_factor)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, channels),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Length)

        Returns:
            Output tensor of shape (Batch, Channels, Length)
        """
        residual1 = x

        # --- Self-Attention ---
        x_norm1 = self.norm1(x.permute(0, 2, 1))  # -> (B, L, C)
        attn_output, _ = self.attention(
            query=x_norm1, key=x_norm1, value=x_norm1, need_weights=False
        )
        x = residual1 + self.dropout1(attn_output.permute(0, 2, 1))  # -> (B, C, L)

        # --- Feed-Forward ---
        residual2 = x
        x_norm2 = self.norm2(x.permute(0, 2, 1))  # -> (B, L, C)
        x = residual2 + self.dropout2(
            self.ffn(x_norm2).permute(0, 2, 1)
        )  # -> (B, C, L)

        return x


class ConditionalUNet1DDDM(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_step_dim=256,
        anthro_dim=23,
        deterministic=False,
    ):
        super().__init__()
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_step_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        self.ffm_srcpos = FourierFeatureMapping(
            num_features=16, dim_data=3, trainable=True
        )

        # Conditional embedding
        self.anthro_mlp = nn.Sequential(
            nn.Linear(anthro_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )

        # Down blocks
        self.down1 = nn.ModuleList(
            [
                ResnetBlock1D(in_channels, 64),  # 64 -> 32
                ResnetBlock1D(64, 64),  # 64 -> 32
                CrossAttentionBlock1D(64),  # 64 -> 32
                Down1D(64, 64),  # Downsampling, 64 -> 32
            ]
        )

        self.down2 = nn.ModuleList(
            [
                ResnetBlock1D(64, 128),  # 64 -> 32, 128 -> 64
                ResnetBlock1D(128, 128),  # 128 -> 64
                CrossAttentionBlock1D(128),  # 128 -> 64
                Down1D(128, 128),  # Downsampling, 128 -> 64
            ]
        )

        # Middle blocks
        self.mid = nn.ModuleList(
            [
                ResnetBlock1D(128, 256),  # 128 -> 64, 256 -> 128
                SelfAttentionBlock1D(256),
                ResnetBlock1D(256, 256),  # 256 -> 128
            ]
        )

        # Up blocks
        self.up1 = nn.ModuleList(
            [
                ResnetBlock1D(256, 128),  # 384 -> 192, 128 -> 64
                ResnetBlock1D(128, 128),  # 128 -> 64
                CrossAttentionBlock1D(128),  # 128 -> 64
                Up1D(256, 128),  # Upsampling, 128 -> 64
            ]
        )

        self.up2 = nn.ModuleList(
            [
                ResnetBlock1D(128, 64),  # or 128,  256 -> 128, 64 -> 32
                ResnetBlock1D(64, 64),  # 64 -> 32
                CrossAttentionBlock1D(64),  # 64 -> 32
                Up1D(128, 64),  # Upsampling, 64 -> 32
            ]
        )

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv1d(64, out_channels, 3, padding=1),  # 64 -> 32
        )

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = 256 // 2  # time_step_dim // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def _apply_block(self, layers, x, t, c=None):
        for layer in layers:  # Iterates through the list/slice passed
            # ic(f.shape if f is not None else None)
            if isinstance(layer, ResnetBlock1D):
                x = layer(x, t)
            elif isinstance(layer, CrossAttentionBlock1D):
                x = layer(x, c)
            elif isinstance(layer, SelfAttentionBlock1D):
                x = layer(x)
        return x

    def forward(self, x, time_step, cond):
        # Time and conditional embeddings
        time_emb = self.time_mlp(time_step)

        cond_emb = None
        if cond is not None:
            srcpos_emb, anthro_emb = cond[:, :3], cond[:, 3:]
            srcpos_emb = self.ffm_srcpos(srcpos_emb)
            anthro_emb = self.anthro_mlp(anthro_emb)
            # ic(srcpos_emb.shape, anthro_emb.shape)
            cond_emb = (
                torch.cat([srcpos_emb, anthro_emb], dim=1)
                .unsqueeze(-1)
                .expand(-1, -1, x.shape[-1])
            )
            # ic(x.shape, cond_emb.shape, time_emb.shape)

        # --- Downsampling Path (Corrected Skip Logic) ---
        skips = []
        # Down 1
        x_res1 = self._apply_block(self.down1[:-1], x, time_emb, cond_emb)
        skips.append(x_res1)
        x = self.down1[-1](x_res1)

        # Down 2
        x_res2 = self._apply_block(self.down2[:-1], x, time_emb, cond_emb)
        skips.append(x_res2)
        x = self.down2[-1](x_res2)

        # Mid
        x = self._apply_block(self.mid, x, time_emb, cond_emb)

        # Up 1
        skip = skips.pop()
        up_layer1 = self.up1[-1]
        res_attn_layers1 = self.up1[:-1]
        x = up_layer1(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate along C
        x = self._apply_block(res_attn_layers1, x, time_emb, cond_emb)

        # Up 2
        skip = skips.pop()
        up_layer2 = self.up2[-1]
        res_attn_layers2 = self.up2[:-1]
        x = up_layer2(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate along C
        x = self._apply_block(res_attn_layers2, x, time_emb, cond_emb)

        # x = self.out_norm(x)
        return self.out(x)
