# networks/sl_module/transformer.py
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_ids, scale=1.0):
        pos = pos_ids.float() * scale
        freqs = torch.einsum("bt,d->btd", pos, self.inv_freq)  # (B,T,dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)               
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dropout=0.0,
        pos_type="rope",          # "none" | "rope"
        rope_base=10000.0,
        time_scale=1.0,
    ):
        super().__init__()
        assert dim % heads == 0

        self.pos_type = pos_type
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

        if pos_type == "rope":
            self.rope = RotaryEmbedding1D(self.head_dim, base=rope_base)
            self.time_scale = time_scale
        else:
            self.rope = None

    def forward(self, x, mask=None, pos_ids=None):
        b, t, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b t (qkv h d) -> qkv b h t d", qkv=3, h=h)

        # --------------------
        # RoPE (if enabled)
        # --------------------
        if self.pos_type == "rope":
            assert pos_ids is not None, "RoPE requires pos_ids"
            cos, sin = self.rope(pos_ids, scale=self.time_scale)  # (B,T,Dh)
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]

            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            key_mask = mask[:, None, None, :].expand(b, h, t, t)
            dots = dots.masked_fill(~key_mask, float("-inf"))

        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class PositionEmbeddingSine1D(nn.Module):
    """
    Standard sine/cosine positional embedding for 1D sequences.

    Returns pos: (B, T, D) where D = dim
    """
    def __init__(self, dim, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.num_pos_feats = dim // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x, mask=None):
        b, t, _ = x.shape

        if mask is None:
            mask = torch.ones((b, t), device=x.device, dtype=torch.bool)
        if mask.dtype != torch.bool:
            mask = mask.bool()

        x_embed = mask.cumsum(1, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            denom = x_embed[:, -1:].clamp(min=eps)
            x_embed = x_embed / denom * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)

        pos = x_embed[:, :, None] / dim_t
        pos = torch.cat([pos.sin(), pos.cos()], dim=-1)
        return pos


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout,
        pos_type="rope",      # "none" | "absolute" | "rope"
        rope_base=1000.0,
        time_scale=0.5,
        pos_drop=0.1,
        pos_alpha_init=0.1,
    ):
        super().__init__()
        self.pos_type = pos_type
        print(f"Using {pos_type} pos_type")

        # Absolute PE
        if pos_type == "absolute":
            self.pos_embed = PositionEmbeddingSine1D(dim)
            self.pos_alpha = nn.Parameter(torch.tensor(float(pos_alpha_init)))
            self.pos_dropout = nn.Dropout(pos_drop)
        else:
            self.pos_embed = None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim,
                                    heads=heads,
                                    dropout=dropout,
                                    pos_type="rope" if pos_type == "rope" else "none",
                                    rope_base=rope_base,
                                    time_scale=time_scale,
                                ),
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout))),
                    ]
                )
            )

    def forward(self, x, mask=None):
        b, t, _ = x.shape

        # Absolute PE
        if self.pos_type == "absolute":
            pos = self.pos_embed(x, mask)
            x = x + self.pos_dropout(self.pos_alpha * pos)

        # RoPE pos ids
        pos_ids = None
        if self.pos_type == "rope":
            pos_ids = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)

        for attn, ff in self.layers:
            x = attn(x, mask=mask, pos_ids=pos_ids)
            x = ff(x)

        return x