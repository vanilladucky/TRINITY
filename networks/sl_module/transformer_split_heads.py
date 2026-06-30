# networks/sl_module/transformer_split_heads.py
"""
Split-Head Transformer Architecture for Multi-Task Learning (3 Tasks: Emotion, Event, Nature)

This architecture implements a transformer with:
- Shared attention heads (4 heads): Learn common knowledge for all tasks
- Emotion-specific heads (4 heads): Trained only with emotion_loss
- Event-specific heads (4 heads): Trained only with event_loss
- Nature-specific heads (4 heads): Trained only with nature_loss

During forward pass:
- Task X: Shared 4 heads + Task X specific 4 heads -> Combined output
"""

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
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pos_ids, scale=1.0):
        inv_freq = self.inv_freq.to(pos_ids.device)
        pos = pos_ids.float() * scale
        freqs = torch.einsum("bt,d->btd", pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


class SplitHeadAttention(nn.Module):
    """
    Attention module that exposes shared and task-specific embeddings separately.
    Now supports 3 tasks: emotion, event, nature.
    """
    def __init__(
        self,
        dim,
        total_heads=8,
        shared_heads=4,
        dropout=0.0,
        pos_type="rope",
        rope_base=10000.0,
        time_scale=1.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.total_heads = total_heads
        self.shared_heads = shared_heads
        self.task_specific_heads = total_heads - shared_heads
        
        assert dim % total_heads == 0, f"dim {dim} must be divisible by total_heads {total_heads}"
        
        self.head_dim = dim // total_heads
        self.scale = self.head_dim ** -0.5
        self.pos_type = pos_type
        
        # Shared heads QKV projection
        shared_dim = self.shared_heads * self.head_dim
        self.shared_to_qkv = nn.Linear(dim, shared_dim * 3, bias=False)
        
        # Task-specific dimension
        task_dim = self.task_specific_heads * self.head_dim

        # 1. Emotion-specific heads QKV
        self.emotion_to_qkv = nn.Linear(dim, task_dim * 3, bias=False)
        
        # 2. Event-specific heads QKV
        self.event_to_qkv = nn.Linear(dim, task_dim * 3, bias=False)

        # 3. Nature-specific heads QKV (NEW)
        self.nature_to_qkv = nn.Linear(dim, task_dim * 3, bias=False)
        
        # Output projections
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        
        # RoPE
        if pos_type == "rope":
            self.rope = RotaryEmbedding1D(self.head_dim, base=rope_base)
            self.time_scale = time_scale
        else:
            self.rope = None
    
    def _apply_rope(self, q, k, pos_ids):
        cos, sin = self.rope(pos_ids, scale=self.time_scale)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    
    def _attention(self, q, k, v, mask, b, t, h):
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            key_mask = mask[:, None, None, :].expand(b, h, t, t)
            dots = dots.masked_fill(~key_mask, float("-inf"))
        
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        return out, attn
    
    def forward(self, x, mask=None, pos_ids=None, task_type="emotion"):
        b, t, _ = x.shape
        
        # === Shared heads ===
        shared_qkv = self.shared_to_qkv(x)
        shared_q, shared_k, shared_v = rearrange(
            shared_qkv, "b t (qkv h d) -> qkv b h t d", 
            qkv=3, h=self.shared_heads
        )
        
        # === Task-specific heads ===
        if task_type == "emotion":
            task_qkv = self.emotion_to_qkv(x)
        elif task_type == "event":
            task_qkv = self.event_to_qkv(x)
        else: # nature
            task_qkv = self.nature_to_qkv(x)
        
        task_q, task_k, task_v = rearrange(
            task_qkv, "b t (qkv h d) -> qkv b h t d",
            qkv=3, h=self.task_specific_heads
        )
        
        # === Apply RoPE ===
        if self.pos_type == "rope" and pos_ids is not None:
            shared_q, shared_k = self._apply_rope(shared_q, shared_k, pos_ids)
            task_q, task_k = self._apply_rope(task_q, task_k, pos_ids)
        
        # === Compute attention ===
        shared_out, shared_attn = self._attention(shared_q, shared_k, shared_v, mask, b, t, self.shared_heads)
        task_out, task_attn = self._attention(task_q, task_k, task_v, mask, b, t, self.task_specific_heads)        

        # === Separate Embeddings ===
        shared_emb = rearrange(shared_out, "b h t d -> b t (h d)")
        task_emb = rearrange(task_out, "b h t d -> b t (h d)")
        
        # === Combined Output ===
        combined_out = torch.cat([shared_out, task_out], dim=1)
        out_reshaped = rearrange(combined_out, "b h t d -> b t (h d)")
        out = self.to_out(out_reshaped)
        
        return out, shared_emb, task_emb, shared_attn, task_attn


class SplitHeadTransformerLayer(nn.Module):
    def __init__(self, dim, total_heads=8, shared_heads=4, mlp_dim=2048, dropout=0.0, pos_type="rope", rope_base=1000.0, time_scale=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = SplitHeadAttention(dim, total_heads, shared_heads, dropout, pos_type, rope_base, time_scale)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x, mask=None, pos_ids=None, task_type="emotion"):
        norm_x = self.norm1(x)
        attn_out, shared_emb, task_emb, shared_attn, task_attn = self.attention(norm_x, mask=mask, pos_ids=pos_ids, task_type=task_type)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, shared_emb, task_emb, shared_attn, task_attn


class SplitHeadTransformer(nn.Module):
    def __init__(self, dim, depth=5, total_heads=8, shared_heads=4, mlp_dim=2048, dropout=0.0, pos_type="rope", rope_base=1000.0, time_scale=0.5):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.total_heads = total_heads
        self.shared_heads = shared_heads
        self.pos_type = pos_type

        self.layers = nn.ModuleList([
            SplitHeadTransformerLayer(dim, total_heads, shared_heads, mlp_dim, dropout, pos_type, rope_base, time_scale)
            for _ in range(depth)
        ])
        print(f"Using {pos_type} pos_type")
    
    def forward(self, x, mask=None, task_type="emotion"):
        b, t, _ = x.shape

        pos_ids = None
        if self.pos_type == "rope":
            pos_ids = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        
        last_shared = None
        last_task = None
        all_shared_attn = [] 
        all_task_attn = []
        
        for layer in self.layers:
            x, shared_emb, task_emb, shared_attn, task_attn = layer(x, mask=mask, pos_ids=pos_ids, task_type=task_type)
            last_shared = shared_emb
            last_task = task_emb
            all_shared_attn.append(shared_attn)
            all_task_attn.append(task_attn)
        
        stacked_shared_attn = torch.stack(all_shared_attn, dim=0)
        stacked_task_attn = torch.stack(all_task_attn, dim=0)
        
        return x, last_shared, last_task, stacked_shared_attn, stacked_task_attn
    
    def get_shared_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.attention.shared_to_qkv.parameters())
            params.extend(layer.attention.to_out.parameters())
            params.extend(layer.norm1.parameters())
            params.extend(layer.norm2.parameters())
            params.extend(layer.ffn.parameters())
        return params
    
    # ... (keep get_emotion_specific_parameters, get_event_specific_parameters, get_nature_specific_parameters as they are) ...
    def get_emotion_specific_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.attention.emotion_to_qkv.parameters())
        return params
    
    def get_event_specific_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.attention.event_to_qkv.parameters())
        return params

    def get_nature_specific_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.attention.nature_to_qkv.parameters())
        return params


class SplitHeadTransformerSeparateFFN(nn.Module):
    """
    Transformer with split attention heads AND separate FFN per task.
    Supports 3 tasks.
    """
    
    def __init__(
        self,
        dim,
        depth=5,
        total_heads=8,
        shared_heads=4,
        mlp_dim=2048,
        dropout=0.0,
        pos_type="rope",
        rope_base=1000.0,
        time_scale=0.5,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.pos_type = pos_type
        
        self.layers = nn.ModuleList([
            SplitHeadTransformerLayerSeparateFFN(
                dim=dim,
                total_heads=total_heads,
                shared_heads=shared_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                pos_type=pos_type,
                rope_base=rope_base,
                time_scale=time_scale,
            )
            for _ in range(depth)
        ])
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask=None, task_type="emotion"):
        b, t, _ = x.shape
        pos_ids = None
        if self.pos_type == "rope":
            pos_ids = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        
        for layer in self.layers:
            x = layer(x, mask=mask, pos_ids=pos_ids, task_type=task_type)
        
        return x
    
    def get_shared_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.attention.shared_to_qkv.parameters())
            params.extend(layer.attention.to_out.parameters())
            params.extend(layer.norm1.parameters())
            params.extend(layer.norm2_shared.parameters())
            params.extend(layer.ffn_shared.parameters())
        return params
    
    def get_emotion_specific_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.attention.emotion_to_qkv.parameters())
            params.extend(layer.norm2_emotion.parameters())
            params.extend(layer.ffn_emotion.parameters())
        return params
    
    def get_event_specific_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.attention.event_to_qkv.parameters())
            params.extend(layer.norm2_event.parameters())
            params.extend(layer.ffn_event.parameters())
        return params

    def get_nature_specific_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.attention.nature_to_qkv.parameters())
            params.extend(layer.norm2_nature.parameters())
            params.extend(layer.ffn_nature.parameters())
        return params


class SplitHeadTransformerLayerSeparateFFN(nn.Module):
    """Transformer layer with split-head attention AND separate FFN per task (3 tasks)."""
    
    def __init__(
        self,
        dim,
        total_heads=8,
        shared_heads=4,
        mlp_dim=2048,
        dropout=0.0,
        pos_type="rope",
        rope_base=1000.0,
        time_scale=0.5,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        
        self.attention = SplitHeadAttention(
            dim=dim,
            total_heads=total_heads,
            shared_heads=shared_heads,
            dropout=dropout,
            pos_type=pos_type,
            rope_base=rope_base,
            time_scale=time_scale,
        )
        
        # Norms
        self.norm2_shared = nn.LayerNorm(dim)
        self.norm2_emotion = nn.LayerNorm(dim)
        self.norm2_event = nn.LayerNorm(dim)
        self.norm2_nature = nn.LayerNorm(dim) # NEW
        
        # FFNs
        self.ffn_shared = FeedForward(dim, mlp_dim // 2, dropout)
        self.ffn_emotion = FeedForward(dim, mlp_dim // 2, dropout)
        self.ffn_event = FeedForward(dim, mlp_dim // 2, dropout)
        self.ffn_nature = FeedForward(dim, mlp_dim // 2, dropout) # NEW
    
    def forward(self, x, mask=None, pos_ids=None, task_type="emotion"):
        x = x + self.attention(self.norm1(x), mask=mask, pos_ids=pos_ids, task_type=task_type)[0]
        
        shared_ffn_out = self.ffn_shared(self.norm2_shared(x))
        
        if task_type == "emotion":
            task_ffn_out = self.ffn_emotion(self.norm2_emotion(x))
        elif task_type == "event":
            task_ffn_out = self.ffn_event(self.norm2_event(x))
        else: # nature
            task_ffn_out = self.ffn_nature(self.norm2_nature(x))
        
        x = x + shared_ffn_out + task_ffn_out
        
        return x