"""ARC Tiny Recursive Model, loop-explicit.

The shared core is one L_level module: two post-norm attention/SwiGLU blocks.
Each H cycle runs that module L_cycles times to update z_L, then once to update z_H.
Evaluation repeats this inner pass for halt_max_steps ACT steps. There is no KV cache;
every call recomputes full non-causal attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models_common import trunc_normal_init_


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(inputs.dtype) if self.bias is not None else None
        return F.linear(inputs, self.weight.to(inputs.dtype), bias=bias)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.embedding(inputs, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_position_embeddings, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.output_size = head_dim * num_heads
        self.qkv_proj = CastedLinear(hidden_size, 3 * self.output_size, bias=False)
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch, seq, 3 * self.num_heads, self.head_dim)
        query, key, value = qkv.chunk(3, dim=2)
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        attn = attn.permute(0, 2, 1, 3).reshape(batch, seq, self.output_size)
        return self.o_proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = (-(round(expansion * hidden_size * 2 / 3) // -256)) * 256
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TRMBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, expansion: float, rms_norm_eps: float):
        super().__init__()
        self.self_attn = Attention(hidden_size, hidden_size // num_heads, num_heads)
        self.mlp = SwiGLU(hidden_size, expansion)
        self.norm_eps = rms_norm_eps

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class LLevel(nn.Module):
    def __init__(self, layers: list[TRMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, cos_sin) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin, hidden_states)
        return hidden_states


@dataclass
class TRMConfig:
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4
    L_layers: int = 2
    L_cycles: int = 4
    H_cycles: int = 3
    seq_len: int = 900
    vocab_size: int = 12
    puzzle_emb_len: int = 16
    puzzle_emb_ndim: int = 512
    num_puzzle_identifiers: int = 1
    halt_max_steps: int = 16
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class PuzzleEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weights = nn.Buffer(torch.zeros(num_embeddings, embedding_dim))


class ARCTinyRecursiveModel(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        scale = math.sqrt(config.hidden_size)
        self.embed_scale = scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, 1.0 / scale, torch.float32)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        self.puzzle_emb_len = config.puzzle_emb_len
        self.puzzle_emb = PuzzleEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_heads,
            config.seq_len + config.puzzle_emb_len,
            config.rope_theta,
        )
        self.L_level = LLevel(
            [TRMBlock(config.hidden_size, config.num_heads, config.expansion, config.rms_norm_eps) for _ in range(config.L_layers)]
        )
        self.H_init = nn.Buffer(torch.zeros(config.hidden_size))
        self.L_init = nn.Buffer(torch.zeros(config.hidden_size))

    def input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(inputs.to(torch.int32))
        puzzle = self.puzzle_emb.weights[puzzle_identifiers]
        pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle.shape[-1]
        if pad_count > 0:
            puzzle = F.pad(puzzle, (0, pad_count))
        puzzle = puzzle.view(-1, self.puzzle_emb_len, self.config.hidden_size)
        return self.embed_scale * torch.cat((puzzle, embedding), dim=-2)

    def init_state(self, batch_size: int, device):
        seq = self.config.seq_len + self.puzzle_emb_len
        z_h = self.H_init.view(1, 1, -1).expand(batch_size, seq, -1).clone()
        z_l = self.L_init.view(1, 1, -1).expand(batch_size, seq, -1).clone()
        return z_h.to(device), z_l.to(device)

    def logits_from_z(self, z_h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(z_h)[:, self.puzzle_emb_len :]


def apply_L(model: ARCTinyRecursiveModel, hidden, injection, rope):
    return model.L_level(hidden, injection, rope)


def one_h_cycle(model, z_h, z_l, embeddings, rope, l_cycles: int):
    for _ in range(l_cycles):
        z_l = apply_L(model, z_l, z_h + embeddings, rope)
    z_h = apply_L(model, z_h, z_l, rope)
    return z_h, z_l


def normalize_state_dict(state: dict) -> dict:
    anchor = "L_level.layers.0.mlp.down_proj.weight"
    matches = [key for key in state if key.endswith(anchor)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one L_level weight, found {matches[:8]}")
    prefix = matches[0][: -len(anchor)]
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def load_arc_trm(path: str, device: torch.device) -> ARCTinyRecursiveModel:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    state = normalize_state_dict(obj)
    weight = state["puzzle_emb.weights"]
    config = TRMConfig(num_puzzle_identifiers=weight.shape[0], puzzle_emb_ndim=weight.shape[1])
    model = ARCTinyRecursiveModel(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [key for key in missing if "rotary_emb" not in key]
    real_unexpected = [key for key in unexpected if not key.startswith("puzzle_emb.local")]
    if real_missing or real_unexpected:
        raise RuntimeError(f"missing={real_missing[:12]} unexpected={real_unexpected[:12]}")
    model.float().eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)
