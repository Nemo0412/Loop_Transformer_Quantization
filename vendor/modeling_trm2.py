"""TRM-LM v2: a small transformer whose middle layers loop. Standalone PyTorch definition + generation.
By SmallAICreator / UltraLabs.

    from modeling_trm2 import TRM2, generate
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("tokenizer.json")
    m = TRM2(); m.load_state_dict(torch.load("trm2_sft.pt", map_location="cpu")); m.eval()
    print(generate(m, tok, "What is the capital of France?", loops=3))

Architecture: embed -> prelude (1 block) -> core (5 blocks) x r loops -> coda (1 block) -> tied head.
Each loop: h = rms(h) * gain[i] + e + tag[i], where e is the prelude output (re-injected every loop).
Blocks are pre-norm (RMSNorm) with RoPE causal attention and SwiGLU. r=1 is an ordinary 7-layer model;
more loops = more computation per token from the same 9.8M parameters.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(VOCAB=16384, D=256, HEADS=4, FF=704, PRELUDE=1, CORE=5, CODA=1, SEQ=1024, MAX_ITERS=8)
USER, ASSISTANT, END, EOT = 3, 4, 5, 0          # chat format: <|user|>...<|end|><|assistant|>...<|end|>


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def rope_tables(T, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.n1 = nn.Parameter(torch.ones(d))
        self.n2 = nn.Parameter(torch.ones(d))
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gu = nn.Linear(d, 2 * ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(rms(x) * self.n1).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        g, u = self.gu(rms(x) * self.n2).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u)


class TRM2(nn.Module):
    def __init__(self, c=CFG):
        super().__init__()
        self.c = c
        d = c["D"]
        self.embed = nn.Embedding(c["VOCAB"], d)
        self.prelude = nn.ModuleList([Block(d, c["HEADS"], c["FF"]) for _ in range(c["PRELUDE"])])
        self.core = nn.ModuleList([Block(d, c["HEADS"], c["FF"]) for _ in range(c["CORE"])])
        self.coda = nn.ModuleList([Block(d, c["HEADS"], c["FF"]) for _ in range(c["CODA"])])
        self.gain = nn.Parameter(torch.ones(c["MAX_ITERS"], d))
        self.tag = nn.Parameter(torch.zeros(c["MAX_ITERS"], d))
        self.nf = nn.Parameter(torch.ones(d))

    def forward(self, ids, loops=3):
        cos, sin = rope_tables(ids.shape[1], self.c["D"] // self.c["HEADS"], ids.device)
        x = self.embed(ids)
        for b in self.prelude:
            x = b(x, cos, sin)
        e = h = x
        for i in range(loops):
            j = min(i, self.c["MAX_ITERS"] - 1)
            h = rms(h) * self.gain[j] + e + self.tag[j]
            for b in self.core:
                h = b(h, cos, sin)
        for b in self.coda:
            h = b(h, cos, sin)
        return (rms(h) * self.nf) @ self.embed.weight.t()


@torch.no_grad()
def generate(model, tok, prompt, loops=3, max_new=160, temperature=0.6, top_p=0.9, rep_penalty=1.2, chat=True):
    ids = ([USER] + tok.encode(prompt, add_special_tokens=False).ids + [END, ASSISTANT]) if chat \
        else tok.encode(prompt, add_special_tokens=False).ids
    out = []
    for _ in range(max_new):
        lg = model(torch.tensor([(ids + out)[-model.c["SEQ"]:]]), loops)[0, -1].float()
        lg[[1, 2, 3, 4]] = -1e9                          # never emit pad / role tokens
        for t in set(out[-64:]):
            lg[t] = lg[t] / rep_penalty if lg[t] > 0 else lg[t] * rep_penalty
        if temperature <= 0:
            nxt = int(lg.argmax())
        else:
            p = torch.softmax(lg / temperature, -1)
            sp, si = p.sort(descending=True)
            keep = (sp.cumsum(0) - sp) < top_p
            nxt = int(si[torch.multinomial(sp * keep, 1)])
        if nxt in (END, EOT):
            break
        out.append(nxt)
    return tok.decode(out)
