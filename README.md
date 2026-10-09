# Loop Transformer Quantization

How weight-only quantization error behaves inside **looped / recurrent-depth**
language models, where the same quantized weight matrix is reused at every loop.

The central question: in a looped model the weight perturbation `δW` is *identical*
at every iteration, so the hidden-state error obeys a recursion

```
e_{t+1} = F(h_t + e_t; W + δW) - F(h_t; W)
        ≈ (∂F/∂h) e_t + (∂F/∂W) δW
        =  J e_t + b
```

Whether the error compounds or saturates is then decided by the spectral radius of
`J`: if `ρ(J) < 1` the error converges to the fixed point `e* = (I - J)^-1 b`
(a plateau); if `ρ(J) ≥ 1` it grows with depth.

## Main subject: Ouro-1.4B

[Ouro](https://ouro-llm.github.io/) (`ByteDance/Ouro-1.4B`) is a pre-trained Looped
Language Model: one shared stack of 24 decoder layers applied `R` times in latent
space (trained at `R = 4`, extrapolates to 8).

Because the loop is purely sequential, a single forward with `total_ut_steps = T`
yields the hidden state at *every* intermediate loop count, so one pass covers
`T = 1..8`.

### Quantization

Weight-only, applied to the **shared looped core** only — the `q/k/v/o/gate/up/down`
projections of all 24 decoder layers (168 Linear layers, 1.23B params). Embeddings,
RMSNorm scales, `lm_head` and the exit gate stay in full precision, so the measured
error comes purely from the looped weights.

| config | scheme |
|---|---|
| INT8 | asymmetric per-group RTN, group 128 |
| INT4 | asymmetric per-group RTN, group 128 |
| FP8  | E4M3, per-group `amax -> 448` scaling, group 128 |

Baseline is **FP32**. This is *fake quant*: weights are dequantized back to floating
point and a normal matmul is used. **No low-bit kernel is involved and no speedup is
claimed** — the study is about numerical error, not throughput.

## Results

### Error across loops (`ouro_results/`)

WikiText-2 raw test, 48 sequences x 1024 tokens. Error at the 8th loop versus FP32:

| quant | err RMS | relative error | cos vs FP | PPL (quant) | PPL (FP32) |
|---|---:|---:|---:|---:|---:|
| INT8 | 0.054 | 0.032 | 0.998 | 10.69 | 10.68 |
| FP8  | 0.137 | 0.094 | 0.990 | 10.92 | 10.68 |
| INT4 | 0.371 | 0.357 | 0.927 | 12.48 | 10.68 |

- `err_rms` is the elementwise RMS of `h_q - h_fp`; `rel_err` is the mean per-token
  `||h_q - h_fp|| / ||h_fp||`.
- INT8 is effectively lossless on perplexity. FP8 (E4M3) costs 0.24 PPL. INT4 costs
  1.8 PPL.
- Error grows with depth for all three but **does not diverge**. INT4 saturates after
  loop 2 at ~0.35-0.37, while INT8 and FP8 are still slowly accumulating at loop 8.
- INT4's weight error is 17x INT8's (matching the expected `2^4`), yet its
  hidden-state error is only ~6.9x — strongly sublinear, i.e. the large INT4
  perturbation sits in a regime where the loop map contracts it more aggressively.

### Error decomposition

`--decompose` splits each loop's error exactly into

```
e_{t+1} = p_t + q_t
p_t = F(h_q_t; Wq) - F(h_fp_t; Wq)   propagation of the previous error  (~ J e_t)
q_t = F(h_fp_t; Wq) - F(h_fp_t; W)   fresh injection by this loop        (~ b)
```

and reports the empirical contraction factor `||p_t|| / ||e_{t-1}||` (an estimate of
`ρ(J)`) plus the plateau it implies, `||q|| / (1 - gain)`.

### Quantizer validation

`verify_quant.py` reads real Ouro weight matrices from the safetensors file and
checks the quantizers are genuinely low-bit:

| quant | relative weight error | distinct levels per group (max) |
|---|---:|---:|
| INT8 | 0.0063 | 85-93 (256) |
| FP8  | 0.0256 | 64-69 |
| INT4 | 0.1046 | 12-16 (16) |

INT4/INT8 weight-error ratio is 17.0, matching the expected `2^4 = 16`.

## Layout

```
run_ouro_error.py       error-vs-loop and error decomposition on Ouro-1.4B
run_ouro_gsm8k.py       GSM8K accuracy under each quantization (8-shot CoT, greedy)
verify_quant.py         proves the quantizers emit real low-bit weights
src/quantize.py         per-group RTN; FP8 E4M3 lives in run_ouro_error.py
src/compensate.py       inference-time error compensators (mean / low-rank / dynamics)
run_compensation.py     compensation study on a small from-scratch looped LM
run_arc_poc.py          the same error study on the ARC Prize Tiny Recursive Model
submit_*.sbatch         SLURM launchers (NYU Greene, L40S)
ouro_results/           Ouro CSVs, plots, report
comp_results/           compensation study output
arc_results/            ARC TRM output
```

## Environment

Ouro's remote code targets `transformers==4.55.0`; newer releases break it
(`ROPE_INIT_FUNCTIONS['default']`, read-only `Cache.key_cache`, the rewritten mask
utilities). The drivers run inside a venv pinned to 4.55 that inherits torch from the
host environment:

```bash
python -m venv --system-site-packages /path/to/ouro_tf455
source /path/to/ouro_tf455/bin/activate
pip install "transformers==4.55.0" "huggingface_hub<1.0" "tokenizers<0.22"
```

## Running

```bash
# error vs loop, INT8 / INT4 / FP8 against FP32
python run_ouro_error.py --configs INT8 INT4 FP8 --loops 8 --eval-seqs 48

# exact propagation / injection split per loop
python run_ouro_error.py --decompose --configs INT8 INT4 FP8 --loops 8

# GSM8K accuracy
python run_ouro_gsm8k.py --configs FP INT8 INT4 FP8 --loops 4 --n-problems 200
```
