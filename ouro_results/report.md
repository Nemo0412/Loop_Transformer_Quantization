# Ouro-1.4B weight-quantization error across loops

Frozen `ByteDance/Ouro-1.4B` (hidden 2048, 24 layers, 8 UT loops). Eval = WikiText-2 raw test, 48 sequences of 1024 tokens. Weight-only per-group quantization (group 128) on the shared looped core (q/k/v/o/gate/up/down of all 24 decoder layers). Baseline = FP32. INT = asymmetric RTN, FP8 = E4M3 per-group scaling. Dequantized FP32 matmul, no low-bit kernel.

Quantized Linears: 168, params 1,233,125,376.

## Error and perplexity at the final loop

| quant | loop | err_rms | rel_err | cosine | PPL (q) | PPL (fp32) | CE gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| INT8 | 8 | 0.0539 | 0.0320 | 0.9982 | 10.69 | 10.68 | 0.0009 |
| INT4 | 8 | 0.3712 | 0.3569 | 0.9274 | 12.48 | 10.68 | 0.1561 |
| FP8 | 8 | 0.1366 | 0.0939 | 0.9895 | 10.92 | 10.68 | 0.0224 |

## Error growth across loops

- INT8: err_rms loop1=0.0166 -> loop8=0.0539 (x3.25); PPL loop8 quant=10.69 vs FP32=10.68.
- INT4: err_rms loop1=0.2505 -> loop8=0.3712 (x1.48); PPL loop8 quant=12.48 vs FP32=10.68.
- FP8: err_rms loop1=0.0375 -> loop8=0.1366 (x3.65); PPL loop8 quant=10.92 vs FP32=10.68.