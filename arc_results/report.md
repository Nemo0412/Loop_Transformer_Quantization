# Cross-loop quantization on the ARC Tiny Recursive Model

Same question as the TRM-LM run, now on the frozen ARC checkpoints from [arcprize/trm_arc_prize_verification](https://huggingface.co/arcprize/trm_arc_prize_verification). No pretrained weight is updated.

## What loops here

`L_level` is two post-norm attention/SwiGLU blocks, shared by `z_L` and `z_H`. Each H cycle applies that module 4 times to update `z_L`, then once to update `z_H`. Embeddings, including the per-puzzle embedding, stay FP32. Every `CastedLinear` (the eight core projections plus the grid head and halt head) is quantized once and reused. Evaluation always runs 16 ACT steps: halting is disabled outside training, and there is no KV cache.

H cycles are swept over 1, 2, 3, 4, and 6. The trained depth is 3. Grid labels use the official encoding (pad 0 ignored, EOS 1, colors 2–11). Cross-entropy is the training stablemax loss, averaged over supervised cells. Perplexity is `exp(CE)` on that 12-way vocabulary. Exact accuracy is the fraction of grids whose supervised cells are all correct. These are single unaugmented puzzles, not the augmented majority vote behind the public 40% / 6.2% ARC numbers.

Quantization is asymmetric per-group round-to-nearest, group size 128. The forward pass multiplies dequantized FP32 weights. **No low-bit kernel is used**, so latency is not an integer-GEMM speedup.

## Baseline

| checkpoint | method | H | CE | PPL | exact acc | logits MSE | ms/example |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | fp | 1 | 0.3760 | 1.456 | 25.0% | 0.0000 | 67.55 |
| v1 | fp | 3 | 0.3729 | 1.452 | 25.0% | 0.0000 | 201.80 |
| v1 | fp | 6 | 0.3713 | 1.450 | 25.0% | 0.0000 | 402.54 |
| v1 | int4 | 1 | 0.3717 | 1.450 | 24.0% | 736.3034 | 67.61 |
| v1 | int4 | 3 | 0.3697 | 1.447 | 24.0% | 782.7676 | 201.08 |
| v1 | int4 | 6 | 0.3705 | 1.449 | 24.0% | 760.7374 | 400.82 |
| v1 | int3 | 1 | 0.3799 | 1.462 | 19.8% | 2339.2988 | 66.96 |
| v1 | int3 | 3 | 0.3927 | 1.481 | 19.8% | 2121.4096 | 200.03 |
| v1 | int3 | 6 | 0.3906 | 1.478 | 19.8% | 2048.5742 | 399.78 |
| v1 | int2 | 1 | 1.6297 | 5.102 | 0.0% | 10207.2875 | 66.33 |
| v1 | int2 | 3 | 1.6981 | 5.463 | 0.0% | 10209.9421 | 197.91 |
| v1 | int2 | 6 | 1.6824 | 5.378 | 0.0% | 10120.3939 | 395.81 |
| v2 | fp | 1 | 1.1065 | 3.024 | 4.2% | 0.0000 | 67.70 |
| v2 | fp | 3 | 1.1390 | 3.124 | 4.2% | 0.0000 | 203.06 |
| v2 | fp | 6 | 1.1257 | 3.082 | 4.2% | 0.0000 | 406.39 |
| v2 | int4 | 1 | 1.0522 | 2.864 | 3.1% | 564.2596 | 67.72 |
| v2 | int4 | 3 | 1.0992 | 3.002 | 3.1% | 483.1136 | 203.10 |
| v2 | int4 | 6 | 1.1183 | 3.060 | 3.1% | 501.6706 | 406.05 |
| v2 | int3 | 1 | 1.1211 | 3.068 | 3.1% | 1797.0913 | 67.36 |
| v2 | int3 | 3 | 1.0992 | 3.002 | 3.1% | 1869.9886 | 202.67 |
| v2 | int3 | 6 | 1.1105 | 3.036 | 3.1% | 1843.0430 | 406.08 |
| v2 | int2 | 1 | 3.1669 | 23.734 | 0.0% | 5092.7250 | 66.36 |
| v2 | int2 | 3 | 3.0161 | 20.411 | 0.0% | 5089.6944 | 198.82 |
| v2 | int2 | 6 | 3.2133 | 24.862 | 0.0% | 5172.8105 | 398.29 |

## Error across H cycles

Plot files: `plots/error_vs_loop_v1.png` and `plots/error_vs_loop_v2.png`. z_H is the state read by the grid head. Local error is one quantized H cycle on the FP state; propagated error is the rest.

| checkpoint | bits | H | RMS | relative | cosine vs previous | local/total |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 4 | 1 | 0.359 | 0.359 | n/a | 0.371 |
| v1 | 4 | 3 | 0.362 | 0.362 | 0.933 | 0.348 |
| v1 | 4 | 6 | 0.365 | 0.365 | 0.963 | 0.340 |
| v1 | 3 | 1 | 0.560 | 0.560 | n/a | 0.508 |
| v1 | 3 | 3 | 0.563 | 0.563 | 0.956 | 0.482 |
| v1 | 3 | 6 | 0.564 | 0.564 | 0.982 | 0.475 |
| v1 | 2 | 1 | 1.081 | 1.081 | n/a | 0.609 |
| v1 | 2 | 3 | 1.094 | 1.094 | 0.975 | 0.594 |
| v1 | 2 | 6 | 1.094 | 1.094 | 0.993 | 0.589 |
| v2 | 4 | 1 | 0.358 | 0.358 | n/a | 0.411 |
| v2 | 4 | 3 | 0.363 | 0.363 | 0.933 | 0.388 |
| v2 | 4 | 6 | 0.364 | 0.364 | 0.961 | 0.381 |
| v2 | 3 | 1 | 0.633 | 0.633 | n/a | 0.501 |
| v2 | 3 | 3 | 0.641 | 0.641 | 0.957 | 0.474 |
| v2 | 3 | 6 | 0.640 | 0.640 | 0.987 | 0.469 |
| v2 | 2 | 1 | 1.216 | 1.216 | n/a | 0.625 |
| v2 | 2 | 3 | 1.220 | 1.220 | 0.969 | 0.607 |
| v2 | 2 | 6 | 1.221 | 1.221 | 0.987 | 0.597 |

## Oracle

At 3 H cycles, alpha 0 and 1:

| checkpoint | bits | mode | alpha | CE | exact acc | CE gap closed |
|---|---:|---|---:|---:|---:|---:|
| v1 | 4 | after_loop_1 | 0.00 | 0.3697 | 24.0% | -0.0% |
| v1 | 4 | after_loop_1 | 1.00 | 0.3755 | 25.0% | 184.0% |
| v1 | 4 | after_loop_2 | 0.00 | 0.3697 | 24.0% | -0.0% |
| v1 | 4 | after_loop_2 | 1.00 | 0.3752 | 24.0% | 174.4% |
| v1 | 4 | every_except_final | 0.00 | 0.3697 | 24.0% | -0.0% |
| v1 | 4 | every_except_final | 1.00 | 0.3752 | 24.0% | 174.4% |
| v1 | 3 | after_loop_1 | 0.00 | 0.3927 | 19.8% | 0.0% |
| v1 | 3 | after_loop_1 | 1.00 | 0.3454 | 21.9% | 238.5% |
| v1 | 3 | after_loop_2 | 0.00 | 0.3927 | 19.8% | 0.0% |
| v1 | 3 | after_loop_2 | 1.00 | 0.3536 | 22.9% | 197.4% |
| v1 | 3 | every_except_final | 0.00 | 0.3927 | 19.8% | 0.0% |
| v1 | 3 | every_except_final | 1.00 | 0.3536 | 22.9% | 197.4% |
| v1 | 2 | after_loop_1 | 0.00 | 1.6981 | 0.0% | 0.0% |
| v1 | 2 | after_loop_1 | 1.00 | 0.5063 | 11.5% | 89.9% |
| v1 | 2 | after_loop_2 | 0.00 | 1.6981 | 0.0% | 0.0% |
| v1 | 2 | after_loop_2 | 1.00 | 0.3808 | 22.9% | 99.4% |
| v1 | 2 | every_except_final | 0.00 | 1.6981 | 0.0% | 0.0% |
| v1 | 2 | every_except_final | 1.00 | 0.3808 | 22.9% | 99.4% |
| v2 | 4 | after_loop_1 | 0.00 | 1.0992 | 3.1% | -0.0% |
| v2 | 4 | after_loop_1 | 1.00 | 1.1153 | 4.2% | 40.5% |
| v2 | 4 | after_loop_2 | 0.00 | 1.0992 | 3.1% | -0.0% |
| v2 | 4 | after_loop_2 | 1.00 | 1.1126 | 4.2% | 33.7% |
| v2 | 4 | every_except_final | 0.00 | 1.0992 | 3.1% | -0.0% |
| v2 | 4 | every_except_final | 1.00 | 1.1126 | 4.2% | 33.7% |
| v2 | 3 | after_loop_1 | 0.00 | 1.0992 | 3.1% | -0.0% |
| v2 | 3 | after_loop_1 | 1.00 | 1.0452 | 3.1% | -135.9% |
| v2 | 3 | after_loop_2 | 0.00 | 1.0992 | 3.1% | -0.0% |
| v2 | 3 | after_loop_2 | 1.00 | 1.0930 | 3.1% | -15.6% |
| v2 | 3 | every_except_final | 0.00 | 1.0992 | 3.1% | -0.0% |
| v2 | 3 | every_except_final | 1.00 | 1.0930 | 3.1% | -15.6% |
| v2 | 2 | after_loop_1 | 0.00 | 3.0161 | 0.0% | 0.0% |
| v2 | 2 | after_loop_1 | 1.00 | 1.3656 | 0.0% | 87.9% |
| v2 | 2 | after_loop_2 | 0.00 | 3.0161 | 0.0% | 0.0% |
| v2 | 2 | after_loop_2 | 1.00 | 1.1399 | 4.2% | 100.0% |
| v2 | 2 | every_except_final | 0.00 | 3.0161 | 0.0% | 0.0% |
| v2 | 2 | every_except_final | 1.00 | 1.1399 | 4.2% | 100.0% |

v1: Oracle recovery is meaningful: INT3 H=2 after_loop_1 alpha=0.5 closes 340.1% of the CE gap and changes exact accuracy by +0.010. v2: Oracle recovery is meaningful: INT2 H=2 after_loop_1 alpha=1.0 closes 100.4% of the CE gap and changes exact accuracy by +0.042.

## Linear corrector

Fit only on calibration demonstrations. Test grids follow the corrected quantized trajectory. The map is `U @ V.T @ state`, separate for `z_H` and `z_L`.

| checkpoint | bits | H | method | rank | CE | exact acc | gap closed | pred cosine | extra params | latency ratio |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 4 | 3 | none | - | 0.3697 | 24.0% | -0.0% | n/a | 0 | 1.000 |
| v1 | 4 | 3 | oracle | - | 0.3752 | 24.0% | 174.4% | n/a | 0 | n/a |
| v1 | 4 | 3 | shared | 16 | 0.3793 | 18.8% | 303.3% | 0.053 | 32768 | 1.002 |
| v1 | 4 | 3 | independent | 16 | 0.3867 | 19.8% | 535.7% | 0.060 | 65536 | 1.003 |
| v1 | 3 | 3 | none | - | 0.3927 | 19.8% | 0.0% | n/a | 0 | 1.000 |
| v1 | 3 | 3 | oracle | - | 0.3536 | 22.9% | 197.4% | n/a | 0 | n/a |
| v1 | 3 | 3 | shared | 16 | 0.4413 | 10.4% | -245.2% | 0.207 | 32768 | 1.008 |
| v1 | 3 | 3 | independent | 16 | 0.4180 | 10.4% | -127.4% | 0.218 | 65536 | 1.009 |
| v1 | 2 | 3 | none | - | 1.6981 | 0.0% | 0.0% | n/a | 0 | 1.000 |
| v1 | 2 | 3 | oracle | - | 0.3808 | 22.9% | 99.4% | n/a | 0 | n/a |
| v1 | 2 | 3 | shared | 16 | 1.6029 | 0.0% | 7.2% | 0.316 | 32768 | 0.999 |
| v1 | 2 | 3 | independent | 16 | 1.5799 | 0.0% | 8.9% | 0.336 | 65536 | 1.002 |
| v2 | 4 | 3 | none | - | 1.0992 | 3.1% | -0.0% | n/a | 0 | 1.000 |
| v2 | 4 | 3 | oracle | - | 1.1126 | 4.2% | 33.7% | n/a | 0 | n/a |
| v2 | 4 | 3 | shared | 16 | 1.0806 | 4.2% | -46.7% | 0.154 | 32768 | 0.999 |
| v2 | 4 | 3 | independent | 16 | 1.0866 | 4.2% | -31.6% | 0.172 | 65536 | 0.998 |
| v2 | 3 | 3 | none | - | 1.0992 | 3.1% | -0.0% | n/a | 0 | 1.000 |
| v2 | 3 | 3 | oracle | - | 1.0930 | 3.1% | -15.6% | n/a | 0 | n/a |
| v2 | 3 | 3 | shared | 16 | 1.1449 | 3.1% | 115.0% | 0.212 | 32768 | 0.999 |
| v2 | 3 | 3 | independent | 16 | 1.1264 | 3.1% | 68.3% | 0.226 | 65536 | 1.000 |
| v2 | 2 | 3 | none | - | 3.0161 | 0.0% | 0.0% | n/a | 0 | 1.000 |
| v2 | 2 | 3 | oracle | - | 1.1399 | 4.2% | 100.0% | n/a | 0 | n/a |
| v2 | 2 | 3 | shared | 16 | 2.8611 | 0.0% | 8.3% | 0.323 | 32768 | 0.999 |
| v2 | 2 | 3 | independent | 16 | 2.8338 | 0.0% | 9.7% | 0.326 | 65536 | 0.999 |

## Is it promising here?

The strongest full-state correction before later H cycles is v1 INT4 at 4 H cycles, closing 1500.0% of the stablemax CE gap and moving exact accuracy by +0.000. The best rank-16 independent map (v1 INT4, 4 H cycles) closes 29307.4% of the CE gap, 1954% of the oracle's recovery, with 98304 extra FP32 parameters. Cross-loop linear feedback is promising enough to study further, still as fake quantization rather than a faster kernel.
