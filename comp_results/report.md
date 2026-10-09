# Inference-time error compensation for the quantized TRM-LM

Frozen `TRM-LM-v2-10M`. Eval = WikiText-2 raw test, 324291 next-token predictions (sha `ff8b6d507cfc`); calibration = the disjoint validation prefix. Weight-only asymmetric per-group RTN, group 128. Dequantized FP32 matmul, no low-bit kernel.

**No compensator reads a full-precision hidden state at inference.** FP states build calibration targets and log prediction quality only. Mean and low-rank maps are fit sequentially on the compensated quantized rollout; dynamics is fit from the uncompensated rollout with one DAgger refit and free-runs at inference, seeded from the observable loop-0 state.

## Error decomposition  e_{t+1} = p_t + q_t

At loops=8. p = propagation (previous error through the quantized core), q = injection (fresh quantization error on the FP state).

| bits | loop | ‖e‖ | ‖p‖ | ‖q‖ | gain ‖p‖/‖e‖ | cos(p,q) | cos(e_t,e_{t+1}) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1.427 | 1.172 | 1.110 | 0.821 | -0.223 | n/a |
| 2 | 2 | 1.721 | 1.487 | 0.890 | 0.864 | -0.010 | 0.891 |
| 2 | 3 | 1.903 | 1.671 | 0.838 | 0.878 | 0.053 | 0.964 |
| 2 | 4 | 1.996 | 1.769 | 0.830 | 0.886 | 0.065 | 0.981 |
| 2 | 5 | 2.020 | 1.803 | 0.858 | 0.893 | 0.038 | 0.986 |
| 2 | 6 | 2.087 | 1.884 | 0.865 | 0.903 | 0.026 | 0.990 |
| 2 | 7 | 2.133 | 1.942 | 0.867 | 0.911 | 0.016 | 0.992 |
| 2 | 8 | 2.166 | 1.984 | 0.868 | 0.916 | 0.008 | 0.994 |
| 3 | 1 | 0.600 | 0.417 | 0.457 | 0.695 | -0.059 | n/a |
| 3 | 2 | 0.704 | 0.526 | 0.361 | 0.747 | 0.239 | 0.867 |
| 3 | 3 | 0.776 | 0.598 | 0.339 | 0.770 | 0.330 | 0.962 |
| 3 | 4 | 0.816 | 0.638 | 0.335 | 0.782 | 0.354 | 0.983 |
| 3 | 5 | 0.827 | 0.648 | 0.346 | 0.783 | 0.337 | 0.989 |
| 3 | 6 | 0.859 | 0.683 | 0.348 | 0.795 | 0.329 | 0.993 |
| 3 | 7 | 0.884 | 0.712 | 0.349 | 0.806 | 0.320 | 0.995 |
| 3 | 8 | 0.903 | 0.735 | 0.350 | 0.815 | 0.311 | 0.997 |
| 4 | 1 | 0.277 | 0.186 | 0.209 | 0.672 | -0.020 | n/a |
| 4 | 2 | 0.322 | 0.236 | 0.163 | 0.734 | 0.283 | 0.859 |
| 4 | 3 | 0.356 | 0.269 | 0.153 | 0.758 | 0.382 | 0.959 |
| 4 | 4 | 0.375 | 0.289 | 0.151 | 0.769 | 0.412 | 0.982 |
| 4 | 5 | 0.383 | 0.294 | 0.156 | 0.769 | 0.403 | 0.987 |
| 4 | 6 | 0.400 | 0.313 | 0.158 | 0.781 | 0.400 | 0.992 |
| 4 | 7 | 0.414 | 0.328 | 0.158 | 0.792 | 0.395 | 0.995 |
| 4 | 8 | 0.425 | 0.341 | 0.158 | 0.802 | 0.389 | 0.996 |

## Compensators

PPL at the largest loop count, with 95% bootstrap CI on CE. Fraction-of-oracle is how much of the naive→oracle CE gap each method closes.

| bits | loops | method | PPL | CE [95% CI] | gap closed | frac oracle | pred cos | extra params |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | 8 | dynamics_r16 | 91.57 | 4.5171 [4.4868, 4.5472] | -309.5% | -407.5% | -0.042 | 67840 |
| 4 | 8 | dynamics_r64 | 107.17 | 4.6744 [4.6452, 4.7044] | -521.8% | -687.0% | -0.011 | 264448 |
| 4 | 8 | lowrank_indep_r16 | 71.07 | 4.2637 [4.2333, 4.2954] | 32.3% | 42.6% | 0.280 | 57344 |
| 4 | 8 | lowrank_indep_r4 | 71.55 | 4.2704 [4.2400, 4.3022] | 23.4% | 30.8% | 0.210 | 14336 |
| 4 | 8 | lowrank_indep_r64 | 70.73 | 4.2589 [4.2284, 4.2905] | 38.9% | 51.3% | 0.350 | 229376 |
| 4 | 8 | lowrank_shared_r16 | 71.29 | 4.2667 [4.2362, 4.2983] | 28.3% | 37.3% | 0.127 | 8192 |
| 4 | 8 | mean | 72.22 | 4.2797 [4.2493, 4.3116] | 10.7% | 14.1% | 0.148 | 1792 |
| 4 | 8 | mean_fitlam | 72.22 | 4.2797 [4.2493, 4.3116] | 10.7% | 14.1% | 0.148 | 1792 |
| 4 | 8 | none | 72.80 | 4.2877 [4.2574, 4.3192] | 0.0% | 0.0% | n/a | 0 |
| 3 | 8 | dynamics_r16 | 103.62 | 4.6408 [4.6092, 4.6723] | -6.8% | -10.2% | 0.043 | 67840 |
| 3 | 8 | dynamics_r64 | 202.28 | 5.3097 [5.2803, 5.3370] | -174.1% | -260.1% | 0.025 | 264448 |
| 3 | 8 | lowrank_indep_r16 | 91.55 | 4.5169 [4.4848, 4.5486] | 24.1% | 36.1% | 0.297 | 57344 |
| 3 | 8 | lowrank_indep_r4 | 94.41 | 4.5476 [4.5155, 4.5793] | 16.5% | 24.6% | 0.228 | 14336 |
| 3 | 8 | lowrank_indep_r64 | 89.08 | 4.4896 [4.4576, 4.5213] | 31.0% | 46.2% | 0.377 | 229376 |
| 3 | 8 | lowrank_shared_r16 | 93.48 | 4.5377 [4.5053, 4.5698] | 18.9% | 28.3% | 0.144 | 8192 |
| 3 | 8 | mean | 96.90 | 4.5737 [4.5421, 4.6047] | 9.9% | 14.8% | 0.167 | 1792 |
| 3 | 8 | mean_fitlam | 96.90 | 4.5737 [4.5421, 4.6047] | 9.9% | 14.8% | 0.167 | 1792 |
| 3 | 8 | none | 100.83 | 4.6134 [4.5821, 4.6445] | 0.0% | 0.0% | n/a | 0 |
| 2 | 8 | dynamics_r16 | 5680.14 | 8.6447 [8.6174, 8.6706] | -68.2% | -91.1% | 0.008 | 67840 |
| 2 | 8 | dynamics_r64 | 5046.51 | 8.5265 [8.5005, 8.5497] | -63.7% | -85.1% | 0.045 | 264448 |
| 2 | 8 | lowrank_indep_r16 | 470.60 | 6.1540 [6.1288, 6.1782] | 26.3% | 35.2% | 0.332 | 57344 |
| 2 | 8 | lowrank_indep_r4 | 567.26 | 6.3408 [6.3169, 6.3646] | 19.2% | 25.7% | 0.247 | 14336 |
| 2 | 8 | lowrank_indep_r64 | 373.53 | 5.9230 [5.8968, 5.9487] | 35.1% | 46.9% | 0.446 | 229376 |
| 2 | 8 | lowrank_shared_r16 | 569.51 | 6.3448 [6.3224, 6.3683] | 19.1% | 25.5% | 0.196 | 8192 |
| 2 | 8 | mean | 649.08 | 6.4756 [6.4525, 6.4988] | 14.1% | 18.9% | 0.184 | 1792 |
| 2 | 8 | mean_fitlam | 649.08 | 6.4756 [6.4525, 6.4988] | 14.1% | 18.9% | 0.184 | 1792 |
| 2 | 8 | none | 941.81 | 6.8478 [6.8226, 6.8748] | 0.0% | 0.0% | n/a | 0 |

## Reading

INT4: naive PPL 72.80, FP 67.60, oracle 68.81; best no-FP compensator = lowrank_indep_r64 closing 38.9% of the CE gap (51.3% of oracle), pred cosine 0.350. INT3: naive PPL 100.83, FP 67.60, oracle 77.14; best no-FP compensator = lowrank_indep_r64 closing 31.0% of the CE gap (46.2% of oracle), pred cosine 0.377. INT2: naive PPL 941.81, FP 67.60, oracle 130.90; best no-FP compensator = lowrank_indep_r64 closing 35.1% of the CE gap (46.9% of oracle), pred cosine 0.446.
