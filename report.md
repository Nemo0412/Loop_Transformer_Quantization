# Cross-loop quantization error feedback

Proof of concept on frozen `TRM-LM-v2-10M` (`trm2.pt`). The question is whether quantization error left by earlier passes through the shared core can be estimated and removed in later passes, without retraining the language model.

## Setup

The network is not a single shared matrix. `modeling_trm2.py` runs embedding, one prelude block, then `loops` passes of a five-block core, then one coda block. Each core pass is `h = RMS(h) * gain[i] + prelude + tag[i]` followed by the same five pre-norm attention/SwiGLU blocks. Those five blocks stay one module list: each Linear is quantized once and reused on every loop.

Weight-only asymmetric per-group round-to-nearest, group size 128, along the input features of every `nn.Linear` in the prelude, core, and coda. The down projection has 704 input features, so its last group has 64 weights. Embeddings, the tied head, RMS scales, and the loop gain/tag stay FP32. Quantized weights are stored conceptually as packed integers plus FP16 scales and INT8 zero-points, but the forward pass multiplies the dequantized FP32 matrix. **No low-bit kernel is used.** TF32 is off. Latency below is that FP32 simulation, not an integer-GEMM speedup.

Evaluation tokens are the WikiText-2 raw test split, packed in non-overlapping windows of 1024 (sha256 `ff8b6d507cfc`, 324291 next-token predictions). Calibration for the linear corrector, when it is fit, is the first 128 windows of the validation split. Seed 42. Cross-entropy is in nats over the 16,384-token BPE; perplexity is `exp(CE)`.

Device: NVIDIA L40S.

## Experiment 1: baseline RTN

| method | loops | CE | PPL | logits MSE | storage MiB | ms/token |
|---|---:|---:|---:|---:|---:|---:|
| fp | 1 | 4.3123 | 74.614 | 0.0000 | 21.438 | 0.0012 |
| fp | 2 | 4.1819 | 65.488 | 0.0000 | 21.438 | 0.0019 |
| fp | 3 | 4.1604 | 64.100 | 0.0000 | 21.438 | 0.0026 |
| fp | 4 | 4.1609 | 64.127 | 0.0000 | 21.438 | 0.0032 |
| fp | 6 | 4.1846 | 65.664 | 0.0000 | 21.438 | 0.0046 |
| int4 | 1 | 4.3655 | 78.691 | 0.1639 | 2.808 | 0.0012 |
| int4 | 2 | 4.2367 | 69.177 | 0.1650 | 2.808 | 0.0019 |
| int4 | 3 | 4.2171 | 67.837 | 0.1714 | 2.808 | 0.0026 |
| int4 | 4 | 4.2200 | 68.035 | 0.1802 | 2.808 | 0.0032 |
| int4 | 6 | 4.2509 | 70.166 | 0.2146 | 2.808 | 0.0046 |
| int3 | 1 | 4.5792 | 97.441 | 0.7112 | 2.138 | 0.0012 |
| int3 | 2 | 4.4794 | 88.185 | 0.7182 | 2.138 | 0.0019 |
| int3 | 3 | 4.4846 | 88.644 | 0.7463 | 2.138 | 0.0026 |
| int3 | 4 | 4.5062 | 90.579 | 0.7829 | 2.138 | 0.0032 |
| int3 | 6 | 4.5633 | 95.902 | 0.9224 | 2.138 | 0.0045 |
| int2 | 1 | 6.2131 | 499.264 | 3.9909 | 1.468 | 0.0012 |
| int2 | 2 | 6.1728 | 479.509 | 4.0423 | 1.468 | 0.0019 |
| int2 | 3 | 6.2807 | 534.181 | 4.1975 | 1.468 | 0.0025 |
| int2 | 4 | 6.4000 | 601.855 | 4.3868 | 1.468 | 0.0032 |
| int2 | 6 | 6.6762 | 793.335 | 4.9943 | 1.468 | 0.0045 |

Storage is the theoretical packed size of the quantized Linear weights, including scales and zero-points, counted once even though the core is executed several times. The FP row uses the FP32 size of those same Linear weights.
Mean quantized-simulation latency is 1.00× the FP32 latency. That ratio is kernel noise around a dense FP32 matmul, not a compression speedup.

## Experiment 2: where the error comes from

Hidden states are recorded after each five-block core pass. Local error is the one-step difference between a quantized core and an FP core on the same FP input state. Propagated error is the rest: the quantized core applied to the live quantized state, minus the quantized core applied to the FP state. The two parts add to the total error.

| bits | loop | RMS | relative RMS | cosine vs previous | local/total | propagated/total |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 0.2767 | 0.1487 | n/a | 0.755 | 0.672 |
| 4 | 2 | 0.3216 | 0.1518 | 0.859 | 0.507 | 0.734 |
| 4 | 3 | 0.3556 | 0.1573 | 0.959 | 0.430 | 0.758 |
| 4 | 4 | 0.3750 | 0.1628 | 0.982 | 0.403 | 0.769 |
| 4 | 5 | 0.3826 | 0.1716 | 0.987 | 0.408 | 0.769 |
| 4 | 6 | 0.4001 | 0.1791 | 0.992 | 0.394 | 0.781 |
| 3 | 1 | 0.6005 | 0.3227 | n/a | 0.761 | 0.695 |
| 3 | 2 | 0.7035 | 0.3320 | 0.867 | 0.513 | 0.747 |
| 3 | 3 | 0.7764 | 0.3434 | 0.962 | 0.437 | 0.770 |
| 3 | 4 | 0.8159 | 0.3542 | 0.983 | 0.411 | 0.782 |
| 3 | 5 | 0.8273 | 0.3710 | 0.989 | 0.418 | 0.783 |
| 3 | 6 | 0.8592 | 0.3846 | 0.993 | 0.406 | 0.795 |
| 2 | 1 | 1.4268 | 0.7667 | n/a | 0.778 | 0.821 |
| 2 | 2 | 1.7215 | 0.8123 | 0.891 | 0.517 | 0.864 |
| 2 | 3 | 1.9034 | 0.8419 | 0.964 | 0.440 | 0.878 |
| 2 | 4 | 1.9963 | 0.8668 | 0.981 | 0.416 | 0.886 |
| 2 | 5 | 2.0198 | 0.9058 | 0.986 | 0.425 | 0.893 |
| 2 | 6 | 2.0867 | 0.9341 | 0.990 | 0.414 | 0.903 |

Plot: `plots/error_vs_loop.png`.

Total RMS at the last loop relative to loop 1: INT2 1.46×, INT3 1.43×, INT4 1.45×.
Mean consecutive-loop error cosine is 0.959. A high value means the error direction is stable enough that a corrector could transfer across loops; a value near zero means each loop's error points somewhere else.

## Experiment 3: oracle state correction

This is an offline diagnostic. The FP hidden state is available only here. After a selected core boundary the run continues from `quantized + alpha * (fp - quantized)`, and every later core pass is quantized again. The injected error is recomputed from the live state, so an earlier correction changes the later target. Alpha 0 reproduces the uncorrected baseline.

PPL at 6 loops, or at the largest loop count in this run:

| bits | mode | alpha | PPL | gap closed | final residual RMS | subsequent core loops |
|---:|---|---:|---:|---:|---:|---:|
| 4 | after_loop_1 | 0.00 | 70.166 | 0.0% | 0.4001 | 5 |
| 4 | after_loop_1 | 0.50 | 69.789 | 8.4% | 0.3809 | 5 |
| 4 | after_loop_1 | 1.00 | 69.499 | 14.8% | 0.3653 | 5 |
| 4 | after_loop_2 | 0.00 | 70.166 | 0.0% | 0.4001 | 4 |
| 4 | after_loop_2 | 0.50 | 69.471 | 15.4% | 0.3656 | 4 |
| 4 | after_loop_2 | 1.00 | 68.957 | 26.8% | 0.3380 | 4 |
| 4 | every_except_final | 0.00 | 70.166 | 0.0% | 0.4001 | 5 |
| 4 | every_except_final | 0.50 | 67.354 | 62.5% | 0.2262 | 5 |
| 4 | every_except_final | 1.00 | 66.787 | 75.1% | 0.1702 | 5 |
| 3 | after_loop_1 | 0.00 | 95.902 | 0.0% | 0.8592 | 5 |
| 3 | after_loop_1 | 0.50 | 93.136 | 9.1% | 0.8204 | 5 |
| 3 | after_loop_1 | 1.00 | 91.127 | 15.8% | 0.7907 | 5 |
| 3 | after_loop_2 | 0.00 | 95.902 | 0.0% | 0.8592 | 4 |
| 3 | after_loop_2 | 0.50 | 91.156 | 15.7% | 0.7902 | 4 |
| 3 | after_loop_2 | 1.00 | 87.904 | 26.5% | 0.7353 | 4 |
| 3 | every_except_final | 0.00 | 95.902 | 0.0% | 0.8592 | 5 |
| 3 | every_except_final | 0.50 | 77.972 | 59.3% | 0.4979 | 5 |
| 3 | every_except_final | 1.00 | 74.742 | 70.0% | 0.3768 | 5 |
| 2 | after_loop_1 | 0.00 | 793.335 | 0.0% | 2.0867 | 5 |
| 2 | after_loop_1 | 0.50 | 623.198 | 23.4% | 1.9684 | 5 |
| 2 | after_loop_1 | 1.00 | 535.827 | 35.4% | 1.9028 | 5 |
| 2 | after_loop_2 | 0.00 | 793.335 | 0.0% | 2.0867 | 4 |
| 2 | after_loop_2 | 0.50 | 533.601 | 35.7% | 1.9040 | 4 |
| 2 | after_loop_2 | 1.00 | 409.626 | 52.7% | 1.7792 | 4 |
| 2 | every_except_final | 0.00 | 793.335 | 0.0% | 2.0867 | 5 |
| 2 | every_except_final | 0.50 | 167.572 | 86.0% | 1.2303 | 5 |
| 2 | every_except_final | 1.00 | 125.640 | 91.8% | 0.9299 | 5 |

Oracle recovery is meaningful: INT2 loops=6 mode=every_except_final alpha=1.0 closes 91.8% of a PPL gap of 727.671.

## Experiment 4: low-rank linear correction

The corrector is `predicted_error = U @ V.T @ h`, fit by reduced-rank regression on validation pairs `(quantized state, fp state - quantized state)` only. Pretrained weights stay frozen. Test evaluation follows the corrected quantized trajectory. It does not teacher-force FP states. The oracle column is alpha-1 correction after every non-final loop.

| bits | loops | method | rank | PPL | gap closed | fraction of oracle | pred MSE | pred cosine | extra params | latency ratio |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3 | none | - | 67.837 | 0.0% | 0.00 | n/a | n/a | 0 | 1.000 |
| 4 | 3 | oracle | - | 65.108 | 73.0% | 1.00 | n/a | n/a | 0 | n/a |
| 4 | 3 | shared | 16 | 66.580 | 33.6% | 0.46 | 0.08048 | 0.261 | 8192 | 1.017 |
| 4 | 3 | independent | 16 | 66.591 | 33.3% | 0.46 | 0.07673 | 0.308 | 16384 | 1.020 |
| 4 | 6 | none | - | 70.166 | 0.0% | 0.00 | n/a | n/a | 0 | 1.000 |
| 4 | 6 | oracle | - | 66.787 | 75.1% | 1.00 | n/a | n/a | 0 | n/a |
| 4 | 6 | shared | 16 | 68.456 | 38.0% | 0.51 | 0.10904 | 0.137 | 8192 | 1.027 |
| 4 | 6 | independent | 16 | 68.462 | 37.8% | 0.50 | 0.10618 | 0.173 | 40960 | 1.031 |
| 3 | 3 | none | - | 88.644 | 0.0% | 0.00 | n/a | n/a | 0 | 1.000 |
| 3 | 3 | oracle | - | 72.677 | 65.1% | 1.00 | n/a | n/a | 0 | n/a |
| 3 | 3 | shared | 16 | 82.115 | 26.6% | 0.41 | 0.37146 | 0.288 | 8192 | 1.024 |
| 3 | 3 | independent | 16 | 82.248 | 26.1% | 0.40 | 0.35504 | 0.331 | 16384 | 1.024 |
| 3 | 6 | none | - | 95.902 | 0.0% | 0.00 | n/a | n/a | 0 | 1.000 |
| 3 | 6 | oracle | - | 74.742 | 70.0% | 1.00 | n/a | n/a | 0 | n/a |
| 3 | 6 | shared | 16 | 87.645 | 27.3% | 0.39 | 0.49727 | 0.171 | 8192 | 1.042 |
| 3 | 6 | independent | 16 | 87.639 | 27.3% | 0.39 | 0.48603 | 0.199 | 40960 | 1.044 |
| 2 | 3 | none | - | 534.181 | 0.0% | 0.00 | n/a | n/a | 0 | 1.000 |
| 2 | 3 | oracle | - | 117.796 | 88.6% | 1.00 | n/a | n/a | 0 | n/a |
| 2 | 3 | shared | 16 | 347.510 | 39.7% | 0.45 | 2.05098 | 0.325 | 8192 | 1.027 |
| 2 | 3 | independent | 16 | 339.562 | 41.4% | 0.47 | 1.94613 | 0.364 | 16384 | 1.029 |
| 2 | 6 | none | - | 793.335 | 0.0% | 0.00 | n/a | n/a | 0 | 1.000 |
| 2 | 6 | oracle | - | 125.640 | 91.8% | 1.00 | n/a | n/a | 0 | n/a |
| 2 | 6 | shared | 16 | 468.795 | 44.6% | 0.49 | 2.63591 | 0.231 | 8192 | 1.038 |
| 2 | 6 | independent | 16 | 456.720 | 46.3% | 0.50 | 2.56693 | 0.242 | 40960 | 1.042 |

Latency ratio is corrected FP32 time over uncorrected quantized FP32 time. Extra FLOPs are `4 * D * rank` per corrected boundary per token (two rank-r matvecs). They are not measured inside an integer kernel.

## Is cross-loop correction promising?

Across bit-widths, total hidden-state RMS changes by 1.43× to 1.46× from the first loop to the last. At the last loop, local one-step error is 0.39–0.41× the total RMS (values above 1 mean the propagated part points against the local part). The strongest full-state cross-loop oracle is INT2 at 6 loops, which closes 91.8% of the uncorrected perplexity gap (793.335 → 125.640, FP 65.664). The best rank-16 independent corrector (INT2, 6 loops) closes 46.3% of the FP gap and 50% of what the oracle recovered, with 40960 extra FP32 parameters. This is promising enough to study further: a frozen low-rank linear map, fit without touching the pretrained weights, captures a real share of the oracle gain. It remains a fake-quant result. It does not show that INT2/INT3/INT4 kernels would be faster, and it is not a training recipe for the 10M model.

