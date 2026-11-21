# RMSNorm — vLLM

Production RMSNorm CUDA kernels from [vLLM](https://github.com/vllm-project/vllm), copied verbatim.

| | |
|---|---|
| Source | [`csrc/layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/f8b29fab414884206460fe63b983ca34fc6e3e9b/csrc/layernorm_kernels.cu) |
| Commit | `f8b29fab414884206460fe63b983ca34fc6e3e9b` |
| License | Apache-2.0 (see [LICENSE](LICENSE)) |

The file is kept as-is for reading; it includes vLLM-internal headers and is not meant to build standalone here.

## What it computes

`y = x / sqrt(mean(x^2) + eps) * w`, one thread block per row (token).

## Where to look

- `rms_norm_kernel` (line 15) — plain RMSNorm. Handles 2D/3D/4D inputs (regular, per-head q/k norm, `[batch, seq, head, dim]`) through a `NUM_DIMS` template parameter. Pass 1 reads the row with vectorized loads and reduces the sum of squares with `cub::BlockReduce`; thread 0 writes `rsqrt` to shared memory; pass 2 scales and writes out.
- `fused_add_rms_norm_kernel` (line 107) — residual add fused with RMSNorm: `residual += x; x = rmsnorm(residual)`, saving one round trip to global memory. Uses packed 16-bit vector types when the width and dtype allow.
- Fallback `fused_add_rms_norm_kernel` (line 180) — scalar path for unaligned or unsupported cases.
- `rms_norm` / `fused_add_rms_norm` (line 221 / 314) — host launchers: pick vector width from alignment, dispatch on dtype, launch.
