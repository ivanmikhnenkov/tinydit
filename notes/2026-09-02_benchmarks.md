# Step-time benchmarks on the RTX PRO 6000 Blackwell Max-Q (300 W), 2026-09-02

`scripts/bench_step.py`: forward + backward + fused AdamW, bf16 autocast, random 32×32×32 latents, random T5 context of length L,
256 image tokens. Peak bf16 matmul measured ~288 TFLOP/s. "110M samples" = 4.1M images × ~27 epochs.
Eager numbers are from the host venv (no C compiler → no torch.compile); compiled numbers from the Docker image (docker/Dockerfile).

| model (dim×depth) | params (per-block adaLN) | L | mode | bs | s/step | img/s | peak GB | MFU | 110M samples |
|---|---|---|---|---|---|---|---|---|---|
| 768×12 (old base) | 158M | 128 | eager | 256 | 0.790 | 324 | 65.7 | 27% | 3.9 d |
| 768×12 | 158M | 128 | **compiled** | 256 | 0.329 | 778 | 34.6 | 66% | 1.6 d |
| 768×16 | 210M | 32 | eager | 256 | 1.010 | 254 | 82.5 | 28% | 5.0 d |
| 768×16 | 210M | 128 | eager | 256 | 1.065 | 240 | 87.1 | 27% | 5.3 d |
| 768×16 | 210M | 32 | compiled | 256 | 0.421 | 607 | 44.0 | 68% | 2.1 d |
| 768×16 | 210M | 64 | compiled | 256 | 0.409 | 626 | 44.6 | 70% | 2.0 d |
| 768×16 | 210M | 128 | compiled | 256 | 0.429 | 596 | 45.8 | 67% | 2.1 d |
| 896×16 | 281M | 128 | eager | 128 | 0.624 | 205 | 53.4 | 31% | 6.2 d |
| 896×16 | 281M | 128 | **compiled** | 192 | 0.416 | 461 | 41.6 | 70% | 2.8 d |
| 1024×16 | 362M | 128 | eager | 128 | 0.740 | 173 | 61.6 | 34% | 7.4 d |

Takeaways: torch.compile = 2.4× and half the memory → on by default (needs gcc → Docker). Text length 32→128 costs ~5% of the DiT step.
adaLN-single removes ~45–60M parameters from these rows without changing compute (768×16 → ~165M, 896×16 → ~220M).
Eager RMSNorm warns "Mismatch dtype between input and weight … cannot dispatch to fused implementation" → cast norm weights to the activation dtype.

`scripts/bench_text.py`: live flan-t5-base encoding of a 256-caption batch, bf16, fixed max_length padding:
L=32 22.7 ms · L=64 36.8 ms · L=128 100.5 ms · L=192 175 ms (54% of the L=128 tokens were padding).
At a 0.43 s compiled step, 100 ms is 23% → encode long and short captions as separate sub-batches at their own lengths and use SDPA/compile for T5.


## Run 1 measured (2026-09-02 21:00)
`run1` config, bs 256, compiled, live T5, cos+dispersive losses: **0.758 s/step = 338 img/s** with the GPU to itself;
1.47 s/step while the pexels2 ingest (FLUX.2 AE encode, 14–56% SM) shared the GPU. Validation at step 500 with the compiled
model triggered 5 extra graph compilations (bs 128 per bucket, ~9 min stall) → val now runs the eager model.
