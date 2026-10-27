# Exercise: DGPO from the loss up

Goal: implement the four pieces of DGPO's loss in a 2-D toy where every effect is visible in seconds,
verify each against a test that states the property in plain words, then compare with the real
implementation in `src/tinydit/rl/dgpo.py`. Read the paper (arXiv 2510.08425) first or alongside;
the report has the derivation in six lines.

## The setup
`toy.py` gives you the DiT stand-in (`Velocity`, a 2-D flow-matching MLP), rectified-flow pretraining on
two modes at x = −2 and x = +2, the 10-step Euler ODE sampler, reward r(x) = x[0] ("prefer the right mode"),
and the round loop `dgpo_round`. You implement:

| function | what it must do | tests |
|---|---|---|
| `advantages(r, G)` | group mean, global std | zero-mean per group; positive and negative mass equal (this is what cancels the partition function) |
| `renoise(x0, ts, trunc, G, gen)` | one step j ≤ trunc and one eps per group, x_t = t·x0 + (1−t)·eps, target = x0 − eps | shared t and eps inside a group; endpoints; all five steps get drawn |
| `dsm(model, x_t, t, target)` | per-sample velocity MSE | shape and value |
| `dgpo_loss(...)` | margin, gate, trust region, KL | gate 0.5 at init; gradient equals the textbook softplus objective (× β); gate closes as the policy learns; clipped samples drop out of the gradient; KL is the velocity gap and pulls back |

Then the end-to-end test: 40 rounds must move mass to the right mode and close the gate.

```
.venv/bin/python -m pytest exercises/dgpo -q                 # your implementation
DGPO_IMPL=solution .venv/bin/python -m pytest exercises/dgpo -q   # the reference (do this after)
.venv/bin/python exercises/dgpo/toy.py 0.02                  # run it; argument = beta_kl; writes toy_dgpo.png
```

## Questions to answer once it passes (write the answers down)
0. The toy uses β = 3 where the paper uses 100. Print the deltas (dsm − ref_dsm) in both settings and explain why the same β cannot serve both.
1. Run `toy.py` with beta_kl 0, 0.02, 0.2. Where does the left mode end up in each case, and what does the gate do? Relate to the stage-1 runs, where the real model gave its gain back.
2. Set `trunc=9` in `dgpo_round` (all timesteps). Does it still move mass? Faster or slower? Why does the paper train only on the noisiest steps?
3. Remove the shared noise (`eps` per sample instead of per group). Watch the margin's variance across rounds.
4. Make the reference an EMA of the policy (the paper's θ⁻ mode): `ref` ← 0.3·ref + 0.7·policy after each round. What happens to the gate, and to the left mode, with beta_kl 0.02?
5. Why does the code's gradient equal the textbook gradient only up to the factor β? Why does that not matter with Adam, and when would it?
6. The trust region compares `exp(old_dsm − dsm)` to 1 ± clip. What is this ratio an estimate of, and why is PPO's clip transplanted onto it?
7. Compare with `src/tinydit/rl/dgpo.py`: what is different because there is a prompt, CFG, and 32×32×32 latents?
