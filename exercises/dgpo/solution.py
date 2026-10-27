"""DGPO in two dimensions: the reference solution. Do not read before finishing toy.py.

A flow-matching model over a 2-D point cloud stands in for the DiT: same t-orientation as tinydit
(t = 0 noise, t = 1 data, v = x0 - eps), the same 10-step Euler ODE sampler, the same DGPO loss as
src/tinydit/rl/dgpo.py. There is no prompt: a group is simply G independent samples. The reward is
r(x) = x[0], "prefer the right-hand mode", so a correct implementation moves probability mass from
the left mode to the right one, the gate sigma(margin) falls from 0.5, and the KL weight decides how
much of the left mode survives.

Implement the four functions marked TODO (advantages, renoise, dsm, dgpo_loss), then

    .venv/bin/python -m pytest exercises/dgpo -q        the tests, one property each
    .venv/bin/python exercises/dgpo/toy.py              reward per round + toy_dgpo.png before/after

solution.py is the reference; the tests run against it with DGPO_IMPL=solution.
"""
from __future__ import annotations
import math, os
import torch, torch.nn as nn


# ---------------------------------------------------------------- the stand-in for the DiT
class Velocity(nn.Module):
    """v(x, t): (N,2),(N,) -> (N,2). Sinusoidal t features, two hidden layers."""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 8, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t):
        f = (2 ** torch.arange(4, device=x.device).float()) * math.pi
        te = torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], 1)
        return self.net(torch.cat([x, te], 1))


def data(n, gen):
    """Two equal modes on the x-axis at -2 and +2."""
    m = torch.where(torch.rand(n, generator=gen) < 0.5, -2.0, 2.0)
    return torch.stack([m, torch.zeros(n)], 1) + 0.3 * torch.randn(n, 2, generator=gen)


def pretrain(model, steps=3000, bs=256, lr=2e-3, seed=0):
    """Rectified-flow pretraining: x_t = (1-t) eps + t z, target z - eps. A few seconds on CPU."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(steps):
        z = data(bs, gen); e = torch.randn(bs, 2, generator=gen); t = torch.rand(bs, generator=gen)
        x_t = (1 - t[:, None]) * e + t[:, None] * z
        loss = ((model(x_t, t) - (z - e)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def timesteps(steps):
    return torch.linspace(0, 1, steps + 1)


@torch.no_grad()
def ode_sample(model, n, steps=10, gen=None):
    """Euler from noise to data. Deterministic given the noise: nothing here has a density."""
    ts = timesteps(steps)
    x = torch.randn(n, 2, generator=gen)
    for i in range(steps):
        x = x + model(x, ts[i].expand(n)) * (ts[i + 1] - ts[i])
    return x


def reward(x):
    return x[:, 0]


# ---------------------------------------------------------------- TODO 1 / 4
def advantages(r, G):
    """r: (N,) rewards of N = groups * G samples, groups contiguous -> A: (N,)
    A_i = (r_i - mean of its group) / (std over all N + 1e-4). Group mean, global std: both papers' default."""
    rg = r.view(-1, G)
    return ((rg - rg.mean(1, keepdim=True)) / (r.std() + 1e-4)).reshape(-1)


# ---------------------------------------------------------------- TODO 2 / 4
def renoise(x0, ts, trunc, G, gen):
    """One sampler step j in {0, ..., trunc} per group and one noise eps per group (shared by its G
    samples); then x_t = t_j x0 + (1 - t_j) eps and target = x0 - eps.
    x0: (N,2) clean ODE samples | ts: the sampler's timesteps | -> x_t (N,2), t (N,), target (N,2)"""
    ng = x0.shape[0] // G
    j = torch.randint(0, trunc + 1, (ng,), generator=gen).repeat_interleave(G)
    eps = torch.randn(ng, x0.shape[1], generator=gen).repeat_interleave(G, 0)
    t = ts[j]
    return t[:, None] * x0 + (1 - t[:, None]) * eps, t, x0 - eps


# ---------------------------------------------------------------- TODO 3 / 4
def dsm(model, x_t, t, target):
    """Per-sample flow-matching loss || v(x_t, t) - target ||^2, averaged over the 2 coordinates -> (N,)"""
    return ((model(x_t, t) - target) ** 2).mean(1)


# ---------------------------------------------------------------- TODO 4 / 4
def dgpo_loss(policy, ref, x_t, t, target, adv, G, old_dsm=None, beta_dpo=3.0, clip=1e-2, beta_kl=0.02):
    """The group preference loss, exactly as in src/tinydit/rl/dgpo.py:
        dsm_i     = dsm(policy)                                     with gradient
        ref_dsm_i = dsm(ref)                                        no gradient
        margin_g  = beta_dpo * sum_{i in g} A_i (dsm_i - ref_dsm_i) / G       detached, one per group
        w_g       = sigmoid(margin_g)                               the gate, 0.5 when policy == ref
        trust region (only when old_dsm is given): ratio_i = exp(old_dsm_i - dsm_i); sample i is clipped when
                      A_i > 0 and ratio_i > 1 + clip, or A_i <= 0 and ratio_i < 1 - clip; clipped samples
                      contribute dsm_i.detach() (no gradient)
        loss      = mean_i( w_g(i) * A_i * dsm_i )  +  beta_kl * mean_i || v_policy(x_t,t) - v_ref(x_t,t) ||^2
    -> loss (scalar tensor), stats dict of floats: margin (mean over groups), w (mean over samples),
       clipfrac (fraction clipped), kl (mean of the velocity gap), dsm (mean)"""
    v = policy(x_t, t)
    with torch.no_grad():
        v_ref = ref(x_t, t)
    d = ((v - target) ** 2).mean(1)
    d_ref = ((v_ref - target) ** 2).mean(1)
    margin = (adv * beta_dpo * (d - d_ref)).detach().view(-1, G).sum(1) / G
    w = torch.sigmoid(margin).repeat_interleave(G)
    if old_dsm is None:
        clipped = torch.zeros_like(adv, dtype=torch.bool)
    else:
        ratio = torch.exp(old_dsm - d.detach())
        clipped = torch.where(adv > 0, ratio > 1 + clip, ratio < 1 - clip)
    d_c = torch.where(clipped, d.detach(), d)
    pref = (w * adv * d_c).mean()
    kl = ((v - v_ref) ** 2).mean(1)
    loss = pref + beta_kl * kl.mean()
    return loss, {"margin": float(margin.mean()), "w": float(w.mean()), "clipfrac": float(clipped.float().mean()),
                  "kl": float(kl.detach().mean()), "dsm": float(d.detach().mean())}


# ---------------------------------------------------------------- the loop (given)
def dgpo_round(policy, ref, opt, P=8, G=32, steps=10, trunc=4, inner=2, beta_dpo=3.0, clip=1e-2, beta_kl=0.02, gen=None):
    """One round: P groups of G ODE samples -> reward -> advantages -> one re-noised timestep per group ->
    `inner` optimizer steps on that batch (the second one is where the trust region can clip).
    beta_dpo is 3 here, not the paper's 100: the deltas of a 2-D loss are ~30x larger than the DiT's per-dimension
    ones, and the margin is beta x delta. With 100 the gate closes in the first round and nothing moves (try it)."""
    N = P * G
    x0 = ode_sample(policy, N, steps, gen)
    r = reward(x0); A = advantages(r, G)
    x_t, t, target = renoise(x0, timesteps(steps), trunc, G, gen)
    with torch.no_grad():
        old = dsm(policy, x_t, t, target)
    for _ in range(inner):
        loss, st = dgpo_loss(policy, ref, x_t, t, target, A, G, old, beta_dpo, clip, beta_kl)
        opt.zero_grad(); loss.backward(); opt.step()
    st.update(reward=float(r.mean()), frac_right=float((x0[:, 0] > 0).float().mean()), loss=float(loss.detach()))
    return st


def main(rounds=60, beta_kl=0.02, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    ref = pretrain(Velocity(), seed=seed); ref.requires_grad_(False)
    policy = Velocity(); policy.load_state_dict(ref.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    before = ode_sample(ref, 2000, gen=gen)
    print(f"reference: reward {float(reward(before).mean()):+.3f}, right-mode fraction {float((before[:, 0] > 0).float().mean()):.2f}")
    for k in range(rounds):
        st = dgpo_round(policy, ref, opt, beta_kl=beta_kl, gen=gen)
        if k % 5 == 0 or k == rounds - 1:
            print(f"round {k:3d}  reward {st['reward']:+.3f}  right-mode {st['frac_right']:.2f}  gate {st['w']:.3f}  "
                  f"margin {st['margin']:+.2f}  kl {st['kl']:.4f}  clipped {st['clipfrac']:.2f}")
    after = ode_sample(policy, 2000, gen=gen)
    print(f"after: reward {float(reward(after).mean()):+.3f}, right-mode fraction {float((after[:, 0] > 0).float().mean()):.2f}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        for a, pts, name in ((ax[0], before, "reference"), (ax[1], after, f"after {rounds} DGPO rounds, beta_kl {beta_kl}")):
            a.scatter(pts[:, 0], pts[:, 1], s=3, alpha=.5); a.set_title(name); a.set_xlim(-4, 4); a.set_ylim(-3, 3)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_dgpo.png"); fig.savefig(out, dpi=120); print("wrote", out)
    except ImportError:
        pass


if __name__ == "__main__":
    import sys
    main(beta_kl=float(sys.argv[1]) if len(sys.argv) > 1 else 0.02)
