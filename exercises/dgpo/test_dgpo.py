"""One property of DGPO per test. Run with the skeleton (default) or the reference:
    .venv/bin/python -m pytest exercises/dgpo -q
    DGPO_IMPL=solution .venv/bin/python -m pytest exercises/dgpo -q
"""
import importlib, math, os, sys
import pytest, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
T = importlib.import_module(os.environ.get("DGPO_IMPL", "toy"))


def models(seed=0, hidden=32):
    torch.manual_seed(seed)
    ref = T.Velocity(hidden); ref.requires_grad_(False)
    pol = T.Velocity(hidden); pol.load_state_dict(ref.state_dict())
    return pol, ref


def batch(P=4, G=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(P * G, 2, generator=gen) * 2
    A = T.advantages(T.reward(x0), G)
    x_t, t, target = T.renoise(x0, T.timesteps(10), 4, G, gen)
    return x0, A, x_t, t, target


def nudge(model, scale, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))


# ---------------------------------------------------------------- advantages
def test_advantages_use_the_group_mean_and_the_global_std():
    G, r = 8, torch.randn(40) * 3 + 1
    A = T.advantages(r, G)
    assert A.shape == r.shape
    assert torch.allclose(A.view(-1, G).sum(1), torch.zeros(5), atol=1e-5)          # zero-mean inside each group
    expect = (r.view(-1, G) - r.view(-1, G).mean(1, keepdim=True)) / (r.std() + 1e-4)
    assert torch.allclose(A, expect.reshape(-1), atol=1e-5)


def test_positive_and_negative_mass_are_equal_inside_a_group():
    # sum_{A>0} |A| == sum_{A<=0} |A|: the property that cancels the partition function of the group preference
    A = T.advantages(torch.randn(64), 16).view(-1, 16)
    assert torch.allclose(A.clamp(min=0).sum(1), (-A.clamp(max=0)).sum(1), atol=1e-5)


# ---------------------------------------------------------------- re-noising
def test_renoise_shares_t_and_noise_inside_a_group():
    G, gen = 4, torch.Generator().manual_seed(1)
    x0, ts = torch.randn(12, 2, generator=gen), T.timesteps(10)
    x_t, t, target = T.renoise(x0, ts, 4, G, gen)
    assert x_t.shape == x0.shape and t.shape == (12,) and target.shape == x0.shape
    assert all(any(math.isclose(float(v), float(ts[j]), abs_tol=1e-6) for j in range(5)) for v in t)   # one of steps 0..4
    assert torch.allclose(t.view(-1, G), t.view(-1, G)[:, :1].expand(-1, G))                             # shared t
    eps = x0 - target                                                                                     # target = x0 - eps
    assert torch.allclose(eps.view(-1, G, 2), eps.view(-1, G, 2)[:, :1].expand(-1, G, -1), atol=1e-6)    # shared eps
    assert torch.allclose(x_t, t[:, None] * x0 + (1 - t[:, None]) * eps, atol=1e-6)


def test_renoise_with_trunc_zero_sits_at_pure_noise():
    gen = torch.Generator().manual_seed(2); x0 = torch.randn(8, 2, generator=gen)
    x_t, t, target = T.renoise(x0, T.timesteps(10), 0, 4, gen)
    assert torch.all(t == 0) and torch.allclose(x_t, x0 - target, atol=1e-6)


def test_renoise_draws_different_steps_for_different_groups():
    gen = torch.Generator().manual_seed(3); x0 = torch.randn(400, 2, generator=gen)
    _, t, _ = T.renoise(x0, T.timesteps(10), 4, 4, gen)
    assert len(set(round(float(v), 4) for v in t)) == 5


# ---------------------------------------------------------------- the loss
def test_dsm_is_the_per_sample_mse():
    pol, _ = models(); x0, A, x_t, t, target = batch()
    d = T.dsm(pol, x_t, t, target)
    assert d.shape == (32,) and torch.allclose(d, ((pol(x_t, t) - target) ** 2).mean(1), atol=1e-6)


def test_gate_is_half_and_kl_zero_when_policy_equals_reference():
    pol, ref = models(); x0, A, x_t, t, target = batch()
    loss, st = T.dgpo_loss(pol, ref, x_t, t, target, A, 8)
    assert abs(st["margin"]) < 1e-6 and abs(st["w"] - 0.5) < 1e-6 and st["kl"] < 1e-10 and st["clipfrac"] == 0


def test_gradient_matches_the_textbook_objective():
    # The paper's loss is softplus(margin) with margin = beta/G * sum_i A_i (dsm_i - ref_dsm_i), gradient
    # sigma(margin) * beta/G * sum_i A_i grad dsm_i. The code computes mean_i(w * A_i * dsm_i) with w = sigma(margin)
    # detached: the same gradient without the constant beta. So grad_code * beta == grad_textbook.
    pol, ref = models(); x0, A, x_t, t, target = batch(P=1, G=16)
    nudge(pol, 0.05)                                            # away from the reference so the margin is not 0
    beta = 100.0
    loss, _ = T.dgpo_loss(pol, ref, x_t, t, target, A, 16, None, beta, 1e-2, 0.0)
    g_code = torch.autograd.grad(loss, list(pol.parameters()))
    d = T.dsm(pol, x_t, t, target); d_ref = T.dsm(ref, x_t, t, target).detach()
    textbook = torch.nn.functional.softplus(beta * (A * (d - d_ref)).sum() / 16)
    g_text = torch.autograd.grad(textbook, list(pol.parameters()))
    for a, b in zip(g_code, g_text):
        assert torch.allclose(a * beta, b, rtol=1e-4, atol=1e-7)


def test_gate_closes_as_the_policy_learns_the_preference():
    pol, ref = models(); x0, A, x_t, t, target = batch(P=2, G=16)
    opt = torch.optim.Adam(pol.parameters(), 1e-2); ws = []
    for _ in range(40):
        loss, st = T.dgpo_loss(pol, ref, x_t, t, target, A, 16, None, 100.0, 1e-2, 0.0)
        opt.zero_grad(); loss.backward(); opt.step(); ws.append(st["w"])
    assert ws[0] == pytest.approx(0.5, abs=1e-6) and ws[-1] < 0.2 and st["margin"] < 0
    d = T.dsm(pol, x_t, t, target).detach(); d_ref = T.dsm(ref, x_t, t, target)
    assert ((d - d_ref)[A > 0]).mean() < 0 < ((d - d_ref)[A <= 0]).mean()   # better on positives, worse on negatives


def test_trust_region_removes_clipped_samples_from_the_gradient():
    pol, ref = models(); x0, A, x_t, t, target = batch(P=1, G=16)
    nudge(pol, 0.05)
    d = T.dsm(pol, x_t, t, target).detach()
    same, _ = T.dgpo_loss(pol, ref, x_t, t, target, A, 16, d.clone(), 100.0, 1e-2, 0.0)   # ratio == 1: nothing clipped
    assert _["clipfrac"] == 0
    old = d.clone(); old[A > 0] += 1.0                          # positives look far more likely than at sampling time
    loss, st = T.dgpo_loss(pol, ref, x_t, t, target, A, 16, old, 100.0, 1e-2, 0.0)
    assert st["clipfrac"] == pytest.approx(float((A > 0).float().mean()))
    g = torch.autograd.grad(loss, list(pol.parameters()))
    dd = T.dsm(pol, x_t, t, target)
    manual = (st["w"] * A * torch.where(A > 0, dd.detach(), dd)).mean()     # same gate, positives switched off
    g_m = torch.autograd.grad(manual, list(pol.parameters()))
    for a, b in zip(g, g_m):
        assert torch.allclose(a, b, atol=1e-7)
    old = d.clone(); old[A <= 0] -= 1.0                         # negatives look far less likely: clipped the other way
    _, st = T.dgpo_loss(pol, ref, x_t, t, target, A, 16, old, 100.0, 1e-2, 0.0)
    assert st["clipfrac"] == pytest.approx(float((A <= 0).float().mean()))


def test_kl_term_is_the_velocity_gap_and_pulls_toward_the_reference():
    pol, ref = models(); x0, A, x_t, t, target = batch()
    nudge(pol, 0.1)
    loss0, st0 = T.dgpo_loss(pol, ref, x_t, t, target, A, 8, None, 100.0, 1e-2, 0.0)
    loss1, st1 = T.dgpo_loss(pol, ref, x_t, t, target, A, 8, None, 100.0, 1e-2, 1.0)
    assert st0["kl"] > 0 and float((loss1 - loss0).detach()) == pytest.approx(st0["kl"], abs=1e-6)
    opt = torch.optim.SGD(pol.parameters(), 0.1)
    for _ in range(20):                                          # zero advantages: only the KL term is left
        l, st = T.dgpo_loss(pol, ref, x_t, t, target, torch.zeros_like(A), 8, None, 100.0, 1e-2, 1.0)
        opt.zero_grad(); l.backward(); opt.step()
    assert st["kl"] < st0["kl"]


# ---------------------------------------------------------------- end to end
def test_toy_moves_mass_to_the_rewarded_mode():
    torch.manual_seed(0); gen = torch.Generator().manual_seed(0)
    ref = T.pretrain(T.Velocity(), steps=800, seed=0); ref.requires_grad_(False)
    pol = T.Velocity(); pol.load_state_dict(ref.state_dict())
    opt = torch.optim.Adam(pol.parameters(), 3e-4)
    base = float(T.reward(T.ode_sample(ref, 2000, gen=gen)).mean())
    for _ in range(60):
        st = T.dgpo_round(pol, ref, opt, beta_kl=0.02, gen=gen)
    after = float(T.reward(T.ode_sample(pol, 2000, gen=gen)).mean())
    assert after > base + 0.5, (base, after)
    assert st["w"] < 0.45
