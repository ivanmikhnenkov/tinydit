"""Step schedules and ODE solvers for rectified-flow sampling.

Two independent knobs, easy to conflate:

  schedule -- WHERE the steps sit in t. Uniform spacing is wasteful here: the residual
              ||x1_hat(t) - x1|| decays like (1-t)^2, so late steps barely move the result.
              Every schedule below concentrates steps near t=0 (the noisy end), where
              global structure is decided.

  solver   -- HOW each step is taken. All three below integrate the same field; they
              differ only in how they estimate the average velocity across a step.

                euler  1st order, 1 call/step. Assumes the start velocity holds all the
                       way across. The rectangle rule.
                heun   2nd order, 2 calls/step. Probes the far end, averages the two.
                       The trapezoid rule. This is EDM's reference sampler.
                ab2    2nd order, 1 call/step. Gets the same curvature information from
                       the *previous* step instead of paying for a second call, so it is
                       the cheaper way to be second-order. Variable-step Adams-Bashforth;
                       the same idea DPM-Solver++(2M) and UniPC build on, though those
                       derive their coefficients in a log-SNR parameterisation and are not
                       identical to this.

              Compare solvers at equal NFE (function evaluations), never at equal steps.

t runs 0 (pure noise) -> 1 (data), the convention used throughout this repo.
"""
from __future__ import annotations
import math
import torch


def uniform(n, p=None, device="cuda"):
    """Even spacing. The baseline, and what the training eval grids use."""
    return torch.linspace(0, 1, n + 1, device=device)


def power(n, k=2.0, device="cuda"):
    """t = u^k. k=2 matches the measured (1-t)^2 decay of the residual."""
    return torch.linspace(0, 1, n + 1, device=device) ** float(k)


def shift(n, s=3.0, device="cuda"):
    """The SD3/FLUX timestep shift, written for our t-orientation.

    tau' = s*tau / (1 + (s-1)*tau) applied to the noise-side variable tau = 1-t.
    Production models raise s with resolution, since more pixels need more of the
    budget spent settling global structure."""
    u = torch.linspace(0, 1, n + 1, device=device)
    tau = 1 - u
    tau = s * tau / (1 + (s - 1) * tau)
    return 1 - tau


def karras(n, rho=7.0, device="cuda"):
    """Karras et al. (EDM, 2022), mapped through sigma = (1-t)/t.

    Designed so the *truncation error per step* is roughly equal along the path, rather
    than the step size. rho=7 is their empirical optimum."""
    t_lo, t_hi = 1e-3, 1 - 1e-3
    s_max, s_min = (1 - t_lo) / t_lo, (1 - t_hi) / t_hi
    i = torch.linspace(0, 1, n + 1, device=device)
    inv = 1.0 / float(rho)
    sig = (s_max ** inv + i * (s_min ** inv - s_max ** inv)) ** float(rho)
    t = 1.0 / (1.0 + sig)
    t[0], t[-1] = 0.0, 1.0
    return t


def cosine(n, p=None, device="cuda"):
    """Dense at *both* ends. Included as a control: if it loses to the one-sided
    schedules, that is direct evidence the noisy end is where the budget belongs."""
    u = torch.linspace(0, 1, n + 1, device=device)
    return (1 - torch.cos(math.pi * u)) / 2


SCHEDULES = {"uniform": (uniform, None), "power": (power, 2.0),
             "shift": (shift, 3.0), "karras": (karras, 7.0), "cosine": (cosine, None)}


def make(name, n, param=None, device="cuda"):
    fn, default = SCHEDULES[name]
    return fn(n, device=device) if default is None else \
        fn(n, param if param is not None else default, device=device)


@torch.no_grad()
def solve(vel, x, ts, solver="euler"):
    """Integrate dx/dt = vel(x,t) over the knots `ts`. Returns (x, nfe, trace).

    trace holds (t, x, v) at each knot so callers can decode the posterior mean
    x1_hat = x + (1-t)*v without a second pass."""
    nfe, trace = 0, []
    v_prev = h_prev = None
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        v0 = vel(x, t0); nfe += 1
        trace.append((float(t0), x, v0))

        if solver == "heun":
            v1 = vel(x + v0 * dt, t1); nfe += 1
            x = x + 0.5 * (v0 + v1) * dt
        elif solver == "ab2" and v_prev is not None:
            # variable-step Adams-Bashforth 2. The extrapolation coefficients depend on the
            # ratio of consecutive step sizes -- with these schedules steps are far from
            # uniform, so the textbook (3/2, -1/2) constants would be wrong.
            r = dt / h_prev
            x = x + dt * ((1 + r / 2) * v0 - (r / 2) * v_prev)
        else:
            x = x + v0 * dt                      # euler, and ab2's first step

        v_prev, h_prev = v0, dt
    return x, nfe, trace


SOLVERS = ("euler", "heun", "ab2")
