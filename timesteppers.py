import torch as pt

from ConvFiLMScore import ConvFiLMScore1D
from SDEs import beta

# Backward simulation of the SDE using Euler-Maruyama in normalized space.
@pt.no_grad()
def sample_sgm_em(score_model : ConvFiLMScore1D,
                  cond_norm: pt.Tensor,
                  dt: float = 5e-3,
                  n_grid : int = 100):
    B = cond_norm.shape[0]
    y = pt.randn((B, 2*n_grid))

    n_steps = int(1.0 / dt)
    for n in range(n_steps):
        print('t =', n*dt)
        s = n * dt
        t = (1.0 - s) * pt.ones((B,))  # network expects "forward time" t
        t = pt.clamp(t, min=1e-3)

        # Compute the score
        score = score_model(y, t, cond_norm)

        # Do one backward EM step
        beta_t = beta(t)[:, None]
        y = y + dt*(0.5*beta_t*y + beta_t*score) + pt.sqrt(beta_t*dt)*pt.randn_like(y)
    return y

@pt.no_grad()
def sample_sgm_heun( score_model : ConvFiLMScore1D,
                     cond_norm: pt.Tensor,
                     dt: float = 5e-3,
                     tmin: float = 1e-4,
                     power: float = 2.0, 
                     n_grid : int = 100) -> pt.Tensor:
    B = cond_norm.shape[0]
    device = cond_norm.device
    dtype = cond_norm.dtype
    y = pt.randn((B, 2 * n_grid), device=device, dtype=dtype)

    # number of steps (keep your original interface)
    n_steps = int(1.0 / dt)

    # Non-uniform time grid: t goes from 1 -> tmin with more resolution near tmin
    # u in [0,1], then t(u) = tmin + (1 - tmin) * (1 - u)^power
    u = pt.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
    t_grid = tmin + (1.0 - tmin) * (1.0 - u).pow(power)

    for n in range(n_steps):
        t0 = t_grid[n].expand(B)       # (B,)
        t1 = t_grid[n + 1].expand(B)   # (B,)
        dt_step = (t0 - t1)            # positive scalar (as tensor): step size in reverse-time

        print(f"step {n:5d}/{n_steps}  t={t0[0].item():.6f}  dt={dt_step[0].item():.6f}")

        beta0 = beta(t0)[:, None]  # (B,1)
        score0 = score_model(y, t0, cond_norm)  # (B, 2*n_grid)
        f0 = 0.5 * beta0 * y + beta0 * score0   # (B, 2*n_grid)

        # Noise for predictor/corrector (Heun for SDE)
        z = pt.randn_like(y)
        g0 = pt.sqrt(beta0 * dt_step[:, None])
        y_pred = y + f0 * dt_step[:, None] + g0 * z

        # Drift at predicted state (y_pred, t1)
        beta1 = beta(t1)[:, None]
        score1 = score_model(y_pred, t1, cond_norm)
        f1 = 0.5 * beta1 * y_pred + beta1 * score1

        # Heun Update
        y = y + 0.5 * (f0 + f1) * dt_step[:, None] + g0 * z

    return y