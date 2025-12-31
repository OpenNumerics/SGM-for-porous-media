import torch as pt

def covariance_1d(x : pt.Tensor,
                  l, sigma = 1.0) -> pt.Tensor:
    dx2 = (x[:, None] - x[None, :])**2
    K = sigma**2 * pt.exp(-0.5 * dx2 / l**2) + 1.e-4 * pt.diag(pt.ones_like(x))
    return K

def sample_gp_1d(x : pt.Tensor,
                 n_fcns : int,
                 l, mu, sigma) -> pt.Tensor:
    K = covariance_1d(x, l, sigma)
    L = pt.linalg.cholesky(K)

    n_x = x.shape[0]
    eps = pt.randn((n_x, n_fcns))
    samples = mu + L @ eps
    return samples