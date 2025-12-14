import torch as pt

def solve_tridiagonal(a: pt.Tensor, 
                      b: pt.Tensor, 
                      c: pt.Tensor, 
                      d: pt.Tensor) -> pt.Tensor:
    """
    Lean Thomas algorithm for tridiagonal systems Ax=d (single RHS).

    a: (N-1,) lower diagonal  (A[i+1, i])
    b: (N,)   main diagonal   (A[i, i])
    c: (N-1,) upper diagonal  (A[i, i+1])
    d: (N,)   RHS
    returns x: (N,)
    """
    n = b.numel()

    # overwrite copies (keep inputs intact)
    cp = c.clone()
    dp = d.clone()
    bp = b.clone()

    # forward sweep
    inv = 1.0 / bp[0]
    cp[0] = cp[0] * inv
    dp[0] = dp[0] * inv
    for i in range(1, n - 1):
        denom = bp[i] - a[i - 1] * cp[i - 1]
        inv = 1.0 / denom
        cp[i] = cp[i] * inv
        dp[i] = (dp[i] - a[i - 1] * dp[i - 1]) * inv

    denom = bp[n - 1] - a[n - 2] * cp[n - 2]
    dp[n - 1] = (dp[n - 1] - a[n - 2] * dp[n - 2]) / denom

    # back substitution
    x = dp
    for i in range(n - 2, -1, -1):
        x[i] = x[i] - cp[i] * x[i + 1]
    return x

if __name__ == '__main__':
    pt.set_default_dtype(pt.float64)

    N = 10_000
    diagonal = pt.randn((N,))
    upper = pt.randn((N-1,))
    lower = pt.randn((N-1,))
    rhs = diagonal = pt.randn((N,))
    sol_thomas = solve_tridiagonal(lower, diagonal, upper, rhs)

    A = pt.diag(upper, 1) + pt.diag(diagonal, 0) + pt.diag(lower, -1)
    print('Residual Norm:', pt.linalg.norm(A @ sol_thomas - rhs) / pt.linalg.norm(rhs))