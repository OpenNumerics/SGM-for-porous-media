import torch as pt

from thomas import solve_tridiagonal

from typing import Callable, Dict, Tuple

def solve_phi(x_cells : pt.Tensor, # size N
              k_eff_cells : pt.Tensor,
              phi_s : Callable[[pt.Tensor], pt.Tensor],
              F_right : float,
              U0 : float,
              k_rn : float,
              a_s: pt.Tensor) -> pt.Tensor:
    dx = x_cells[1] - x_cells[0]
    N = len(x_cells)

    # Evaluate k_eff on the non-exterior faces.
    k_eff_interior_faces = 2.0 * k_eff_cells[0:-1] * k_eff_cells[1:] / (k_eff_cells[0:-1] + k_eff_cells[1:]) # (N-1,)

    # Start completing the big tridiagonal system
    diagonal = pt.zeros((N,), device=x_cells.device)
    diagonal[1:-1] = -(k_eff_interior_faces[0:-1] + k_eff_interior_faces[1:]) - a_s[1:-1] * k_rn * dx**2
    diagonal[0]  = -k_eff_interior_faces[0]  - a_s[0] * k_rn * dx**2
    diagonal[-1] = -k_eff_interior_faces[-1] - a_s[-1] * k_rn * dx**2

    lower = pt.zeros((N-1,), device=x_cells.device)
    lower[:] = k_eff_interior_faces[:]

    upper = pt.zeros((N-1,), device=x_cells.device)
    upper[:] = k_eff_interior_faces[:]

    # Build the right-hand side
    b = -a_s * k_rn * (phi_s(x_cells) - U0) * dx**2
    b[-1] += -dx * F_right

    # Solve the tridiagonal system
    phi = solve_tridiagonal(lower, diagonal, upper, b)

    # Normalize phi by subtracting phi(x=0)
    return phi - phi[0]

def compute_j(x_cells : pt.Tensor, # (N+1,)
              phi_s : Callable[[pt.Tensor], pt.Tensor],
              phi : pt.Tensor,
              U0 : float,
              k_rn : float) -> pt.Tensor:
    return k_rn * (phi_s(x_cells) - phi - U0)

def step_c( x_cells: pt.Tensor,                      # (N+1,)
            eps_cells: pt.Tensor,   # porosity ε(x) at cell centers
            D_eff_cells: pt.Tensor, # effective diffusivity D_eff(x) at cell centers
            c_n: pt.Tensor,                          # (N,) concentration at time n (cell centers)
            j_n: pt.Tensor,                          # (N,) reaction current at time n (cell centers)
            dt: float,
            c_right: float,                          # Dirichlet BC at x=L: c(L)=c_right
            a_s: pt.Tensor,
            t_plus: float,
            F: float = 96485.33212,                  # Faraday constant (C/mol)
) -> pt.Tensor:
    """
    One timestep for:
      ε ∂_t c = ∂_x(D_eff ∂_x c) + (1 - t_plus)/F * a_s * j

    Discretization:
      - FVM in space, cell-centered unknowns
      - implicit diffusion, explicit source j_n
      - left Neumann (no-flux), right Dirichlet (fixed concentration)

    Returns c_{n+1} (N,).
    """
    dx = x_cells[1] - x_cells[0]
    N = x_cells.numel()

    # harmonic averages to interior faces (N-1,)
    D_faces = 2.0 * D_eff_cells[:-1] * D_eff_cells[1:] / (D_eff_cells[:-1] + D_eff_cells[1:])

    # Source term at time n (explicit)
    S_n = (1.0 - t_plus) / F * a_s * j_n  # (N,)

    # Build tridiagonal system for c_{n+1}:
    #  -D_{i-1/2}/dx^2 * c_{i-1}^{n+1}
    # + (eps_i/dt + (D_{i-1/2}+D_{i+1/2})/dx^2) * c_i^{n+1}
    #  -D_{i+1/2}/dx^2 * c_{i+1}^{n+1}
    # = eps_i/dt * c_i^n + S_i^n

    lower = pt.zeros((N - 1,), device=x_cells.device)
    upper = pt.zeros((N - 1,), device=x_cells.device)
    diag  = pt.zeros((N,),     device=x_cells.device)
    rhs   = (eps_cells / dt) * c_n + S_n

    inv_dx2 = 1.0 / (dx * dx)

    # Interior rows i=1..N-2
    lower[:] = -D_faces[:] * inv_dx2
    upper[:] = -D_faces[:] * inv_dx2

    diag[1:-1] = (eps_cells[1:-1] / dt) + (D_faces[:-1] + D_faces[1:]) * inv_dx2

    # Left boundary i=0: no-flux => F_{1/2} = 0
    # Discrete: eps0/dt (c0^{n+1}-c0^n) = (F_{-1/2} - F_{1/2})/dx + S
    # with F_{-1/2}=0 and F_{1/2} = -D_{1/2} (c1-c0)/dx
    # => diffusion contribution = D_{1/2}(c1 - c0)/dx^2
    diag[0] = (eps_cells[0] / dt) + D_faces[0] * inv_dx2
    upper[0] = -D_faces[0] * inv_dx2  # already set, but explicit is fine

    # Right boundary i=N-1: Dirichlet c_{N-1}^{n+1} = c_right
    diag[-1] = 1.0
    rhs[-1] = pt.as_tensor(c_right, device=x_cells.device)
    lower[-1] = 0.0  # decouple from c_{N-2}

    # Solve tri-diagonal system
    c_np1 = solve_tridiagonal(lower, diag, upper, rhs)
    return c_np1

def simulateFVM(eps_cells : pt.Tensor,
                k_eff_cells : pt.Tensor,
                D_eff_cells : pt.Tensor,
                x_cells : pt.Tensor,
                c0 : pt.Tensor,
                T : float,
                dt : float,
                parameters : Dict,
                phi_s : Callable[[pt.Tensor], pt.Tensor],
                c_right : float,
                F_right : float) -> Tuple[pt.Tensor, pt.Tensor]:
    c = pt.clone(c0)
    a_s = parameters["a_s"] * (1.0 - eps_cells)
    
    n_steps = int(T / dt)
    for n in range(1, n_steps+1):
        print('t =', n*dt)
        phi = solve_phi(x_cells, k_eff_cells, phi_s, F_right, parameters["U0"], parameters["k_rn"], a_s)
        j = compute_j(x_cells, phi_s, phi, parameters["U0"], parameters["k_rn"])
        c = step_c(x_cells, eps_cells, D_eff_cells, c, j, dt, c_right, a_s, parameters["t_plus"])

    return c, phi