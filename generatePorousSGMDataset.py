import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch as pt
import torch.distributions as dist

pt.set_grad_enabled(False)

# ---- worker must be top-level for multiprocessing pickling ----
def worker(n, n_eps, x_cells, l_values, U0_values, F_right_values,
            eps_min, eps_max, kappa0, D0, b, t_plus, a_s, k_rn, c0, c_right,
            T, dt, G):
        print('n =', n)

        # local imports to avoid pickling issues
        from gp import sample_gp_1d
        from fvm import simulateFVM

        phi_s = lambda x: -G * x

        try:
            l = float(l_values[n])
            eps_values = sample_gp_1d(x_cells, n_eps, l, 0.0, 1.0)
        except Exception as e:
            traceback.print_exc()
            return n, False, None, None

        eps_values = eps_min + (eps_max - eps_min) * pt.sigmoid(eps_values)
        k_eff = kappa0 * eps_values**b
        D_eff = D0 * eps_values**b

        U0 = float(U0_values[n])
        F_right = float(F_right_values[n])

        c0_tensor = c0 * pt.ones_like(x_cells)

        parameters = {"t_plus": t_plus, "a_s": a_s, "k_rn": k_rn, "U0": U0}

        c_solutions = pt.zeros((n_eps, pt.numel(x_cells)))
        phi_solutions = pt.zeros((n_eps, pt.numel(x_cells)))
        for eps_idx in range(n_eps):
            c_T, phi_T = simulateFVM(eps_values[:,eps_idx], k_eff[:,eps_idx], D_eff[:,eps_idx], x_cells, c0_tensor,
                                    T, dt, parameters, phi_s, c_right, F_right)
            c_solutions[eps_idx,:] = c_T
            phi_solutions[eps_idx,:] = phi_T
        
        good = not ((pt.isnan(c_solutions).any() or pt.isnan(phi_solutions).any() or
                     pt.isinf(c_solutions).any() or pt.isinf(phi_solutions).any()))
        
        return n, good, c_solutions, phi_solutions

if __name__ == '__main__':
    # Define the grid
    L = 1.e-4
    n_grid = 100
    x_faces = pt.linspace(0.0, L, n_grid+1)
    x_cells = 0.5 * (x_faces[1:] + x_faces[0:-1])

    # Model parameters
    b = 1.5
    D0 = 3e-11
    kappa0 = 1.0
    a_s = 5e5
    k_rn = 0.1
    G = 1.e3
    phi_s = lambda x: -G*x
    c0 = 1000.0
    c0_tensor = c0 * pt.ones_like(x_cells)
    c_right = c0
    t_plus = 0.4

    # Timestepping parameters
    dt = 0.1
    T = 100.0

    # Sample l log-uniformly, U0 uniformly and F_right also uniformly
    N_samples = 100
    class LogUniform(dist.TransformedDistribution):
        def __init__(self, lb, ub):
            super(LogUniform, self).__init__(dist.Uniform(math.log(lb), math.log(ub)), dist.ExpTransform())

    l_min = L / 50.0
    l_max = L / 5.0
    l_sampler = LogUniform(l_min, l_max)
    l_values = l_sampler.sample((N_samples,))
    U0_values = dist.Uniform(-0.2, 0.2).sample((N_samples,))
    F_right_values = dist.Uniform(0.0, 50.0).sample((N_samples,))

    # For each of these samples, generate 25 eps(x) and solve the PDE
    # ---- main ----
    n_eps = 25
    eps_min, eps_max = 0.2, 0.5
    c_data = pt.zeros((N_samples * n_eps, n_grid))
    phi_data = pt.zeros((N_samples * n_eps, n_grid))
    gp_success = [True] * (N_samples * n_eps)

    # Make sure x_cells / l_values / ... are CPU tensors or plain lists for mp
    x_cells_cpu = x_cells.detach().cpu()
    l_cpu = l_values.detach().cpu()
    U0_cpu = U0_values.detach().cpu()
    F_cpu = F_right_values.detach().cpu()

    max_workers = min(8, os.cpu_count())  # or set explicitly
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(worker, n, n_eps, x_cells_cpu, l_cpu, U0_cpu, F_cpu,
                    eps_min, eps_max, kappa0, D0, b, t_plus, a_s, k_rn, c0, c_right,
                    T, dt, G)
            for n in range(N_samples)
        ]

        for fut in as_completed(futures):
            n, ok, c_T, phi_T = fut.result()
            if not ok:
                print('Failed')
                gp_success[n*n_eps:(n+1)*n_eps] = [False] * n_eps
                continue
            c_data[n*n_eps:(n+1)*n_eps, :] = c_T
            phi_data[n*n_eps:(n+1)*n_eps, :] = phi_T

    # Store the data
    parameters = pt.cat((l_values[:,None], U0_values[:,None], F_right_values[:,None]), dim=1)
    pt.save(pt.Tensor(gp_success), './data/gp_multieps.pt')
    pt.save(parameters.repeat_interleave(n_eps, dim=0), './data/parameters_multieps.pt')
    pt.save(c_data, './data/c_data_multieps.pt')
    pt.save(phi_data, './data/phi_data_multieps.pt')