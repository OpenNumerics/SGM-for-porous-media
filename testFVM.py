import torch as pt
pt.set_default_dtype(pt.float64)

from fvm import simulateFVM
from plotFVM import plot_solution

def testFVM():
    L = 1.e-4
    N = 100
    x_faces = pt.linspace(0.0, L, N+1)
    x_cells = 0.5 * (x_faces[1:] + x_faces[0:-1])

    dt = 0.1
    T = 100.0

    c0 = 1000.0
    c0_tensor = c0 * pt.ones_like(x_cells)
    c_right = c0
    t_plus = 0.4

    b = 1.5

    eps = 0.35 * pt.ones_like(x_cells)
    D0 = 1e-10
    kappa0 = 1.0
    k_eff = kappa0 * eps**b
    D_eff = D0 * eps**b

    a_s = 5e5
    k_rn = 0.1
    U0 = 0.0
    G = 1.e3
    phi_s = lambda x: -G*x

    # Right bc for phi is the ionic current density i_c(L)
    F_right = 10.0

    # Run the FVM simulator
    parameters = {"t_plus" : t_plus, "a_s" : a_s, "k_rn" : k_rn, "U0" : U0}
    c_T, phi_T = simulateFVM(eps, k_eff, D_eff, x_cells, c0_tensor, 
                             T, dt, parameters, phi_s, c_right, F_right)
    
    # Plot the solution field
    plot_solution(x_cells, c_T, phi_T)

if __name__ == '__main__':
    testFVM()