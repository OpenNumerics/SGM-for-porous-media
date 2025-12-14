import torch as pt
pt.set_default_dtype(pt.float64)
import matplotlib.pyplot as plt

from fvm import simulateFVM

def testFVM():
    L = 1.e-4
    N = 100
    x_faces = pt.linspace(0.0, L, N+1)
    x_cells = 0.5 * (x_faces[1:] + x_faces[0:-1])

    dt = 0.01
    T = 100.0

    c0 = 1000.0
    c0_tensor = c0 * pt.ones_like(x_cells)
    c_right = c0
    t_plus = 0.4

    b = 1.5

    eps = lambda x: 0.35 * pt.ones_like(x)
    D0 = 1e-10
    kappa0 = 1.0
    k_eff = lambda x : kappa0 * eps(x)**b
    D_eff = lambda x : D0 * eps(x)**b

    a_s = 5e5
    k_rn = 0.01
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

def plot_solution(x_cells, c_T, phi_T):
    # Convert to numpy for plotting
    x = x_cells.cpu().numpy()
    c = c_T.cpu().numpy()
    phi = phi_T.cpu().numpy()

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Left axis: concentration
    ax1.plot(1e6 * x, c, color="tab:blue", lw=2)
    ax1.set_xlabel("x [μm]")
    ax1.set_ylabel("Concentration c [mol/m³]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Right axis: potential
    ax2 = ax1.twinx()
    ax2.plot(1e6 * x, phi, color="tab:red", lw=2, linestyle="--")
    ax2.set_ylabel("Electrolyte potential φ [V]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    testFVM()