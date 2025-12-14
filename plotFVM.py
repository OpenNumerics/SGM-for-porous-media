import matplotlib.pyplot as plt

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