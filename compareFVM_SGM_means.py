import math
import torch as pt
import matplotlib.pyplot as plt

from PorousSGMDataset import PorousDataset
from SDEs import beta
from ConvFiLMScore import ConvFiLMScore1D
from timesteppers import sample_sgm_heun
from solveFVM import getPDEParameters
from fvm import simulateFVM
from gp import sample_gp_1d

pt.set_grad_enabled(False)
device = pt.device("cpu")
dtype = pt.float32
pt.set_default_device(device)
pt.set_default_dtype(dtype)

# Load the dataset for normalization
dataset = PorousDataset(device, dtype)

# Load the model
n_grid = 100
n_embeddings = 16
score_model = ConvFiLMScore1D(n_grid, n_time_freq=n_embeddings)
score_model.load_state_dict(pt.load("./models/porous_score_model_convfilm_multiple_best_validated.pth", weights_only=True))
score_model.eval()

# Useful model parameters for backward simulation
L = 1e-4
c0 = 1000.0
x_faces = pt.linspace(0.0, L, n_grid+1)
x_cells = 0.5 * (x_faces[1:] + x_faces[0:-1])
c0_tensor = c0 * pt.ones_like(x_cells)
c_right = c0

# Backward simulation of the SDE in normalized space.
@pt.inference_mode()
def sample_sgm(cond_norm, dt):
    B = cond_norm.shape[0]
    y = pt.randn((B, 2*n_grid))

    n_steps = int(1.0 / dt)
    for n in range(n_steps):
        print('t =', n*dt)
        s = n * dt
        t = (1.0 - s) * pt.ones((B,))  # network expects "forward time" t
        t = pt.clamp(t, min=1e-4)

        # Compute the score
        score = score_model(y, t, cond_norm)

        # Do one backward EM step
        beta_t = beta(t)[:, None]
        y = y + dt*(0.5*beta_t*y + beta_t*score) + pt.sqrt(beta_t*dt)*pt.randn_like(y)

    return y

# Sample a good initial (l, U0, F_right)
n_eps = 500
l = L / 10.0
log_l = math.log(l)
U0 = 0.0
F_right = 25.0
log_l_norm = (log_l - dataset.min_log_l) / (dataset.max_log_l - dataset.min_log_l)
U0_norm = (U0 - dataset.min_U0) / (dataset.max_U0 - dataset.min_U0)
F_right_norm = (F_right - dataset.min_F_right) / (dataset.max_F_right - dataset.min_F_right)
cond_norm = pt.tensor([[log_l_norm, U0_norm, F_right_norm]], dtype=dtype).repeat(n_eps, 1) # (n_eps,3)

# Generate the SGM solution
print('Backward SDE Simulation..')
dt = 1e-3
y = sample_sgm_heun(score_model, cond_norm, dt, n_grid=n_grid)
c = dataset.mean_c + y[:,:n_grid] * dataset.std_c
phi = dataset.mean_phi + y[:,n_grid:] * dataset.std_phi

# Solve the PDE by generating eps(x) randomly
print('Solving Many PDEs')
parameters, phi_s = getPDEParameters()
parameters["U0"] = U0
x_faces = pt.linspace(0.0, L, n_grid+1)
x_cells = 0.5 * (x_faces[1:] + x_faces[0:-1])
eps_min, eps_max = 0.2, 0.5
eps_values = sample_gp_1d(x_cells, n_eps, l, 0.0, 1.0)
eps_values = eps_min + (eps_max - eps_min) * pt.sigmoid(eps_values)

# Right bc for phi is the ionic current density i_c(L)
b = 1.5
D0 = 3e-11
kappa0 = 1.0
k_eff = kappa0 * eps_values**b
D_eff = D0 * eps_values**b

# Run the FVM simulator
dt = 0.1
T = 100.0
c_pde = pt.zeros(n_eps, pt.numel(x_cells))
phi_pde = pt.zeros(n_eps, pt.numel(x_cells))
for eps_idx in range(n_eps):
    print('eps_idx =', eps_idx)
    c_T, phi_T = simulateFVM(eps_values[:,eps_idx], k_eff[:,eps_idx], D_eff[:,eps_idx], x_cells, c0_tensor, 
                                T, dt, parameters, phi_s, c_right, F_right)
    c_pde[eps_idx,:] = c_T
    phi_pde[eps_idx,:] = phi_T

# Average and compare
mean_c_pde = pt.mean(c_pde, dim=0)
mean_phi_pde = pt.mean(phi_pde, dim=0)
mean_c_sgm = pt.mean(c, dim=0)
mean_phi_sgm = pt.mean(phi, dim=0)
pt.save(pt.stack((c_pde, phi_pde), dim=1) , './models/pde_realization_multiples.pt')
pt.save(pt.stack((c, phi), dim=1) , './models/sgm_realizations_multiple.pt')

# Plot both on separate axis
fig, ax1 = plt.subplots(figsize=(9,5))
ax2 = ax1.twinx()

# concentration (left axis)
ax1.plot(1e6 * x_cells, mean_c_pde,  label="PDE Mean c(x)", color="red", linewidth=2)
ax1.plot(1e6 * x_cells, mean_c_sgm,  "--", label="SGM Mean c(x)", color="red", linewidth=2)
ax1.set_xlabel("x [μm]")
ax1.set_ylabel("Concentration c [mol/m³]")

# potential (right axis)
ax2.plot(1e6 * x_cells, mean_phi_pde, label="PDE Mean φ(x)", color="blue", linewidth=2)
ax2.plot(1e6 * x_cells, mean_phi_sgm, "--", label="SGM Mean φ(x)", color="blue", linewidth=2)
ax2.set_ylabel("Electrolyte potential φ [V]")

# single combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.tight_layout()
plt.show()