import torch as pt
import matplotlib.pyplot as plt

# Load the SGM CI
sgm = pt.load('./models/sgm_realizations.pt', weights_only=True)
pde = pt.load('./models/pde_realizations.pt', weights_only=True)
n_grid = sgm.shape[2]

# Compute the average function
c_sgm = sgm[:,0,:]
phi_sgm = sgm[:,1,:]
c_pde = pde[:,0,:]
phi_pde = pde[:,1,:]
mean_c_pde = pt.mean(c_pde, dim=0)
mean_phi_pde = pt.mean(phi_pde, dim=0)
mean_c_sgm = pt.mean(c_sgm, dim=0)
mean_phi_sgm = pt.mean(phi_sgm, dim=0)
c_mean_diff = mean_c_pde - mean_c_sgm
phi_mean_diff = mean_phi_pde - mean_phi_sgm

# Compute the 95% confidence interval.
lower_ci_c = pt.quantile(c_sgm, 0.025, dim=0) + c_mean_diff
upper_ci_c = pt.quantile(c_sgm, 0.975, dim=0) + c_mean_diff
lower_ci_phi = pt.quantile(phi_sgm, 0.025, dim=0) + phi_mean_diff
upper_ci_phi = pt.quantile(phi_sgm, 0.975, dim=0) + phi_mean_diff

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2)

L = 1e-4
x_faces = pt.linspace(0.0, L, n_grid+1)
x_cells = 0.5 * (x_faces[1:] + x_faces[0:-1])

ax1.plot(1e6 * x_cells, mean_c_pde.flatten(), label='Mean PDE', color='blue', linewidth=2)
ax1.plot(1e6 * x_cells, mean_c_sgm.flatten(), label='Mean SGM', color='red', linewidth=2)
ax1.fill_between( 1e6 * x_cells, lower_ci_c.flatten(), upper_ci_c.flatten(), color='blue',  alpha=0.2, label='95% Confidence Interval')
ax1.set_xlabel("x [μm]")
ax1.set_ylabel("Concentration [mol/m³]")
ax1.set_title(r'Elektrolyte Concentration $c(x)$')
ax1.legend()

ax2.plot(1e6 * x_cells, mean_phi_pde.flatten(), label='Mean PDE', color='blue', linewidth=2)
ax2.plot(1e6 * x_cells, mean_phi_sgm.flatten(), label='Mean SGM', color='red', linewidth=2)
ax2.fill_between( 1e6 * x_cells, lower_ci_phi.flatten(), upper_ci_phi.flatten(), color='blue',  alpha=0.2, label='95% Confidence Interval')
ax2.set_xlabel("x [μm]")
ax2.set_ylabel("Voltage [V]")
ax2.set_title(r'Elektrolyte Potential $\varphi(x)$')
ax2.legend()

plt.show()