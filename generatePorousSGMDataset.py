import math
import torch as pt
import torch.distributions as dist

from gp import sample_gp_1d
from fvm import simulateFVM

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

# Sample l log-normally, U0 uniformly and F_right also uniformly
N_samples = 10_000
class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(dist.Uniform(math.log(lb), math.log(ub)), dist.ExpTransform())

l_min = L / 50.0
l_max = L / 5.0
l_sampler = LogUniform(l_min, l_max)
l_values = l_sampler.sample((N_samples,))
U0_values = dist.Uniform(-0.2, 0.2).sample((N_samples,))
F_right_values = dist.Uniform(0.0, 50.0).sample((N_samples,))

# For each of these samples, generate eps(x) and solve the PDE
eps_min, eps_max = 0.2, 0.5
c_data = pt.zeros((N_samples, n_grid))
phi_data = pt.zeros((N_samples, n_grid))
for n in range(N_samples):
    print('n =', n)
    l = float(l_values[n])
    eps_values = sample_gp_1d(x_cells, l, 0.0, 1.0)
    eps_values = eps_min + (eps_max - eps_min) * pt.sigmoid(eps_values)
    k_eff = kappa0 * eps_values**b
    D_eff = D0 * eps_values**b

    U0 = float(U0_values[n])
    F_right = float(F_right_values[n])

    # Run the FVM simulator
    parameters = {"t_plus" : t_plus, "a_s" : a_s, "k_rn" : k_rn, "U0" : U0}
    c_T, phi_T = simulateFVM(eps_values, k_eff, D_eff, x_cells, c0_tensor, 
                             T, dt, parameters, phi_s, c_right, F_right)
    c_data[n,:] = c_T
    phi_data[n,:] = phi_T

# Store the data
parameters = pt.cat((l_values, U0_values, F_right_values), dim=1)
pt.save(parameters, './data/parameters.pt')
pt.save(c_data, './data/c_data.pt')
pt.save(phi_data, './data/phi_data.pt')