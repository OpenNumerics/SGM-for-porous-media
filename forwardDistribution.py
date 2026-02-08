import math
import torch as pt
import matplotlib.pyplot as plt

from PorousSGMDataset import PorousDataset
from ConvFiLMScore import ConvFiLMScore1D
from timesteppers import sample_sgm_heun
from solveFVM import getPDEParameters
from fvm import simulateFVM
from gp import sample_gp_1d
from SDEs import forwardSDE

# Check if the forward distribution for our beta and T = 1 is sufficiently Gaussian ( in the marginals )
device = pt.device("cpu")
dtype = pt.float64
dataset = PorousDataset( device, dtype, is_test=True)
c_values = dataset.norm_c_data[0:100,:]
phi_values = dataset.norm_phi_data[0:100,:]

# Generate N = 10_000 samples of c and phi
N = 10_000
idx_c = pt.randint(0, c_values.shape[0], (N,))
idx_phi = pt.randint(0, phi_values.shape[0], (N,))
x_samples = pt.cat( (c_values[idx_c,:], phi_values[idx_phi,:]), dim=1 )

# Propagate forward
print('Propagating particles forward')
dt = 1e-3
xT_samples = forwardSDE( x_samples, dt )
print(xT_samples.mean(dim=0))

# Display
dim = 189
plt.hist( xT_samples[:, dim].detach().numpy(), bins=500, density=True )
plt.show()