import math
import torch as pt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from SDEs import sampleInitial, mean_factor_tensor, var_tensor
from SGMNetwork import Score

from typing import Tuple

pt.set_default_dtype(pt.float32)
pt.set_default_device("mps")

# Dataset class
class PorousDataset (Dataset):
    def __init__(self):
        super().__init__()
        parameters = pt.load('./data/parameters.pt', map_location=pt.device("mps")).to(dtype=pt.float32)
        c_data = pt.load('./data/c_data.pt', map_location=pt.device("mps")).to(dtype=pt.float32)
        phi_data = pt.load('./data/phi_data.pt', map_location=pt.device("mps")).to(dtype=pt.float32)
        self.N_samples = int(parameters.shape[0])

        # Normalize the input data
        mean_c = pt.mean(c_data)
        std_c = pt.std(c_data)
        mean_phi = pt.mean(phi_data)
        std_phi = pt.std(phi_data)
        self.norm_c_data = (c_data - mean_c) / std_c
        self.norm_phi_data = (phi_data - mean_phi) / std_phi

        # Normalize the input parameters
        log_l = pt.log(parameters[:,0])
        self.min_log_l = pt.min(log_l)
        self.max_log_l = pt.max(log_l)
        self.log_l_values = (log_l - self.min_log_l) / (self.max_log_l - self.min_log_l)
        self.min_U0 = pt.min(parameters[:,1])
        self.max_U0 = pt.max(parameters[:,1])
        self.U0_values = (parameters[:,1] - self.min_U0) / (self.max_U0 - self.min_U0)
        self.min_F_right = pt.min(parameters[:,2])
        self.max_F_right = pt.max(parameters[:,2])
        self.F_right_values = (parameters[:,2] - self.min_F_right) / (self.max_F_right - self.min_F_right)

    def __len__(self):
        return self.N_samples
    
    def __getitem__(self, index) -> Tuple[pt.Tensor, pt.Tensor]:
        return pt.cat((self.norm_c_data[index,:], self.norm_phi_data[index,:])), pt.stack((self.log_l_values[index], self.U0_values[index], self.F_right_values[index]))

# Load the full dataset
B = 4096
dataset = PorousDataset()
loader = DataLoader(dataset, B, shuffle=True)

n_grid = 100
n_embeddings = 16
layers = [2 + 2*n_embeddings + 3, 256, 256, 256, 256, 2*n_grid]
score_model = Score(layers)

lr = 1e-4
optimizer = optim.Adam(score_model.parameters(), lr=lr)

def time_embedding(t : pt.Tensor):
    freqs = 2 * math.pi * (2 ** pt.arange(n_embeddings, device=t.device, dtype=t.dtype))
    t = t[:, None]
    emb = pt.cat([pt.sin(freqs * t), pt.cos(freqs * t)], dim=1)
    return emb

def loss_fn(x0: pt.Tensor,          # (B, 2*n_grid)
            cond: pt.Tensor,        # (B, 3)
            ) -> pt.Tensor:
    """
    Evaluate the score-based loss function based on random samples.
    """
    B_ = x0.shape[0]
    assert(x0.shape[0] == cond.shape[0])

    # Sample t uniformly
    t = pt.rand((B_,), device=pt.get_default_device())
    embed_t = time_embedding(t) # Shape (B, 2*n_freq)

    # Forward diffusion
    mt = mean_factor_tensor(t)[:,None]
    vt = var_tensor(t)[:,None].clamp_min(1.e-4)
    stds = pt.sqrt(vt)
    noise = pt.randn_like(x0)
    xt = x0 * mt + noise * stds

    # Propage the noise through the network
    input = pt.cat((xt, embed_t, cond), dim=1)
    output = score_model(input)

    # Compute the loss
    ref_output = -(xt - x0 * mt) / vt
    loss = pt.mean(vt * (output - ref_output)**2)
    
    return loss

def getGradientNorm():
    grads = []
    for param in score_model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grads = pt.cat(grads)
    return pt.norm(grads).item()

n_epochs = 10_000
counter = []
losses = []
grad_norms = []

score_model.train()
for epoch in range(n_epochs):
    for batch_idx, (x0, cond) in enumerate(loader):
        optimizer.zero_grad()

        loss = loss_fn(x0, cond)
        loss.backward()
        grad_norm = getGradientNorm()
        optimizer.step()

        counter.append((1.0*batch_idx)/len(loader) + epoch)
        losses.append(loss.item())
        grad_norms.append(grad_norm)

    print('Train Epoch: {} \tLoss: {:.6f} \t Gradient Norm {:.6f}'.format(  epoch, loss.item(), grad_norm ))

# Save the network weights on file    
pt.save(score_model.state_dict(), "./models/porous_score_model_embedded.pth")

# Plot the loss and grad norm
plt.semilogy(counter, losses, label='Losses', alpha=0.5)
plt.semilogy(counter, grad_norms, label='Gradient Norms', alpha=0.5)
plt.xlabel('Epoch')
plt.legend()
plt.show()