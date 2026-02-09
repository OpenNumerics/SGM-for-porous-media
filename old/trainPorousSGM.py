import math
import torch as pt
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from PorousSGMDataset import PorousDataset
from SDEs import mean_factor_tensor, var_tensor
from SGMNetwork import Score

from typing import Tuple

# Set global device and dtype, except for the dataloader
dtype = pt.float32
device = pt.device("mps")

# Load the full dataset
B = 512
dataset = PorousDataset(pt.device("cpu"), dtype)
loader = DataLoader(dataset, B, shuffle=True)

n_grid = 100
n_embeddings = 16
layers = [2*n_grid + 2*n_embeddings + 3, 256, 256, 256, 256, 2*n_grid]
score_model = Score(layers).to(device=device)

lr = 1e-4
optimizer = optim.Adam(score_model.parameters(), lr=lr)

def time_embedding(t : pt.Tensor):
    freqs = 2 * math.pi * (2 ** pt.arange(n_embeddings, device=t.device, dtype=t.dtype))
    t = t[:, None]
    emb = pt.cat([pt.sin(freqs * t), pt.cos(freqs * t)], dim=1)
    return emb

lam = 1e-3
def lap1d(u):  # u: (B, D) with D=2*n_grid
    return u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]
def loss_fn(x0: pt.Tensor,          # (B, 2*n_grid)
            cond: pt.Tensor,        # (B, 3)
            ) -> pt.Tensor:
    """
    Evaluate the score-based loss function based on random samples.
    """
    x0 = x0.to(device=device)
    cond = cond.to(device=device)

    B_ = x0.shape[0]
    assert(x0.shape[0] == cond.shape[0])

    # Sample t uniformly
    t = pt.rand((B_,), device=device, dtype=dtype)
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

    # Regularize the output to improve smoothness
    x0_hat = (xt + vt * output) / mt          # mt, vt: (B,1)
    c_hat  = x0_hat[:, :n_grid]
    phi_hat = x0_hat[:, n_grid:]
    smooth = (lap1d(c_hat).pow(2).mean() + lap1d(phi_hat).pow(2).mean())
    
    return loss + smooth

def getGradientNorm():
    grads = []
    for param in score_model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grads = pt.cat(grads)
    return pt.norm(grads).item()

n_epochs = 50_000
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