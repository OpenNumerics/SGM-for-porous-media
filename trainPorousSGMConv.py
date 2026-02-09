import torch as pt
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from PorousSGMDataset import PorousDataset
from SDEs import mean_factor_tensor, var_tensor
from ConvFiLMScore import ConvFiLMScore1D

# Set global device and dtype, except for the dataloader
dtype = pt.float32
device = pt.device("mps")

# Load the full dataset
B = 1024
train_dataset = PorousDataset(pt.device("cpu"), dtype)
train_loader = DataLoader(train_dataset, B, shuffle=True)
test_dataset = PorousDataset(pt.device("cpu"), dtype, is_test=True)
test_loader = DataLoader(test_dataset, len(test_dataset))

# Build the complicated FiLM Scoring Network
n_grid = 100
n_embeddings = 16
score_model = ConvFiLMScore1D(n_grid, n_time_freq=n_embeddings).to(device=device)
n_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {n_params:,}")

lr = 2e-4
n_epochs = 5_000
optimizer = optim.Adam(score_model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
    eta_min=1e-6
)

def loss_fn(x0: pt.Tensor,          # (B, 2*n_grid)
            cond: pt.Tensor,        # (B, 3)
            ) -> pt.Tensor:
    """
    Evaluate the score-based loss function based on random samples.
    """
    x0 = x0.to(device=device)
    cond = cond.to(device=device)

    # Sample t quadratically
    B_ = x0.shape[0]
    tmin = 1e-4
    t = tmin + (1 - tmin) * pt.rand((B_,), device=device, dtype=dtype)**2

    # Forward diffusion
    mt = mean_factor_tensor(t)[:,None]
    vt = var_tensor(t)[:,None]
    stds = pt.sqrt(vt)
    noise = pt.randn_like(x0)
    xt = x0 * mt + noise * stds

    # Propage the noise through the network
    output = score_model(xt, t, cond)

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

counter = []
losses = []
grad_norms = []
test_counter = []
test_losses = []

best_loss = float("inf")
for epoch in range(n_epochs):
    current_lr = optimizer.param_groups[0]["lr"]

    score_model.train()
    for batch_idx, (x0, cond) in enumerate(train_loader):
        optimizer.zero_grad()

        loss = loss_fn(x0, cond)
        loss.backward()
        grad_norm = getGradientNorm()
        optimizer.step()

        counter.append((1.0*batch_idx)/len(train_loader) + epoch)
        losses.append(loss.item())
        grad_norms.append(grad_norm)
    scheduler.step()
    print('Train Epoch: {} \tLoss: {:.6f} \t Gradient Norm {:.6f} \t Learning Rate {:.2E}'.format(  epoch, loss.item(), grad_norm, current_lr ))

    score_model.eval()
    with pt.no_grad():
        for batch_idx, (x0, cond) in enumerate(test_loader):
            test_loss = loss_fn(x0, cond)
            test_counter.append((1.0*batch_idx)/len(test_loader) + epoch)
            test_losses.append(test_loss.item())

        if test_loss.item() < best_loss:
            pt.save(score_model.state_dict(), "./models/porous_score_model.pth")
            best_loss = test_loss.item()

    print('Test Loss {:.6f}'.format( test_loss.item() ))

# Save the final network weights on file    
pt.save(score_model.state_dict(), "./models/porous_score_model.pth")

# Plot the loss and grad norm
plt.semilogy(counter, losses, label='Losses', alpha=0.5)
plt.semilogy(counter, grad_norms, label='Gradient Norms', alpha=0.5)
plt.semilogy(test_counter, test_losses, label='Test Losses', alpha=0.5)
plt.xlabel('Epoch')
plt.legend()
plt.show()