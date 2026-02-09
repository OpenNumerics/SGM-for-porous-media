import math
import torch as pt
import torch.optim as optim
import matplotlib.pyplot as plt

from SDEs import sampleInitial, mean_factor_tensor, var_tensor
from SGMNetwork import Score

pt.set_default_dtype(pt.float32)
pt.set_default_device("mps")

n_embeddings = 8
layers = [2 + 2*n_embeddings, 128, 128, 128, 128, 2]
score_model = Score(layers)

lr = 1e-4
optimizer = optim.Adam(score_model.parameters(), lr=lr)

def time_embedding(t):
    freqs = 2 * math.pi * (2 ** pt.arange(n_embeddings))
    t = t[:, None]
    emb = pt.cat([pt.sin(freqs * t), pt.cos(freqs * t)], dim=1)
    return emb

B = 4096
def loss_fn():
    """
    Evaluate the score-based loss function based on random samples.
    """
    
    t = pt.rand((B,), device=pt.get_default_device())
    embed_t = time_embedding(t)
    x0 = sampleInitial(B)

    mt = mean_factor_tensor(t)[:,None]
    vt = var_tensor(t)[:,None].clamp_min(1.e-4)
    stds = pt.sqrt(vt)

    noise = pt.randn_like(x0)
    xt = x0 * mt + noise * stds
    input = pt.cat((xt, embed_t), dim=1)

    ref_output = -(xt - x0 * mt) / vt

    output = score_model(input)
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
losses = []
grad_norms = []
for epoch in range(n_epochs):
    optimizer.zero_grad()

    loss = loss_fn()
    loss.backward()
    grad_norm = getGradientNorm()
    optimizer.step()

    losses.append(loss.item())
    grad_norms.append(grad_norm)
    if epoch % 100 == 0:
        print('Train Epoch: {} \tLoss: {:.6f} \t Gradient Norm {:.6f}'.format(
            epoch, loss.item(), grad_norm))

# Save the network weights on file    
pt.save(score_model.state_dict(), "./models/score_model_embedded.pth")

# Plot the loss and grad norm
epoch_list = pt.arange(1, n_epochs+1)
plt.semilogy(epoch_list.cpu().numpy(), losses, label='Losses', alpha=0.5)
plt.semilogy(epoch_list.cpu().numpy(), grad_norms, label='Gradient Norms', alpha=0.5)
plt.xlabel('Epoch')
plt.legend()
plt.show()