import math
import torch as pt
import matplotlib.pyplot as plt

from SDEs import beta
from SGMNetwork import Score

pt.set_default_dtype(pt.float32)
pt.set_default_device("cpu")
pt.set_grad_enabled(False)

# Load the model from file
test_embedded = True
if test_embedded:
    n_embeddings = 8
    def time_embedding(t, N):
        freqs = 2 * math.pi * (2 ** pt.arange(n_embeddings))
        t = t * pt.ones((N,1))
        emb = pt.cat([pt.sin(freqs * t), pt.cos(freqs * t)], dim=1)
        return emb
    layers = [2 + 2*n_embeddings, 128, 128, 128, 128, 2]
    score_model = Score(layers)
    score_model.load_state_dict(pt.load("./models/score_model_embedded.pth", weights_only=True))
else:
    def time_embedding(t, N):
        return t * pt.ones((N,1))
    layers = [3, 128, 128, 128, 128, 2]
    score_model = Score(layers)
    score_model.load_state_dict(pt.load("./models/score_model.pth", weights_only=True))

# Setup the backward SDE
def backwardSDE(Y : pt.Tensor,
                dt : float) -> pt.Tensor:
    n_steps = int(1.0 / dt)
    assert(abs(n_steps * dt - 1) < 1e-10)

    B = Y.shape[0]
    for n in range(n_steps):
        s = n * dt
        t = max(1e-2, 1.0 - s)
        print('s =', s)

        # Evaluate the drift
        input = pt.cat((Y, time_embedding(t, B)), dim=1)
        drift = score_model(input)

        # Propagate the particles
        beta_t = beta(t)
        Y = Y + dt * beta_t * (0.5 * Y + drift) + math.sqrt(beta_t * dt) * pt.randn_like(Y)

    return Y

# Run the backward simulation, starting with random noise
N = 10_000
Y0 = pt.randn((N, 2), requires_grad=False)
dt = 1e-4
Y_1 = backwardSDE(Y0, dt)

# Make a scatter plot of Y_1
plt.scatter(Y_1.detach().numpy()[:,0], Y_1.detach().numpy()[:,1], label=r"$Y(s=1)$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.legend()
plt.show()