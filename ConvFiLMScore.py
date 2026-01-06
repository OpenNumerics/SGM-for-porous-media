import math
import torch as pt
import torch.nn as nn

class TimeEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for scalar t in [0,1].
    Output shape: (B, 2*n_freq)
    """
    def __init__(self, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq

    def forward(self, t: pt.Tensor) -> pt.Tensor:
        # t: (B,)
        # frequencies: (n_freq,)
        freqs = (2.0 * math.pi) * (2.0 ** pt.arange(self.n_freq, device=t.device, dtype=t.dtype))
        t = t[:, None]  # (B,1)
        emb = pt.cat([pt.sin(freqs[None, :] * t), pt.cos(freqs[None, :] * t)], dim=1)
        return emb  # (B, 2*n_freq)


class FiLM(nn.Module):
    """
    FiLM modulator: takes conditioning vector z (B, z_dim) of time embeddings and conditioning variables.
    outputs (gamma, beta) each of shape (B, C), to modulate Conv features (B, C, N).
    """
    def __init__(self, z_dim: int, # = 2 * n_freq + 3
                       channels: int, 
                       hidden: int = 128, 
                       n_layers: int = 2):
        super().__init__()

        layers = []
        in_dim = z_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 2 * channels)]  # gamma and beta
        self.net = nn.Sequential(*layers)

        # Optional: init last layer small-ish to start near identity modulation
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor]:
        # z: (B, z_dim)
        gb = self.net(z)  # (B, 2C)
        gamma, beta = pt.chunk(gb, chunks=2, dim=1)  # each (B,C)
        return gamma, beta


class ConvBlockFiLM(nn.Module):
    """
    One residual conv block with FiLM conditioning:
      h -> Conv -> FiLM -> SiLU -> Conv -> FiLM -> SiLU -> +residual
    """
    def __init__(self, channels: int, kernel_size: int, film: FiLM):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, padding_mode='replicate')
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, padding_mode='replicate')

        self.act = nn.SiLU()
        self.film = film  # provided externally (shared or per-block)

    def apply_film(self, h: pt.Tensor, z: pt.Tensor) -> pt.Tensor:
        # h: (B,C,N), z: (B,z_dim)
        gamma, beta = self.film(z)         # (B,C), (B,C)
        h = h * (1.0 + gamma[:, :, None])  # broadcast over N; "1+gamma" starts near identity
        h = h + beta[:, :, None]
        return h

    def forward(self, h: pt.Tensor, z: pt.Tensor) -> pt.Tensor:
        res = h

        h = self.conv1(h)
        h = self.apply_film(h, z)
        h = self.act(h)

        h = self.conv2(h)
        h = self.apply_film(h, z)
        h = self.act(h)

        return h + res


class ConvFiLMScore1D(nn.Module):
    """
    Score network for 1D fields with 2 channels (c, phi).

    Forward signature matches your training setup:
      input: x0_or_xt_flat (B, 2*N)
      cond: (B, n_cond)  e.g. (log_l_norm, U0_norm, F_right_norm)
      t: (B,)

    output: score_flat (B, 2*N)
    """
    def __init__(
        self,
        n_grid: int,
        n_cond: int = 3,
        n_time_freq: int = 16,
        base_channels: int = 64,
        n_blocks: int = 10,
        kernel_size: int = 7,
        film_hidden: int = 128,
        film_layers: int = 4,
    ):
        super().__init__()
        self.n_grid = n_grid
        self.n_cond = n_cond

        self.time_emb = TimeEmbedding(n_freq=n_time_freq)
        z_dim = n_cond + 2 * n_time_freq

        # Lift 2 channels -> base_channels
        self.in_proj = nn.Conv1d(2, base_channels, kernel_size=1)

        # FiLM modulators: either one per block or one shared
        self.film_blocks = nn.ModuleList([FiLM(z_dim=z_dim, channels=base_channels, hidden=film_hidden, n_layers=film_layers) for _ in range(n_blocks)])

        # Make n_blocks (default 10) conv-film-conv-film blocks.
        self.blocks = nn.ModuleList([
            ConvBlockFiLM(channels=base_channels, kernel_size=kernel_size, film=self.film_blocks[b]) for b in range(n_blocks)
        ])

        # Project back to 2 channels
        self.out_proj = nn.Conv1d(base_channels, 2, kernel_size=1)
        
        # z -> (gain_c, gain_phi, bias_c, bias_phi)
        self.out_affine = nn.Linear(z_dim, 4)
        nn.init.zeros_(self.out_affine.weight)
        nn.init.zeros_(self.out_affine.bias)

    def forward(self, x_flat: pt.Tensor, t: pt.Tensor, cond: pt.Tensor) -> pt.Tensor:
        """
        x_flat: (B, 2*N)
        t:      (B,)
        cond:   (B, n_cond)
        """
        B = x_flat.shape[0]
        N = self.n_grid
        assert x_flat.shape[1] == 2 * N, f"Expected x_flat second dim {2*N}, got {x_flat.shape[1]}"
        assert cond.shape[0] == B
        assert t.shape[0] == B

        # reshape to (B,2,N)
        x = x_flat.view(B, 2, N).contiguous()

        # conditioning z = [cond, time_embedding(t)]
        emb_t = self.time_emb(t)          # (B, 2*n_time_freq)
        z = pt.cat([cond, emb_t], dim=1)  # (B, z_dim)

        # conv net
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, z)
        out = self.out_proj(h)  # (B,2,N)

        # apply z-dependent affine on the *2 channels*
        gb = self.out_affine(z)           # (B,4)
        gain_off, bias = gb[:, :2], gb[:, 2:]   # each (B,2)

        gain = 1.0 + gain_off             # start near 1
        out = out * gain[:, :, None] + bias[:, :, None]

        return out.view(B, 2 * N)