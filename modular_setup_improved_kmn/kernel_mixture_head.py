import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KernelMixtureHead(nn.Module):
    """
    Same idea as your current 50-bin softmax, but each "bin" is a smooth
    kernel (Gaussian by default) instead of a hard indicator function.
    Output size is unchanged (n_centers), so this is close to a drop-in
    replacement for your current final layer.
    """
 
    def __init__(self, in_features: int, centers: torch.Tensor, bandwidth: float = None,
                 learnable_bandwidth: bool = True):
        """
        centers: (n_centers,) tensor of fixed kernel centers in log10(time)
                 space (e.g. same locations as your current bin midpoints).
        bandwidth: initial kernel std. If None, defaults to ~1.5x the
                   spacing between centers (enough overlap to smooth
                   bin-boundary gradients without blurring true peaks).
        """
        super().__init__()
        self.register_buffer("centers", centers)
        n = len(centers)
        self.proj = nn.Linear(in_features, n)
 
        if bandwidth is None:
            spacing = (centers[1:] - centers[:-1]).mean().item()
            bandwidth = 1.5 * spacing
 
        if learnable_bandwidth:
            self.log_bw = nn.Parameter(torch.log(torch.tensor(float(bandwidth))))
        else:
            self.register_buffer("log_bw", torch.log(torch.tensor(float(bandwidth))))
 
    def weights(self, h):
        return F.softmax(self.proj(h), dim=-1)  # (B, n_centers)
 
    def density(self, h, y_grid):
        """Evaluate predicted density on an arbitrary grid y_grid: (G,)."""
        w = self.weights(h)                                # (B, n_centers)
        bw = self.log_bw.exp().clamp_min(1e-3)
        y_grid = y_grid.view(1, 1, -1).to(h.device)         # (1,1,G)
        centers = self.centers.view(1, -1, 1)               # (1,n_centers,1)
        kernel = torch.exp(-0.5 * ((y_grid - centers) / bw) ** 2)
        kernel = kernel / (bw * np.sqrt(2 * np.pi))
        density = (w.unsqueeze(-1) * kernel).sum(dim=1)     # (B,G)
        return density
 
    def log_bin_probs(self, h, bin_edges, eps=1e-8):
        """
        Returns log-probabilities per bin, (B, n_bins), suitable for direct
        use with nn.KLDivLoss(reduction='batchmean')(log_bin_probs, target_hist)
        -- this is the drop-in replacement for your current
        F.log_softmax(logits) call.
        """
        w = self.weights(h)                                 # (B,n_centers)
        bw = self.log_bw.exp().clamp_min(1e-3)
        edges = bin_edges.to(h.device).view(1, 1, -1)        # (1,1,n_bins+1)
        centers = self.centers.view(1, -1, 1)                # (1,n_centers,1)
        z = (edges - centers) / (bw * np.sqrt(2.0))
        cdf = 0.5 * (1.0 + torch.erf(z))                     # (1,n_centers,n_bins+1)
        bin_mass_per_kernel = cdf[..., 1:] - cdf[..., :-1]   # (1,n_centers,n_bins)
        pred_hist = (w.unsqueeze(-1) * bin_mass_per_kernel).sum(dim=1)  # (B,n_bins)
        pred_hist = pred_hist.clamp_min(eps)
        pred_hist = pred_hist / pred_hist.sum(dim=-1, keepdim=True)
        return pred_hist.log()
