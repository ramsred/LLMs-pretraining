# structured_noise.py
import torch, torch.nn as nn

class StructuredNoise(nn.Module):
    """
    Adds (1) channel dropout  and  (2) per-dim scaled Gaussian noise.
    Only active during .train().
    """
    def __init__(self, p_drop=0.15, sigma=0.02, momentum=0.1, feat_dim=1024):
        super().__init__()
        self.p_drop = p_drop
        self.sigma = sigma
        # running estimate of per-dim stdev for heteroskedastic scaling
        self.register_buffer("running_std", torch.ones(feat_dim))
        self.momentum = momentum

    def forward(self, x: torch.Tensor):
        if not self.training or self.sigma == 0.0 and self.p_drop == 0.0:
            return x

        # -------- 1. Channel drop-mask ----------
        if self.p_drop > 0:
            keep_mask = torch.rand_like(x) > self.p_drop    # Bernoulli(1-p)
            x = x * keep_mask

        # -------- 2. Heteroskedastic Gaussian ----
        if self.sigma > 0:
            # update running std (on current batch)
            batch_std = x.detach().std(dim=0)
            self.running_std = (1 - self.momentum) * self.running_std + \
                               self.momentum * batch_std
            noise = torch.randn_like(x) * self.running_std * self.sigma
            x = x + noise
        return x
