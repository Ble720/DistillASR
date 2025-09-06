import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param=5,
        time_mask_param=15,
        num_freq_masks=1,
        num_time_masks=1,
        time_warp=True,
        max_time_warp=5
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.time_warp = time_warp
        self.max_time_warp = max_time_warp

    def forward(self, x):
        """
        x: (B, D, T) - batch, features, time
        """
        B, D, T = x.shape
        dev = x.device
        cpu = torch.device("cpu")

        # --------- Frequency Masking ----------
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (B,), device=cpu)
            f0 = torch.empty(B, dtype=torch.long, device=cpu)
            for b in range(B):
                high = max(D - f[b].item(), 1)
                f0[b] = torch.randint(0, high, (1,), device=cpu)

            arange = torch.arange(D, device=cpu).view(1, D, 1)
            mask = ((arange >= f0.view(B, 1, 1)) & (arange < (f0 + f).view(B, 1, 1))).float()
            x = x * (1 - mask.to(dev))  # apply mask

        # --------- Time Masking ----------
        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (B,), device=cpu)
            t0 = torch.empty(B, dtype=torch.long, device=cpu)
            for b in range(B):
                high = max(T - t[b].item(), 1)
                t0[b] = torch.randint(0, high, (1,), device=cpu)

            arange = torch.arange(T, device=cpu).view(1, 1, T)
            mask = ((arange >= t0.view(B, 1, 1)) & (arange < (t0 + t).view(B, 1, 1))).float()
            x = x * (1 - mask.to(dev))  # apply mask

        # --------- Time Warping ----------
        if self.time_warp and T > 1:
            warp = torch.randint(-self.max_time_warp, self.max_time_warp + 1, (B,), device=cpu)
            idx = torch.arange(T, device=cpu).view(1, T).repeat(B, 1) + warp.view(B, 1)
            idx = idx.clamp(0, T - 1).to(dev)
            x = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, D, -1))

        return x