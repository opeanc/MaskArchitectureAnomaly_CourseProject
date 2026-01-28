import torch
import torch.nn as nn
import torch.nn.functional as F

class RbALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask_logits, class_logits, outlier_mask):
        # Align Outlier Mask with Logits Resolution
        if outlier_mask.shape[-2:] != mask_logits.shape[-2:]:
            outlier_mask = F.interpolate(
                outlier_mask.unsqueeze(1).float(),
                size=mask_logits.shape[-2:],
                mode='nearest'
            ).squeeze(1)

        # Get Probabilities
        mask_probs = mask_logits.sigmoid()
        class_probs = F.softmax(class_logits, dim=-1)[..., :-1] # Exclude void

        # Calculate Sum of Known Class Probabilities
        pixel_class_probs = torch.einsum("bqk, bqhw -> bkhw", class_probs, mask_probs)
        sum_known_activations = pixel_class_probs.sum(dim=1) # [B, H, W]

        # 4. Filter for Outlier Pixels
        outlier_mask_flat = outlier_mask.view(-1) > 0.5

        if not outlier_mask_flat.any():
            return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)

        outlier_vals = sum_known_activations.view(-1)[outlier_mask_flat]

        # Loss Calculation
        loss = torch.mean(outlier_vals ** 2)

        return loss