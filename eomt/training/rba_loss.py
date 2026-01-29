import torch
import torch.nn as nn
import torch.nn.functional as F

class RbALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask_logits, class_logits, outlier_mask):
        """
        Args:
            mask_logits: mask logits [B, Q = 100, H, W]
            class_logits: class logits including void class [B, Q = 100, num_classes + 1]
            outlier_mask: ground truth mask for outlier pixels [B, H, W]
        """
        # resizing Outlier Mask to Logits Resolution (from 1024x1024 to 256x256)
        if outlier_mask.shape[-2:] != mask_logits.shape[-2:]:
            outlier_mask = F.interpolate(
                outlier_mask.unsqueeze(1).float(),
                size=mask_logits.shape[-2:],
                mode='nearest'
            ).squeeze(1)

        # probs that a query is active in a certain pixel
        mask_probs = mask_logits.sigmoid()
        # prob that a query belongs to a certain class
        class_probs = F.softmax(class_logits, dim=-1)[..., :-1] # Exclude void

        # -- PIXEL-WISE RECONSTRUCTION --
        # prob that a certain pixel belongs to a certain class
        pixel_class_probs = torch.einsum("bqk, bqhw -> bkhw", class_probs, mask_probs)
        # sum all all the probabilities of known classes for each pixel
        sum_known_activations = pixel_class_probs.sum(dim=1) # [B, H, W]

        # -- ENERGY MINIMIZATION --
        # looking only pixels where there is the anomaly ( > 0.5)
        # Gamma_out
        outlier_mask_flat = outlier_mask.view(-1) > 0.5

        if not outlier_mask_flat.any():
            return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)

        # take only the predicted pixels where there is an anomaly (i ∈ Gamma_out)
        outlier_vals = sum_known_activations.view(-1)[outlier_mask_flat]

        # Loss_RBA
        loss = torch.mean(outlier_vals ** 2)

        return loss