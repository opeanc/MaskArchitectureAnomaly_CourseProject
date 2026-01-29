import torch
import torch.nn.functional as F

class KLLoss(torch.nn.Module):

    def __init__(self, num_classes=19):
        super().__init__()
        self.num_classes = num_classes
        self.kl_loss = torch.nn.KLDivLoss(reduction='none', log_target=False)

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
        class_probs = F.softmax(class_logits[..., :-1], dim=-1)

        # prob that a certain pixel belongs to a certain class
        pixel_class_mass = torch.einsum("bqk, bqhw -> bkhw", class_probs, mask_probs)
        # sum all the probabilities of known classes for each pixel
        total_mass = pixel_class_mass.sum(dim=1, keepdim=True) + 1e-6
        # normalized distribution over classes for each pixel
        pixel_dist = pixel_class_mass / total_mass

        # KL Loss expects log-probabilities for the predictions (+ small constant for numerical stability)
        log_pred_dist = torch.log(pixel_dist + 1e-8)
        # create uniform target distribution (ideal case for outlier pixels)
        uniform_target = torch.full_like(pixel_dist, 1.0 / self.num_classes)

        # distance between the predicted distribution and the uniform distribution
        loss_map = self.kl_loss(log_pred_dist, uniform_target).sum(dim=1)

        outlier_mask_flat = outlier_mask.view(-1) > 0.5
        if not outlier_mask_flat.any():
            return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)

        # selecting only the outlier pixels and return the mean loss
        return loss_map.view(-1)[outlier_mask_flat].mean()