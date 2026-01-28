import torch
import torch.nn.functional as F

class KLLoss(torch.nn.Module):

    def __init__(self, num_classes=19):
        super().__init__()
        self.num_classes = num_classes
        self.kl_loss = torch.nn.KLDivLoss(reduction='none', log_target=False)

    def forward(self, mask_logits, class_logits, outlier_mask):
        if outlier_mask.shape[-2:] != mask_logits.shape[-2:]:
            outlier_mask = F.interpolate(
                outlier_mask.unsqueeze(1).float(),
                size=mask_logits.shape[-2:],
                mode='nearest'
            ).squeeze(1)

        mask_probs = mask_logits.sigmoid()
        class_probs = F.softmax(class_logits[..., :-1], dim=-1)

        pixel_class_mass = torch.einsum("bqk, bqhw -> bkhw", class_probs, mask_probs)
        total_mass = pixel_class_mass.sum(dim=1, keepdim=True) + 1e-6
        pixel_dist = pixel_class_mass / total_mass

        log_pred_dist = torch.log(pixel_dist + 1e-8)
        uniform_target = torch.full_like(pixel_dist, 1.0 / self.num_classes)

        loss_map = self.kl_loss(log_pred_dist, uniform_target).sum(dim=1)

        outlier_mask_flat = outlier_mask.view(-1) > 0.5
        if not outlier_mask_flat.any():
            return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)

        return loss_map.view(-1)[outlier_mask_flat].mean()