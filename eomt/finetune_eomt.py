import importlib
import os
import sys
import torch
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F
from training.rba_loss import RbALoss
from training.kll_loss import KLLoss
from training.mask_classification_loss import MaskClassificationLoss
from datasets.anomaly_cityscapes import CityscapesAnomalyDataset
from torch.utils.data import DataLoader
import yaml
import importlib


### GLOBAL VARIABLES AND PATHS ###
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


### HELPER FUNCTIONS ###
def freeze_lower_layers(model):
    # freezing all the layers
    for param in model.parameters():
        param.requires_grad = False
    unfrozen_layers = []

    print("\n[INFO] configuring fine-tuning...")
    targets = ["class_head", "mask_head"]

    for name, param in model.named_parameters():
        if any(t in name for t in targets):
            param.requires_grad = True
            unfrozen_layers.append(name)
            print(f" -> unfreeze: {name}")

    if len(unfrozen_layers) == 0:
        print("   WARNING: No layer 'class_embed' or 'mask_embed' found!")
    else:
        print(f"  Success! Unlocked {len(unfrozen_layers)} parameter tensors.")

    return model


def split_batch_left_right(images, masks):
    
    if masks.dim() == 3:
        masks = masks.unsqueeze(1) # from [B, H, W] to [B, 1, H, W]

    h, w = images.shape[2], images.shape[3]
    half_w = w // 2 # cut point

    # images and masks are concatenated left-right in the batch dimension
    # i.e. [B*2, C, H, W//2]
    images_out = torch.cat([images[..., :half_w], images[..., half_w:]], dim=0)
    masks_out = torch.cat([masks[..., :half_w], masks[..., half_w:]], dim=0)
    return images_out, masks_out


def prepare_targets_for_criterion(targets, device):
    """
    Converts the ground truth masks in EoMT model output

    Args:
        targets: ground truth mask
        device: CUDA or CPU
    """
    new_targets = []

    for i in range(targets.shape[0]): # targets.shape[0] = Batch
        gt = targets[i] # we take one mask at time from the batch
        if gt.dim() == 3: gt = gt.squeeze(0) 
        labels = torch.unique(gt) # classes contained in the mask
        labels = labels[labels != 255] # removing void

        if len(labels) == 0: # avoiding training crash (if the crop contains only void) by returning empty tensor
            new_targets.append({
                "labels": torch.tensor([], device=device, dtype=torch.long),
                "masks": torch.zeros((0, gt.shape[0], gt.shape[1]), device=device, dtype=torch.bool)
            })
            continue

        masks_list = []
        for label in labels:
            masks_list.append(gt == label) # creating one binary mask (True/False) for each class - ONE FOR EACH CLASS
        masks_tensor = torch.stack(masks_list, dim=0) # [num_classes, H, W]

        new_targets.append({
            "labels": labels.long().to(device),
            "masks": masks_tensor.to(device)
        })
    return new_targets


def get_optimizer(model, learning_rate=1e-4, weight_decay=0.05):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found!")
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def mask_to_tensor_transform(x):
    # converts PIL Image or array into Long tensor [1, H, W]
    return torch.from_numpy(np.array(x)).long().unsqueeze(0)

### TRAINING SETTINGS ###
class TrainConfig:
    def __init__(self, args):
        # paths taken from arguments
        self.CITYSCAPES_DIR = args.cityscapes_dir
        self.OBJ_DIR = args.obj_dir
        self.CHECKPOINT_DIR = args.save_dir
        
        # other hyperparameters taken from arguments or default
        self.BATCH_SIZE = args.batch_size
        self.NUM_WORKERS = args.num_workers
        self.EPOCHS = args.epochs
        self.LR = args.lr
        
        self.LAMBDA_RBA = 0.5
        self.ANOMALY_ID = 254
        self.VOID_ID = 255
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.SUBSET = args.subset


input_transform_cityscapes = Compose(
    [
        Resize((1024, 2048), Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform_cityscapes = Compose(
    [
        Resize((1024, 2048), Image.NEAREST),
        mask_to_tensor_transform
    ]
)


### MAIN FUNCTION ###
def main():
    parser = ArgumentParser()
    
    # arguments for paths
    parser.add_argument('--cityscapes_dir', required=True, help="Path to Cityscapes root directory")
    parser.add_argument('--obj_dir', required=True, help="Path to Anomaly Objects directory")
    parser.add_argument('--save_dir', default=os.path.join(PROJECT_ROOT, "trained_models"), help="Directory to save checkpoints")
    parser.add_argument('--config_path', default=os.path.join(PROJECT_ROOT, "eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"))
    
    # arguments for loading pretrained weights
    parser.add_argument('--pretrained_weights', required=True, help="Path to .bin or .pth pretrained weights")
    
    # arguments for training
    parser.add_argument('--subset', default="train")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    cfg = TrainConfig(args)

    print(f"[INFO] Project Root detected: {PROJECT_ROOT}")
    print(f"[INFO] Saving models to: {cfg.CHECKPOINT_DIR}")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # encoder instance (ViT-Adapter)
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    # 1024x1024
    encoder = encoder_cls(img_size=(1024, 1024), **encoder_cfg.get("init_args", {}))

    # network instance (EoMT)
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    model = network_cls(
        masked_attn_enabled=False,
        num_classes=19,
        encoder=encoder,
        **network_kwargs,
    )

    if cfg.DEVICE.type == 'cuda':
        # wrap in DataParallel if using CUDA
        model = torch.nn.DataParallel(model).to(cfg.DEVICE)
        print(f"Model moved to GPU: {cfg.DEVICE}")
    else:
        model = model.to(cfg.DEVICE) # move to CPU
        print(f"Model moved to CPU: {cfg.DEVICE}")

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            clean_name = name.replace("network.", "").replace("module.", "")
            found = False

            for k_model in own_state.keys():
                if k_model == clean_name or k_model.endswith(clean_name):
                    if own_state[k_model].shape == param.shape:
                        own_state[k_model].copy_(param)
                        found = True
                        break

            if not found and "criterion" not in name:
                print(f"Parameter not found in the model: {name}")

        return model
    
    # Load checkpoint using the determined device
    checkpoint = torch.load(args.pretrained_weights, map_location=cfg.DEVICE)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model = load_my_state_dict(model, state_dict)
    model = freeze_lower_layers(model)
    model.train()
    print ("Model and weights LOADED successfully")

    # --- CONFIGURAZIONE LOSS ---
    useRbaLoss = True  # True = RbA Loss, False = KL Loss

    REAL_BATCH_SIZE = cfg.BATCH_SIZE
    TARGET_BATCH_SIZE = 16
    ACCUMULATION_STEPS = max(1, TARGET_BATCH_SIZE // REAL_BATCH_SIZE)
    print(f"[Main] Gradient Accumulation: step ogni {ACCUMULATION_STEPS} batch.")

    dataset = CityscapesAnomalyDataset(cfg.CITYSCAPES_DIR, cfg.OBJ_DIR, input_transform_cityscapes, target_transform_cityscapes, subset=cfg.SUBSET)
    loader = DataLoader(dataset, num_workers=cfg.NUM_WORKERS, batch_size=cfg.BATCH_SIZE, shuffle=True)

    optimizer = get_optimizer(model, learning_rate=cfg.LR)

    # --- SELEZIONE LOSS OOD ---
    if useRbaLoss:
        print("[Main] Selected Loss: RbALoss (RbA Maximization)")
        ood_loss_fn = RbALoss()
        lambda_ood = 0.5
    else:
        print("[Main] Selected Loss: KLLoss (MSP Maximization - Entropy Maximization)")
        ood_loss_fn = KLLoss(num_classes=19)
        lambda_ood = 0.5

    clean_criterion = MaskClassificationLoss(
        num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75,
        mask_coefficient=5.0, dice_coefficient=5.0, class_coefficient=2.0,
        num_labels=19, no_object_coefficient=0.1
    ).to(cfg.DEVICE)

    print("[Main] Begin Training...")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(cfg.EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(pbar):
            images = images.to(cfg.DEVICE)
            masks = masks.to(cfg.DEVICE)
            images_in, masks_in = split_batch_left_right(images, masks)

            mask_logits_list, class_logits_list = model(images_in)
            pred_mask_logits = mask_logits_list[-1] # taking only the output of the final layer
            pred_class_logits = class_logits_list[-1] # taking only the output of the final layer

            # determine if there are outlier pixels in the batch
            is_outlier_batch = (masks_in == cfg.ANOMALY_ID).any()

            if is_outlier_batch:
                # prepares anomaly binary mask
                outlier_mask = (masks_in == cfg.ANOMALY_ID).float().squeeze(1)
                # calculate OOD loss
                ood_val = ood_loss_fn(pred_mask_logits, pred_class_logits, outlier_mask)
                loss = lambda_ood * ood_val

                # Update log string
                loss_type = "RbA_OE" if useRbaLoss else "MSP_OE"
            else:
                # transform ground truth masks to EoMT format
                clean_targets = prepare_targets_for_criterion(masks_in, cfg.DEVICE)

                # standard segmentation loss
                loss_dict = clean_criterion(
                    masks_queries_logits=pred_mask_logits,
                    class_queries_logits=pred_class_logits,
                    targets=clean_targets
                )

                loss = (loss_dict["loss_mask"] * 5.0 +
                        loss_dict["loss_dice"] * 5.0 +
                        loss_dict["loss_cross_entropy"] * 2.0)
                loss_type = "Clean"

            # Gradient Accumulation
            loss = loss / ACCUMULATION_STEPS
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward() # calculate gradients
                current_loss_val = loss.item() * ACCUMULATION_STEPS
            else:
                current_loss_val = 0.0

            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
                optimizer.step() # update weights
                optimizer.zero_grad() # reset gradients
                pbar.set_postfix({"Type": loss_type, "Loss": f"{current_loss_val:.4f} (Step)"})
            else:
                pbar.set_postfix({"Type": loss_type, "Loss": f"{current_loss_val:.4f} (Acc)"})

        save_path = os.path.join(cfg.CHECKPOINT_DIR, f"finetuned_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved: {save_path}")


if __name__ == "__main__":
    main()