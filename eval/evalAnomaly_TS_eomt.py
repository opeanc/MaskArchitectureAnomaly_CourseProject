import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
current_dir = os.path.dirname(os.path.abspath(__file__))
weightspath = os.path.abspath(os.path.join(current_dir, "..", "trained_models", "eomt_cityscapes.bin"))
import cv2
import glob
import torch
torch.set_num_threads(1)
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import gc
import torch.nn.functional as F
from ood_metrics import fpr_at_95_tpr 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
eomt_path = os.path.join(project_root, 'eomt')
import sys
if project_root not in sys.path:
    sys.path.append(project_root)
if eomt_path not in sys.path:
    sys.path.append(eomt_path)
from eomt.models.vit import ViT
from eomt.models.eomt import EoMT
import yaml
import importlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

def to_per_pixel_logits_semantic(mask_logits, class_logits):
    # merge transformer's queries and masks
    # Sigmoid(mask) * Softmax(class)
    return torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1], # Esclude l'ultima classe 'null'
    )

def window_inference(model, img, img_size=(640, 640)):
    x = img.float() / 255.0 if img.max() > 1.0 else img.float()
    
    # cropping
    # left piece (0:1024), righe piece (1024:2048)
    crops = torch.cat([x[:, :, :, 0:1024], x[:, :, :, 1024:2048]], dim=0) # [2, 3, 512, 512]
    
    mask_logits_list, class_logits_list = model(crops)
    
    mask_logits = mask_logits_list[-1] # [2, 100, H_m, W_m]
    class_logits = class_logits_list[-1] # [2, 100, 20]
    
    return mask_logits, class_logits

def load_ground_truth(path, target_transform):
    """Helper function to load and process masks consistent with RoadObstacle21/LostAndFound"""
    pathGT = path.replace("images", "labels_masks")
    
    # Fix estensioni file
    if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:      pathGT = pathGT.replace("jpg", "png")
    if "RoadAnomaly" in pathGT:    pathGT = pathGT.replace("jpg", "png")

    if not os.path.exists(pathGT):
        return None

    mask = Image.open(pathGT)
    mask = target_transform(mask)
    ood_gts = np.array(mask)

    # Mappatura classi specifiche dei dataset
    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts==2), 1, ood_gts)
    if "LostAndFound" in pathGT:
        ood_gts = np.where((ood_gts==0), 255, ood_gts)
        ood_gts = np.where((ood_gts==1), 0, ood_gts)
        ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
    if "Streethazard" in pathGT:
        ood_gts = np.where((ood_gts==14), 255, ood_gts)
        ood_gts = np.where((ood_gts<20), 0, ood_gts)
        ood_gts = np.where((ood_gts==255), 1, ood_gts)

    return ood_gts

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="./RoadObsticle21/images/*.webp", nargs="+")  
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val") 
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--best-temperature', action='store_true', help="Find best T on the dataset")
    
    args = parser.parse_args()
    
    # Setup Output File
    if not os.path.exists('results_TS_eomt.txt'): open('results_TS_eomt.txt', 'w').close()
    file = open('results_TS_eomt.txt', 'a')

    # Load Model
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights
    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)
    print(f"Using Temperature: {args.temperature}")

    config_path = os.path.join(project_root, "eomt", "configs", "dinov2", "cityscapes", "semantic", "eomt_base_640.yaml")
    with open(config_path, "r") as f:
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

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            # removing network. and module.
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weightspath, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model = load_my_state_dict(model, state_dict)
    print ("Model and weights LOADED successfully")
    model.eval()

    # Prepare File List
    input_files = args.input
    if len(input_files) == 1 and ('*' in input_files[0] or '?' in input_files[0]):
        file_list = glob.glob(os.path.expanduser(input_files[0]))
    else:
        file_list = input_files
    print(f"Found {len(file_list)} images.")

    # -------------------------------------------------------------------------
    # MODALITÀ 1: STANDARD (Una sola temperatura)
    # -------------------------------------------------------------------------
    if not args.best_temperature:
        msp_anomaly_score_list = []
        ood_gts_list = []
        
        for path in file_list:
            print(f"Processing: {path}")
            try:
                img = Image.open(path).convert('RGB')
                images = input_transform(img).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    net = model.module if hasattr(model, 'module') else model
                    m_logits, c_logits = window_inference(net, images)

                    # semantic fusion (Sigmoid * Softmax)
                    pixel_logits_crops = to_per_pixel_logits_semantic(m_logits, c_logits)

                    # stitching
                    # 1024x1024 | 1024x1024 -> 1024x2048
                    full_logits = torch.cat([pixel_logits_crops[0:1], pixel_logits_crops[1:2]], dim=-1)

                    # upsample
                    full_logits = F.interpolate(
                        full_logits, 
                        size=(1024, 2048),
                        mode="bilinear", 
                        align_corners=False
                    )

                    scaled_logits = full_logits / args.temperature
                    probs = F.softmax(scaled_logits, dim=1)
                    
                    probs_np = probs.squeeze(0).cpu().numpy()   # (20, H, W)
                
                msp_score = 1.0 - np.max(probs_np, axis=0)

                ood_gts = load_ground_truth(path, target_transform)
                if ood_gts is None or 1 not in np.unique(ood_gts): 
                    continue
                
                valid_mask = (ood_gts == 0) | (ood_gts == 1)
                
                ood_gts_list.append(ood_gts[valid_mask])
                msp_anomaly_score_list.append(msp_score[valid_mask])

                del logits, probs, probs_np, msp_score, ood_gts, valid_mask, images

            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if len(ood_gts_list) > 0:
            print("Calculating metrics...")
            all_labels = np.concatenate(ood_gts_list)
            all_msp = np.concatenate(msp_anomaly_score_list)
            
            auprc = average_precision_score(all_labels, all_msp)
            fpr95 = fpr_at_95_tpr(all_msp, all_labels)  # <--- AGGIUNTO QUI
            
            print(f"Temp={args.temperature} | MSP AuPRC: {auprc*100:.2f}% | FPR95: {fpr95*100:.2f}%")
            file.write(f"Subset: {args.subset} T={args.temperature} -> MSP AuPRC: {auprc*100:.2f} | FPR95: {fpr95*100:.2f}\n")
        else:
            print("No valid data found to calculate metrics.")

    # -------------------------------------------------------------------------
    # MODALITÀ 2: BEST TEMPERATURE SEARCH (EoMT Adapted)
    # -------------------------------------------------------------------------
    else:
        print("\n--- MODE: BEST TEMPERATURE SEARCH - EoMT ---")
        all_logits_list = []
        all_labels_list = []

        print("Extracting logits from model (EoMT Inference)...")
        for i, path in enumerate(file_list):
            if i % 10 == 0: print(f"   Image {i}/{len(file_list)}")
            
            # Load GT first to check validity
            ood_gts = load_ground_truth(path, target_transform)
            if ood_gts is None or 1 not in np.unique(ood_gts): continue

            try:
                img = Image.open(path).convert('RGB')
                images = input_transform(img).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    # --- EoMT INFERENCE LOGIC START ---
                    net = model.module if hasattr(model, 'module') else model
                    m_logits, c_logits = window_inference(net, images)

                    # semantic fusion
                    pixel_logits_crops = to_per_pixel_logits_semantic(m_logits, c_logits)

                    # stitching
                    full_logits = torch.cat([pixel_logits_crops[0:1], pixel_logits_crops[1:2]], dim=-1)

                    # upsample
                    full_logits = F.interpolate(
                        full_logits, 
                        size=(1024, 2048),
                        mode="bilinear", 
                        align_corners=False
                    ) # [1, 19, 1024, 2048]
                    # --- EoMT INFERENCE LOGIC END ---
                    
                    # Flattening spaziale per salvare RAM e CPU
                    # Usiamo full_logits.shape[1] per gestire le classi (dovrebbe essere 19)
                    num_cls = full_logits.shape[1]
                    logits_flat = full_logits.squeeze(0).permute(1, 2, 0).reshape(-1, num_cls).cpu().numpy()
                    
                # Flatten mask
                mask_flat = ood_gts.flatten()
                
                # Filtra pixel validi subito
                valid_idx = (mask_flat == 0) | (mask_flat == 1)
                
                all_logits_list.append(logits_flat[valid_idx])
                all_labels_list.append(mask_flat[valid_idx])
                
            except Exception as e:
                print(f"Skipping {path}: {e}")

        # Pulizia RAM
        del model
        if 'images' in locals(): del images
        torch.cuda.empty_cache()
        gc.collect()
        print("Model deleted from GPU to free RAM.")

        print("Concatenating Arrays...")
        if len(all_logits_list) == 0:
            print("No valid data found.")
            return

        all_logits = np.concatenate(all_logits_list, axis=0)
        all_labels = np.concatenate(all_labels_list, axis=0)
        
        print(f"Total Pixels Analyzed: {all_labels.shape[0]}")
        
        print("Finding the best temperature...")
        candidates = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 100.0, 1000.0]
        best_t, best_score = 1.0, 0.0

        for t in candidates:
            # Scalatura
            scaled = all_logits / t
            
            # Softmax stabile
            shift = np.max(scaled, axis=1, keepdims=True)
            exp_l = np.exp(scaled - shift)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            
            # MSP Score (1 - max_prob)
            scores = 1.0 - np.max(probs, axis=1)
            
            # Calcolo AUPRC
            auprc = average_precision_score(all_labels, scores)
            print(f"   T={t:<5} -> MSP AuPRC: {auprc*100:.2f}%")
            
            if auprc > best_score:
                best_score = auprc
                best_t = t

        # CALCOLO FINALE DI FPR95 PER IL BEST T
        print(f"\nRecalculating FPR95 for Best T = {best_t}...")
        scaled_best = all_logits / best_t
        shift_best = np.max(scaled_best, axis=1, keepdims=True)
        exp_l_best = np.exp(scaled_best - shift_best)
        probs_best = exp_l_best / np.sum(exp_l_best, axis=1, keepdims=True)
        scores_best = 1.0 - np.max(probs_best, axis=1)
        
        best_fpr95 = fpr_at_95_tpr(scores_best, all_labels)

        print(f"Best T = {best_t} | AuPRC: {best_score*100:.2f}% | FPR95: {best_fpr95*100:.2f}%")
        file.write(f"BEST T SEARCH -> T={best_t} | AuPRC: {best_score*100:.2f} | FPR95: {best_fpr95*100:.2f}\n")

    file.close()

if __name__ == '__main__':
    main()