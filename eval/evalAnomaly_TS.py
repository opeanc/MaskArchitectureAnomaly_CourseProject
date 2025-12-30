# Copyright (c) OpenMMLab. All rights reserved.

# for macos - FIX CRITICO PER OPENMP
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# IMPOSTAZIONE DEVICE: FORZIAMO CPU PER STABILITÀ SU MAC
device = torch.device("cpu")

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val") 
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=0) 
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    
    # --- NUOVO ARGOMENTO TEMPERATURE SCALING ---
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Temperature scaling value (default=1.0 which means no scaling)')
    
    args = parser.parse_args()
    
    msp_anomaly_score_list = []
    maxlogit_anomaly_score_list = []
    entropy_anomaly_score_list = []
    
    ood_gts_list = []

    if not os.path.exists('results_TS.txt'):
        open('results_TS.txt', 'w').close()
    file = open('results_TS.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)
    print(f"Using Temperature: {args.temperature}")

    model = ERFNet(NUM_CLASSES)
    model = model.to(device)

    def load_my_state_dict(model, state_dict): 
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    key = name.split("module.")[-1]
                    if key in own_state:
                        own_state[key].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))
    print("Model and weights LOADED successfully")
    model.eval()
    
    input_files = args.input
    if len(input_files) == 1 and ('*' in input_files[0] or '?' in input_files[0]):
        file_list = glob.glob(os.path.expanduser(input_files[0]))
    else:
        file_list = input_files

    print(f"Found {len(file_list)} images to process.")

    for path in file_list:
        print(f"Processing: {path}")
        
        try:
            img = Image.open(path).convert('RGB')
            images = input_transform(img).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                # 1. Otteniamo i Logits (Tensor PyTorch)
                result = model(images) 
                
                # --- APPLICAZIONE TEMPERATURE SCALING ---
                # Dividiamo i logits per T prima della softmax.
                # Questo influenza MSP e Entropia. Non influenza MaxLogit (solo scala).
                scaled_result = result / args.temperature

                # 2. Calcoliamo le Probabilità (Softmax) sui logits scalati
                probs = torch.nn.functional.softmax(scaled_result, dim=1)

                # Spostiamo su CPU e convertiamo in numpy
                # Usiamo i logits ORIGINALI per MaxLogit (per purezza, anche se il ranking non cambia)
                logits_np = result.squeeze(0).data.cpu().numpy()
                probs_np = probs.squeeze(0).data.cpu().numpy()

            # ==========================================
            # METODO 1: MSP (Baseline Classica)
            # ==========================================
            msp_score = 1.0 - np.max(probs_np, axis=0)

            # ==========================================
            # METODO 2: Max Logit
            # ==========================================
            max_logit_score = - np.max(logits_np, axis=0)

            # ==========================================
            # METODO 3: Max Entropy
            # ==========================================
            entropy_score = -np.sum(probs_np * np.log(probs_np + 1e-8), axis=0)

            
            msp_anomaly_result = msp_score
            maxlogit_anomaly_result = max_logit_score
            entropy_anomaly_result = entropy_score
            
            # Logica Ground Truth (invariata)
            pathGT = path.replace("images", "labels_masks")                
            if "RoadObsticle21" in pathGT:
               pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT:
               pathGT = pathGT.replace("jpg", "png")                
            if "RoadAnomaly" in pathGT:
               pathGT = pathGT.replace("jpg", "png")  

            if not os.path.exists(pathGT):
                print(f"Warning: Ground Truth not found for {pathGT}, skipping.")
                continue

            mask = Image.open(pathGT)
            mask = target_transform(mask)
            ood_gts = np.array(mask)

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

            if 1 not in np.unique(ood_gts):
                continue              
            else:
                ood_gts_list.append(ood_gts)
                msp_anomaly_score_list.append(msp_anomaly_result)
                maxlogit_anomaly_score_list.append(maxlogit_anomaly_result)
                entropy_anomaly_score_list.append(entropy_anomaly_result)
            
            del result, msp_anomaly_result, max_logit_score, entropy_anomaly_result, ood_gts, mask
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    file.write("\n")

    if len(ood_gts_list) == 0:
        print("No valid images processed or no anomalies found in GT.")
        return
    
    # === MSP RESULTS ===
    print("Calculating metrics with msp for dataset: " + args.subset)
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(msp_anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc_msp = average_precision_score(val_label, val_out)
    fpr_msp = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc_msp*100.0}')
    print(f'FPR@TPR95: {fpr_msp*100.0}')

    # === MAX LOGIT RESULTS ===
    print("Calculating metrics with Max Logit for dataset: " + args.subset)
    anomaly_scores = np.array(maxlogit_anomaly_score_list)
    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    val_out = np.concatenate((ind_out, ood_out))
    prc_auc_logit = average_precision_score(val_label, val_out)
    fpr_logit = fpr_at_95_tpr(val_out, val_label)
    print(f'AUPRC score: {prc_auc_logit*100.0}')
    print(f'FPR@TPR95: {fpr_logit*100.0}')

    # === ENTROPY RESULTS ===
    print("Calculating metrics with Entropy for dataset: " + args.subset)
    anomaly_scores = np.array(entropy_anomaly_score_list)
    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    val_out = np.concatenate((ind_out, ood_out))
    prc_auc_entropy = average_precision_score(val_label, val_out)
    fpr_entropy = fpr_at_95_tpr(val_out, val_label)
    print(f'AUPRC score: {prc_auc_entropy*100.0}')
    print(f'FPR@TPR95: {fpr_entropy*100.0}')

    file.write(f"--- Processing Subset: {args.subset} [Temp={args.temperature}] ---\n")
    file.write(f"MSP       -> AUPRC: {prc_auc_msp*100.0:.2f} | FPR@95: {fpr_msp*100.0:.2f}\n")
    file.write(f"MaxLogit  -> AUPRC: {prc_auc_logit*100.0:.2f} | FPR@95: {fpr_logit*100.0:.2f}\n")
    file.write(f"Entropy   -> AUPRC: {prc_auc_entropy*100.0:.2f} | FPR@95: {fpr_entropy*100.0:.2f}\n")
    file.write("--------------------------------------------\n\n")

    file.close()

if __name__ == '__main__':
    main()