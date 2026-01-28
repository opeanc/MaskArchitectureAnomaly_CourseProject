# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
weightspath = os.path.abspath(os.path.join(current_dir, "..", "trained_models", "eomt_cityscapes.bin"))
import sys
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
eomt_path = os.path.join(project_root, 'eomt')
if project_root not in sys.path:
    sys.path.append(project_root)
if eomt_path not in sys.path:
    sys.path.append(eomt_path)
from eomt.models.vit import ViT
from eomt.models.eomt import EoMT
import yaml
import importlib
import matplotlib.pyplot as plt

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

# --- CONFIGURAZIONE ODIN ---
ODIN_TEMP = 1000.0  # Temperatura alta
ODIN_EPS = 0.0014   # Magnitudo perturbazione (ImageNet value)

input_transform = Compose(
    [
        Resize((1024, 2048), Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform = Compose(
    [
        Resize((1024, 2048), Image.NEAREST),
    ]
)


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


def window_inference_odin_query_level_vis(model, img):
    """
    Versione per visualizzazione: Ritorna anche i crops perturbati
    """
    x = img.float() / 255.0 if img.max() > 1.0 else img.float()
    crops = torch.cat([x[:, :, :, 0:1024], x[:, :, :, 1024:2048]], dim=0)
    
    # 1. SETUP PERTURBAZIONE
    crops_input = crops.clone().detach()
    crops_input.requires_grad = True
    
    # 2. PRIMA PASSATA (Clean)
    mask_logits_list, class_logits_list = model(crops_input)
    class_logits = class_logits_list[-1] 
    class_logits_scaled = class_logits / ODIN_TEMP 

    with torch.enable_grad(): 
        probs = F.softmax(class_logits_scaled, dim=-1) 
        max_vals, max_indices = torch.max(probs, dim=-1)
        
        VOID_IDX = 19 
        valid_queries_mask = (max_indices != VOID_IDX) & (max_vals > 0.1)
        
        if valid_queries_mask.sum() > 0:
            selected_probs = max_vals[valid_queries_mask]
            loss = torch.log(selected_probs + 1e-10).sum()
            model.zero_grad()
            loss.backward()
            perturbation = ODIN_EPS * crops_input.grad.sign()
            crops_perturbed = (crops_input + perturbation).detach()
        else:
            crops_perturbed = crops_input.detach()

    # 3. SECONDA PASSATA (Perturbed)
    with torch.no_grad():
        m_logits_p_list, c_logits_p_list = model(crops_perturbed)
        m_logits_p = m_logits_p_list[-1]
        c_logits_p = c_logits_p_list[-1]

    # Ritorna anche crops_input (originale crop) e crops_perturbed (con rumore)
    return m_logits_p, c_logits_p / ODIN_TEMP, crops_input, crops_perturbed


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="epoch_106-step_19902_eomt.ckpt")
    parser.add_argument('--loadModel', default="../eomt/mofrld/eomt.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="../../cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    anomaly_score_list = {
        "MSP": list(),
        "ODIN": list()
    }
    ood_gts_list = []

    if not os.path.exists('results_eomt.txt'):
        open('results_eomt.txt', 'w').close()
    file = open('results_eomt.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

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


    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        #images = images.permute(0,3,1,2)
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

            full_probs = F.softmax(full_logits, dim=1)
            
            
            # MSP
            anomaly_result_MSP = 1.0 - torch.max(full_probs, dim=1)[0].squeeze().cpu().numpy()
            

        # --- ODIN INFERENCE (Perturbed Input) ---
        # Richiede gradiente -> No torch.no_grad() globale, ma gestito in window_inference_odin
        net = model.module if hasattr(model, 'module') else model
    
        # CHIAMATA ALLA FUNZIONE _VIS
        m_logits_odin, c_logits_odin, crops_orig, crops_pert = window_inference_odin_query_level_vis(net, images)

        with torch.no_grad(): 
            pixel_logits_crops_odin = to_per_pixel_logits_semantic(m_logits_odin, c_logits_odin)
            
            # Stitching LOGITS ODIN
            full_logits_odin = torch.cat([pixel_logits_crops_odin[0:1], pixel_logits_crops_odin[1:2]], dim=-1)
            full_logits_odin = F.interpolate(full_logits_odin, size=(1024, 2048), mode="bilinear", align_corners=False)
            
            # Calcolo Anomaly Map ODIN
            anomaly_result_ODIN = 1.0 - torch.max(full_logits_odin, dim=1)[0].squeeze().cpu().numpy()

            # --- SEZIONE VISUALIZZAZIONE PER IL PAPER ---
            # Salviamo un'immagine di esempio (es. la prima o una specifica che sai avere problemi)
            if True: # Metti una condizione se vuoi farlo solo per alcune immagini
                
                # A. Ricostruzione Immagine Perturbata (Stitching dei crop)
                # crops_pert è [2, 3, 1024, 1024]. Uniamoli per formare l'immagine intera
                pert_left = crops_pert[0].permute(1, 2, 0).cpu().numpy()
                pert_right = crops_pert[1].permute(1, 2, 0).cpu().numpy()
                orig_left = crops_orig[0].permute(1, 2, 0).cpu().numpy()
                orig_right = crops_orig[1].permute(1, 2, 0).cpu().numpy()
                
                full_img_pert = np.concatenate([pert_left, pert_right], axis=1) # [1024, 2048, 3]
                full_img_orig = np.concatenate([orig_left, orig_right], axis=1)
                
                # Calcolo del Rumore (Amplificato per visualizzazione)
                noise_map = (full_img_pert - full_img_orig)
                # Normalizziamo il noise tra 0 e 1 per vederlo bene
                noise_vis = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min() + 1e-8)
                
                # B. Differenza Mappe (ODIN - CLEAN)
                # Dove è POSITIVO (Rosso): ODIN è più anomalo del Clean (Falso Positivo?)
                # Dove è NEGATIVO (Blu): ODIN è più sicuro del Clean (Miglioramento?)
                # Se vedi bordi rossi/blu intorno agli oggetti, è la prova dell'espansione impropria/jitter.
                diff_map = anomaly_result_ODIN - anomaly_result_MSP # Assicurati che siano stesse dimensioni
                
                # PLOTTING
                plt.figure(figsize=(20, 10))
                
                # 1. Immagine Originale con GT sovrapposta (Opzionale, qui metto immagine raw)
                plt.subplot(2, 3, 1)
                plt.imshow(full_img_orig)
                plt.title("Input Image")
                plt.axis('off')
                
                # 3. Clean Anomaly Map (MSP)
                plt.subplot(2, 3, 4)
                plt.imshow(anomaly_result_MSP, cmap='jet', vmin=0, vmax=1)
                plt.title("Before ODIN (Standard MSP)")
                plt.colorbar()
                plt.axis('off')
                
                # 4. ODIN Anomaly Map
                plt.subplot(2, 3, 5)
                plt.imshow(anomaly_result_ODIN, cmap='jet', vmin=0, vmax=1)
                plt.title("After ODIN (Query-Level)")
                plt.colorbar()
                plt.axis('off')
                
                # 5. Mappa delle Differenze (Artifact Map)
                plt.subplot(2, 3, 6)
                # Usa cmap 'seismic' o 'bwr' per vedere positivo/negativo
                # Valori vicini a 0 sono bianchi (nessun cambio).
                plt.imshow(diff_map, cmap='seismic', vmin=-0.5, vmax=0.5) 
                plt.title("Difference (ODIN - Clean)\nRed=Artifacts/Increased Anomaly")
                plt.colorbar()
                plt.axis('off')
                
                # Salva
                save_name = os.path.basename(path).replace('.webp', '_analysis.png').replace('.png', '_analysis.png')
                plt.tight_layout()
                plt.savefig(f"./debug_odin/{save_name}") # Assicurati che la cartella esista
                plt.close()
                print(f"Visualizzazione salvata: {save_name}")

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask).squeeze()

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
            ood_gts_list.append(ood_gts.flatten().astype(np.uint8))
            anomaly_score_list["MSP"].append(anomaly_result_MSP.flatten().astype(np.float16))
            anomaly_score_list["ODIN"].append(anomaly_result_ODIN.flatten().astype(np.float16))
        del anomaly_result_MSP, anomaly_result_ODIN, ood_gts, mask
        torch.cuda.empty_cache()

    for method in anomaly_score_list.keys():
        file.write( f"\n\t\t{method} => ")

        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list[method])

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        print(f'{method} AUPRC score: {prc_auc*100.0}')
        print(f'{method} FPR@TPR95: {fpr*100.0}')

        file.write(('\t\t\tAUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()
