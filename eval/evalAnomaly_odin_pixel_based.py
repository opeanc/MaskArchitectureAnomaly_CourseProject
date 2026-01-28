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


def window_inference_odin(model, img):
    """
    Esegue l'inferenza ODIN con perturbazione dell'input 'Per-Crop'.
    """
    x = img.float() / 255.0 if img.max() > 1.0 else img.float()
    
    # Creazione Crops
    crops = torch.cat([x[:, :, :, 0:1024], x[:, :, :, 1024:2048]], dim=0)
    
    # --- 1. Calcolo Perturbazione ---
    # Creiamo una copia del tensore che supporti il gradiente
    crops_input = crops.clone().detach()
    crops_input.requires_grad = True
    
    # Forward Pass iniziale
    mask_logits_list, class_logits_list = model(crops_input)
    mask_logits = mask_logits_list[-1]
    class_logits = class_logits_list[-1]
    
    # Applica Temperature Scaling sui Logits per il calcolo del gradiente
    # Questo serve a calcolare la 'confidenza' calibrata
    class_logits_scaled = class_logits / ODIN_TEMP
    
    # Calcoliamo le probabilità per-pixel (Mask * Softmax(Scaled_Logits))
    # Riutilizziamo la funzione helper. Softmax verrà applicata dentro.
    pixel_probs = to_per_pixel_logits_semantic(mask_logits, class_logits_scaled)
    
    # ODIN Loss: Massimizzare la log-probabilità della classe vincente (In-Distribution Confidence)
    max_scores, _ = torch.max(pixel_probs, dim=1) # [N_crops, H, W]
    loss = torch.log(max_scores + 1e-10).sum()
    
    # Calcolo Gradiente
    model.zero_grad()
    loss.backward()
    
    # Creazione Immagine Perturbata (Gradient Ascent: vogliamo AUMENTARE lo score ID)
    # x_new = x + eps * sign(grad)
    perturbation = ODIN_EPS * crops_input.grad.sign()
    crops_perturbed = (crops_input + perturbation).detach()
    
    # --- 2. Inference Finale su Immagine Perturbata ---
    with torch.no_grad():
        m_logits_p_list, c_logits_p_list = model(crops_perturbed)

    m_logits_p = m_logits_p_list[-1]
    c_logits_p = c_logits_p_list[-1]
    
    # Ritorniamo i logit. 
    # NOTA: Dividiamo i class_logits per T qui, così la successiva 'softmax' (in to_per_pixel...)
    # calcolerà effettivamente softmax(logits / T).
    return m_logits_p, c_logits_p / ODIN_TEMP


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
        m_logits_odin, c_logits_odin = window_inference_odin(net, images)

        with torch.no_grad(): # Post-processing ODIN non richiede gradiente
            # c_logits_odin è già diviso per T. La funzione sotto fa softmax.
            pixel_logits_crops_odin = to_per_pixel_logits_semantic(m_logits_odin, c_logits_odin)
            
            # stitching ODIN
            full_logits_odin = torch.cat([pixel_logits_crops_odin[0:1], pixel_logits_crops_odin[1:2]], dim=-1)
            full_logits_odin = F.interpolate(full_logits_odin, size=(1024, 2048), mode="bilinear", align_corners=False)
            
            # ODIN Score = MSP su output perturbato e scalato
            # full_logits_odin sono già probabilità pesate (sigm * softmax).
            # Ma attenzione: to_per_pixel restituisce PROBABILITÀ (einsum di probs).
            # Quindi full_logits_odin contiene già valori ~ [0,1].
            # L'MSP si calcola come (1 - max(pixel_probs)).
            
            anomaly_result_ODIN = 1.0 - torch.max(full_logits_odin, dim=1)[0].squeeze().cpu().numpy()

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
