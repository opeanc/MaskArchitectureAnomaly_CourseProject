import os
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import math

# --- FIX PATH ---
sys.path.append(os.getcwd())

# --- IMPORT CLASSI MODELLO ---
try:
    from models.vit import ViT
    from models.eomt import EoMT
except ImportError as e:
    print("ERRORE CRITICO DI IMPORTAZIONE:")
    print(f"{e}")
    sys.exit(1)

# --- IMPORT METRICHE ---
try:
    from ood_metrics import fpr_at_95_tpr
    from sklearn.metrics import average_precision_score
except ImportError:
    print("Error: Librerie metriche mancanti. Esegui: pip install ood-metrics scikit-learn")
    sys.exit(1)

from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def interpolate_pos_embed(checkpoint_pos_embed, model_pos_embed):
    """
    Interpola i positional embeddings del checkpoint per adattarli alla dimensione attuale del modello.
    checkpoint_pos_embed: [1, N_ckpt, C]
    model_pos_embed: [1, N_model, C]
    """
    if checkpoint_pos_embed.shape == model_pos_embed.shape:
        return checkpoint_pos_embed

    print(f"Interpolating pos_embed: {checkpoint_pos_embed.shape} -> {model_pos_embed.shape}")
    
    # Rimuovi class token se presente (DINOv2 di solito usa register tokens o solo patch tokens)
    # Assumiamo struttura standard ViT [1, N_patches, Dim]
    
    N_ckpt = checkpoint_pos_embed.shape[1]
    N_model = model_pos_embed.shape[1]
    dim = checkpoint_pos_embed.shape[2]
    
    # Calcola le dimensioni della griglia quadrata
    grid_size_ckpt = int(math.sqrt(N_ckpt))
    grid_size_model = int(math.sqrt(N_model))
    
    # Reshape a griglia [1, Dim, Grid, Grid] per interpolazione
    # Permute: [1, N, C] -> [1, C, N] -> [1, C, H, W]
    pos_embed_grid = checkpoint_pos_embed.permute(0, 2, 1).reshape(1, dim, grid_size_ckpt, grid_size_ckpt)
    
    # Interpolazione Bicubica
    pos_embed_resized = F.interpolate(
        pos_embed_grid, 
        size=(grid_size_model, grid_size_model), 
        mode='bicubic', 
        align_corners=False
    )
    
    # Torna a sequenza [1, N_new, C]
    pos_embed_new = pos_embed_resized.flatten(2).transpose(1, 2)
    
    return pos_embed_new

def load_checkpoint(model, ckpt_path):
    """Carica i pesi gestendo i prefissi e interpolando pos_embed se necessario."""
    print(f"Loading weights from {ckpt_path}...")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        k_clean = k.replace("network.", "").replace("model.", "").replace("net.", "").replace("module.", "")
        new_state_dict[k_clean] = v

    # --- FIX CRITICO: Interpolazione Positional Embeddings ---
    if "encoder.backbone.pos_embed" in new_state_dict:
        ckpt_pos = new_state_dict["encoder.backbone.pos_embed"]
        model_pos = model.encoder.backbone.pos_embed
        
        # Se le dimensioni non coincidono, interpola!
        if ckpt_pos.shape != model_pos.shape:
            new_state_dict["encoder.backbone.pos_embed"] = interpolate_pos_embed(ckpt_pos, model_pos)

    # Caricamento strict=False
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}")
    return model

def get_rba_score(mask_logits, class_logits):
    """
    Calcola l'Anomaly Score usando RbA (Rejected by All).
    """
    # 1. Softmax sulle classi (escludendo l'ultima classe background/void)
    scores = F.softmax(class_logits[0], dim=-1)[:, :-1] # [Q, 19]
    
    # Max score per ogni query
    max_scores, _ = scores.max(dim=-1) # [Q]
    
    # 2. Sigmoid sulle maschere
    mask_probs = mask_logits[0].sigmoid() # [Q, H, W]
    
    # 3. RbA Logic: Contribution = MaskProb * ClassConfidence
    mask_contribution = mask_probs * max_scores[:, None, None]
    
    # Per ogni pixel, prendiamo il valore massimo tra tutte le query
    per_pixel_confidence, _ = mask_contribution.max(dim=0) # [H, W]
    
    # Anomaly Score = 1 - Confidence
    anomaly_map = 1.0 - per_pixel_confidence
    
    return anomaly_map.cpu().numpy()

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Ignorato")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt")
    parser.add_argument("--input", type=str, required=True, help="Glob path to input images")
    parser.add_argument("--subset", type=str, default="Val")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. COSTRUZIONE MODELLO MANUALE ---
    print("Building EoMT model manually...")
    
    # Usiamo 644x644 che è una dimensione comoda e gestibile da Colab T4
    # Grazie alla funzione interpolate_pos_embed, ora funzionerà con QUALSIASI peso.
    IMG_SIZE = (644, 644)
    
    try:
        # A. Costruiamo l'Encoder
        encoder = ViT(
            backbone_name="vit_base_patch14_reg4_dinov2",
            img_size=IMG_SIZE 
        )
        
        # B. Costruiamo EoMT
        model = EoMT(
            encoder=encoder,
            num_classes=19,
            num_q=100,
            num_blocks=3,
            masked_attn_enabled=True
        )
    except Exception as e:
        print(f"Errore durante la costruzione del modello: {e}")
        sys.exit(1)

    # C. Carichiamo i pesi (Con interpolazione automatica!)
    model = load_checkpoint(model, args.ckpt)
    model.to(device)
    model.eval()
    
    # --- 2. DATASET ---
    files = glob.glob(args.input)
    print(f"Found {len(files)} images in {args.input}")
    
    if len(files) == 0:
        print("Error: No images found.")
        return

    transform = Compose([
        Resize(IMG_SIZE),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    anomaly_scores = []
    ood_gts = []
    
    # --- 3. INFERENZA ---
    print("Starting inference...")
    for path in tqdm(files):
        try:
            img = Image.open(path).convert("RGB")
            
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                
                mask_logits = outputs[0][-1]
                class_logits = outputs[1][-1]
                
                anomaly_map = get_rba_score(mask_logits, class_logits)
                
                # Resize a 1024x512 per valutazione coerente
                anomaly_map = Image.fromarray(anomaly_map)
                anomaly_map = anomaly_map.resize((1024, 512), Image.BILINEAR)
                anomaly_map = np.array(anomaly_map)
            
            # --- GROUND TRUTH LOGIC ---
            pathGT = path.replace("images", "labels_masks")
            if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")
            if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")
            
            if not os.path.exists(pathGT):
                continue
            
            mask = Image.open(pathGT)
            mask = mask.resize((1024, 512), Image.NEAREST)
            gt = np.array(mask)
            
            if "RoadAnomaly" in pathGT: gt = np.where((gt==2), 1, gt)
            if "LostAndFound" in pathGT: 
                gt = np.where((gt==0), 255, gt)
                gt = np.where((gt==1), 0, gt)
                gt = np.where((gt>1)&(gt<201), 1, gt)
            if "Streethazard" in pathGT:
                gt = np.where((gt==14), 255, gt)
                gt = np.where((gt<20), 0, gt)
                gt = np.where((gt==255), 1, gt)
            
            if 1 not in np.unique(gt): continue
            
            ood_gts.append(gt)
            anomaly_scores.append(anomaly_map)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
            
    # --- 4. CALCOLO METRICHE ---
    if not ood_gts:
        print("Nessuna immagine valida processata.")
        return

    print("Calculating metrics...")
    ood_gts = np.array(ood_gts)
    anomaly_scores = np.array(anomaly_scores)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    
    print("------------------------------------------------")
    print(f"Dataset: {args.subset}")
    print(f"Method: EoMT + RbA (DINOv2 backbone)")
    print(f"AUPRC: {prc_auc*100:.2f}%")
    print(f"FPR@95: {fpr*100:.2f}%")
    print("------------------------------------------------")

if __name__ == "__main__":
    main()