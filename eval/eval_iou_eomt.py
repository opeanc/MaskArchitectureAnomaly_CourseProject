#mIoU EOMT
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
eomt_path = os.path.join(project_root, 'eomt')
if project_root not in sys.path:
    sys.path.append(project_root)
if eomt_path not in sys.path:
    sys.path.append(eomt_path)
from eomt.models.vit import ViT
from eomt.models.eomt import EoMT
import torch.nn.functional as F
import importlib
import time
import timm
import yaml

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

# Definizione automatica del device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((1024, 2048), Image.BILINEAR),
    ToTensor(), 
    # normalization happens in window_inference
])
target_transform_cityscapes = Compose([
    Resize((1024, 2048), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),
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

def main(args):

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
    
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print("Using GPU (CUDA)")
    else:
        model = model.to(device)
        print("Using CPU")

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

    checkpoint = torch.load(weightspath, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model = load_my_state_dict(model, state_dict)
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    # num_workers=0 is more stable
    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
                        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    print(f"Images fonud in the loader: {len(loader)}")
    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

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
                size=labels.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            preds = full_logits.argmax(dim=1).unsqueeze(1)
        
        iouEvalVal.addBatch(preds.data, labels)
        
        filenameSave = filename[0].split("leftImg8bit/")[1] 

        print (step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i].item())+'{:0.2f}'.format(iou_classes[i].item()*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal.item())+'{:0.2f}'.format(iouVal.item()*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="epoch_106-step_19902_eomt.ckpt")
    parser.add_argument('--loadModel', default="eomt.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())