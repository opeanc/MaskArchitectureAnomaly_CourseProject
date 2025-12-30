# Code to calculate IoU (mean and per-class) in a dataset
# Optimized for Multi-Temperature Evaluation on Mac
#######################

# python eval_iou.py --datadir /Users/francescodedominicis/Desktop/POLITO/Advanced_ML/Project/cityscapes --subset val

import numpy as np
import torch
import torch.nn.functional as F
import os
import time

# ### MAC FIX 1: Risolve il conflitto OpenMP ###
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ### MAC FIX 2: Threading limit ###
torch.set_num_threads(1)

from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage

# Assicurati che questi import funzionino nel tuo path
from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

# ### MAC FIX 3: CPU Device ###
device = torch.device("cpu")

NUM_CHANNELS = 3
NUM_CLASSES = 20

input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   # ignore label to 19
])

def main(args):
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print(f"Loading model: {modelpath}")
    print(f"Loading weights: {weightspath}")

    # --- DEFINIZIONE TEMPERATURE ---
    temperature_list = [0.5, 0.75, 1.0, 1.1]
    print(f"Evaluating Temperatures: {temperature_list}")

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
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))
    print("Model loaded successfully")
    model.eval()

    if not os.path.exists(args.datadir):
        print("Error: datadir could not be loaded")
        return

    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), 
                        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # --- Creiamo un valutatore per OGNI temperatura ---
    # Usiamo un dizionario per tenerli separati: {0.5: iouEval(...), 1.0: iouEval(...)}
    evaluators = {t: iouEval(NUM_CLASSES) for t in temperature_list}

    start = time.time()

    print(f"Starting evaluation on {len(loader)} batches...")

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        inputs = Variable(images)
        
        with torch.no_grad():
            # 1. Inferenza PESANTE (fatta una volta sola)
            logits = model(inputs)

            # 2. Loop LEGGERO sulle temperature
            for t in temperature_list:
                # Applica scaling
                scaled_logits = logits / t
                
                # Calcola la classe vincente (argmax)
                preds = scaled_logits.max(1)[1].unsqueeze(1).data
                
                # Aggiorna il valutatore specifico per questa temperatura
                evaluators[t].addBatch(preds, labels)

        if step % 10 == 0:
            filenameSave = filename[0].split("leftImg8bit/")[1] 
            print(f"Step {step}: {filenameSave}")

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    
    # --- STAMPA COMPATTA FINALE ---
    print("\n\n#######################################")
    print("       FINAL mIoU SUMMARY")
    print("#######################################")
    print(f"{'Temperature':<15} | {'mIoU (%)':<10}")
    print("-" * 30)

    for t in temperature_list:
        iouVal, iou_classes = evaluators[t].getIoU()
        miou_percent = iouVal * 100
        
        print(f"T = {t:<11} | {miou_percent:.2f}%")
    
    print("#######################################\n")

    # Dettaglio per T=1.0 (Baseline)
    print("Detailed Class IoU for T=1.0 (Baseline):")
    _, iou_classes_1 = evaluators[1.0].getIoU()
    classes_names = ['Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic Light', 
                     'Traffic Sign', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Rider', 'Car', 
                     'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle']
    
    for i, name in enumerate(classes_names):
        val = iou_classes_1[i] * 100
        print(f"{name:15s}: {val:.2f}%")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())