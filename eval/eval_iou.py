# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

# python eval_iou.py --datadir /Users/francescodedominicis/Desktop/POLITO/Advanced_ML/Project/cityscapes --subset val
# export KMP_DUPLICATE_LIB_OK=TRUE && python eval_iou.py --datadir /Users/francescodedominicis/Desktop/POLITO/Advanced_ML/Project/cityscapes --subset val

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

# ### MAC FIX 1: Risolve il conflitto OpenMP (Error #15) ###
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ### MAC FIX 2: Evita crash MKL limitando i thread ###
torch.set_num_threads(1)

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

# Assicurati che questi file esistano nella cartella eval/
from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

# ### MAC FIX 3: Definizione esplicita del device (CPU) ###
device = torch.device("cpu")

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)
    
    # ### MAC FIX 4: Spostamento su CPU invece di CUDA ###
    model = model.to(device)

    # ### MAC FIX 5: Disabilitato DataParallel ###
    # DataParallel su CPU causa Segmentation Fault su macOS.
    # model = torch.nn.DataParallel(model)
    
    # Il codice originale usava questo check per CUDA, lo commentiamo per sicurezza
    # if (not args.cpu):
    #     model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    # Rimuove il prefisso 'module.' se presente (perché abbiamo tolto DataParallel)
                    key = name.split("module.")[-1]
                    if key in own_state:
                         own_state[key].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    # ### MAC FIX 6: map_location force to CPU ###
    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))
    print ("Model and weights LOADED successfully")


    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    # DataLoader inizializzato con i parametri args
    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        # ### MAC FIX 7: Spostamento tensori su CPU ###
        images = images.to(device)
        labels = labels.to(device)
        
        # Codice originale CUDA rimosso/commentato
        # if (not args.cpu):
        #    images = images.cuda()
        #    labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        # getColorEntry potrebbe richiedere modifiche se usa codici colore non standard, 
        # ma di solito è solo string formatting
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
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
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    # ### MAC FIX 8: Default num-workers a 0 ###
    # Multiprocessing su Mac è lento e instabile per questo task
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())