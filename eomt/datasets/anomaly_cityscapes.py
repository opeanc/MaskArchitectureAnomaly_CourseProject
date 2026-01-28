import cv2
import torch
from torch.utils.data import Dataset
import random
import os
import pandas as pd
from PIL import Image
from .custom_transformations import load_sample, inject_anomaly


EXTENSIONS = ['.jpg', '.png']

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_cityscapes_label(filename):
    return filename.endswith("_labelTrainIds.png")


class CityscapesAnomalyDataset(Dataset):
    def __init__(self, city_root, obj_root, image_transform=None, mask_transform=None, p_anomaly=0.5, subset='train'):
        """
        Args:
            city_root: Cartella Cityscapes
            obj_root: Cartella con gli oggetti da utilizzare (img + mask)
            image_transform: Trasformazioni PyTorch per immagine
            mask_transform: Trasformazioni PyTorch per maschera
            p_anomaly: Probabilità di inserire un'anomalia (0.5 default)
            subset: Subset utilizzato per costruire il dataset
        """

        self.obj_root = obj_root

        # Cityscapes
        self.city_images = os.path.join(city_root, 'leftImg8bit/' + subset)
        self.city_masks = os.path.join(city_root, 'gtFine/' + subset)

        self.city_filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.city_images)) for f in fn if is_image(f)]
        self.city_filenames.sort()

        self.city_masks_filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.city_masks)) for f in fn if is_cityscapes_label(f)]
        self.city_masks_filenames.sort()

        # Transformations & other logic
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.p_anomaly = p_anomaly
        self.annotations_path = os.path.join(self.obj_root, 'annotations.csv')
        self.annotations = pd.read_csv(self.annotations_path)

    def __getitem__(self, idx):

        # load Cityscapes (Clean)
        city_img_path = self.city_filenames[idx]
        city_mask_path = self.city_masks_filenames[idx]
        print(city_img_path)
        city_image = cv2.imread(city_img_path)
        city_mask = cv2.imread(city_mask_path, cv2.IMREAD_UNCHANGED)

        # Anomaly injection
        if random.random() < self.p_anomaly:

            # random aomaly sampling
            obj_path, mask_path, obj_label = load_sample(self.annotations)
            obj_path = os.path.join(self.obj_root, obj_path)
            mask_path = os.path.join(self.obj_root, mask_path)

            city_image, city_mask = inject_anomaly(
                city_image, city_mask,
                obj_path, mask_path, obj_label,
                anomaly_id=254
            )

        # from BGR image (OpenCV) to RGB (PIL)
        img_rgb_tmp = cv2.cvtColor(city_image, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(img_rgb_tmp)
        final_mask = Image.fromarray(city_mask)

        # transformations
        if self.image_transform:
            final_image = self.image_transform(final_image)
        if self.mask_transform:
            final_mask = self.mask_transform(final_mask)

        return final_image, final_mask

    def __len__(self):
        return len(self.city_filenames)
