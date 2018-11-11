# Adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py

import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

pascal_labels = np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

class PascalDatasetLoader(Dataset):
    def __init__(self, path, split, img_size=500):
        self.path = path
        self.img_size = img_size
        self.split = split
        self.num_classes = 21
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = defaultdict(list)

        for split in ["train", "val", "traincrf"]:
            self._get_files(split)

        self.tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = os.path.join(self.path, "JPEGImages", img_name + ".jpg")
        lbl_path = os.path.join(self.path, "SegmentationClass", img_name + ".png")
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        img, lbl = self.transform(img, lbl)
        return img, lbl
    
    def _get_files(self, split):
        file_list_path = os.path.join(self.path, "ImageSets/Segmentation", split + ".txt")
        file_list = tuple(open(file_list_path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[split] = file_list
        return file_list
    
    def transform(self, img, lbl):
        img = img.resize((self.img_size, self.img_size))
        lbl = lbl.resize((self.img_size, self.img_size))
        img = self.tform(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = 0 
        return img, lbl
    
    def mask_to_labels(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(pascal_labels):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask
    
    def labels_to_mask(self, labels): 
        r = labels.copy()
        g = labels.copy()
        b = labels.copy()
        for ll in range(0, self.num_classes):
            r[labels == ll] = pascal_labels[ll, 0]
            g[labels == ll] = pascal_labels[ll, 1]
            b[labels == ll] = pascal_labels[ll, 2]
        rgb = np.zeros((labels.shape[0], labels.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0


if __name__ == "main":
    path = '/home/jimiolaniyan/Documents/Research/VOCdevkit/VOC2012'
    data = PascalDatasetLoader(path)