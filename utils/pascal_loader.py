# Adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py

import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


class PascalDatasetLoader(Dataset):
    def __init__(self, path, ):
        self.path = path
        self.num_classes = 21
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = self._get_files()

        self.tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = os.path.join(self.path, "JPEGImages", img_name + ".jpg")
        lbl_path = os.path.join(self.path, "SegmentationClass", img_name + ".png")
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        img, lbl = self.transform(img, lbl)
        return img, lbl
    
    def _get_files(self):
        file_list_path = os.path.join(self.path, "ImageSets/Segmentation/val.txt")
        file_list = tuple(open(file_list_path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        return file_list
    
    def transform(self, img, lbl):
        img = self.tform(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = 0 
        return img, lbl
    
    def encode_segmentation(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)

if __name__ == "main":
    path = '/home/jimiolaniyan/Documents/Research/VOCdevkit/VOC2012'
    data = PascalDatasetLoader(path)