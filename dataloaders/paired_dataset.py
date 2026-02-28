import glob
import os

import torch
from PIL import Image
import random
import numpy as np

from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

from .realesrgan import RealESRGAN_degradation

class PairedCaptionDataset(data.Dataset):
    def __init__(self, split=None, args=None):
        super(PairedCaptionDataset, self).__init__()
        self.args = args
        self.split = split

        txt_path = self.args.txt_path[0]

        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []

        with open(txt_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                
                if len(parts) >= 3:
                    self.gt_list.append(parts[0])          
                    self.lr_list.append(parts[1])   
                    self.tag_path_list.append(' '.join(parts[2:]))   
                else:
                    print(f"skip: {line.strip()}")

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)
        indices = list(range(len(self.lr_list)))
        random.shuffle(indices)
        self.lr_list = [self.lr_list[i] for i in indices]
        self.gt_list = [self.gt_list[i] for i in indices]
        self.tag_path_list = [self.tag_path_list[i] for i in indices]

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):


        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        
        if random.random() < self.args.deg_prob:
            lq_path = self.lr_list[index]
            lq_img = Image.open(lq_path).convert('RGB')
            lq_img = lq_img.resize(gt_img.size, Image.Resampling.LANCZOS)
        else:
            width, height = gt_img.size
            lq_img = gt_img.resize((width // 4, height // 4), resample=Image.Resampling.BICUBIC)
            lq_img = lq_img.resize((width, height), resample=Image.Resampling.BICUBIC)

        gt_img = self.img_preproc(gt_img)
        lq_img = self.img_preproc(lq_img)

        tag = self.tag_path_list[index]

        example = dict()

        example["neg_prompt"] = self.args.neg_prompt
        example["null_prompt"] = ""
        example["prompt"] = tag
        example["output_pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["conditioning_pixel_values"] = lq_img.squeeze(0) * 2.0 - 1.0


        return example

    def __len__(self):
        return len(self.gt_list)