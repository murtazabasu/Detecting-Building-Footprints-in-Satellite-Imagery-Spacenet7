import os
import glob
import torch

import numpy as np
import pandas as pd

from PIL import Image, ImageFile

from collections import defaultdict
from torchvision import transforms

from albumentations import (
    Compose, 
    OneOf,
    RandomBrightness,
    RandomBrightnessContrast,
    RandomGamma, 
    ShiftScaleRotate
)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SpaceNetBuildingDetection(torch.utils.data.Dataset):
    def __init__(self,
        image_ids,
        arguments,
        transform=True,
        preprocessing_fn=None):
        """
        Dataset class for segmentation problem
        -> image_ids: ids of the image, list
        -> transform: True/False, no transform in validation
        -> preprocessing_fn: a function for preprocessing image
        """
        self.data = defaultdict(dict)
        self.counter = 0
        
        # get the arguments
        self.args = arguments

        # for augmentation
        self.transform = transform

        # preprocessing function 
        self.preprocessing_fn = preprocessing_fn

        # albumentation augmentation
        # have shift, scale & rotate features
        # and is applied with 80% probability.
        # Gamma controls the and brightness /
        # contrast i.e. applied to the image.
        # Albumentation takes care of which augmentation
        # is applied to image and mask
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10, p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit=(90, 110)
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.1,
                        ),
                    ],
                    p=0.5,
                ),
            ]
        )
        # going pver all image_ids to store 
        # image and mask paths
        for imgid in image_ids:
            rasterImg = "global_monthly_2019_08_mosaic_" + imgid + ".tif"
            maskImg = "mask_" + imgid + ".tif" 
            self.data[self.counter] = {
                "img_path": os.path.join(
                    self.args.raster_path, rasterImg
                ),
                "mask_path": os.path.join(
                    self.args.output_path, maskImg
                ),
            }
            self.counter+=1
        
    def __len__(self):
        # return length of dataset
        return len(self.data)
    
    def __getitem__(self, item):
        # for a givenm item index,
        # return image and mask tensors
        # read image and mask paths
        img_path = self.data[item]["img_path"]
        #print("img_path", img_path)
        mask_path = self.data[item]["mask_path"]

        # read image and convert to RGB
        img = Image.open(img_path)
        img = img.resize((self.args.resize_img, self.args.resize_img))
        img = img.convert("RGB")
        # PIL image to numpy array
        img = np.array(img)

        # read mask image
        mask = Image.open(mask_path)
        mask= mask.resize((self.args.resize_img, self.args.resize_img))
        mask = np.array(mask)

        # Convert to binary float matrix
        mask = (mask>=1).astype("float32")

        # if this is training data, apply transforms
        if self.transform is True:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        # image normalization using preprocessing
        # tensors.
        img = self.preprocessing_fn(img)

        # return image and mask tensors
        return {
            "image": transforms.ToTensor()(img),
            "mask": transforms.ToTensor()(mask).float(),
        } 