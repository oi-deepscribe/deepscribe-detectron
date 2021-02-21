# pytorch dataset containing hotspots and sign sequence


import json
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor


class SquarePad:
    def __call__(self, image: Image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


def convert_to_gray(imgs):
    for i in range(len(imgs)):
        # convert to gray
        gray_img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        imgs[i] = gray_img
        # resize to 50
        resize = 50
        old_size = imgs[i].shape[:2]

        delta_w = max(old_size) - old_size[1]
        delta_h = max(old_size) - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        imgs[i] = cv2.copyMakeBorder(
            imgs[i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        imgs[i] = cv2.resize(imgs[i], (resize, resize))
        imgs[i] = imgs[i].astype("float32")
    return imgs


class SignSequenceDataset(Dataset):
    def __init__(
        self,
        hotspot_file: str,
        signs_file: str,
        image_folder: str,
        augment: List = None,
        top_frequency=None,
    ):
        with open(hotspot_file, "r") as hfile:
            self.image_data = json.load(hfile)

        # load sign frequency data
        self.sign_data = pd.read_csv(signs_file)

        self.image_folder = image_folder

        transform_list = [SquarePad(), Resize((50, 50))]

        if augment is not None:
            transform_list.extend(augment)

        self.transforms = Compose(transform_list + [ToTensor()])

        if top_frequency is not None:
            # only keep the top top_frequency signs

            sorted_inds = np.argsort(self.sign_data["frequency"])[::-1]

            top_inds = list(sorted_inds[:top_frequency])
            # unk_inds = list(sorted_inds[top_frequency:])
            # assign
            for img_dat in self.image_data:
                for box in img_dat["annotations"]:
                    if box["category_id"] in top_inds:
                        box["category_id"] = top_inds.index(box["category_id"])
                    else:
                        box["category_id"] = top_frequency  # UNK token

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List]:

        # load image from disk
        img = cv2.imread(f"{self.image_folder}/{self.image_data[idx]['file_name']}")
        # img = Image.open(f"{self.image_folder}/{self.image_data[idx]['file_name']}")

        # get hostpot cutouts
        cutouts = []
        labels = []
        for box in self.image_data[idx]["annotations"]:
            x1, y1, x2, y2 = [int(coord) for coord in box["bbox"]]
            hotspot = img[y1:y2, x1:x2, :]

            # convert to PIL image for use in PyTorch functions
            hotspot_transformed = cv2.cvtColor(hotspot, cv2.COLOR_BGR2RGB)
            cutouts.append(Image.fromarray(hotspot_transformed))
            labels.append(int(box["category_id"]))

        # transform hotspot cutouts + concatenate

        # add channel dimension
        images = torch.stack([self.transforms(hotspot) for hotspot in cutouts])

        return (
            images,
            torch.Tensor(labels),
            len(labels),
        )
