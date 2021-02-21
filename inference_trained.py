# running HMM training and prediction.

import os

import pytorch_lightning as pl
import sklearn_crfsuite
import torch
import torch.nn.functional as F
from nltk.tag.hmm import HiddenMarkovModelTrainer
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from tqdm import trange
from sklearn_crfsuite import metrics
import pickle

from resnet import ResNet18
from dsdetectron.sequence import SignSequenceDataset

top_frequency = 50

dset_train = SignSequenceDataset(
    "/local/ecw/deepscribe-detectron/data_dec_2020/hotspots_train.json",
    "/local/ecw/deepscribe-detectron/data_dec_2020/signs_dec_2020.csv",
    "/local/ecw/deepscribe-detectron/data_dec_2020/images_cropped",
    augment=[transforms.Grayscale(num_output_channels=1)],
    top_frequency=top_frequency,
)

dset_val = SignSequenceDataset(
    "/local/ecw/deepscribe-detectron/data_dec_2020/hotspots_val.json",
    "/local/ecw/deepscribe-detectron/data_dec_2020/signs_dec_2020.csv",
    "/local/ecw/deepscribe-detectron/data_dec_2020/images_cropped",
    augment=[transforms.Grayscale(num_output_channels=1)],
    top_frequency=top_frequency,
)

# ckpt = "/local/ecw/deepscribe-detectron/deepscribe-torch/1w52nq8i/checkpoints/epoch=56.ckpt"

ckpt = "/local/ecw/deepscribe-detectron/lightning_logs/version_120/checkpoints/epoch=65.ckpt"

trained = ResNet18.load_from_checkpoint(ckpt, n_classes=top_frequency + 1)


def inference_dset(model: pl.LightningModule, dset: SignSequenceDataset, k=5):
    # list of sequences with elements (pred_class, true_class)
    topk_predictions = []
    labels = []
    for i in trange(len(dset)):
        images, targets, _ = dset[i]
        logits = model(images)

        labels.append([int(targ) for targ in targets.tolist()])
        topk_predictions.append(logits.tolist())

    return topk_predictions, labels


topk_train, labels_train = inference_dset(trained, dset_train)
topk_test, labels_test = inference_dset(trained, dset_val)

with open("inference_latest_63_top1.pkl", "wb") as outf:
    pickle.dump(
        {
            "topk_train": topk_train,
            "topk_test": topk_test,
            "labels_train": labels_train,
            "labels_test": labels_test,
        },
        outf,
    )