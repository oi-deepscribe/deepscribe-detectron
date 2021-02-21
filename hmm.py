# running HMM training and prediction.

import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from nltk.tag.hmm import HiddenMarkovModelTrainer
from tqdm import trange

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

ckpt = "/local/ecw/deepscribe-detectron/lightning_logs/version_120/checkpoints/epoch=65.ckpt"

trained = ResNet18.load_from_checkpoint(ckpt, n_classes=top_frequency + 1)


# should be able to batch this.
def inference_dset(model: pl.LightningModule, dset: SignSequenceDataset):
    # list of sequences with elements (pred_class, true_class)
    labeled = []
    for i in trange(len(dset)):
        images, targets, _ = dset[i]
        logits = model(images)

        _, pred = logits.topk(1, dim=1)

        # print(pred)

        predicted_signs = pred[:, 0].tolist()

        # print(predicted_signs)

        target_ints = [int(targ) for targ in targets.tolist()]

        labeled.append(list(zip(predicted_signs, target_ints)))

    return labeled


labeled_train = inference_dset(trained, dset_train)
labeled_val = inference_dset(trained, dset_val)


hmm_trainer = HiddenMarkovModelTrainer()

hmm_tagger = hmm_trainer.train(labeled_sequences=labeled_train)

hmm_tagger.test(labeled_val, verbose=False)
