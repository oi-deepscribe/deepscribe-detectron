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
def inference_dset(model: pl.LightningModule, dset: SignSequenceDataset, k=5):
    # list of sequences with elements (pred_class, true_class)
    topk_predictions = []
    labels = []
    for i in trange(len(dset)):
        images, targets, _ = dset[i]
        logits = model(images)

        _, pred = logits.topk(k, dim=1)

        # print(pred.tolist())

        # predicted_signs = pred[:, 0].tolist()

        # print(predicted_signs)

        labels.append([str(targ) for targ in targets.tolist()])
        topk_predictions.append(pred.tolist())

    return topk_predictions, labels


def topk_to_features(sent):

    feats = []

    for topk in sent:
        feats.append({f"top-{i}": str(label) for i, label in enumerate(topk)})

    return feats


topk_train, labels_train = inference_dset(trained, dset_train)
topk_test, labels_test = inference_dset(trained, dset_val)

crf_train = [topk_to_features(sent) for sent in topk_train]
crf_test = [topk_to_features(sent) for sent in topk_test]

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.2,
    c2=0.2,
    max_iterations=10000,
    all_possible_transitions=True,
)

crf.fit(crf_train, labels_train)

crf_test_pred = crf.predict(crf_test)
print(metrics.flat_accuracy_score(labels_test, crf_test_pred))

# hmm_trainer = HiddenMarkovModelTrainer()

# hmm_tagger = hmm_trainer.train(labeled_sequences=labeled_train)

# hmm_tagger.test(labeled_val, verbose=False)
