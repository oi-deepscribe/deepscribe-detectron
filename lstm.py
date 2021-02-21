# just trying LSTM on its own to figure out what's going on

import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import pickle
from resnet import AccuracyTopK
from torch.utils.data import Dataset
from typing import Tuple

from dsdetectron.sequence import SignSequenceDataset


class RNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim=51,
        hidden_size=512,
        bidirectional=False,
        residual_connection=False,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.input1 = nn.Linear(input_dim, hidden_size)

        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=1,
            dropout=0.00,
            bidirectional=self.bidirectional,
        )

        self.output1 = nn.Linear(
            hidden_size * 2 if self.bidirectional else hidden_size, input_dim
        )
        self.output2 = nn.Linear(input_dim, input_dim)
        # self.output2 = nn.Linear(n_classes, hidden_size)

        self.residual_connection = residual_connection

        self.valid_top1 = AccuracyTopK(top_k=1)
        self.valid_top3 = AccuracyTopK(top_k=3)
        self.valid_top5 = AccuracyTopK(top_k=5)

    def forward(self, sequence):

        bsize, seq_len, seq_dim = sequence.size()

        input_projected = self.input1(sequence.reshape(-1, seq_dim))

        lstm_out, _ = self.rnn(
            F.relu(input_projected.view(bsize, seq_len, input_projected.size(1)))
        )

        # [bsize*seq_len, hidden_dim]
        final_in = lstm_out.reshape(-1, lstm_out.size(2))

        # categories_residual = self.output2(F.relu(self.output2(sequence.reshape(-1, seq_dim)).reshape(bsize, seq_len, -1))

        categories = self.output1(F.relu(final_in)).view(bsize, seq_len, -1)

        if self.residual_connection:
            categories = categories + sequence

        categories = self.output2(F.relu(categories.reshape(-1, seq_dim))).view(
            bsize, seq_len, -1
        )

        return categories

    def validation_step(self, batch, batch_idx):

        sequence, labels = batch

        bsize, seq_len, _ = sequence.size()
        out = self(sequence)

        loss = F.cross_entropy(
            out.view(bsize * seq_len, -1),
            labels.view(bsize * seq_len),
            ignore_index=-1,
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.valid_top1(out.view(bsize * seq_len, -1), labels.view(bsize * seq_len))
        self.valid_top3(out.view(bsize * seq_len, -1), labels.view(bsize * seq_len))
        self.valid_top5(out.view(bsize * seq_len, -1), labels.view(bsize * seq_len))

        # # self.log("valid_acc", self.valid_acc, on_epoch=True, prog_bar=True)
        self.log("valid_top1", self.valid_top1, on_epoch=True, prog_bar=True)
        self.log("valid_top3", self.valid_top3, on_epoch=True, prog_bar=True)
        self.log("valid_top5", self.valid_top5, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        sequence, labels = batch

        bsize, seq_len, _ = sequence.size()
        out = self(sequence)

        loss = F.cross_entropy(
            out.view(bsize * seq_len, -1),
            labels.view(bsize * seq_len),
            ignore_index=-1,
        )

        self.log("train_loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class LogitsPredictionDataset(Dataset):
    def __init__(self, predictions_list, labels_list):
        self.preds = [torch.tensor(lst).float() for lst in predictions_list]
        self.labels = [torch.tensor(lst) for lst in labels_list]

    def __len__(self) -> int:
        return len(self.preds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.preds[idx], self.labels[idx]


def collate(elements):
    # unzip batch components
    (
        sequences,
        labels,
    ) = zip(*elements)

    padded_seq = nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0.0
    )

    padded_labels = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-1
    )

    return padded_seq, padded_labels


with open("inference_latest_63_top1.pkl", "rb") as inf:
    dat = pickle.load(inf)

dset_train = LogitsPredictionDataset(dat["topk_train"], dat["labels_train"])

dset_val = LogitsPredictionDataset(dat["topk_test"], dat["labels_test"])

dataloader_train = DataLoader(
    dset_train, batch_size=32, collate_fn=collate, num_workers=12
)
dataloader_val = DataLoader(dset_val, batch_size=32, collate_fn=collate, num_workers=12)


model = RNNClassifier(residual_connection=True, bidirectional=True)

callbacks = [
    pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    pl.callbacks.LearningRateMonitor(),
]
trainer = pl.Trainer(max_epochs=500, gpus=1, callbacks=callbacks)


trainer.fit(model, train_dataloader=dataloader_train, val_dataloaders=dataloader_val)
