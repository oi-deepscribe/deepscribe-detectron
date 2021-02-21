import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms

from dsdetectron.sequence import SignSequenceDataset
from resnet import AccuracyTopK, ResNet18


class CNNSequence(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        freeze_cnn: bool = False,
        ckpt: str = None,
        hidden_size=512,
    ):
        super().__init__()

        self.cnn = ResNet18(n_classes=hidden_size)

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
        )

        self.output = nn.Linear(hidden_size * 2, n_classes)

        self.valid_top1 = AccuracyTopK(top_k=1)
        self.valid_top3 = AccuracyTopK(top_k=3)
        self.valid_top5 = AccuracyTopK(top_k=5)

    def forward(self, padded_data: torch.Tensor):

        # run CNN on padded data.

        batch_size, timesteps, C, H, W = padded_data.size()

        # run CNN over all timesteps
        output_cnn = self.cnn(padded_data.view(batch_size * timesteps, C, H, W))

        # print(output_cnn.size())

        lstm_output, _ = self.lstm(F.relu(output_cnn.view(batch_size, timesteps, -1)))

        # print(lstm_output.size())

        output_linear = self.output(
            F.relu(lstm_output.reshape(-1, lstm_output.size(2)))
        )

        # print(output_linear.size())

        return output_linear.view((batch_size, timesteps, -1))

    def training_step(self, batch, batch_idx):

        sequence_padded, targets_padded, lengths = batch

        batch_size, timesteps, C, H, W = sequence_padded.size()

        pred = self(sequence_padded)

        loss = F.cross_entropy(
            pred.view(batch_size * timesteps, -1),
            targets_padded.view(batch_size * timesteps).long(),
            ignore_index=-1,
        )
        self.log("train_loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sequence_padded, targets_padded, lengths = batch

        batch_size, timesteps, C, H, W = sequence_padded.size()

        pred = self(sequence_padded)

        loss = F.cross_entropy(
            pred.view(batch_size * timesteps, -1),
            targets_padded.view(batch_size * timesteps).long(),
            ignore_index=-1,
        )
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.valid_top1(pred.view(batch_size * timesteps, -1), targets_padded.view(-1))
        self.valid_top3(pred.view(batch_size * timesteps, -1), targets_padded.view(-1))
        self.valid_top5(pred.view(batch_size * timesteps, -1), targets_padded.view(-1))

        self.log("valid_top1", self.valid_top1, on_epoch=True, prog_bar=True)
        self.log("valid_top3", self.valid_top3, on_epoch=True, prog_bar=True)
        self.log("valid_top5", self.valid_top5, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def collate_fn(elements):

    # unzip batch components
    sequences, labels, lengths = zip(*elements)

    padded_seq = nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0.0
    )

    padded_labels = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-1
    )

    return padded_seq, padded_labels, lengths


if __name__ == "__main__":

    augmentation_set = [
        transforms.RandomAffine(20, translate=(0.2, 0.2)),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.RandomPerspective(),
        transforms.Grayscale(num_output_channels=1),
    ]

    # augmentation_set = []

    top_frequency = 50

    # dset_train = SignSequenceDataset(
    #     "/local/ecw/deepscribe-detectron/data_nov_2020/hotspots_train.json",
    #     "/local/ecw/deepscribe-detectron/data_nov_2020/signs_nov_2020.csv",
    #     "/local/ecw/deepscribe-detectron/data_nov_2020/images_cropped",
    #     augment=augmentation_set,
    #     top_frequency=top_frequency,
    # )

    dset_train = SignSequenceDataset(
        "/local/ecw/deepscribe-detectron/data_dec_2020/hotspots_train.json",
        "/local/ecw/deepscribe-detectron/data_dec_2020/signs_dec_2020.csv",
        "/local/ecw/deepscribe-detectron/data_dec_2020/images_cropped",
        augment=augmentation_set,
        top_frequency=top_frequency,
    )

    dset_val = SignSequenceDataset(
        "/local/ecw/deepscribe-detectron/data_dec_2020/hotspots_val.json",
        "/local/ecw/deepscribe-detectron/data_dec_2020/signs_dec_2020.csv",
        "/local/ecw/deepscribe-detectron/data_dec_2020/images_cropped",
        augment=[transforms.Grayscale(num_output_channels=1)],
        top_frequency=top_frequency,
    )

    # dset_val = SignSequenceDataset(
    #     "/local/ecw/deepscribe-detectron/data_nov_2020/hotspots_val.json",
    #     "/local/ecw/deepscribe-detectron/data_nov_2020/signs_nov_2020.csv",
    #     "/local/ecw/deepscribe-detectron/data_nov_2020/images_cropped",
    #     top_frequency=top_frequency,
    # )

    dataloader_train = DataLoader(
        dset_train, batch_size=20, collate_fn=collate_fn, num_workers=12
    )
    dataloader_val = DataLoader(
        dset_val, batch_size=20, collate_fn=collate_fn, num_workers=12
    )

    # create logger
    # logger = pl.loggers.WandbLogger(project="deepscribe-torch")

    # load ResNet from disk

    # ckpt = "/local/ecw/deepscribe-detectron/deepscribe-torch/1w52nq8i/checkpoints/epoch=56.ckpt"

    # trained = ResNet18.load_from_checkpoint(ckpt, n_classes=top_frequency + 1).cuda()

    # del trained

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        pl.callbacks.LearningRateMonitor(),
    ]
    trainer = pl.Trainer(max_epochs=500, gpus=1, callbacks=callbacks)

    model = CNNSequence(top_frequency + 1)

    trainer.fit(
        model, train_dataloader=dataloader_train, val_dataloaders=dataloader_val
    )
