import os
from typing import Type, List, Tuple, Union, Optional, Callable
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms

from dsdetectron.sequence import SignSequenceDataset


class AccuracyTopK(pl.metrics.Metric):
    # from https://github.com/PyTorchLightning/pytorch-lightning/pull/3822
    def __init__(self, top_k=1, dist_sync_on_step=False, ignore_index=-1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = top_k
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, y):

        mask_arr = y.ne(self.ignore_index)

        y = y.masked_select(mask_arr)
        logits = logits[mask_arr, :]

        _, pred = logits.topk(self.k, dim=1)
        pred = pred.t()

        corr = pred.eq(y.view(1, -1).expand_as(pred))

        self.correct += corr[: self.k].sum()
        self.total += y.numel()

    def compute(self):
        return self.correct.float() / self.total


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResnetSingleChannel(models.ResNet):
    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResnetSingleChannel, self).__init__(block, layers)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


class ResNet18(pl.LightningModule):
    def __init__(self, n_classes: int):
        super().__init__()
        # here n_classes just defines
        # self.cnn = models.resnet18(num_classes=n_classes)

        self.cnn = ResnetSingleChannel(BasicBlock, [2, 2, 2, 2], num_classes=512)

        fc_layers = []

        for _ in range(3):
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Linear(512, 512))

        self.fc = nn.Sequential(*fc_layers)

        self.output = nn.Linear(512, n_classes)

        # self.valid_acc = pl.metrics.Accuracy()
        self.valid_top1 = AccuracyTopK(top_k=1)
        self.valid_top3 = AccuracyTopK(top_k=3)
        self.valid_top5 = AccuracyTopK(top_k=5)

    # tensor of [B, C, H, W]
    def forward(self, imgs: torch.Tensor):

        output_cnn = self.cnn(imgs)

        fc_output = self.fc(output_cnn)

        return self.output(F.relu(fc_output))

    def training_step(self, batch, batch_idx):

        images, targets, _ = batch

        pred = self(images)

        loss = F.cross_entropy(pred, targets.long(), ignore_index=-1)
        self.log("train_loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        images, targets, lengths = batch

        pred = self(images)

        loss = F.cross_entropy(pred, targets.long(), ignore_index=-1)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        # self.valid_acc(pred, targets)
        self.valid_top1(pred, targets)
        self.valid_top3(pred, targets)
        self.valid_top5(pred, targets)

        # self.log("valid_acc", self.valid_acc, on_epoch=True, prog_bar=True)
        self.log("valid_top1", self.valid_top1, on_epoch=True, prog_bar=True)
        self.log("valid_top3", self.valid_top3, on_epoch=True, prog_bar=True)
        self.log("valid_top5", self.valid_top5, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def collate_nopack(elements):

    # unzip batch components
    sequences, labels, lengths = zip(*elements)

    return torch.cat(sequences), torch.cat(labels), lengths


if __name__ == "__main__":

    augmentation_set = [
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomPerspective(),
        transforms.Grayscale(num_output_channels=1),
    ]
    top_frequency = 50

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

    dataloader_train = DataLoader(
        dset_train, batch_size=10, collate_fn=collate_nopack, num_workers=12
    )
    dataloader_val = DataLoader(
        dset_val, batch_size=10, collate_fn=collate_nopack, num_workers=12
    )

    # create logger
    # logger = pl.loggers.WandbLogger(project="deepscribe-torch")

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        pl.callbacks.LearningRateMonitor(),
    ]
    trainer = pl.Trainer(max_epochs=500, gpus=1, callbacks=callbacks)

    model = ResNet18(top_frequency + 1)

    trainer.fit(
        model, train_dataloader=dataloader_train, val_dataloaders=dataloader_val
    )
