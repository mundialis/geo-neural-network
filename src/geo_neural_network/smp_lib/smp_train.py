#!/usr/bin/env python3
"""############################################################################
#
# MODULE:      lib for semantic segmentation training with smp
# AUTHOR(S):   Markus Metz, mundialis
# PURPOSE:     Train a model from segmentation_models.pytorch.
#
# COPYRIGHT:   (C) 2025 by mundialis GmbH & Co. KG
#
# This program is free software. You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
#############################################################################
"""

# train a model from segmentation_models.pytorch
# use segmentation_models_pytorch tools and pytorch lightning for training
# https://github.com/qubvel-org/segmentation_models.pytorch

# based on
# https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/
# examples/camvid_segmentation_multiclass.ipynb

import os
import shutil
import sys
from pathlib import Path

import albumentations
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from osgeo import gdal
from pytorch_lightning.loggers import CSVLogger
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


def read_image_gdal(filename):
    """Args:
    filename (string): path to file to read with GDAL.
    """
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f"Unable to open file: {filename}")

    img = ds.ReadAsArray()
    # PyTorch works only on CHW format (Channel, Height, Width)
    # GDAL should by default return multiband raster data in CHW format
    # singleband raster are returned as HW, no channel dimension
    # print(f"GDAL image shape: {img.shape}")

    # close GDAL dataset
    ds = None

    return img


# Dataset for training and validation for semantic segmentation:
# assign a label to each pixel
# https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
class GdalImageDataset(BaseDataset):
    """Pytorch dataset using GDAL to read raster files."""

    def __init__(self, img_dir, lbl_dir, augmentation=None) -> None:
        """Initialize the dataset."""
        # directory listing
        self.ids = os.listdir(img_dir)
        self.images_fps = [
            os.path.join(img_dir, image_id) for image_id in self.ids
        ]

        # file names of images and masks can have different endings
        mask_ids = []
        for image_id in self.ids:
            if image_id.endswith("_image.vrt"):
                mask_id = image_id.replace("_image.vrt", "_label.tif")
            elif image_id.endswith("_image.tif"):
                mask_id = image_id.replace("_image.tif", "_label.tif")
            elif image_id.endswith(".vrt"):
                mask_id = image_id.replace(".vrt", ".tif")
            else:
                mask_id = image_id

            # file exists?
            if not Path(os.path.join(lbl_dir, mask_id)).exists():
                print(
                    f"ERROR: label file <{os.path.join(lbl_dir, mask_id)}> "
                    "does not exist",
                    file=sys.stderr,
                )
                sys.exit(1)

            mask_ids.append(mask_id)

        self.labels_fps = [
            os.path.join(lbl_dir, mask_id) for mask_id in mask_ids
        ]
        # for a huge number of files, read a textfile with filenames

        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.ids)

    def __getitem__(self, i):
        """Get next item
        Returns:
            image, mask pair.
        """
        image = read_image_gdal(self.images_fps[i])
        mask = read_image_gdal(self.labels_fps[i])

        if self.augmentation:
            # convert image from CHW to HWC
            image = image.transpose(1, 2, 0)
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
            # convert image back to CHW
            image = image.transpose(2, 0, 1)
        return image, mask


# training set images augmentation
def get_training_augmentation(img_size=512):
    """Define training augmentation."""
    train_transform = [
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Affine(
            scale=(0.5, 1.5),
            rotate=(-180, 180),
            shear=(-10, 10),
            translate_percent=(-0.1, 0.1),
            border_mode=0,
            p=0.4,
        ),
        albumentations.PadIfNeeded(min_height=img_size, min_width=img_size),
        albumentations.RandomCrop(height=img_size, width=img_size, p=0.5),
        albumentations.GaussNoise(p=0.5),
        albumentations.Perspective(p=0.5),
        albumentations.OneOf(
            [
                # albumentations.CLAHE(p=1), # only grayscale or RGB
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.5,
                    contrast_limit=0.5,
                    p=0.5,
                ),
                albumentations.RandomGamma(p=0.5),
            ],
            p=0.9,
        ),
        albumentations.OneOf(
            [
                albumentations.Sharpen(p=0.5),
                albumentations.Blur(blur_limit=3, p=0.5),
                albumentations.MotionBlur(blur_limit=3, p=0.5),
            ],
            p=0.9,
        ),
        albumentations.OneOf(
            [
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.5,
                    contrast_limit=0.5,
                    p=0.5,
                ),
                # only grayscale or RGB
                # albumentations.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albumentations.Compose(train_transform)


def get_validation_augmentation(img_size=512):
    """Add paddings to make image shape divisible by 32."""
    test_transform = [
        albumentations.PadIfNeeded(img_size, img_size),
    ]
    return albumentations.Compose(test_transform)


class PlModule(pl.LightningModule):
    """Pytorch lightning module for training."""

    def __init__(
        self,
        model,
        out_classes,
        model_path_base,
        t_max,
    ) -> None:
        """Initialize the module."""
        super().__init__()
        self.model = model

        # Preprocessing parameters for image normalization
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        # self.register_buffer(
        #     "std", torch.tensor(params["std"]).view(1, 3, 1, 1)
        # )
        # self.register_buffer(
        #     "mean", torch.tensor(params["mean"]).view(1, 3, 1, 1)
        # )
        self.mean = 125.5
        self.std = 100.2

        if out_classes > 1:
            # Loss function for multi-class segmentation
            self.loss_fn = smp.losses.JaccardLoss(
                smp.losses.MULTICLASS_MODE,
                from_logits=True,
            )
        else:
            # Loss function for binary segmentation
            self.loss_fn = smp.losses.JaccardLoss(
                smp.losses.BINARY_MODE,
                from_logits=True,
            )

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # custom best model saving
        self.best_loss = 1000
        self.current_loss = 1000
        self.best_model_path = None
        self.best_epoch = -1
        self.model_path_base = model_path_base.rstrip("/")
        self.t_max = t_max

    def forward(self, image):
        """Forward."""
        # Normalize image
        image = image.float()
        image = (image - self.mean) / self.std
        return self.model(image)

    # ruff:noqa:ARG002 # Unused method argument
    def shared_step(self, batch, stage):
        """Check mask as steps for train and apply data."""
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        if self.number_of_classes > 1:
            assert mask.ndim == 3  # [batch_size, H, W]
        else:
            mask = mask.unsqueeze(1)
            assert mask.ndim == 4  # [batch_size, C, H, W]

            # Check that mask values in between 0 and 1, NOT 0 and 255 for
            # binary segmentation
            assert mask.max() <= 1.0
            assert mask.min() >= 0

        # Predict mask logits
        logits_mask = self.forward(image)

        if self.number_of_classes > 1:
            assert (
                logits_mask.shape[1] == self.number_of_classes
            )  # [batch_size, number_of_classes, H, W]

            # Ensure the logits mask is contiguous
            logits_mask = logits_mask.contiguous()

        # Compute loss using given loss fn (pass original mask, not one-hot
        # encoded)
        loss = self.loss_fn(logits_mask, mask)

        if self.number_of_classes > 1:
            # Apply softmax to get probabilities for multi-class segmentation
            prob_mask = logits_mask.softmax(dim=1)

            # Convert probabilities to predicted class labels
            pred_mask = prob_mask.argmax(dim=1)

            # Compute true positives, false positives, false negatives,
            # and true negatives
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask,
                mask,
                mode="multiclass",
                num_classes=self.number_of_classes,
            )
        else:
            # first convert mask values to probabilities, then
            # apply thresholding
            prob_mask = logits_mask.sigmoid()
            # pred_mask = (prob_mask > 0.5).float()
            pred_mask = (prob_mask > 0.5).long()

            # Compute true positives, false positives, false negatives,
            # and true negatives
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask,
                mask,
                mode="binary",
            )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        """Share epoch end."""
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # metric calculations
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

        # loss
        loss = torch.stack([x["loss"] for x in outputs])
        dataset_loss = torch.mean(loss)

        metrics = {
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_dataset_accuracy": dataset_acc,
            f"{stage}_dataset_f1_score": dataset_f1,
            f"{stage}_dataset_loss": dataset_loss,
        }

        self.current_loss = dataset_loss
        self.log_dict(metrics, prog_bar=False)

    # ruff:noqa:ARG002 # Unused method argument
    def training_step(self, batch, batch_idx):
        """Train step."""
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        """Train epoch end."""
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    # ruff:noqa:ARG002 # Unused method argument
    def validation_step(self, batch, batch_idx):
        """Validate step."""
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        """Validate epoch end."""
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

        # replacement for broken callback ModelCheckpoint
        # save best model monitoring validation loss
        if self.current_epoch > 4 and self.best_loss > self.current_loss:
            self.best_loss = self.current_loss
            best_model_path = (
                f"{self.model_path_base}_epoch{self.current_epoch}"
            )
            print("\nsaving new best model...\n", file=sys.stderr)
            self.model.save_pretrained(best_model_path, push_to_hub=False)
            if self.best_model_path:
                try:
                    Path(self.best_model_path).rmdir()
                except Exception:
                    shutil.rmtree(self.best_model_path, ignore_errors=True)

            self.best_model_path = best_model_path
            self.best_epoch = self.current_epoch

    # ruff:noqa:ARG002 # Unused method argument
    def test_step(self, batch, batch_idx):
        """Test step."""
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        """Test epoch end."""
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizers."""
        # weight_decay should be in the range 0, 0.05
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=2e-4,
            weight_decay=0.0,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
            eta_min=0.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def smp_train(
    data_dir,
    img_size,
    in_channels,
    out_classes=2,
    model_arch="Unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    input_model_path=None,
    output_model_path=None,
    output_train_metrics_path=None,
    epochs=50,
    batch_size=8,
):
    """See https://smp.readthedocs.io/en/latest/encoders.html.

    Args:
        data_dir (string): root folder with training data
        img_size (int): size of the training images in pixels, e.g. 512
        in_channels (int): number of input channels
        out_classes (int): number of output classes
        model_arch (string): model architecture,
            see https://smp.readthedocs.io/en/latest/models.html
        encoder_name (string): name of the encoder,
            see https://smp.readthedocs.io/en/latest/encoders.html
        encoder_weights (string): name of pretrained weights, default
                                  "imagenet"
        input_model_path (string): path to trained and locally saved model
        output_model_path (string): path to save new model
        output_train_metrics_path (string): path to save training metrics
        epochs (int): number of epochs for training
        batch_size (int): batch size for training

    """
    if output_model_path is None:
        print("ERROR: output model path is required.", file=sys.stderr)
        sys.exit(1)

    # dataset definitions
    x_train_dir = os.path.join(data_dir, "train_images")
    y_train_dir = os.path.join(data_dir, "train_masks")

    x_valid_dir = os.path.join(data_dir, "val_images")
    y_valid_dir = os.path.join(data_dir, "val_masks")

    torch.set_float32_matmul_precision("medium")

    gdal.UseExceptions()

    # pytorch datasets
    train_dataset = GdalImageDataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(img_size),
    )

    valid_dataset = GdalImageDataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(img_size),
    )

    # pytorch dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    out_classes_model = out_classes
    if out_classes == 2:
        out_classes_model = 1

    # loading the model
    if input_model_path:
        if not Path(input_model_path).exists():
            print(
                f"ERROR: input model path {input_model_path} does not exist.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Loading model saved at {input_model_path} ...")
        model = smp.from_pretrained(input_model_path)

        if not model:
            print(
                f"ERROR: failed to load input model from {input_model_path}.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print(
            f"loading model {model_arch} with encoder {encoder_name} ...",
            file=sys.stderr,
        )
        # handling special cases
        model_kwargs = {}
        if batch_size < 6 and model_arch.lower() in {"upernet", "manet"}:
            model_kwargs["decoder_use_norm"] = False
        # img_size=XXX, needed for swin and other transformer encoders
        if (
            encoder_name.lower()[:7] == "tu-swin"
            or encoder_name.lower()[:8] == "tu-hiera"
            or encoder_name.lower()[:9] == "tu-mvitv2"
        ):
            model_kwargs["img_size"] = img_size

        model = smp.create_model(
            model_arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes_model,
            **model_kwargs,
        )

    # The number of iteration per epoch is calculated by number_of_samples /
    # batch_size ??
    # the number of iterations per epoch is reported for each epoch during
    # training: cross-check
    # T_MAX = EPOCHS * math.ceil(len(train_loader) / BATCH_SIZE)
    # no, T_MAX too low, undulating training curves
    t_max = epochs * len(train_loader)

    # arguments for smp model
    mymodule = PlModule(
        model,
        out_classes=out_classes_model,
        model_path_base=output_model_path,
        t_max=t_max,
    )
    # small batchsizes: do not use batchnorm because pytorch batch_norm fails
    # with small batch sizes

    print("setting up pl trainer ...", file=sys.stderr)

    # logger for training metrics
    p_abs = Path(output_train_metrics_path).absolute()
    p_base = Path(p_abs).name
    p_dir = Path(p_abs).parent
    logger = CSVLogger(
        p_dir,
        name=None,
        version=f"{p_base}",
    )

    # checkpoint callback
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks
    # .ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint
    # After training finishes, use best_model_path to retrieve the path to the
    # best checkpoint file and best_model_score to retrieve its score.

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # !!! the ModelCheckpoint destroys the metrics,
    # there is no improvement during training when using the
    # callback ModelCheckpoint validation loss remains constant
    # validation iou, precision, recall are all and always 0
    # without this callback, metrics improve as expected
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )

    print("training ...", file=sys.stderr)
    trainer.fit(
        mymodule,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    if mymodule.best_model_path:
        # rename best_model_path to output_model_path
        Path(mymodule.best_model_path).rename(output_model_path)
    else:
        mymodule.model.save_pretrained(output_model_path, push_to_hub=False)

    if mymodule.best_epoch > 0:
        print(f"best epoch: {mymodule.best_epoch}", file=sys.stderr)
        with open(
            os.path.join(output_model_path, "README.md"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write("\n\n## Saved model")
            f.write(f"\nSaved model with best epoch: {mymodule.best_epoch}\n")
