#!/usr/bin/env python3
"""############################################################################
#
# MODULE:      smp_test.py
# AUTHOR(S):   Markus Metz, mundialis
# PURPOSE:     Test a trained and saved model from segmentation_models.pytorch.
#
# COPYRIGHT:   (C) 2025 by mundialis GmbH & Co. KG
#
# This program is free software; you can redistribute it and/or modify
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

# test a saved model: use a test dataset to compute a confusion matrix
# and IoU per class


import os
import sys
from pathlib import Path

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from osgeo import gdal
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
)


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


class GdalImageDataset(BaseDataset):
    """pytorch dataset using GDAL to read raster files."""

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


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32."""
    test_transform = [
        albumentations.PadIfNeeded(512, 512),
    ]
    return albumentations.Compose(test_transform)


def evaluate_model(
    model,
    test_loader,
    num_classes,
    device="cuda",
    class_names=None,
    save_plot_path=None,
    save_norm_plot_path=None,
):
    """Evaluate segmentation model and compute confusion matrix."""
    # Set model to evaluation mode
    model.eval()
    model.to(device)

    # Initialize confusion matrix
    if num_classes > 2:
        confmat = MulticlassConfusionMatrix(
            num_classes=num_classes,
            normalize="none",
        ).to(device)
    else:
        confmat = BinaryConfusionMatrix(normalize="none").to(device)

    with torch.no_grad():
        for im, ma in test_loader:
            # Move data to device
            images = im.to(device)

            # Normalize images
            images = images.float()
            images = (images - 125.5) / 100.2

            masks = ma.to(device)

            # Forward pass
            outputs = model(images)

            # Get predicted labels (B x H x W)
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            preds = outputs.argmax(dim=1) if num_classes > 2 else outputs

            # Update confusion matrix (flatten spatial dimensions H x W)
            confmat.update(preds.flatten(), masks.flatten())

    # Compute final confusion matrix on device
    cm_tensor = confmat.compute()

    # Save confusion matrix plot if requested
    if save_plot_path:
        fig, ax = confmat.plot(
            labels=class_names,
            add_text=True,
        )  # Set add_text=True to show values in cells
        fig.set_size_inches(12, 10)
        # fig.tight_layout()
        plt.savefig(save_plot_path, bbox_inches="tight", dpi=300)
        plt.close(fig)  # Important: close figure to free memory

    if save_norm_plot_path:
        cm_tensor_norm = cm_tensor / cm_tensor.sum(dim=-1, keepdim=True)
        fig, ax = confmat.plot(
            val=cm_tensor_norm,
            labels=class_names,
            add_text=True,
        )  # Set add_text=True to show values in cells
        fig.set_size_inches(12, 10)
        # fig.tight_layout()
        plt.savefig(save_norm_plot_path, bbox_inches="tight", dpi=300)
        plt.close(fig)  # Important: close figure to free memory

    return cm_tensor


def smp_test(
    data_dir,
    input_model_path,
    num_classes,
    class_names,
    output_path,
):
    """Args:
    data_dir (string): root folder with training data
    input_model_path (string): path to trained and locally saved model
    num_classes (int): number of output classes
    class_names (string): comma-separated list of class names
    output_path (string): path where to save figures and statistics.

    """
    class_names = [x.strip() for x in class_names.split(",")]
    if len(class_names) != num_classes:
        print("Number of class names does not match number of classes!")

    if not Path(output_path).exists():
        Path(output_path).mkdir()

    # hard-coded output file names
    plot_path = os.path.join(output_path, "confusion_matrix.png")
    norm_plot_path = os.path.join(
        output_path,
        "confusion_matrix_normalized.png",
    )
    iou_path = os.path.join(output_path, "metrics_per_class")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_test_dir = os.path.join(data_dir, "test_images")
    y_test_dir = os.path.join(data_dir, "test_masks")

    # Load your model and test_loader here
    model = smp.from_pretrained(input_model_path)

    gdal.UseExceptions()

    test_dataset = GdalImageDataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
    )

    # Compute confusion matrix and save plot
    print("evaluating the model ...")
    cm = evaluate_model(
        model,
        test_loader,
        num_classes=num_classes,
        device=device,
        class_names=class_names,
        save_plot_path=plot_path,
        save_norm_plot_path=norm_plot_path,
    )

    # Convert to numpy for analysis
    cm_np = cm.cpu().numpy()

    # Calculate metrics
    diagonal = np.diag(cm_np)
    accuracy = diagonal.sum() / cm_np.sum()
    precision_per_class = diagonal / cm_np.sum(0)
    recall_per_class = diagonal / cm_np.sum(1)
    fscore_per_class = (
        2
        * (precision_per_class * recall_per_class)
        / (precision_per_class + recall_per_class)
    )
    iou_per_class = diagonal / (cm_np.sum(0) + cm_np.sum(1) - diagonal)

    # normalize cm by target (truth), from
    # https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/classification/confusion_matrix.py
    # `"true"` will divide by the sum of the column dimension (truth)
    cm_np_norm = cm_np / cm_np.sum(axis=-1, keepdims=True)
    # `"pred"` will divide by the sum of the row dimension (predictions)
    # cm_np_norm = cm_np / cm_np.sum(dim=-2, keepdim=True)

    with open(iou_path, "w", encoding="utf-8") as f:
        print(f"Overall Accuracy: {accuracy:.4f}", file=f)
        print(f"mIoU: {np.nanmean(iou_per_class):.4f}", file=f)
        for i, iou in enumerate(iou_per_class):
            class_name = class_names[i] if class_names else f"Class {i}"
            print(f"IoU for {class_name}: {iou:.4f}", file=f)
        print(f"mF-score: {np.nanmean(fscore_per_class):.4f}", file=f)
        for i, fscore in enumerate(fscore_per_class):
            class_name = class_names[i] if class_names else f"Class {i}"
            print(f"F-score for {class_name}: {fscore:.4f}", file=f)

    outfile = os.path.join(output_path, "confusion_matrix.csv")
    np.savetxt(outfile, cm_np, delimiter=";")
    outfile = os.path.join(output_path, "confusion_matrix_normalized.csv")
    np.savetxt(outfile, cm_np_norm, delimiter=";")
    # print("\nvalues", file=f)
    # print(cm_np, file=f)
    # print("\nnormalized", file=f)
    # print(cm_np_norm, file=f)

    print(f"IoU per class saved to: {iou_path}")
