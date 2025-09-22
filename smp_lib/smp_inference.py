#!/usr/bin/env python3
"""############################################################################
#
# MODULE:      smp_inference.py
# AUTHOR(S):   Markus Metz, mundialis
# PURPOSE:     Apply a saved model from segmentation_models.pytorch.
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

# after a model from segmentation_models.pytorch has been trained (and
# finetuned) use this locally saved model for inference

# adapted from
# https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/
# examples/upernet_inference_pretrained.ipynb

import os
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from osgeo import gdal


def read_image_gdal(filename, driver, output_file):
    """Args:
    filename (string): path to file to read with GDAL
    driver (GDAL raster driver): GDAL driver to use for creating output raster
    output_file (string): path to output raster file.
    """
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f"Unable to open file: {filename}")

    img = ds.ReadAsArray()
    # PyTorch works only on CHW format (Channel, Height, Width)
    # GDAL should by default return multiband raster data in CHW format
    # singleband raster are returned as HW, no channel dimension
    # print(f"GDAL image shape: {img.shape}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    # channel = ds.RasterCount  # should be 1
    trans = ds.GetGeoTransform()
    proj = ds.GetProjection()

    seg_map = driver.Create(
        output_file,
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=LZW"],
    )
    seg_map.SetGeoTransform(trans)
    seg_map.SetProjection(proj)
    # Set no data to 255 -> assuming no classification with 255 classes
    seg_map.GetRasterBand(1).SetNoDataValue(255)

    # close GDAL dataset
    ds = None

    return img, seg_map


def smp_infer(data_dir, input_model_path, num_classes, output_path):
    """Args:
    data_dir (string): root folder with training data
    input_model_path (string): path to trained and locally saved model
    num_classes (int): number of output classes
    output_path (string): path where to save results.

    """
    x_test_dir = data_dir

    if not Path(output_path).exists():
        Path(output_path).mkdir()

    gdal.UseExceptions()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model and preprocessing function
    model = smp.from_pretrained(input_model_path).eval().to(device)
    # preprocessing = A.Compose.from_pretrained(MODEL_PATH)
    # print("Preprocessing:\n", preprocessing)

    mean = 125.5
    std = 100.2

    # GDAL driver to write outputs
    driver = gdal.GetDriverByName("GTiff")

    # directory listing
    ids = os.listdir(x_test_dir)
    images_fps = [os.path.join(x_test_dir, image_id) for image_id in ids]

    # only tif, jp2 and vrt
    # TODO: Statt alle tif, vrt oder jp2 Dateien in einem Ordner zu nehmen,
    # eine Textdatei mit den Dateinamen einlesen
    # Note: in der Textdatei könnten auch GDAL subdatasets,
    # z.B. NETCDF:"sst.nc":tos definiert werden.
    """
    os.chdir(x_test_dir)
    ids = sorted(
        list(pathlib.Path(".").glob('*.tif'))
        + list(pathlib.Path(".").glob('*.vrt'))
    )
    images_fps = [os.path.join(x_test_dir, image_id) for image_id in ids]
    """

    print("inference ...", file=sys.stderr)
    # loop over images (image file paths
    for filename, image_fp in zip(ids, images_fps):
        output_file = os.path.join(output_path, filename)
        if output_file.endswith(".vrt"):
            output_file = output_file.replace(".vrt", ".tif")
        elif output_file.endswith(".jp2"):
            output_file = output_file.replace(".jp2", ".tif")
        elif not output_file.endswith(".tif") and not output_file.endswith(
            ".jp2",
        ):
            # only process tif, jp2 and vrt images
            continue
        # Load image
        image, seg_map = read_image_gdal(image_fp, driver, output_file)
        # print(f"image shape: {image.shape}")
        # Preprocess image
        # normalized_image = preprocessing(image=image)["image"]
        normalized_image = (image.astype(np.float32) - mean) / std

        input_tensor = torch.as_tensor(normalized_image)
        # image is already CHW
        # input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        #       HWC -> BCHW
        input_tensor = input_tensor.unsqueeze(0)  # CHW -> BCHW
        input_tensor = input_tensor.to(device)

        # Perform inference
        with torch.inference_mode():
            output_mask = model(input_tensor)

        # Postprocess mask
        mask = torch.nn.functional.interpolate(
            output_mask,
            size=image.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        # Evaluate model output to discrete classes
        # + preserve nan values as no-data value
        # (here the no-data value is 255)
        if num_classes > 2:
            nan_mask = np.all(np.isnan(mask[0].cpu().numpy()), 0)
            mask = mask[0].argmax(0).cpu().numpy()
            mask[nan_mask] = 255
        else:
            # mask = mask[0].sigmoid()
            nan_mask = np.isnan(mask[0].cpu().numpy())
            mask_filter = (mask[0] > 0.5).cpu().numpy()
            # object type to allow saving no data value (not only boolean)
            mask = mask_filter.astype(object)
            mask[nan_mask] = 255

        # write output with GDAL
        seg_map.WriteArray(mask)
        # close GDAL dataset
        seg_map = None

    print("done", file=sys.stderr)
