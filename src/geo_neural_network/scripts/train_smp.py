#!/usr/bin/env python3
"""
#############################################################################
#
# MODULE:      wrapper script for smp_train from geo-neural-network
# AUTHOR(S):   Markus Metz, mundialis
#
# PURPOSE:     Train a model from segmentation_models.pytorch
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
############################################################################
"""

# train a model from segmentation_models.pytorch
# use segmentation_models_pytorch tools and pytorch lightning for training
# https://github.com/qubvel-org/segmentation_models.pytorch

# based on
# https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/camvid_segmentation_multiclass.ipynb

import argparse
import configparser
from geo_neural_network.smp_lib.smp_train import smp_train


def main(config):
    # dataset definitions
    DATA_DIR = config["data_dir"]
    IN_CHANNELS = config["in_channels"]
    OUT_CLASSES = config["out_classes"]
    IMG_SIZE = config["img_size"]

    # model definition
    MODEL_ARCH = config["model_arch"]
    # see https://smp.readthedocs.io/en/latest/encoders.html
    ENCODER_NAME = config["encoder_name"]
    # weights can also be None
    ENCODER_WEIGHTS = config["encoder_weights"]

    # path to folder with saved, trained model, can be None
    IN_MODEL_PATH = config["input_model_path"]

    # path to folder to save the trained model
    OUT_MODEL_PATH = config["output_model_path"]

    # some training hyperparameters
    EPOCHS = config["epochs"]
    # do not use batch normalisation in the model with batch size < 8
    # add decoder_use_norm=False when initialising the plModule
    # applies to upernet, manet, not to segformer
    BATCH_SIZE = config["batch_size"]

    # end of configuration

    smp_train(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
        model_arch=MODEL_ARCH,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        input_model_path=IN_MODEL_PATH,
        output_model_path=OUT_MODEL_PATH,
        output_train_metrics_path=None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model from segmentation_models.pytorch"
    )
    parser.add_argument("configfile", help="Path to configfile.")

    args = parser.parse_args()

    confparser = configparser.ConfigParser()
    confparser.read(args.configfile)

    config = {}
    config["data_dir"] = confparser.get("settings.dataset", "data_dir")
    config["in_channels"] = int(
        confparser.get("settings.dataset", "in_channels")
    )
    config["out_classes"] = int(
        confparser.get("settings.dataset", "out_classes")
    )
    config["img_size"] = int(confparser.get("settings.dataset", "img_size"))

    config["model_arch"] = confparser.get("settings.model", "model_arch")
    config["encoder_name"] = confparser.get("settings.model", "encoder_name")
    config["encoder_weights"] = confparser.get(
        "settings.model", "encoder_weights"
    )
    config["epochs"] = int(confparser.get("settings.model", "epochs"))
    config["batch_size"] = int(confparser.get("settings.model", "batch_size"))

    config["input_model_path"] = None
    config["input_model_path"] = confparser.get(
        "settings.model", "input_model_path", fallback=None
    )
    config["output_model_path"] = confparser.get(
        "settings.model", "output_model_path"
    )
    config["output_train_metrics_path"] = None
    config["output_train_metrics_path"] = confparser.get(
        "settings.model", "output_train_metrics_path", fallback=None
    )

    main(config)
