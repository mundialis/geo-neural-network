#!/usr/bin/env python3
"""#############################################################################
#
# MODULE:      wrapper script for smp_train from geo-neural-network
# AUTHOR(S):   Markus Metz, mundialis.
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
    """Pass arguments from config file to smp_train."""
    # dataset definitions
    data_dir = config["data_dir"]
    in_channels = config["in_channels"]
    out_classes = config["out_classes"]
    img_size = config["img_size"]

    # model definition
    model_arch = config["model_arch"]
    # see https://smp.readthedocs.io/en/latest/encoders.html
    encoder_name = config["encoder_name"]
    # weights can also be None
    encoder_weights = config["encoder_weights"]

    # path to folder with saved, trained model, can be None
    in_model_path = config["input_model_path"]

    # path to folder to save the trained model
    out_model_path = config["output_model_path"]

    # path to folder to save training metrics
    output_train_metrics_path = config["output_train_metrics_path"]

    # some training hyperparameters
    epochs = config["epochs"]
    # do not use batch normalisation in the model with batch size < 8
    # add decoder_use_norm=False when initialising the plModule
    # applies to upernet, manet, not to segformer
    batch_size = config["batch_size"]

    smp_train(
        data_dir=data_dir,
        img_size=img_size,
        in_channels=in_channels,
        out_classes=out_classes,
        model_arch=model_arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        input_model_path=in_model_path,
        output_model_path=out_model_path,
        output_train_metrics_path=output_train_metrics_path,
        epochs=epochs,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model from segmentation_models.pytorch",
    )
    parser.add_argument("configfile", help="Path to configfile.")

    args = parser.parse_args()

    confparser = configparser.ConfigParser()
    confparser.read(args.configfile)

    config = {}
    config["data_dir"] = confparser.get("dataset", "data_dir")
    config["in_channels"] = int(confparser.get("dataset", "in_channels"))
    config["out_classes"] = int(confparser.get("dataset", "out_classes"))
    config["img_size"] = int(confparser.get("dataset", "img_size"))

    config["model_arch"] = confparser.get("model", "model_arch")
    config["encoder_name"] = confparser.get("model", "encoder_name")
    config["encoder_weights"] = confparser.get("model", "encoder_weights")
    config["epochs"] = int(confparser.get("model", "epochs"))
    config["batch_size"] = int(confparser.get("model", "batch_size"))

    config["input_model_path"] = None
    config["input_model_path"] = confparser.get(
        "model",
        "input_model_path",
        fallback=None,
    )
    config["output_model_path"] = confparser.get("output", "output_model_path")
    config["output_train_metrics_path"] = None
    config["output_train_metrics_path"] = confparser.get(
        "output",
        "output_train_metrics_path",
        fallback=None,
    )

    main(config)
