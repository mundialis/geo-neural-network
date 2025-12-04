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
import os

import matplotlib.pyplot as plt
import pandas as pd

from geo_neural_network.smp_lib.smp_train import smp_train


def plot_curve(epoch, y_train, y_val, y_label, outpath, ylim=None):
    """Plot metric curves."""
    plt.figure()
    # query non-Nan values -> val and train metrics saved alternating per row
    plt.plot(
        epoch.drop_duplicates(),
        y_train.dropna(),
        "blue",
        label="Training",
    )
    plt.plot(
        epoch.drop_duplicates(),
        y_val.dropna(),
        "orange",
        label="Validation",
    )
    if ylim:
        plt.ylim(ylim)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(outpath)


def main(config):
    """Pass arguments from config file to smp_train."""
    # dataset definitions
    data_dir = config["data_dir"]
    data_version = config["data_version"]
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

    # base path to folder to save the trained model
    output_base_path = config["output_base_path"]
    output_base_dir = config["output_base_dir"]
    out_model_dir = f"model_{model_arch.replace('-','_')}_{encoder_name.replace('-','_')}_{output_base_dir}_{data_version}"

    out_model_path = os.path.join(
        output_base_path,
        out_model_dir,
        "model"
    )

    # path to folder to save training metrics
    output_train_metrics_path = os.path.join(output_base_path, out_model_dir, "metrics")

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

    # Create plots from train metrics
    train_metrics_file = os.path.join(
        output_train_metrics_path,
        "metrics.csv",
    )
    train_metrics = pd.read_csv(train_metrics_file, sep=",", header=0)
    # Loss curve
    plot_curve(
        train_metrics["epoch"],
        train_metrics["train_dataset_loss"],
        train_metrics["valid_dataset_loss"],
        "Loss",
        os.path.join(output_train_metrics_path, "loss.png"),
    )
    # Accuracy curve
    plot_curve(
        train_metrics["epoch"],
        train_metrics["train_dataset_accuracy"],
        train_metrics["valid_dataset_accuracy"],
        "Accuracy",
        os.path.join(output_train_metrics_path, "accuracy.png"),
        ylim=[0, 1],
    )
    # IoU curve
    plot_curve(
        train_metrics["epoch"],
        train_metrics["train_dataset_iou"],
        train_metrics["valid_dataset_iou"],
        "Intersection over Union (IoU)",
        os.path.join(output_train_metrics_path, "iou.png"),
        ylim=[0, 1],
    )
    # F1 curve
    plot_curve(
        train_metrics["epoch"],
        train_metrics["train_dataset_f1_score"],
        train_metrics["valid_dataset_f1_score"],
        "F1 score",
        os.path.join(output_train_metrics_path, "f1_score.png"),
        ylim=[0, 1],
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
    config["data_version"] = confparser.get("dataset", "data_version")
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
    config["output_base_path"] = confparser.get("output", "output_base_path")
    config["output_base_dir"] = confparser.get("output", "output_base_dir")

    main(config)
