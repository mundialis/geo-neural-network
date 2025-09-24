#!/usr/bin/env python3
"""
#############################################################################
#
# MODULE:      wrapper script for smp_test from geo-neural-network
# AUTHOR(S):   Markus Metz, mundialis
#
# PURPOSE:     Test a trained and saved model from segmentation_models.pytorch
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

# test a saved model: use a test dataset to compute a confusion matrix
# and IoU per class


import os
import argparse
import configparser
from geo_neural_network.smp_lib.smp_test import smp_test


def main(config):
    DATA_DIR = config["data_dir"]

    NUM_CLASSES = config["num_classes"]
    CLASS_NAMES = [x.strip() for x in config["class_names"].split(",")]
    if len(CLASS_NAMES) != NUM_CLASSES:
        print("Number of class names does not match number of classes!")

    MODEL_PATH = config["model_path"]

    OUTPUT_PATH = config["output_path"]

    smp_test(
        data_dir=DATA_DIR,
        input_model_path=MODEL_PATH,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="finetune a saved model from segmentation_models.pytorch"
    )
    parser.add_argument("configfile", help="Path to configfile.")

    args = parser.parse_args()

    confparser = configparser.ConfigParser()
    confparser.read(args.configfile)

    config = {}
    config["data_dir"] = confparser.get("settings.dataset", "data_dir")

    config["model_path"] = confparser.get("settings.model", "model_path")
    config["num_classes"] = int(
        confparser.get("settings.dataset", "num_classes")
    )
    config["class_names"] = confparser.get("settings.dataset", "class_names")

    config["output_path"] = confparser.get("settings.output", "output_path")

    main(config)
