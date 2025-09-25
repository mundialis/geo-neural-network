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


import argparse
import configparser
from geo_neural_network.smp_lib.smp_test import smp_test


def main(config):
    """
    pass arguments from config file to smp_test
    """
    data_dir = config["data_dir"]

    num_classes = config["num_classes"]
    class_names = [x.strip() for x in config["class_names"].split(",")]
    if len(class_names) != class_names:
        print("Number of class names does not match number of classes!")

    model_path = config["model_path"]

    output_path = config["output_path"]

    smp_test(
        data_dir=data_dir,
        input_model_path=model_path,
        num_classes=num_classes,
        class_names=class_names,
        output_path=output_path,
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
    config["data_dir"] = confparser.get("dataset", "data_dir")

    config["model_path"] = confparser.get("model", "model_path")
    config["num_classes"] = int(confparser.get("dataset", "num_classes"))
    config["class_names"] = confparser.get("dataset", "class_names")

    config["output_path"] = confparser.get("output", "output_path")

    main(config)
