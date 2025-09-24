#!/usr/bin/env python3
"""
#############################################################################
#
# MODULE:      wrapper script for smp_inference from geo-neural-network
# AUTHOR(S):   Markus Metz, mundialis
#
# PURPOSE:     Apply a saved model from segmentation_models.pytorch
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

# after a model from segmentation_models.pytorch has been trained (and finetuned)
# use this locally saved model for inference

# adapted from
# https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/upernet_inference_pretrained.ipynb

import os
import argparse
import configparser
from geo_neural_network.smp_lib.smp_inference import smp_inference


def main(config):

    data_dir = config["data_dir"]

    num_classes = config["num_classes"]

    # path to folder with saved smp model
    model_path = config["model_path"]

    output_path = config["output_path"]

    smp_infer(
        data_dir=data_dir,
        input_model_path=model_path,
        num_classes=num_classes,
        output_path=output_path,
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
    config["num_classes"] = int(
        confparser.get("settings.dataset", "num_classes")
    )
    config["model_path"] = confparser.get("settings.model", "model_path")
    config["output_path"] = confparser.get("settings.output", "output_path")

    main(config)
