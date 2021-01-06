#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import tensorflow as tf
import argparse
from deep_learning.Imagery.Predict import Predict

if __name__ == "__main__":
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
    except Exception as e:
        print(e)
    # Argument parsing
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('cfg_file', action="store", type=str, help="Configuration file")
    parser.add_argument('--model', '-m', action="store", type=str, help="Path to a model.h5 file to be used. If none is"
                                                                        "given, the newest available one will be used.")
    parser.add_argument('--prob_threshold', action="store", type=float,
                        help="Apply a specific probability threshold. Do a value search otherwise.")
    args_cmd = parser.parse_args()
    # main
    p = Predict(args_cmd.cfg_file,
                model=args_cmd.model,
                prob_threshold=args_cmd.prob_threshold)
    p.run()
