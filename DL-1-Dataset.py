#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import argparse
import os
from deep_learning.Imagery.Dataset import Dataset


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Orchestration')
    parser.add_argument('cfg_file', action="store", type=str, help="Configuration file")
    parser.add_argument('--backup_json', action="store_true",
                        help="Backup temporary json file containing img and mask pairs found.", default=True)
    parser.add_argument('--pickup_json', action="store", type=str,
                        help="Use temp json file containing img and mask pairs found.", required=False)
    parser.add_argument('--mode', action="store", type=str,
                        help="Should be 'training' or 'validation'", required=True)

    arg = parser.parse_args()
    if not os.path.isfile(arg.cfg_file):
        raise OSError("Cannot find config file %s" % arg.cfg_file)
    d = Dataset(arg.cfg_file, arg.mode, arg.backup_json, arg.pickup_json)
    d.run()
