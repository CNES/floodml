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
import sys

# Import relative modules; Override 'Common' folder in test-directory
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')))


def main_display(args):
    from Common import demo_tools as dtool
    bname = os.path.basename(args.input)
    date, tile = bname.split("_")[3:5]
    try:
        orbit = bname.split("_")[5].split(".")[0]
    except IndexError:
        orbit = None
    tile = tile[1:] if tile[0] == "T" else tile
    static_display_out = os.path.basename(args.input.replace("Inference", "RapidMapping").replace(".tif", ".png"))
    dtool.static_display(args.input, tile, date, orbit, static_display_out, gswo_dir=args.gsw, sentinel=args.sentinel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation scheduler')

    parser.add_argument('-i', '--input', help='Input raster', type=str, required=True)
    parser.add_argument('-g', '--gsw', help='GSW dir', type=str, required=True)
    parser.add_argument('-s', '--sentinel', help='Sentinel (Should be 1 or 2)', type=int, required=True)

    arg = parser.parse_args()

    main_display(arg)
