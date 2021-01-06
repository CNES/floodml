#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import os
from osgeo import ogr
import glob

os.environ["PATH"] = os.environ["PATH"].split(";")[-1]

# Initialization ######################
s2tiles = '/work/scratch/fatrasc/Hand_proc/Sentinel2_tiles.shp'

input = '/work/scratch/fatrasc/Hand_proc/World_hand.vrt'
output = '/work/scratch/fatrasc/Hand_proc/Hand_tiled'

# Tiling ##############################
ogrl = ogr.Open(s2tiles)
layer = ogrl.GetLayer()

for og in layer:
    x_min, x_max, y_min, y_max = og.GetGeometryRef().GetEnvelope()
    tile = og.GetField('Name')
    print("Tile: ", tile)
    if y_max <= 60 and y_min >= -60:
        out = os.path.join(output, tile+'.tif')
        gdalw = 'gdalwarp -te {0:5f} {1:5f} {2:5f} {3:5f} {4:s} {5:s}'.format(x_min, y_min, x_max, y_max, input, out)
        os.system(gdalw)

