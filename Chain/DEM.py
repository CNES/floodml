#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import os


def get_copdem_codes(demdir, ul, lr):
    """
    Get the list of Copernicus DEM GLO-30 files (1deg x 1deg) for a given site.

    :param demdir: The directory where all Copernicus DEM files are stored in.
    No subfolders are allowed, all files need to be in the same directory
    :param ul: Upper left coordinate (lat, lon) of the site expressed in WGS-84 (EPSG 4326)
    :param lr: Lower right coordinate (lat, lon) of the site expressed in WGS-84 (EPSG 4326)
    :return: The list of filenames needed in order to cover to whole site.
    """
    import math
    if ul[1] > 0:
        f_ul2 = math.floor
    else:
        f_ul2 = math.ceil
    if ul[0] > 0:
        f_ul1 = math.floor
    else:
        f_ul1 = math.ceil
    ul_latlon = [f_ul1(ul[1]), f_ul2(ul[0])]

    if lr[1] > 0:
        f_lr2 = math.ceil
    else:
        f_lr2 = math.floor
    if lr[0] > 0:
        f_lr1 = math.ceil
    else:
        f_lr1 = math.floor
    lr_latlon = [f_lr1(lr[1]), f_lr2(lr[0])]
    dem_files = []
    for y in range(lr_latlon[1], ul_latlon[1]):
        for x in range(ul_latlon[0], lr_latlon[0]):
            code_lat = "N" if y >= 0 else "S"
            code_lon = "E" if x >= 0 else "W"
            demfile = os.path.join(demdir,
                                   "Copernicus_DSM_10_%s%02d_00_%s%03d_00_DEM.dt2" % (code_lat, abs(y),
                                                                                      code_lon, abs(x)))
            assert os.path.isfile(demfile), "Cannot find Copernicus-DEM file: %s" % demfile
            dem_files.append(demfile)
    return dem_files

