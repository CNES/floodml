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
    :param ul: Upper left coordinate (lon, lat) of the site expressed in WGS-84 (EPSG 4326)
    :param lr: Lower right coordinate (lon, lat) of the site expressed in WGS-84 (EPSG 4326)
    :return: The list of filenames needed in order to cover to whole site.
    """
    ul_lonlat = [int(ul[0]), int(ul[1])]
    lr_lonlat = [int(lr[0]) + 1, int(lr[1]) + 1]

    dem_files = []
    for x in range(int(ul_lonlat[1]), int(lr_lonlat[1] + 1)):
        for y in range(int(ul_lonlat[0]), int(lr_lonlat[0] + 1)):
            code_lat = "N" if y >= 0 else "S"
            code_lon = "E" if x >= 0 else "W"
            demfile = os.path.join(demdir,
                                   "Copernicus_DSM_10_%s%02d_00_%s%03d_00_DEM.dt2" % (code_lat, abs(y),
                                                                                      code_lon, abs(x)))
            assert os.path.isfile(demfile), "Cannot find Copernicus-DEM file: %s" % demfile
            dem_files.append(demfile)
    return dem_files

