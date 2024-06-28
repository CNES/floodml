#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS - All Rights Reserved
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
    ul_latlon = [math.floor(ul[1]), math.ceil(ul[0])]
    lr_latlon = [math.ceil(lr[1]), math.floor(lr[0])]
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

def get_gswo_codes(gswdir, ul, lr):
    """
    Get the list of GSWO files (10deg x 10deg) for a given site.

    :param gswdir: The directory where all Global Surface Water Occurrence files are stored in.
    No subfolders are allowed, all files need to be in the same directory
    :param ul: Upper left coordinate (lat, lon) of the site expressed in WGS-84 (EPSG 4326)
    :param lr: Lower right coordinate (lat, lon) of the site expressed in WGS-84 (EPSG 4326)
    :return: The list of filenames needed in order to cover to whole site.
    """
    import math

    ul_latlon = [math.ceil(ul[0]/10)*10, math.floor(ul[1]/10)*10] # 10°x10° data selection
    lr_latlon = [math.floor(lr[0]/10)*10, math.ceil(lr[1]/10)*10] # 10°x10° data selection

    gsw_files = []
    for y in range(lr_latlon[0]+10, ul_latlon[0]+1, 10):
        for x in range(ul_latlon[1], lr_latlon[1], 10):
            code_lat = "N" if y >= 0 else "S"
            code_lon = "E" if x >= 0 else "W"
            gswfile = os.path.join(gswdir,
                                   "occurrence_{}{}_{}{}.tif".format( abs(x),code_lon, abs(y), code_lat))
            assert os.path.isfile(gswfile), "Cannot find GSWO file: %s" % gswfile
            if not os.path.isfile(gswfile):
                print("Cannot find GSWO file: %s" % gswfile)
            else:
                gsw_files.append(gswfile)
    return gsw_files

def get_esawc_codes(indir, ul, lr):
    """
    Get the list of ESA landcover (3deg x 3deg) for a given site.

    :param wcdir: The directory where all ESA world cover files are stored in.
    No subfolders are allowed, all files need to be in the same directory
    :param ul: Upper left coordinate (lat, lon) of the site expressed in WGS-84 (EPSG 4326)
    :param lr: Lower right coordinate (lat, lon) of the site expressed in WGS-84 (EPSG 4326)
    :return: The list of filenames needed in order to cover to whole site.
    """
    import math

    ul_latlon = [math.floor(ul[1]/3)*3, math.ceil(ul[0]/3)*3] # 3°x3° data selection
    lr_latlon = [math.ceil(lr[1]/3)*3, math.floor(lr[0]/3)*3] # 3°x3° data selection

    wc_files = []
    for y in range(lr_latlon[1], ul_latlon[1], 3):
        for x in range(ul_latlon[0], lr_latlon[0], 3):
            code_lat = "N" if y >= 0 else "S"
            code_lon = "E" if x >= 0 else "W"
            wcfile = os.path.join(indir,
                                   "ESA_WorldCover_10m_2021_v200_{}{}{}{}_Map.tif".format(code_lat, str(abs(y)).zfill(2), code_lon, str(abs(x)).zfill(3)))
            assert os.path.isfile(wcfile), "Cannot find ESA worldcover file: %s" % wcfile
            wc_files.append(wcfile)
    return wc_files
