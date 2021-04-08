#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Author:         Peter KETTIG <peter.kettig@cnes.fr>
"""

from Common.GDalDatasetWrapper import GDalDatasetWrapper
import numpy as np


class DEMInfo:
    def __init__(self, site, dem_full_res):
        self.name = site.nom
        self.epsg = site.epsg_str
        self.ulx = site.ul[0]
        self.uly = site.ul[1]
        self.resx = site.res_x
        self.resy = site.res_y
        self.lx = site.px
        self.ly = site.py
        self.alt = dem_full_res
        driver = GDalDatasetWrapper.from_file(dem_full_res)
        self.mean_alt = np.mean(driver.array)
        self.std_dev_alt = np.std(driver.array)
        self.short_description = driver.utm_description
        self.nodata = driver.nodata_value
        res_arr = driver.resolution
        self.dem_subsampling_ratio = str(int(float(res_arr[0]) / float(site.res_x)))

