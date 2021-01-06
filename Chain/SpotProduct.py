#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
import re
from datetime import datetime
import numpy as np
from Chain.Product import MajaProduct
from Common.FileSystem import symlink
from prepare_mnt.mnt.SiteInfo import Site
from Common import FileSystem, ImageApps
from Common.GDalDatasetWrapper import GDalDatasetWrapper


class Spot5Muscate(MajaProduct):
    """
    A Spot 5 muscate product
    """

    base_resolution = (15, -15)
    coarse_resolution = (240, -240)

    @property
    def platform(self):
        return "spot5"

    @property
    def short_name(self):
        return "sp5"

    @property
    def type(self):
        return "muscate"

    @property
    def level(self):
        return "l1c"

    @property
    def nodata(self):
        return 0

    @property
    def tile(self):
        site = self.base.split("_")[-3]
        tile = re.search(self.reg_tile, site)
        if tile:
            return tile.group()[1:]
        return site

    @property
    def metadata_file(self):
        return self.find_file("*MTD_ALL.xml")[0]

    @property
    def date(self):
        str_date = self.base.split("_")[1]
        # Datetime has troubles parsing milliseconds, so it's removed:
        str_date_no_ms = str_date[:str_date.rfind("-")]
        return datetime.strptime(str_date_no_ms, "%Y%m%d-%H%M%S")

    @property
    def validity(self):
        if os.path.exists(self.metadata_file):
            return True
        return False

    def link(self, link_dir):
        symlink(self.fpath, os.path.join(link_dir, self.base))

    @property
    def mnt_site(self):
        try:
            band_bx = self.find_file(r"*_XS1*.tif")[0]
        except IOError as e:
            raise e
        return Site.from_raster(self.tile, band_bx)

    @property
    def mnt_resolutions_dict(self):
        return [{"name": "XS",
                "val": str(self.mnt_resolution[0]) + " " + str(self.mnt_resolution[1])}]

    def get_synthetic_band(self, synthetic_band, **kwargs):
        wdir = kwargs.get("wdir", self.fpath)
        output_folder = os.path.join(wdir, self.base)
        FileSystem.create_directory(output_folder)
        output_bname = "_".join([self.base, synthetic_band.upper() + ".tif"])
        output_filename = kwargs.get("output_filename", os.path.join(wdir, output_bname))
        max_value = kwargs.get("max_value", 10000)
        # Skip existing:
        if os.path.exists(output_filename):
            return output_filename
        if synthetic_band.lower() == "ndvi":
            xs1 = self.find_file(pattern=r"*XS1*.tif$")[0]
            xs3 = self.find_file(pattern=r"*XS3*.tif$")[0]
            ds_1 = GDalDatasetWrapper.from_file(xs1)
            ds_2 = GDalDatasetWrapper.from_file(xs3)
            ds_ndvi = ImageApps.get_ndvi(ds_1, ds_2, vrange=(0, max_value), dtype=np.int16)
            ds_ndvi.write(output_filename, options=["COMPRESS=DEFLATE"])

        elif synthetic_band.lower() == "ndsi":
            xs2 = self.find_file(pattern=r"*XS2*.tif$")[0]
            swir = self.find_file(pattern=r"*SWIR*.tif$")[0]
            ds_1 = GDalDatasetWrapper.from_file(xs2)
            ds_2 = GDalDatasetWrapper.from_file(swir)
            ds_ndsi = ImageApps.get_ndvi(ds_1, ds_2, vrange=(0, max_value), dtype=np.int16)
            ds_ndsi.write(output_filename, options=["COMPRESS=DEFLATE"])

        else:
            raise ValueError("Unknown synthetic band %s" % synthetic_band)
        return output_filename


class Spot4Muscate(MajaProduct):
    """
    A Spot 4 muscate product
    """

    base_resolution = (15, -15)

    @property
    def platform(self):
        return "spot4"

    @property
    def short_name(self):
        return "sp4"

    @property
    def type(self):
        return "muscate"

    @property
    def level(self):
        return "l1c"

    @property
    def nodata(self):
        return 0

    @property
    def tile(self):
        site = self.base.split("_")[-3]
        tile = re.search(self.reg_tile, site)
        if tile:
            return tile.group()[1:]
        return site

    @property
    def metadata_file(self):
        return self.find_file("*MTD_ALL.xml")[0]

    @property
    def date(self):
        str_date = self.base.split("_")[1]
        # Datetime has troubles parsing milliseconds, so it's removed:
        str_date_no_ms = str_date[:str_date.rfind("-")]
        return datetime.strptime(str_date_no_ms, "%Y%m%d-%H%M%S")

    @property
    def validity(self):
        if os.path.exists(self.metadata_file):
            return True
        return False

    def link(self, link_dir):
        symlink(self.fpath, os.path.join(link_dir, self.base))

    @property
    def mnt_site(self):
        try:
            band_bx = self.find_file(r"*_XS1*.tif")[0]
        except IOError as e:
            raise e
        return Site.from_raster(self.tile, band_bx)

    @property
    def mnt_resolutions_dict(self):
        return [{"name": "XS",
                "val": str(self.mnt_resolution[0]) + " " + str(self.mnt_resolution[1])}]

    def get_synthetic_band(self, synthetic_band, **kwargs):
        raise NotImplementedError
