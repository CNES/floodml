#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""



import re
import os
import numpy as np
from datetime import datetime, timedelta
from Chain.Product import MajaProduct
from Common.FileSystem import symlink
from Common import ImageIO, FileSystem, ImageTools, XMLTools, ImageApps
from Common import FileSystem, XMLTools
from Common.GDalDatasetWrapper import GDalDatasetWrapper


class Landsat8Natif(MajaProduct):
    """
    A Landsat-8 natif product
    """

    base_resolution = (30, -30)
    coarse_resolution = (240, -240)

    @property
    def platform(self):
        return "landsat8"

    @property
    def short_name(self):
        return "l8"

    @property
    def type(self):
        return "natif"

    @property
    def level(self):
        return "l1c"

    @property
    def nodata(self):
        return 0

    @property
    def tile(self):
        site = self.base.split("_")[4]
        tile = re.search(self.reg_tile, site)
        if tile:
            return tile.group()[1:]
        return site

    @property
    def metadata_file(self):
        metadata_filename = self.base.split(".")[0] + ".HDR"
        return self.find_file(path=os.path.join(self.fpath, ".."), pattern=metadata_filename)[0]

    @property
    def validity(self):
        if os.path.exists(self.metadata_file):
            return True
        return False

    def link(self, link_dir):
        symlink(self.fpath, os.path.join(link_dir, self.base))
        mtd_file = self.metadata_file
        symlink(mtd_file, os.path.join(link_dir, os.path.basename(mtd_file)))

    @property
    def date(self):
        str_date = self.base.split(".")[0].split("_")[-1]
        # Add a timedelta of 12hrs in order to compensate for the missing H/M/S:
        return datetime.strptime(str_date, "%Y%m%d") + timedelta(hours=12)

    def get_synthetic_band(self, synthetic_band, **kwargs):
        raise NotImplementedError


class Landsat8Muscate(MajaProduct):
    """
    A Landsat-8 muscate product
    """

    base_resolution = (30, -30)

    @property
    def platform(self):
        return "landsat8"

    @property
    def short_name(self):
        return "l8"

    @property
    def type(self):
        return "muscate"

    @property
    def level(self):
        if self.base.find("_L1C_") >= 0:
            return "l1c"
        elif self.base.find("_L2A_") >= 0:
            return "l2a"
        raise ValueError("Unknown product level for %s" % self.base)

    @property
    def nodata(self):
        return 0

    @property
    def tile(self):
        site_basic = self.base.split("_")[3]
        tile = re.search(self.reg_tile, self.base)
        if tile:
            return tile.group()[1:]
        return site_basic

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
        if self.level == "l1c" and os.path.exists(self.metadata_file):
            return True
        if self.level == "l2a":
            try:
                jpi = FileSystem.find_single("*JPI_ALL.xml", self.fpath)
            except ValueError:
                return False
            validity_xpath = "./Processing_Flags_And_Modes_List/Processing_Flags_And_Modes/Value"
            processing_flags = XMLTools.get_xpath(jpi, validity_xpath)
            validity_flags = [flg.text for flg in processing_flags]
            if "L2VALD" in validity_flags:
                return True
        return False

    def link(self, link_dir):
        symlink(self.fpath, os.path.join(link_dir, self.base))

    def get_synthetic_band(self, synthetic_band, **kwargs):
        raise NotImplementedError


class Landsat8LC1(MajaProduct):
    """
    A Landsat-8 ssc product
    """

    base_resolution = (30, -30)
    coarse_resolution = (240, -240)

    @property
    def platform(self):
        return "landsat8"

    @property
    def short_name(self):
        return "l8"

    @property
    def type(self):
        return "natif"

    @property
    def level(self):
        return "l1c"

    @property
    def nodata(self):
        return 0

    @property
    def tile(self):
        return self.base[3:9]

    @property
    def metadata_file(self):
        return self.find_file("*_MTL.txt")[0]

    @property
    def date(self):
        year_doy = self.base[9:15]
        # Add a timedelta of 12hrs in order to compensate for the missing H/M/S:
        return datetime.strptime(year_doy, "%Y%j") + timedelta(hours=12)

    @property
    def validity(self):
        if os.path.exists(self.metadata_file):
            return True
        return False

    def link(self, link_dir):
        symlink(self.fpath, os.path.join(link_dir, self.base))

    def get_synthetic_band(self, synthetic_band, **kwargs):
        raise NotImplementedError


class Landsat8LC2(MajaProduct):
    """
    A Landsat-8 ssc product
    """

    base_resolution = (30, -30)

    @property
    def platform(self):
        return "landsat8"

    @property
    def short_name(self):
        return "l8"

    @property
    def type(self):
        return "natif"

    @property
    def level(self):
        return "l1c"

    @property
    def nodata(self):
        return 0

    @property
    def tile(self):
        return self.base.split("_")[2]

    @property
    def metadata_file(self):
        return self.find_file("*_MTL.txt")[0]

    @property
    def date(self):
        str_date = self.base.split("_")[3]
        # Add a timedelta of 12hrs in order to compensate for the missing H/M/S:
        return datetime.strptime(str_date, "%Y%m%d") + timedelta(hours=12)

    @property
    def validity(self):
        if os.path.exists(self.metadata_file):
            return True
        return False

    def link(self, link_dir):
        symlink(self.fpath, os.path.join(link_dir, self.base))

    def get_synthetic_band(self, synthetic_band, **kwargs):
        output_folder = kwargs.get("wdir", os.path.join(self.fpath, "index"))
        output_bname = "_".join([self.base.split(".")[0], synthetic_band.upper() + ".tif"])
        output_filename = kwargs.get("output_filename", os.path.join(output_folder, output_bname))
        print(output_filename)
        max_value = kwargs.get("max_value", 5000)
        # Skip existing:
        if os.path.exists(output_filename):
            return output_filename
        if synthetic_band.lower() == "ndvi":
            FileSystem.create_directory(output_folder)
            b4 = self.find_file(pattern=r"*B4.TIF", depth=5)[0]
            b5 = self.find_file(pattern=r"*B5.TIF", depth=5)[0]
            ds_red  = GDalDatasetWrapper.from_file(b4)
            ds_nir  = GDalDatasetWrapper.from_file(b5)
            ds_red.array = np.multiply(ds_red.array, 2.75e-5)-0.2 # rescaling
            ds_nir.array = np.multiply(ds_nir.array, 2.75e-5)-0.2 # rescaling
            ds_ndvi = ImageApps.get_ndvi(ds_red, ds_nir, vrange=(-max_value, max_value), dtype=np.int16)
            ds_ndvi.write(output_filename, options=["COMPRESS=DEFLATE"])
        elif synthetic_band.lower() == "mndwi":
            FileSystem.create_directory(output_folder)
            b3 = self.find_file(pattern=r"*B3.TIF", depth=5)[0]
            b6 = self.find_file(pattern=r"*B6.TIF", depth=5)[0]
            ds_green = GDalDatasetWrapper.from_file(b3)
            ds_swir = GDalDatasetWrapper.from_file(b6)
            ds_green.array = np.multiply(ds_green.array, 2.75e-5)-0.2 # rescaling
            ds_swir.array = np.multiply(ds_swir.array, 2.75e-5)-0.2 # rescaling
            ds_ndsi = ImageApps.get_ndsi(ds_green, ds_swir, vrange=(-max_value, max_value), dtype=np.int16)
            ds_ndsi.write(output_filename, options=["COMPRESS=DEFLATE"])
        else:
            raise ValueError("Unknown synthetic band %s" % synthetic_band)
        return output_filename
