#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
import re
import numpy as np
from datetime import datetime, timedelta
from Chain.Product import MajaProduct
from Common import ImageIO, FileSystem, ImageTools, XMLTools, ImageApps
from Common.FileSystem import symlink
from Common.GDalDatasetWrapper import GDalDatasetWrapper


class Sentinel2Natif(MajaProduct):
    """
    A Sentinel-2 natif product
    """

    base_resolution = (10, -10)
    coarse_resolution = (240, -240)

    @property
    def platform(self):
        return "sentinel2"

    @property
    def short_name(self):
        return "s2"

    @property
    def type(self):
        return "natif"

    @property
    def level(self):
        return self.base.split("_")[1][-3:].lower()

    @property
    def nodata(self):
        return 0

    @property
    def tile(self):
        tile = re.search(self.reg_tile, self.base)
        if tile:
            return tile.group()[1:]
        raise ValueError("Cannot determine tile ID: %s" % self.base)

    @property
    def metadata_file(self):
        return self.find_file("MTD_MSIL1C.xml")[0]

    @property
    def date(self):
        str_date = self.base.split("_")[2]
        return datetime.strptime(str_date, "%Y%m%dT%H%M%S")

    @property
    def rel_orbit(self):
        return self.base.split("_")[4]

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
        max_value = kwargs.get("max_value", 5000.)
        # Skip existing:
        if os.path.exists(output_filename):
            return output_filename
        if synthetic_band.lower() == "ndvi":
            FileSystem.create_directory(output_folder)
            b4 = self.find_file(pattern=r"*B0?4(_10m)?.jp2$", depth=5)[0]
            b8 = self.find_file(pattern=r"*B0?8(_10m)?.jp2$", depth=5)[0]
            ds_red = GDalDatasetWrapper.from_file(b4)
            ds_nir = GDalDatasetWrapper.from_file(b8)
            ds_ndvi = ImageApps.get_ndvi(ds_red, ds_nir, vrange=(-max_value, max_value), dtype=np.int16)
            ds_ndvi.write(output_filename, options=["COMPRESS=DEFLATE"])
        elif synthetic_band.lower() == "mndwi":
            FileSystem.create_directory(output_folder)
            b3 = self.find_file(pattern=r"*B0?3(_10m)?.jp2$", depth=5)[0]
            b11 = self.find_file(pattern=r"*B11(_20m)?.jp2$", depth=5)[0]
            ds_green = ImageTools.gdal_translate(b3, tr="10 10", r="cubic")
            ds_swir = GDalDatasetWrapper.from_file(b11)
            ds_ndsi = ImageApps.get_ndsi(ds_green, ds_swir, vrange=(-max_value, max_value), dtype=np.int16)
            ds_ndsi.write(output_filename, options=["COMPRESS=DEFLATE"])
        elif synthetic_band.lower() == "mca_sim":
            FileSystem.create_directory(output_folder)
            b4 = self.find_file(pattern=r"*B0?4(_10m)?.jp2$", depth=5)[0]
            b3 = self.find_file(pattern=r"*B0?3(_10m)?.jp2$", depth=5)[0]
            img_red, drv = ImageIO.tiff_to_array(b4, array_only=False)
            img_green = ImageIO.tiff_to_array(b3)
            img_mcasim = (img_red + img_green) / 2
            ImageIO.write_geotiff_existing(img_mcasim, output_filename, drv, options=["COMPRESS=DEFLATE"])
        else:
            raise ValueError("Unknown synthetic band %s" % synthetic_band)
        return output_filename


class Sentinel2Muscate(MajaProduct):
    """
    A Sentinel-2 muscate product
    """

    base_resolution = (10, -10)
    coarse_resolution = (240, -240)

    @property
    def platform(self):
        return "sentinel2"

    @property
    def short_name(self):
        return "s2"

    @property
    def type(self):
        return "muscate"

    @property
    def level(self):
        if self.base.find("_L1C_") >= 0:
            return "l1c"
        elif self.base.find("_L2A_") >= 0:
            return "l2a"
        elif self.base.find("_L3A_") >= 0:
            return "l3a"
        raise ValueError("Unknown product level for %s" % self.base)

    @property
    def nodata(self):
        return -10000

    @property
    def tile(self):
        tile = re.search(self.reg_tile, self.base)
        if tile:
            return tile.group()[1:]
        raise ValueError("Cannot determine tile ID: %s" % self.base)

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


class Sentinel2SSC(MajaProduct):
    """
    A Sentinel-2 ssc product
    """

    base_resolution = (10, -10)

    @property
    def platform(self):
        return "sentinel2"

    @property
    def short_name(self):
        return "s2"

    @property
    def type(self):
        return "natif"

    @property
    def level(self):
        if self.base.find("_L1VALD") >= 0:
            return "l1c"
        elif self.base.find("_L2VALD") >= 0:
            return "l2a"
        raise ValueError("Unknown product level for %s" % self.base)

    @property
    def nodata(self):
        return 0

    @property
    def tile(self):
        tile = re.search(self.reg_tile[1:], self.base)
        if tile:
            return tile.group()
        raise ValueError("Cannot determine tile ID: %s" % self.base)

    @property
    def metadata_file(self):
        metadata_filename = self.base.split(".")[0] + ".HDR"
        return self.find_file(path=os.path.join(self.fpath, ".."), pattern=metadata_filename)[0]

    @property
    def date(self):
        str_date = self.base.split(".")[0].split("_")[-1]
        # Add a timedelta of 12hrs in order to compensate for the missing H/M/S:
        return datetime.strptime(str_date, "%Y%m%d") + timedelta(hours=12)

    @property
    def validity(self):
        if os.path.exists(self.metadata_file):
            return True
        return False

    def link(self, link_dir):
        symlink(self.fpath, os.path.join(link_dir, self.base))
        mtd_file = self.metadata_file
        symlink(mtd_file, os.path.join(link_dir, os.path.basename(mtd_file)))

    def get_synthetic_band(self, synthetic_band, **kwargs):
        raise NotImplementedError
