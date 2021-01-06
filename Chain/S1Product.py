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
import numpy as np
from datetime import datetime
from Chain.Product import MajaProduct
from Common import ImageIO, FileSystem
from Common.FileSystem import symlink
from prepare_mnt.mnt.SiteInfo import Site


class Sentinel1Tiled(MajaProduct):
    """
    A Sentinel-1 l1c product
    """

    base_resolution = (10, -10)
    coarse_resolution = (240, -240)

    def __init__(self, filepath, **kwargs):
        """
        S1Tiling products do not come with any other information apart from _vv_ and _vh_ .tif files.
        :param filepath: The path to the _vv_ file
        :param kwargs: Optional arguments
        """
        super(Sentinel1Tiled, self).__init__(filepath, **kwargs)
        self.fpath = os.path.dirname(filepath)
        reg_vh = r"^%s$" % self.base.replace("_vv_", "_vh_")
        self._vh = self.find_file(pattern=reg_vh)[0]
        self._vv = filepath
        self.base = os.path.splitext(self.base)[0]

    @property
    def platform(self):
        return "sentinel1"

    @property
    def short_name(self):
        return "s1"

    @property
    def type(self):
        return "s1tiling"

    @property
    def level(self):
        return "l1c"

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
        raise NotImplementedError

    @property
    def date(self):
        str_date = os.path.splitext(self.base)[0].split("_")[-1]
        if "x" in str_date[-5:]:
            return datetime.strptime(str_date.split("t")[0], "%Y%m%d")
        return datetime.strptime(str_date, "%Y%m%dt%H%M%S")

    @property
    def validity(self):
        if os.path.exists(self._vh) and os.path.exists(self._vv):
            return True
        return False

    def link(self, link_dir):
        symlink(self._vv, os.path.join(link_dir, os.path.basename(self._vv)))
        symlink(self._vh, os.path.join(link_dir, os.path.basename(self._vh)))

    @property
    def mnt_site(self):
        return Site.from_raster(self.tile, self._vv)

    @property
    def mnt_resolutions_dict(self):
        return [{"name": "R1",
                "val": str(self.mnt_resolution[0]) + " " + str(self.mnt_resolution[1])}]

    def get_synthetic_band(self, synthetic_band, **kwargs):
        wdir = kwargs.get("wdir", self.fpath)
        output_folder = os.path.join(wdir, self.base)
        output_bname = "%s.tif" % self.base.replace("_vv_", "_%s_" % synthetic_band.lower())
        output_filename = kwargs.get("output_filename", os.path.join(output_folder, output_bname))
        # Skip existing:
        if os.path.exists(output_filename):
            return output_filename
        if synthetic_band.lower() == "vvovervh":
            FileSystem.create_directory(output_folder)
            vv, drv = ImageIO.tiff_to_array(self._vv, array_only=False)
            vh = ImageIO.tiff_to_array(self._vh)
            out = np.where(np.abs(vh) > 0, np.abs(vv) / np.abs(vh), 0)
            ImageIO.write_geotiff_existing(out, output_filename, drv, options=["COMPRESS=DEFLATE"])
        elif synthetic_band.lower() == "vhovervv":
            FileSystem.create_directory(output_folder)
            vv, drv = ImageIO.tiff_to_array(self._vv, array_only=False)
            vh = ImageIO.tiff_to_array(self._vh)
            out = np.where(np.abs(vv) > 0, np.abs(vh) / np.abs(vv), 0)
            ImageIO.write_geotiff_existing(out, output_filename, drv, options=["COMPRESS=DEFLATE"])
        elif synthetic_band.lower() == "vvplusvh":
            FileSystem.create_directory(output_folder)
            vv, drv = ImageIO.tiff_to_array(self._vv, array_only=False)
            vh = ImageIO.tiff_to_array(self._vh)
            out = np.array(vv + vh, dtype=np.float32)
            ImageIO.write_geotiff_existing(out, output_filename, drv, options=["COMPRESS=DEFLATE"])
        elif synthetic_band.lower() == "vhplusvv":
            return self.get_synthetic_band("vvplusvh", **kwargs)
        else:
            raise ValueError("Unknown synthetic band %s" % synthetic_band)
        return output_filename

    @property
    def rgb_values(self):
        """
        Get bands and scaling for each of them in order to create an RGB image

        :return: List of RGB bands as well as their scaling, to be used in :func:`Common.ImageTools.gdal_translate`
        """
        return ["VV", "VH", "VHplusVV"], {"scale_1": "0 .2 0 255",
                                          "scale_2": "0 .05 0 255",
                                          "scale_3": "0 .2 0 255"}

    def __eq__(self, other):
        return self.date == other.date and \
               self.level == other.level and \
               self.tile == other.tile and \
               self.platform == other.platform
