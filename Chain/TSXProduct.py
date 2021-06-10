#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import os
from datetime import datetime
from Chain.Product import MajaProduct
from Common.FileSystem import symlink
from prepare_mnt.mnt.SiteInfo import Site
from Common import XMLTools


class TerraSarXRadiometricallyEnhanced(MajaProduct):
    """
    A TSX RE (radiometrically enhanced) product
    """

    def __init__(self, filepath, **kwargs):
        """
        S1Tiling products do not come with any other information apart from _vv_ and _vh_ .tif files.
        :param filepath: The path to the _vv_ file
        :param kwargs: Optional arguments
        """
        super(TerraSarXRadiometricallyEnhanced, self).__init__(filepath, **kwargs)
        self.base = os.path.splitext(self.base)[0]
        self._xml_file = os.path.join(filepath, "%s.xml" % self.base)
        self.images_in_imgdata = XMLTools.get_xpath(self._xml_file, "./productComponents/imageData/file/location/filename")
        self.files = [im.text for im in self.images_in_imgdata]

        #IMAGE_HH_SRA_stripNear_010.tif
        self.polarisations = [im.text.split('_')[1] for im in self.images_in_imgdata]
        #print('_files', self._files)
        #print('_polarisations', self._polarisations)

        res = XMLTools.get_res(self._xml_file, "./productInfo/imageDataInfo/imageRaster/rowSpacing")
        #print(res)
        self.base_resolution = (res, res)
        self.mnt_resolution = self.base_resolution
        self.orbit = int(XMLTools.get_xpath(self._xml_file, "./productInfo/missionInfo/relOrbit")[0].text)

    @property
    def platform(self):
        return "TerraSAR/TanDEM-X"

    @property
    def short_name(self):
        return "tsx"

    @property
    def type(self):
        return "EEC"

    @property
    def level(self):
        return "l1c"

    @property
    def nodata(self):
        # TODO Is it always 0? If not, get that info from the file that is also used for base_resolution
        return 0 

    @property
    def tile(self):
        #raise ValueError("Cannot determine tile ID on a TerraSarX product: %s" % self.base)
        return "No tile for TSX file"

    @property
    def metadata_file(self):
        return self._xml_file

    @property
    def date(self):
        # The start date of the acq should be used systematically in the project.
        str_date = os.path.splitext(self.base)[0].split("_")[-2]
        if "x" in str_date[-5:]:
            return datetime.strptime(str_date.split("t")[0], "%Y%m%d")
        return datetime.strptime(str_date, "%Y%m%dt%H%M%S")

    @property
    def rel_orbit(self):
        raise NotImplementedError

    @property
    def validity(self):
        for f in self._polarisations:
            if not os.path.exists(f):
                return False
        return True

    def link(self, link_dir):
        for f in self._polarisations:
            symlink(f, os.path.join(link_dir, os.path.basename(f)))

    @property
    def mnt_site(self):
        return Site.from_raster(self.base, self._polarisations[0])

    @property
    def mnt_resolutions_dict(self):
        return [{"name": "R1",
                "val": str(self.mnt_resolution[0]) + " " + str(self.mnt_resolution[1])}]

    def get_synthetic_band(self, synthetic_band, **kwargs):
        raise NotImplementedError

    @property
    def rgb_values(self):
        """
        Get bands and scaling for each of them in order to create an RGB image

        :return: List of RGB bands as well as their scaling, to be used in :func:`Common.ImageTools.gdal_translate`
        """
        raise NotImplementedError

    def __eq__(self, other):
        return self.date == other.date and \
               self.level == other.level and \
               self.tile == other.tile and \
               self.platform == other.platform and \
               self.base_resolution == other.base_resolution
