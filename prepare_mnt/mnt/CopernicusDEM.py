#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

from prepare_mnt.mnt.MNTBase import MNT
import os
from Common import ImageTools


class CopernicusDEM(MNT):
    """
    Base class to get an CopernicusDEM for a given site.
    """

    def __init__(self, site, **kwargs):
        """
        Initialise an Copernicus-DEM type.

        :param site: The :class:`prepare_mnt.mnt.SiteInfo` struct containing the basic information.
        :param kwargs: Forwarded parameters to :class:`prepare_mnt.mnt.MNTBase`
        """
        super(CopernicusDEM, self).__init__(site, **kwargs)
        self.dem_codes = self.get_copdem_codes(self.site)
        if not self.dem_version:
            self.dem_version = 3001

    def get_raw_data(self):
        """
        Get the DEM raw-data from a given directory. If not existing, raise FileNotFoundError.
        :return:
        """
        dem_paths = []
        for code in self.dem_codes:
            tile_path = os.path.join(self.raw_dem, code)
            assert os.path.exists(tile_path), "Cannot find CopernicusDEM Tile: %s" % tile_path
            dem_paths.append(tile_path)
        return dem_paths

    def prepare_mnt(self):
        """
        Prepare the copernicus-dem files.

        :return: Path to the full resolution DEM file.gsw
        :rtype: str
        """
        # Fusion of all DEM files
        mnt_max_res = self.get_raw_data()

        concat = ImageTools.gdal_buildvrt(*mnt_max_res, vrtnodata=-32767)
        # Set nodata to 0
        nodata = ImageTools.gdal_warp(concat,
                                      srcnodata=-32767,
                                      dstnodata=0,
                                      multi=True)
        # Combine to image of fixed extent
        copdem_full_res = os.path.join(self.wdir, "copdem_%sm.tif" % int(self.site.res_x))
        ImageTools.gdal_warp(nodata, dst=copdem_full_res,
                             r="cubic",
                             te=self.site.te_str,
                             t_srs=self.site.epsg_str,
                             tr=self.site.tr_str,
                             dstnodata=0,
                             srcnodata=0,
                             multi=True)
        return copdem_full_res

    @staticmethod
    def get_copdem_codes(site):
        """
        Get the list of Copernicus DEM GLO-30 files (1deg x 1deg) for a given site.

        :param site: A site-class object. See class:`prepare_mnt.mnt.SiteInfo` for more information.
        """
        import math
        ul_latlon = [math.floor(site.ul_latlon[1]), math.ceil(site.ul_latlon[0])]
        lr_latlon = [math.ceil(site.lr_latlon[1]), math.floor(site.lr_latlon[0])]
        dem_files = []
        for y in range(lr_latlon[1], ul_latlon[1]):
            for x in range(ul_latlon[0], lr_latlon[0]):
                code_lat = "N" if y >= 0 else "S"
                code_lon = "E" if x >= 0 else "W"
                demfile = "Copernicus_DSM_10_%s%02d_00_%s%03d_00_DEM.dt2" % (code_lat, abs(y), code_lon, abs(x))
                dem_files.append(demfile)
        return dem_files


if __name__ == "__main__":
    pass
