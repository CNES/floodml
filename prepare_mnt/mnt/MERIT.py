#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Author:         Peter KETTIG <peter.kettig@cnes.fr>
"""

from prepare_mnt.mnt.MNTBase import MNT
import os


class MERIT(MNT):
    """
    Base class to get the necessary mnt for a given site.
    """

    def __init__(self, site, **kwargs):
        super(MERIT, self).__init__(site, **kwargs)
        if not self.dem_version:
            self.dem_version = 2001

    def get_raw_data(self):
        """
        Get the DEM raw-data from a given directory. If not existing, an attempt will be made to download
        it automatically.
        :return:
        """
        from Common import FileSystem
        reg_tile = "T?%s.tif" % self.site.nom
        merit_path = FileSystem.find_single(pattern=reg_tile, path=self.raw_dem)
        return merit_path

    def prepare_mnt(self):
        """
        Prepare the merit files.
        :return:
        """
        from Common import ImageTools
        mnt_max_res = self.get_raw_data()
        mnt_cropped = os.path.join(self.wdir, "mnt_cropped_%s.tif" % self.site.nom)
        ImageTools.gdal_warp(mnt_max_res, dst=mnt_cropped,
                             r="cubic",
                             te=self.site.te_str,
                             t_srs=self.site.epsg_str,
                             tr=self.site.tr_str,
                             srcnodata=-9999,
                             dstnodata=0)

        return mnt_cropped


if __name__ == "__main__":
    pass
