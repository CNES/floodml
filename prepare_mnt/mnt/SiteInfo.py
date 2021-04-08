#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Author:         Peter KETTIG <peter.kettig@cnes.fr>
"""

from Common import ImageIO
from Common.GDalDatasetWrapper import GDalDatasetWrapper
from osgeo import osr, gdal


class Site:
    """
    Stores all necessary information in order to create an MNT
    """
    def __init__(self, nom, epsg, ul, lr, **kwargs):
        self.nom = nom
        self.epsg = epsg
        self.px = kwargs.get("px", None)
        self.py = kwargs.get("py", None)
        self.res_x = kwargs.get("res_x", None)
        self.res_y = kwargs.get("res_y", None)
        self.ul = ul
        self.lr = lr
        self.ul_latlon, self.lr_latlon = self.latlon_minmax

    @property
    def latlon_minmax(self):
        """
        Get lat and lon min and max values
        :return: latmin, latmax, lonmin, lonmax of the current sites
        """
        ul_latlon = ImageIO.transform_point(self.ul, self.epsg, new_epsg=4326)
        lr_latlon = ImageIO.transform_point(self.lr, self.epsg, new_epsg=4326)
        return ul_latlon, lr_latlon

    @property
    def projwin(self):
        return " ".join([str(self.ul[1]), str(self.ul[0]), str(self.lr[1]), str(self.lr[0])])

    @property
    def te_str(self):
        y_val = list(sorted([self.lr[1], self.ul[1]]))
        x_val = list(sorted([self.lr[0], self.ul[0]]))
        return " ".join([str(x_val[0]), str(y_val[0]), str(x_val[1]), str(y_val[1])])

    @property
    def epsg_str(self):
        return "EPSG:" + str(self.epsg)

    @property
    def tr_str(self):
        return str(self.res_x) + " " + str(self.res_y)

    def to_driver(self, path, n_bands=1):
        # create the raster file
        dst_ds = gdal.GetDriverByName('GTiff').Create(path, self.py, self.px, n_bands, gdal.GDT_Byte)
        geotransform = (self.ul[1], self.res_x, 0, self.ul[0], 0, self.res_y)
        dst_ds.SetGeoTransform(geotransform)  # specify coords
        srs = osr.SpatialReference()  # establish encoding
        srs.ImportFromEPSG(self.epsg)  # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        return dst_ds

    @staticmethod
    def from_raster(name, raster, **kwargs):
        """
        Create site from a raster on disk
        :param name: The name of the site
        :param raster: The gdal raster
        Optional arguments:
        - shape_index_y: Select the band index for the Y-size
        - shape_index_x: Select the band index for the X-size
        :return: A site class given the infos from the raster.
        """
        driver = GDalDatasetWrapper.from_file(raster)
        shape_index_y = kwargs.get("shape_index_y", 0)
        shape_index_x = kwargs.get("shape_index_x", 1)
        ny, nx = driver.array.shape[shape_index_y], driver.array.shape[shape_index_x]
        epsg = driver.epsg
        ulx, uly, lrx, lry = driver.ul_lr
        ul = (ulx, uly)
        lr = (lrx, lry)
        xmin, xres, skx, ymax, sky, yres = driver.geotransform
        return Site(name, epsg, ul=ul, lr=lr,  px=nx, py=ny, res_x=xres, res_y=yres)
