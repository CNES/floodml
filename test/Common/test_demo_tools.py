#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import unittest
from Common import demo_tools
from Common.GDalDatasetWrapper import GDalDatasetWrapper
import os
import numpy as np
import matplotlib.pyplot as plt
from Common import FileSystem


class TestDemoTools(unittest.TestCase):

    def setUp(self):
        self.coordinates = (300000.0, 10.0, 0,
                            4900020.0, 0, -10.0)
        self.projection = 'PROJCS["WGS 84 / UTM zone 31N",\
                           GEOGCS["WGS 84",DATUM["WGS_1984",\
                           SPHEROID["WGS 84",6378137,298.257223563,\
                           AUTHORITY["EPSG","7030"]],\
                           AUTHORITY["EPSG","6326"]],\
                           PRIMEM["Greenwich",0,\
                           AUTHORITY["EPSG","8901"]],\
                           UNIT["degree",0.0174532925199433,\
                           AUTHORITY["EPSG","9122"]],\
                           AUTHORITY["EPSG","4326"]],\
                           PROJECTION["Transverse_Mercator"],\
                           PARAMETER["latitude_of_origin",0],\
                           PARAMETER["central_meridian",3],\
                           PARAMETER["scale_factor",0.9996],\
                           PARAMETER["false_easting",500000],\
                           PARAMETER["false_northing",0],\
                           UNIT["metre",1,AUTHORITY["EPSG","9001"]],\
                           AXIS["Easting",EAST],AXIS["Northing",NORTH],\
                           AUTHORITY["EPSG","32631"]]'
        self.height = 1000
        self.width = 1000
        self.tile = "31TFJ"
        self.orbit = "060"
        self.date = "20180101"

        # Create GSW dir
        self.gsw_dir = os.path.join(os.getcwd(), "gsw_dir")
        FileSystem.create_directory(self.gsw_dir)
        gsw = np.zeros((self.width, self.height), dtype=np.uint8)
        gsw[50:100, 50:100] = 128
        gsw[100:150, 100:150] = 250

        ds = GDalDatasetWrapper(array=gsw, projection=self.projection, geotransform=self.coordinates)
        ds.write(os.path.join(self.gsw_dir, "%s.tif" % self.tile))

        # Static display variables
        self.infile = "./test_static_display.tif"
        self.outfile = self.infile.replace("test", "out")

    def tearDown(self):
        FileSystem.remove_directory(self.gsw_dir)
        FileSystem.remove_file(self.outfile)
        FileSystem.remove_file(self.infile)

    def test_static_display(self):
        arr = np.arange(0, 1000000).reshape(1000, 1000)
        arr = np.where(arr > 100000, 1, 0)
        ds = GDalDatasetWrapper(array=arr, projection=self.projection, geotransform=self.coordinates)
        ds.write(self.infile, dtype=np.uint8)
        plt_return = demo_tools.static_display(self.infile, outfile=self.outfile, tile=self.tile,
                                               orbit=self.orbit, date=self.date, gswo_dir=self.gsw_dir)
        self.assertIsNotNone(plt_return)

    def test_draw_legend(self):
        ax = plt.gca()
        demo_tools.draw_legend(ax)
        plt.show()

    def test_draw_disclaimer(self):
        ax = plt.gca()
        demo_tools.draw_disclaimer(ax)
        plt.show()

    def test_draw_data_source(self):
        ax = plt.gca()
        demo_tools.draw_data_source(ax)
        plt.show()


if __name__ == '__main__':
    unittest.main()
