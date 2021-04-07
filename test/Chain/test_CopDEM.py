#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import unittest
from Chain.DEM import get_copdem_codes
from Common.TestFunctions import touch
from Common.FileSystem import remove_file


class TestCopDEM(unittest.TestCase):

    dem_codes_europe = ["./Copernicus_DSM_10_N40_00_E000_00_DEM.dt2"]
    dem_codes_saragosse = ["./Copernicus_DSM_10_N41_00_W001_00_DEM.dt2",
                           "./Copernicus_DSM_10_N42_00_W001_00_DEM.dt2",
                           "./Copernicus_DSM_10_N41_00_W002_00_DEM.dt2",
                           "./Copernicus_DSM_10_N42_00_W002_00_DEM.dt2"]
    dem_codes_australie = ["./Copernicus_DSM_10_S35_00_E143_00_DEM.dt2",
                           "./Copernicus_DSM_10_S34_00_E144_00_DEM.dt2",
                           "./Copernicus_DSM_10_S35_00_E144_00_DEM.dt2",
                           "./Copernicus_DSM_10_S34_00_E143_00_DEM.dt2"]
    dem_codes_bolivia = ["./Copernicus_DSM_10_S18_00_W069_00_DEM.dt2",
                         "./Copernicus_DSM_10_S18_00_W068_00_DEM.dt2",
                         "./Copernicus_DSM_10_S17_00_W069_00_DEM.dt2",
                         "./Copernicus_DSM_10_S17_00_W068_00_DEM.dt2"]

    def setUp(self) -> None:
        for dem in self.dem_codes_europe + self.dem_codes_saragosse + self.dem_codes_australie + self.dem_codes_bolivia:
            touch(dem)

    def tearDown(self) -> None:
        for dem in self.dem_codes_europe + self.dem_codes_saragosse + self.dem_codes_australie + self.dem_codes_bolivia:
            remove_file(dem)

    def test_get_copdem_codes_europe(self):
        ul = (41.0, 0.0)
        lr = (40.0, 1.0)
        dem_codes_europe = get_copdem_codes(".", ul, lr)
        self.assertEqual(dem_codes_europe, self.dem_codes_europe)

    def test_get_copdem_codes_saragosse(self):
        ul = (42.44624086219904, -1.7840029773310966)
        lr = (41.436336379646605, -0.4888634531229668)
        dem_codes = get_copdem_codes(".", ul, lr)
        self.assertEqual(sorted(dem_codes), sorted(self.dem_codes_saragosse))

    def test_get_copdem_codes_australia(self):
        ul = (-33.397079450234656, 143.77448663451733)
        lr = (-34.41180718585893, 144.93030119816075)
        dem_codes = get_copdem_codes(".", ul, lr)
        self.assertEqual(sorted(dem_codes), sorted(self.dem_codes_australie))

    def test_get_copdem_codes_bolivia(self):
        ul = (-16.507875, -68.101762)
        lr = (-17.507875, -67.101762)
        dem_codes = get_copdem_codes(".", ul, lr)
        self.assertEqual(sorted(dem_codes), sorted(self.dem_codes_bolivia))


if __name__ == '__main__':
    unittest.main()
