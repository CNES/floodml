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

    dem_codes_tls = ["./Copernicus_DSM_10_S41_00_E000_00_DEM.dt2"]
    dem_codes_saragosse = ["./Copernicus_DSM_10_N41_00_W001_00_DEM.dt2",
                           "./Copernicus_DSM_10_N42_00_W001_00_DEM.dt2",
                           "./Copernicus_DSM_10_N41_00_W002_00_DEM.dt2",
                           "./Copernicus_DSM_10_N42_00_W002_00_DEM.dt2"]

    def setUp(self) -> None:
        for dem in self.dem_codes_tls + self.dem_codes_saragosse:
            touch(dem)

    def tearDown(self) -> None:
        for dem in self.dem_codes_tls + self.dem_codes_saragosse:
            remove_file(dem)

    def test_get_copdem_codes_tls(self):
        ul = (-40.0, 0.0)
        lr = (-41.0, 1.0)
        dem_codes_tls = get_copdem_codes(".", ul, lr)
        self.assertEqual(dem_codes_tls, self.dem_codes_tls)

    def test_get_copdem_codes_saragosse(self):
        ul = (42.44624086219904, -1.7840029773310966)
        lr = (41.436336379646605, -0.4888634531229668)
        dem_codes = get_copdem_codes(".", ul, lr)
        self.assertEqual(sorted(dem_codes), sorted(self.dem_codes_saragosse))


if __name__ == '__main__':
    unittest.main()
