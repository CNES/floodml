#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import unittest
from prepare_mnt.mnt.CopernicusDEM import CopernicusDEM
from prepare_mnt.mnt import SiteInfo


class TestCopDEM(unittest.TestCase):

    dem_codes_europe = ["Copernicus_DSM_10_N40_00_E000_00_DEM.dt2"]
    dem_codes_saragosse = ["Copernicus_DSM_10_N41_00_W001_00_DEM.dt2",
                           "Copernicus_DSM_10_N42_00_W001_00_DEM.dt2",
                           "Copernicus_DSM_10_N41_00_W002_00_DEM.dt2",
                           "Copernicus_DSM_10_N42_00_W002_00_DEM.dt2"]
    dem_codes_australie = ["Copernicus_DSM_10_S35_00_E143_00_DEM.dt2",
                           "Copernicus_DSM_10_S34_00_E144_00_DEM.dt2",
                           "Copernicus_DSM_10_S35_00_E144_00_DEM.dt2",
                           "Copernicus_DSM_10_S34_00_E143_00_DEM.dt2"]
    dem_codes_bolivia = ["Copernicus_DSM_10_S18_00_W069_00_DEM.dt2",
                         "Copernicus_DSM_10_S18_00_W068_00_DEM.dt2",
                         "Copernicus_DSM_10_S17_00_W069_00_DEM.dt2",
                         "Copernicus_DSM_10_S17_00_W068_00_DEM.dt2"]

    def test_get_copdem_codes_europe(self):
        site = SiteInfo.Site("Europe", 4326,
                             ul=(41.0, 0.0),
                             lr=(40.0, 1.0))
        dem_codes_europe = CopernicusDEM.get_copdem_codes(site)
        self.assertEqual(dem_codes_europe, self.dem_codes_europe)

    def test_get_copdem_codes_saragosse(self):
        site = SiteInfo.Site("30TXM", 4326,
                             ul=(42.44624086219904, -1.7840029773310966),
                             lr=(41.436336379646605, -0.4888634531229668))
        dem_codes = CopernicusDEM.get_copdem_codes(site)
        self.assertEqual(sorted(dem_codes), sorted(self.dem_codes_saragosse))

    def test_get_copdem_codes_australia(self):
        site = SiteInfo.Site("55HBC", 32755,
                             ul=(199980.000, 6300040.000),
                             lr=(309780.000, 6190240.000))
        dem_codes = CopernicusDEM.get_copdem_codes(site)
        self.assertEqual(sorted(dem_codes), sorted(self.dem_codes_australie))

    def test_get_copdem_codes_bolivia(self):
        site = SiteInfo.Site("Bolivia", 4326,
                             ul=(-16.507875, -68.101762),
                             lr=(-17.507875, -67.101762))
        dem_codes = CopernicusDEM.get_copdem_codes(site)
        self.assertEqual(sorted(dem_codes), sorted(self.dem_codes_bolivia))


if __name__ == '__main__':
    unittest.main()
