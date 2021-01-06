#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""



import unittest
from Common import TestFunctions, FileSystem
from Chain.Product import MajaProduct
from Chain.S1Product import Sentinel1Tiled
import os


class TestS1Product(unittest.TestCase):

    prods_s1tiled = ["s1a_55HBD_vv_DES_016_20160925txxxxxx.tif",
                     "s1b_12ABC_vv_ASC_053_20160928t090345.tif"]
    prods_other = ["SPOT4-HRG2-XS_20120622-083239-738_L1C_115-354-0_D_V1-0",
                   "SPOT4-HRG2-XS_20120622-083231-220_L1C_115-353-0_D_V1-0",
                   "SPOT5-HRG2-XS_20120626-103330-813_L1C_048-326-0_D_V1-0",
                   "SPOT5-HRG2-XS_20120617-082907-795_L1C_123-357-0_D_V1-0",
                   "LANDSAT8-OLITIRS-XSTHPAN_20170501-103532-111_L1C_T31TCH_C_V1-0",
                   "LANDSAT8_20170501-103532-111_L2A_T31TCH_C_V1-0",
                   "LC80390222013076EDC00",
                   "LC08_L1TP_199029_20170527_20170615_01_T1",
                   "L8_TEST_L8C_L2VALD_198030_20130626.DBL.DIR",
                   "S2A_MSIL1C_20170412T110621_N0204_R137_T29RPQ_20170412T111708.SAFE",
                   "S2B_MSIL1C_20180316T103021_N0206_R108_T32TMR_20180316T123927.SAFE",
                   "S2A_OPER_PRD_MSIL1C_PDMC_20161109T171237_R135_V20160924T074932_20160924T081448.SAFE",
                   "S2A_OPER_SSC_L2VALD_36JTT____20160914.DBL.DIR",
                   "S2B_OPER_SSC_L1VALD_21MXT____20180925.DBL.DIR",
                   "SENTINEL2B_20171008-105012-463_L1C_T31TCH_C_V1-0",
                   "SENTINEL2A_20161206-105012-463_L2A_T31TCH_C_V1-0",
                   "SENTINEL2X_20190415-000000-000_L3A_T31UFR_C_V1-1",
                   "VENUS-XS_20180201-051359-000_L1C_KHUMBU_C_V1-0",
                   "VENUS_20180201-051359-000_L2A_KHUMBU_C_V1-0",
                   "VENUS-XS_20180201-051359-000_L3A_KHUMBU_C_V1-0",
                   "VE_VM01_VSC_L2VALD_ISRAW906_20180317.DBL.DIR",
                   "VE_OPER_VSC_L1VALD_UNH_20180329.DBL.DIR"]

    @classmethod
    def setUpClass(cls):
        """
        Simulate the basic folder + metadata_file structure
        :return:
        """
        for root in cls.prods_s1tiled:
            TestFunctions.touch(root)
            vh = root.replace("_vv_", "_vh_")
            TestFunctions.touch(vh)

    @classmethod
    def tearDownClass(cls):
        import os
        for root in cls.prods_s1tiled:
            os.remove(root)
            vh = root.replace("_vv_", "_vh_")
            os.remove(vh)

    def test_reg_s1_tiled(self):
        tiles = ["55HBD", "12ABC"]
        levels = ["l1c", "l1c"]
        dates = ["20160925T000000", "20160928T090345"]
        validity = [True, True]
        for prod, tile, date, level, valid in zip(self.prods_s1tiled, tiles, dates, levels, validity):
            p = MajaProduct.factory(prod)
            self.assertIsInstance(p, Sentinel1Tiled)
            self.assertEqual(p.level, level)
            self.assertEqual(p.platform, "sentinel1")
            self.assertEqual(p.type, "s1tiling")
            self.assertEqual(p.tile, tile)
            self.assertEqual(p.nodata, 0)
            self.assertEqual(p.date.strftime("%Y%m%dT%H%M%S"), date)
            self.assertEqual(p.validity, valid)
            link_dir = "linkdir"
            FileSystem.create_directory(link_dir)
            p.link(link_dir)
            self.assertTrue(os.path.islink(os.path.join(link_dir, "%s.tif" % p.base)))
            self.assertEqual(p.mnt_resolutions_dict, [{"name": "R1", "val": "10 -10"}])
            self.assertEqual(p, p)
            FileSystem.remove_directory(link_dir)

        # Other prods:
        for prod in self.prods_other:
            p = MajaProduct.factory(prod)
            self.assertNotIsInstance(p, Sentinel1Tiled)
