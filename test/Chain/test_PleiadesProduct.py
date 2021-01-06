#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""



import unittest
from Common import FileSystem
from Chain.Product import MajaProduct
from Chain.PleiadesProduct import PleiadesTheiaXS
import os


class TestPleiadesProduct(unittest.TestCase):

    prod_pleiades_nat = ["FCGC600108580"]
    prods_other = ["LANDSAT8-OLITIRS-XSTHPAN_20170501-103532-111_L1C_T31TCH_C_V1-0",
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
                   "VENUS-XS_20180201-000000-000_L3A_KHUMBU_C_V1-0"]

    @classmethod
    def setUpClass(cls):
        """
        Simulate the basic folder + metadata_file structure
        :return:
        """
        import os
        for root in cls.prod_pleiades_nat:
            from Common import XMLTools
            from xml.etree import ElementTree
            os.makedirs(root)
            ms_folder = os.path.join(root, "IMG_PHR1A_MS_002")
            os.makedirs(ms_folder)
            metadata = os.path.join(ms_folder, "DIM_PHR1B_MS_201311271105384_SEN_758149101-004.XML")
            top = ElementTree.Element('Dimap_Document')
            comment = ElementTree.Comment('Generated for unittesting')
            top.append(comment)
            dataset = ElementTree.SubElement(top, 'Dataset_Identification')
            dataset_name = ElementTree.SubElement(dataset, "DATASET_NAME", version="1.0")
            dataset_name.text = "DS_PHR1B_201311271105034_FR1_PX_E000N43_0205_03459"
            XMLTools.write_xml(top, metadata)

    @classmethod
    def tearDownClass(cls):
        import shutil
        for root in cls.prod_pleiades_nat:
            shutil.rmtree(root)

    def test_reg_pleiades_theia(self):
        tiles = ["E000N43_0205"]
        levels = ["l1c"]
        dates = ["20131127T110503"]
        validity = [True]
        for prod, tile, date, level, valid in zip(self.prod_pleiades_nat, tiles, dates, levels, validity):
            p = MajaProduct.factory(prod)
            self.assertIsInstance(p, PleiadesTheiaXS)
            self.assertEqual(p.level, level)
            self.assertEqual(p.platform, "pleiades")
            self.assertEqual(p.type, "raw")
            self.assertEqual(p.tile, tile)
            self.assertEqual(p.nodata, 0)
            self.assertEqual(p.date.strftime("%Y%m%dT%H%M%S"), date)
            self.assertTrue(os.path.basename(p.metadata_file).endswith(".XML"))
            self.assertTrue(os.path.exists(p.metadata_file))
            self.assertEqual(p.validity, valid)
            link_dir = "linkdir"
            FileSystem.create_directory(link_dir)
            p.link(link_dir)
            self.assertTrue(os.path.islink(os.path.join(link_dir, p.base)))
            self.assertEqual(p.mnt_resolutions_dict, [{'name': 'XS', 'val': '10 -10'}])
            self.assertEqual(p, p)
            FileSystem.remove_directory(link_dir)
        # Other prods:
        for prod in self.prods_other:
            p = MajaProduct.factory(prod)
            self.assertNotIsInstance(p, PleiadesTheiaXS)


if __name__ == '__main__':
    unittest.main()
