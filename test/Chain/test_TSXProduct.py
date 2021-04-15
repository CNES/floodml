#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import unittest
from xml.etree import ElementTree
from Common import XMLTools
from Common import FileSystem
from Chain.Product import MajaProduct
from Chain.TSXProduct import TerraSarXRadiometricallyEnhanced
from Common.GDalDatasetWrapper import GDalDatasetWrapper
import numpy as np
import os


class TestS1Product(unittest.TestCase):

    prods_tsx = ["TSX1_SAR__EEC_RE___SM_D_SRA_20171202T155851_20171202T155856",
                 "TSX1_SAR__EEC_RE___SM_S_SRA_20151213T064323_20151213T064331"]
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
                   "VE_OPER_VSC_L1VALD_UNH_20180329.DBL.DIR",
                   "s1a_55HBD_vv_DES_016_20160925txxxxxx.tif",
                   "s1b_12ABC_vv_ASC_053_20160928t090345.tif"]

    proj = """PROJCS["WGS 84 / UTM zone 29N",
    GEOGCS["WGS 84",DATUM["WGS_1984",
    SPHEROID["WGS 84",6378137,298.257223563,
    AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,
    AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],
    PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",-9],PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],PARAMETER["false_northing",0],
    UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],
    AXIS["Northing",NORTH],AUTHORITY["EPSG","32629"]]
"""

    geotransforms = [(386498.125, 3.75, 0, 4552501.875, 0, -3.75),
                     (519498.500, 2, 0, 5875001.500, 0, 2)]

    @classmethod
    def setUpClass(cls):
        """
        Simulate the basic folder + metadata_file structure
        :return:
        """
        for root, transform in zip(cls.prods_tsx, cls.geotransforms):
            FileSystem.create_directory(root)
            imgdata = "IMAGEDATA"
            vv = "IMAGE_VV_SRA_stripNear_010.tif"
            hh = "IMAGE_HH_SRA_stripNear_010.tif"
            metadata = os.path.join(root, "%s.xml" % root)
            top = ElementTree.Element('level1Product')
            comment = ElementTree.Comment('Generated for unittesting')
            top.append(comment)
            components = ElementTree.SubElement(top, 'productComponents')

            image_data = ElementTree.SubElement(components, "imageData", layerIndex="1")
            pol_layer = ElementTree.SubElement(image_data, "polLayer")
            pol_layer.text = "HH"
            file_grp = ElementTree.SubElement(image_data, "file")
            location = ElementTree.SubElement(file_grp, "location")
            host = ElementTree.SubElement(location, "host")
            host.text = "."
            path = ElementTree.SubElement(location, "path")
            path.text = imgdata
            filename = ElementTree.SubElement(location, "filename")
            filename.text = hh
            size = ElementTree.SubElement(location, "size")
            size.text = "1234"

            image_data2 = ElementTree.SubElement(components, "imageData", layerIndex="2")
            pol_layer2 = ElementTree.SubElement(image_data2, "polLayer")
            pol_layer2.text = "VV"
            file_grp2 = ElementTree.SubElement(image_data2, "file")
            location2 = ElementTree.SubElement(file_grp2, "location")
            host2 = ElementTree.SubElement(location2, "host")
            host2.text = "."
            path2 = ElementTree.SubElement(location2, "path")
            path2.text = imgdata
            filename2 = ElementTree.SubElement(location2, "filename")
            filename2.text = vv
            size2 = ElementTree.SubElement(location2, "size")
            size2.text = "4321"
            XMLTools.write_xml(top, metadata)
            FileSystem.create_directory(os.path.join(root, imgdata))
            ds = GDalDatasetWrapper(projection=cls.proj, geotransform=transform,
                                    array=np.arange(0, 9).reshape((3, 3)),
                                    nodata_value=0)
            ds.write(os.path.join(root, imgdata, hh))
            ds.write(os.path.join(root, imgdata, vv))

    @classmethod
    def tearDownClass(cls):
        for root in cls.prods_tsx:
            FileSystem.remove_directory(root)

    def test_reg_s1_tiled(self):
        levels = ["l1-notclassified", "l1-notclassified"]
        dates = ["20171202T155851", "20151213T064323"]
        validity = [True, True]
        resolutions = [(3.5, 3.5), (2, 2)]
        for prod, transform, date, level, res, valid in zip(self.prods_tsx, self.geotransforms,
                                                            dates, levels, resolutions, validity):
            p = MajaProduct.factory(prod)
            self.assertIsInstance(p, TerraSarXRadiometricallyEnhanced)
            self.assertTrue(os.path.isfile(p.metadata_file))
            self.assertEqual(p.level, level)
            self.assertEqual(p.platform, "terrasarx")
            self.assertEqual(p.type, "tsxre")
            self.assertEqual(p.nodata, 0)
            self.assertEqual(p.date.strftime("%Y%m%dT%H%M%S"), date)
            self.assertEqual(p.validity, valid)
            self.assertEqual(p.base_resolution, res)
            self.assertEqual(p._polarisations, ["IMAGE_HH_SRA_stripNear_010.tif", "IMAGE_VV_SRA_stripNear_010.tif"])
            link_dir = "linkdir"
            FileSystem.create_directory(link_dir)
            self.assertEqual(p, p)
            FileSystem.remove_directory(link_dir)

            # Test product content:
            ds_hh = GDalDatasetWrapper.from_file(p._polarisations[0])
            ds_vv = GDalDatasetWrapper.from_file(p._polarisations[1])
            self.assertEqual(ds_vv.epsg, 32629)
            self.assertEqual(ds_hh.epsg, 32629)
            self.assertEqual(ds_vv.geotransform, transform)
            self.assertEqual(ds_hh.geotransform, transform)
            np.testing.assert_almost_equal(ds_vv.array, np.arange(0, 9).reshape((3, 3)))
            np.testing.assert_almost_equal(ds_hh.array, np.arange(0, 9).reshape((3, 3)))

        # Other prods:
        for prod in self.prods_other:
            p = MajaProduct.factory(prod)
            self.assertNotIsInstance(p, TerraSarXRadiometricallyEnhanced)
