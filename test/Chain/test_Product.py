#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import unittest
from Common import TestFunctions
from Chain.Product import MajaProduct
import os
from os import path as p


class TestProduct(unittest.TestCase):

    root = "S2A_MSIL1C_20170412T110621_N0204_R137_T29RPQ_20170412T111708.SAFE"

    def setUp(self):
        os.makedirs(self.root)
        TestFunctions.touch(os.path.join(self.root, "MTD_MSIL1C.xml"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def test_get_file_depth1(self):
        product = MajaProduct.factory(self.root)
        expected = os.path.join(os.path.abspath(self.root), "MTD_MSIL1C.xml")
        dirnames_e = p.normpath(expected).split(os.sep)
        calculated = product.find_file(path=self.root, pattern="^MTD_MSIL1C.xml$", depth=1)
        self.assertEqual(len(calculated), 1)
        dirnames_c = p.normpath(calculated[0]).split(os.sep)
        for exp, calc in zip(dirnames_c[-1:], dirnames_e[-1:]):
            self.assertEqual(exp[:-1], calc[:-1])
        self.assertEqual(expected, calculated[0])

    def test_eq(self):
        product = MajaProduct.factory(self.root)
        self.assertEqual(product, product)


if __name__ == '__main__':
    unittest.main()
