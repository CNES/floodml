#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import unittest
from Chain import DummyFiles
import os


class TestDummyFiles(unittest.TestCase):
    root = os.path.join(os.getcwd(), "DummyFiles")
    platforms = ["sentinel2", "landsat8", "venus"]

    @classmethod
    def setUpClass(cls):
        from Common import FileSystem
        FileSystem.create_directory(cls.root)

    @classmethod
    def tearDownClass(cls):
        from Common import FileSystem
        FileSystem.remove_directory(cls.root)

    def test_s2_l1_generation(self):
        import shutil
        gen = DummyFiles.L1Generator(self.root, platform="sentinel2")
        prod = gen.generate()
        self.assertTrue(os.path.exists(gen.mtd))
        self.assertTrue(os.path.exists(gen.prod))
        self.assertTrue("MTD_MSIL1C.xml" in os.path.basename(gen.mtd))
        self.assertTrue("MSIL1C" in gen.prod)
        self.assertEqual(prod, prod)
        shutil.rmtree(gen.prod)
        self.assertFalse(os.path.exists(gen.prod))

    def test_spot4_l1_generation(self):
        import shutil
        gen = DummyFiles.L1Generator(self.root, platform="spot4")
        prod = gen.generate()
        self.assertTrue(os.path.exists(gen.mtd))
        self.assertTrue(os.path.exists(gen.prod))
        self.assertTrue("MTD_ALL.xml" in os.path.basename(gen.mtd))
        self.assertTrue("SPOT4" in gen.prod)
        self.assertEqual(prod, prod)
        shutil.rmtree(gen.prod)
        self.assertFalse(os.path.exists(gen.prod))

    def test_spot5_l1_generation(self):
        import shutil
        gen = DummyFiles.L1Generator(self.root, platform="spot5")
        prod = gen.generate()
        self.assertTrue(os.path.exists(gen.mtd))
        self.assertTrue(os.path.exists(gen.prod))
        self.assertTrue("MTD_ALL.xml" in os.path.basename(gen.mtd))
        self.assertTrue("SPOT5" in gen.prod)
        self.assertEqual(prod, prod)
        shutil.rmtree(gen.prod)
        self.assertFalse(os.path.exists(gen.prod))

    def test_l2_generation(self):
        import shutil
        gen = DummyFiles.L2Generator(self.root)
        prod = gen.generate()
        self.assertTrue(os.path.exists(gen.mtd))
        self.assertTrue(os.path.exists(gen.prod))
        self.assertTrue("_MTD_ALL.xml" in os.path.basename(gen.mtd))
        self.assertTrue(os.path.dirname(gen.mtd), gen.prod)
        self.assertEqual(prod, prod)
        shutil.rmtree(gen.prod)
        self.assertFalse(os.path.exists(gen.prod))


if __name__ == '__main__':
    unittest.main()
