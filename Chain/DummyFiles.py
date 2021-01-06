#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os


def random_tile(platform="sentinel2"):
    """
    Create a random tile string for a given platform.
    For 'sentinel2': T12ABC, for 'landsat8': 192820, for 'venus': ABCDE
    :param platform: Return a tile corresponding to the given platform
    :return: String compliant with the given platform
    """
    import string
    import random

    letters = string.ascii_uppercase
    tile = "T" + str(random.randint(0, 99)).zfill(2) + ''.join(random.choice(letters) for _ in range(3))
    if platform == "sentinel2":
        return tile
    elif platform == "landsat8":
        number = str(random.randint(0, 999999)).zfill(6)
        return random.choice([number, tile])
    elif platform == "venus":
        return ''.join(random.choice(letters) for _ in range(5))
    elif "SPOT" in platform:
        tile_id_1 = str(random.randint(0, 999)).zfill(3)
        tile_id_2 = str(random.randint(0, 999)).zfill(3)
        tile_id_3 = str(random.randint(0, 9)).zfill(1)
        return "-".join([tile_id_1, tile_id_2, tile_id_3])

    return tile


def random_platform(product_level=None):
    import random
    if product_level == "L1C":
        return random.choice(["sentinel2", "spot4", "spot5"])
    return random.choice(["sentinel2", "venus", "landsat8"])


def random_date():
    import random
    from datetime import datetime, timedelta
    date = datetime(2015, 1, 1) + timedelta(days=random.randint(0, 10000),
                                            hours=random.randint(10, 12),
                                            minutes=random.randint(0, 60),
                                            seconds=random.randint(0, 60))
    return date


class DummyGenerator(object):
    """
    Base class for all dummy generators
    """
    def __init__(self, root, date=None, tile=None, platform=random_platform()):
        self.root = root
        if not date:
            self.date = random_date()
        else:
            self.date = date
        if platform.lower() == "sentinel2":
            self.platform = "sentinel2"
        elif platform.lower() == "landsat8":
            self.platform = "landsat8"
        elif platform.lower() == "venus":
            self.platform = "venus"
        elif platform.lower() == "spot4":
            self.platform = "spot4"
        elif platform.lower() == "spot5":
            self.platform = "spot5"
        else:
            raise ValueError("Unknown platform found: %s" % platform)
        if not tile:
            self.tile = random_tile(platform=platform)
        else:
            self.tile = tile

    def generate(self, **kwargs):
        raise NotImplementedError


class DummyEarthExplorer(DummyGenerator):
    """
    Base class for all EarthExplorer-like files
    """

    mission_choices = {"tm": {"sentinel2": "SENTINEL-2"},
                       "muscate": {"sentinel2": "SENTINEL2", "landsat8": "LANDSAT8", "venus": "VENUS",
                                   "spot4": "SPOT4", "spot5": "SPOT5"},
                       "natif": {"sentinel2": "SENTINEL-2", "landsat8": "LANDSAT_8", "venus": "VENuS"}
                       }

    mission_short = {"sentinel2": "S2_",
                     "venus": "VE",
                     "landsat8": "L8",
                     "spot4": "SPOT4",
                     "spot5": "SPOT5"}

    def __init__(self, root, date=None, tile=None, platform=random_platform()):
        super(DummyEarthExplorer, self).__init__(root, date, tile, platform)
        self.hdr = None
        self.dbl = None

    def get_mission(self):
        """
        Return a random Mission name
        :return:
        """
        import random
        mtype = random.choice(["muscate", "natif"])
        return self.mission_choices[mtype][self.platform]

    def create_dummy_hdr(self, file_path, mission=None):
        """
        Create a dummy HDR file with only the "Mission" field in it.
        :param file_path: The full path to the file that should be written
        :param mission: The content of the 'Mission' field to be set.
        """
        import random
        from xml.etree import ElementTree
        platform_hdr = {"sentinel2": random.choice(["SENTINEL2_", "SENTINEL-2_"]),
                        "venus": random.choice(["VENUS", "VENuS"]),
                        "landsat8": random.choice(["LANDSAT8", "LANDSAT_8"]),
                        "spot4": "SPOT4",
                        "spot5": "SPOT5"}
        mission = mission if mission else platform_hdr[self.platform]
        root = ElementTree.Element("Earth_Explorer_Header")
        sub = ElementTree.SubElement(root, "Fixed_Header")
        ElementTree.SubElement(sub, "Mission").text = mission
        ElementTree.SubElement(root, "Variable_Header")
        tree = ElementTree.ElementTree(root)
        tree.write(file_path)

    def generate(self, **kwargs):
        # Pass this on
        raise NotImplementedError


class ProductGenerator(DummyGenerator):
    platform_options = {"L1C": {"sentinel2": ["S2A", "S2B"],
                                "spot4": ["SPOT4-HRG2-XS"],
                                "spot5": ["SPOT5-HRG2-XS"]},
                        "L2A": {"sentinel2": ["SENTINEL2B", "SENTINEL2A"],
                                "landsat8": ["LANDSAT8-OLITIRS"],
                                "venus": ["VENUS-XS"]}}

    def __init__(self, root, date=None, tile=None, platform=random_platform()):
        super(ProductGenerator, self).__init__(root, date, tile, platform)
        self.prod = None
        self.mtd = None

    def generate(self, **kwargs):
        raise NotImplementedError


class L1Generator(ProductGenerator):
    """
    Class to create a dummy L1C product
    """
    def __init__(self, root, date=None, tile=None, platform=random_platform(product_level="L1C")):
        super(L1Generator, self).__init__(root, date, tile, platform)

    def generate(self, **kwargs):
        import random
        from Chain import Product
        from Common import TestFunctions, FileSystem
        platform_specifier = random.choice(self.platform_options["L1C"][self.platform])
        orbit = kwargs.get("orbit", random.randint(0, 999))
        version_orbit = kwargs.get("version", random.randint(0, 9))
        if "sentinel2" in self.platform:
            date_str = self.date.strftime("%Y%m%dT%H%M%S")
            product_name = "_".join([platform_specifier,
                                     "MSIL1C",
                                     date_str,
                                     "N" + str(orbit).zfill(4),
                                     "R" + str(version_orbit).zfill(3),
                                     self.tile,
                                     date_str + ".SAFE"])
            product_path = os.path.join(self.root, product_name)
            metadata_path = os.path.join(product_path, "MTD_MSIL1C.xml")
        else:
            date_str = self.date.strftime("%Y%m%d-%H%M%S")
            product_name = "_".join([platform_specifier,
                                     date_str + "-000",
                                     "L2A",
                                     self.tile,
                                     random.choice("DC"),
                                     "V" + str(version_orbit) + "-" + str(version_orbit)])
            product_path = os.path.join(self.root, product_name)
            metadata_path = os.path.join(product_path, product_name + "_MTD_ALL.xml")

        self.prod = product_path
        self.mtd = metadata_path
        FileSystem.create_directory(product_path)
        TestFunctions.touch(metadata_path)
        return Product.MajaProduct.factory(self.prod)


class L2Generator(ProductGenerator):
    """
    Class to create a dummy L2A product
    """
    def __init__(self, root, date=random_date(), tile=None, platform=random_platform(product_level="L2A")):
        super(L2Generator, self).__init__(root, date, tile, platform)

    def generate(self, **kwargs):
        import random
        from Chain import Product
        from Common import TestFunctions, FileSystem
        platform_specifier = random.choice(self.platform_options["L2A"][self.platform])
        ms = kwargs.get("ms", random.randint(0, 999))
        version = kwargs.get("version", random.randint(0, 9))
        date_str = self.date.strftime("%Y%m%d-%H%M%S-") + str(ms).zfill(3)
        product_name = "_".join([platform_specifier,
                                 date_str,
                                 "L2A",
                                 self.tile,
                                 random.choice("DC"),
                                 "V" + str(version) + "-" + str(version)])
        product_path = os.path.join(self.root, product_name)
        self.prod = product_path
        metadata_path = os.path.join(product_path, product_name + "_MTD_ALL.xml")
        self.mtd = metadata_path
        FileSystem.create_directory(product_path)
        TestFunctions.touch(metadata_path)
        return Product.MajaProduct.factory(self.prod)
