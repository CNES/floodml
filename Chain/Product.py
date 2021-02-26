#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
import re
import shutil
import tempfile
from Common import ImageTools
from Common.FileSystem import find
from prepare_mnt.mnt.MNTFactory import MNTFactory
from Common import FileSystem
from Common.GDalDatasetWrapper import GDalDatasetWrapper


class MajaProduct(object):
    """
    Class to store all necessary information for a single L1- or L2- product
    """
    reg_tile = r"T\d{2}[a-zA-Z]{3}"

    base_resolution = (None, None)
    coarse_resolution = (None, None)

    def __init__(self, filepath, **kwargs):
        """
        Set the path to the root product folder
        :param filepath: The full path to the root product folder
        """
        self.fpath = os.path.realpath(filepath)
        self.base = os.path.basename(self.fpath)
        self.mnt_resolution = kwargs.get("mnt_resolution", self.base_resolution)

    def __str__(self):
        return "\n".join(["Product:   " + self.base,
                          "Acq-Date:  " + self.date.strftime("%Y-%m-%d %H:%M:%S"),
                          "Platform:  " + self.platform,
                          "Level:     " + self.level,
                          "Tile/Site: " + self.tile,
                          ""])

    def __repr__(self):
        return self.__str__()

    @classmethod
    def factory(cls, filepath, **kwargs):
        """
        Detect the underlying product
        :return:
        """
        from Chain.S2Product import Sentinel2SSC, Sentinel2Muscate, Sentinel2Natif
        from Chain.L8Product import Landsat8LC1, Landsat8LC2, Landsat8Muscate, Landsat8Natif
        from Chain.VSProduct import VenusNatif, VenusMuscate
        from Chain.SpotProduct import Spot4Muscate, Spot5Muscate
        from Chain.PleiadesProduct import PleiadesTheiaXS, PleiadesPreprojected
        from Chain.S1Product import Sentinel1Tiled

        fpath = filepath
        base = os.path.basename(fpath)
        reg_s2_nat = r"^S2[AB]_MSIL(1C|2A)_\d+T\d+_N\d+_R\d+_T\d{2}[a-zA-Z]{3}\_\d+T\d+.SAFE$"
        reg_s2_mus = r"^SENTINEL2[ABX]_[-\d]+_L(1C|2A|3A)_T\d{2}[a-zA-Z]{3}_\w_V[\d-]+$"
        reg_s2_ssc = r"^S2[AB]_OPER_SSC_L[12]VALD_\d{2}[a-zA-Z]{3}_\w+.DBL.DIR"
        reg_s2_prd = r"^S2[AB]_OPER_PRD_MSIL1C_PDMC_\w+_R\d+_V\w+.SAFE$"
        reg_l8_lc1 = r"^LC8\w+$"
        reg_l8_lc2 = r"^LC08_L\w+$"
        reg_l8_mus = r"^LANDSAT8(-OLITIRS|-OLI-TIRS|-OLITIRS-XSTHPAN)?" \
                     r"_(\d{8})-\d{6}-\d{3}_L(1C|2A)_T?\w+_[DC]_V\d*-\d*$"
        reg_l8_nat = r"^L8_\w{4}_L8C_L[12]VALD_[\d_]+.DBL.DIR$"
        reg_vs_mus = r"^VENUS(-XS)?_\d{8}-\d{6}-\d{3}_L(1C|2A|3A)_\w+_[DC]_V\d*-\d*$"
        reg_vs_nat = r"^VE_\w{4}_VSC_L[12]VALD_\w+.DBL.DIR$"
        reg_s5_mus = r"^SPOT5-HR\w+-XS_(\d{8})-\d{6}-\d{3}_L(1C|2A)_[\w-]+_[DC]_V\d*-\d*$"
        reg_s4_mus = r"^SPOT4-HR\w+-XS_(\d{8})-\d{6}-\d{3}_L(1C|2A)_[\w-]+_[DC]_V\d*-\d*$"
        reg_s1_til = r"^s1(a|b)_\d{2}[A-Z]{3}_vv_[A-Z]{3}_\d{3}_\d{8}t\w{6}.tif$"
        reg_pleiades_theia = r"FCGC\d*(-\d)?"
        reg_pleiades_reprojected = r"DS_PHR\d[A-Z]_\d{15}_\w+_[WE]\d{3}[NS]\d{2}_\d{4}_\d{4}"

        # Sentinel-2
        if re.search(reg_s2_nat, base):
            return Sentinel2Natif(fpath, **kwargs)
        if re.search(reg_s2_mus, base):
            return Sentinel2Muscate(fpath, **kwargs)
        if re.search(reg_s2_ssc, base):
            return Sentinel2SSC(fpath, **kwargs)
        if re.search(reg_s2_prd, base):
            print("WARNING: S2 PRD products currently not supported.")
        # Landsat-8
        if re.search(reg_l8_nat, base):
            return Landsat8Natif(fpath, **kwargs)
        if re.search(reg_l8_mus, base):
            return Landsat8Muscate(fpath, **kwargs)
        if re.search(reg_l8_lc1, base):
            return Landsat8LC1(fpath, **kwargs)
        if re.search(reg_l8_lc2, base):
            return Landsat8LC2(fpath, **kwargs)
        # Venus
        if re.search(reg_vs_mus, base):
            return VenusMuscate(fpath, **kwargs)
        if re.search(reg_vs_nat, base):
            return VenusNatif(fpath, **kwargs)
        # Spot
        if re.search(reg_s5_mus, base):
            return Spot5Muscate(fpath, **kwargs)
        if re.search(reg_s4_mus, base):
            return Spot4Muscate(fpath, **kwargs)
        # Pleiades
        if re.search(reg_pleiades_theia, base):
            return PleiadesTheiaXS(fpath, **kwargs)
        if re.search(reg_pleiades_reprojected, base):
            return PleiadesPreprojected(fpath, **kwargs)
        # Sentinel-1
        if re.search(reg_s1_til, base):
            return Sentinel1Tiled(fpath, **kwargs)
        pass

    @property
    def platform(self):
        raise NotImplementedError

    @property
    def short_name(self):
        raise NotImplementedError

    @property
    def tile(self):
        raise NotImplementedError

    @property
    def type(self):
        raise NotImplementedError

    @property
    def type_xml_maja(self):
        platform = self.platform
        ptype = self.type
        types = {"sentinel2": {"natif": "SENTINEL-2_", "muscate": "SENTINEL2_", "ssc": "SENTINEL-2_"},
                 "landsat8": {"lc1": "LANDSAT_8", "lc2": "LANDSAT_8", "muscate": "LANDSAT8"},
                 "venus": {"natif": "VENuS", "muscate": "VENUS"},
                 "spot5": {"muscate": "SPOT5"},
                 "spot4": {"muscate": "SPOT4"},
                 "pleiades": {"muscate": "PLEIADES"},
                 "sentinel1": {"s1tiling": "SENTINEL-2_"}
                 }
        return types[platform][ptype]

    def find_file(self, pattern, **kwargs):
        """
        Find file in the root folder
        :param pattern: The pattern to search for
        :keyword depth: The filedepth to search in
        :return: The path to the files found
        """
        depth = kwargs.get("depth", 1)
        path = kwargs.get("path", self.fpath)
        return find(path=path, pattern=pattern, depth=depth)

    @property
    def metadata_file(self):
        raise NotImplementedError

    @property
    def level(self):
        raise NotImplementedError

    @property
    def date(self):
        raise NotImplementedError

    @property
    def rel_orbit(self):
        raise NotImplementedError

    @property
    def validity(self):
        raise NotImplementedError

    def link(self, link_dir):
        raise NotImplementedError

    @property
    def mnt_site(self):
        raise NotImplementedError

    @property
    def mnt_resolutions_dict(self):
        raise NotImplementedError

    @property
    def platform_str(self):
        platform_choices = {"sentinel2": "S2_",
                            "landsat8": "L8",
                            "venus": "VE",
                            "spot5": "SPOT5",
                            "spot4": "SPOT4",
                            "pleiades": "PLEIADES",
                            "sentinel1": "S2_"}

        return platform_choices[self.platform]

    def get_mnt(self, **kwargs):
        try:
            coarse_res = kwargs.pop("coarse_res")
        except KeyError:
            coarse_res = self.coarse_resolution
        return MNTFactory(site=self.mnt_site, platform_id=self.platform_str,
                          mission_field=self.type_xml_maja, mnt_resolutions=self.mnt_resolutions_dict,
                          coarse_res=coarse_res, **kwargs).factory()

    def get_synthetic_band(self, synthetic_band, **kwargs):
        raise NotImplementedError

    def _reproject_to_epsg(self, img, outpath, epsg):
        tmpfile = tempfile.mktemp(prefix="reproject_", suffix=".tif")
        ImageTools.gdal_warp(tmpfile, img, t_srs="EPSG:%s" % epsg,
                             tr=" ".join(map(str, self.base_resolution)),
                             q=True)
        shutil.move(tmpfile, outpath)

    def reproject(self, **kwargs):
        out_dir = kwargs.get("out_dir", self.fpath)
        assert os.path.isdir(out_dir)
        out_dir = os.path.join(out_dir, self.base)
        FileSystem.create_directory(out_dir)
        patterns = kwargs.get("patterns", [r".(tif|jp2)$"])
        imgs = [self.find_file(pattern=p) for p in patterns]
        epsg = kwargs.get("epsg", None)
        # Flatten
        imgs = [i for img in imgs for i in img]
        for img in imgs:
            if not epsg:
                drv = GDalDatasetWrapper.from_file(img)
                epsg = drv.epsg
            outpath = os.path.join(out_dir, os.path.basename(img))
            self._reproject_to_epsg(img, outpath, epsg)
        return out_dir

    def rgb_values(self):
        raise NotImplementedError

    def __lt__(self, other):
        return self.date < other.date

    def __eq__(self, other):
        return self.date == other.date and \
               self.level == other.level and \
               self.metadata_file == other.metadata_file and \
               self.tile == other.tile and \
               self.platform == other.platform
