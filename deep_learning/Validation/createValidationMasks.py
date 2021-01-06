#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import os.path as p
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')) #Import relative modules

import numpy as np

class MaskCreator():
    level1CProductFolder = r"S2A_OPER_MSI_L1C_TL_SGS__\w+_A\w+"
    def __init__(self, h5file, prods, output_dir):
        assert args.input[-3:] == ".h5"
        self.h5file = h5file
        if not p.isdir(prods):
            raise OSError("Cannot find %s" % prods)
        self.prods = prods
        self.output_dir = output_dir
        return
    
    def getTileListFromDir(self, prods):
        """
        Get list of dates and tiles from product directory
        """
        import re
        tiles = []
        for path, dirs, files in os.walk(prods, followlinks=True):
            for file in files:
                if(re.search(self.level1CProductFolder + "_B01.jp2", file)):
                    filename = file.split("_")
                    date = filename[7].split("T")[0]
                    tile = filename[9]
                    tiles.append((date, tile, os.path.join(path, file)))
        return tiles
    
    @staticmethod
    def getH5File(filename):
        """
        Read a HDF5 File and its keys
        """
        import h5py
        if not p.isfile(filename):
            raise OSError("Not a file: %s" % filename)
        f = h5py.File(filename)
        keys = list(f.keys())
        return f,keys

    @staticmethod
    def getXY(lat, lon, drv):
        """
        Transform lat, lon into pixel_x, pixel_y using the geotransform
        """
        # GetGeoTransform gives the (x, y) origin of the top left pixel,
        # the x and y resolution of the pixels, and the rotation of the
        # raster. If the raster is rotated (i.e. the rotation values are
        # anything other than 0) this method will not work.
        from osgeo import osr
        src = osr.SpatialReference()
        src.SetWellKnownGeogCS("WGS84")
        prj = drv.GetProjection()
        dst = osr.SpatialReference(prj)
        tr = osr.CoordinateTransformation(src, dst)
        TL_x, x_res, rotx, TL_y, roty, y_res = drv.GetGeoTransform()
        coordx, coordy, _ = tr.TransformPoint(float(lon), float(lat))
        if rotx != 0 or roty != 0:
            raise ValueError("Cannot get (X,Y) from rotated image: %s" % drv.GetGeoTransform())
        # Divide the difference between the x value of the point and origin,
        # and divide this by the resolution to get the raster index
        px = (coordx - TL_x) / x_res
        py = (coordy - TL_y) / y_res
        return int(px), int(py)

    def createMassiveDataFrame(self, h5file):
        """
        Create the data frame containing data from the hdf5 file.
        For each pixel (row), the following infos are added as columns:
            - Lat
            - Lon
            - Tile
            - Date
            - Class
        """
        keys = ["latitude", "longitude", "granule_id", "dates", "classes"]
        f, _ = self.getH5File(h5file)
        filtered = np.array([f[key][:] for key in keys])
        return filtered
    
    def filterDataFrame(self, df, tiles):
        """
        Filter the dataframe by the existing tiles and dates
        """
        df_reduced = []
        df_new = np.swapaxes(df, 0, 1)
        tiles_filtered = [(date, tile) for date, tile, _ in tiles]
        for lat, lon, tile, date, val in df_new:
            date_formatted = date.decode("utf-8").split(" ")[0].replace(".", "")
            tile = "T" + tile.decode("utf-8")
            fpath = list(set([path for d, t, path in tiles if d == date_formatted and tile == t]))
            if(not fpath):
                continue
            elif(len(fpath) > 1):
                raise ValueError("More than one filename found for tile %s and date %s: 'n %s" % (tile, date_formatted, fpath))
            df_reduced.append([lat.decode("utf-8"), lon.decode("utf-8"), val.decode("utf-8"), fpath[0]])

        return np.array(df_reduced)

    def createMasks(self, df, output, translate=None):
        """
        Create Masks using the provided lat/lon values
        :param df: The dataframe with columns lat,lon, value and filepath
        :param output: The path to write the new raster to
        :param translate: Dict to translate the original values (0:10:60 to another system).
                          Set this to None in order to keep the original values.
        :return: None
        """
        from Common import ImageIO
        filenames = list(set([fn for fn in df[:, 3]]))
        for fn in filenames:
            img, drv = ImageIO.tiffToArray(fn, arrayOnly=False)
            mask = np.zeros_like(img, dtype=np.uint8)
            for lat, lon, val, fpath in df:
                if fpath != fn:
                    continue
                x, y = self.getXY(lat, lon, drv)

                mask[y, x] = translate[int(val)] if translate else int(val)
            output_path = p.join(output, p.splitext(p.basename(fn))[0][:-4] + "_gt.tif")
            print("Writing mask %s" % output_path)
            ImageIO.writeGTiffExisting(mask, drv, output_path)
        return

    def run(self):
        """
        Run the whole thing:
            - Get list of available tiles
            - Read HDF5 file
            - Filter file by available tiles
            - Create masks
        """
        #Translate values to same system as used by O.Hagolle's dataset
        NODATA,CLEAR,WATER,SHADOW,CIRRUS,CLOUD,SNOW = 0,10,20,30,40,50,60
        value_translator = {NODATA : 1,
                            CLEAR : 5,
                            WATER : 6,
                            SHADOW : 4,
                            CIRRUS : 3,
                            CLOUD : 2,
                            SNOW : 7}

        from Common import FileSystem
        FileSystem.createDirectory(self.output_dir)

        tiles = self.getTileListFromDir(self.prods)
        df_all = self.createMassiveDataFrame(self.h5file)
        df = self.filterDataFrame(df_all, tiles)
        self.createMasks(df, self.output_dir, translate=value_translator)
        return


if __name__ == "__main__":
    assert sys.version_info[:2] >= (2,7)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",help="The Input filename", required=True, type=str)
    parser.add_argument("-p", "--prods",help="The input product dir", required=True, type=str)
    parser.add_argument("-o", "--output",help="The output dir", required=True, type=str)
    args = parser.parse_args()
    m = MaskCreator(args.input, args.prods, args.output)
    m.run()
