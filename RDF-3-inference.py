#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import os
import joblib
import numpy as np
import argparse
from random_forest.common import RDF_tools
import tempfile
import progressbar
from datetime import datetime
import Common.demo_tools as dtool
from Common import FileSystem
from deep_learning.Imagery.Dataset import Dataset
from Common.GDalDatasetWrapper import GDalDatasetWrapper


def main_inference(args):

    input_folder = args.input
    dir_output = args.Inf_ouput
    merit_dir = args.meritdir
    sat = args.sentinel
    db_path = args.db_path
    gsw_dir = args.gsw
    products = Dataset.get_available_products(root=input_folder)
    print("Number of files found:", len(products))
    [print(f) for f in products]

    for prod in products:

        # TMP folder
        tmp_dir = tempfile.mkdtemp(dir=os.getcwd())

        tile = prod.tile
        print("Tile:", tile)
        # TODO Extend to S2
        filename = prod._vv
        date = prod.date.strftime("%Y-%m-%d")
        orbit = prod.base.split("_")[4]
        ds_in = GDalDatasetWrapper.from_file(filename)
        epsg = str(ds_in.epsg)
        print("EPSG:", epsg)

        extent_str = ds_in.extent(dtype=str)

        if sat == 1:  # Sentinel-1 case
            # MERIT topography file for corresponding tile (S1 case)
            topo_name = os.path.join(merit_dir, tile + ".tif")
            print("\t\t MERIT_file: %s" % topo_name)
            slp_norm, _ = RDF_tools.slope_creator(tmp_dir, epsg, extent_str, topo_name)
            slp_norm[slp_norm <= 0] = 0.01  # To avoid planar over detection (slp=0 and nodata values set to 0.01)
            v_stack = RDF_tools.s1_inf_stack_builder(filename, slp_norm)
        elif sat == 2:  # Sentinel-2 case
            v_stack = RDF_tools.s2_inf_stack_builder(filename)
        else:
            raise ValueError("Unknown Sentinel Satellite. Has to be 1 or 2.")

        n_divisions = 20
        windows = np.array_split(v_stack, n_divisions, axis=0)
        predictions = []

        # RANDOM FOREST
        print('\tLoading RDF model...')
        rdf = joblib.load(db_path)  # /path to be changed
        for idx in progressbar.progressbar(range(len(windows))):
            # Remove NaN & predict
            current = windows[idx]
            current[np.isnan(current)] = 0
            rdf_pred = rdf.predict(current)
            predictions.append(rdf_pred)
        # Output image
        ds_filename = GDalDatasetWrapper.from_file(filename)

        dim = ds_filename.array.shape[:2]

        vec_out = np.concatenate(predictions).reshape(dim[1], dim[0])
        exout = np.array(vec_out, dtype=np.bool)

        # Apply nodata
        exout[ds_in.array == 0] = 1

        # Invert values 0 -> 1 and 1 -> 0
        exout = np.invert(exout)

        # Export
        FileSystem.create_directory(dir_output)

        if sat == 1:
            nexout = os.path.join(dir_output,
                                  'Inference_RDF_%s_T%s_%s_%s.tif' % (sat, tile, str(date), orbit))
        else:
            nexout = os.path.join(dir_output,
                                  'Inference_RDF_%s_T%s_%s.tif' % (sat, tile, str(date)))

        ds_out = GDalDatasetWrapper(array=exout,
                                    projection=ds_filename.projection,
                                    geotransform=ds_filename.geotransform)
        ds_out.write(nexout)

        static_display_out = nexout.replace("Inference", "RapidMapping").replace(".tif", ".png")
        dtool.static_display(nexout, tile, date, orbit, static_display_out, gswo_dir=gsw_dir)

        FileSystem.remove_directory(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation scheduler')

    parser.add_argument('-i', '--input', help='Input EMSR folder', type=str, required=True)
    parser.add_argument('-o', '--Inf_ouput', help='Output folder', type=str, required=True)
    parser.add_argument('-m', '--meritdir', help='MERIT files folder (for slope)', required=True)
    parser.add_argument('--sentinel', help='S1 or S2', type=int, required=True, choices=[1, 2])
    parser.add_argument('-db', '--db_path', help='Learning database filepath', type=str, required=True)
    parser.add_argument('-tmp', '--tmp_dir', help='Global DB output folder ', type=str, required=False)
    parser.add_argument('-g', '--gsw', help='Tiled GSW folder', type=str, required=True)

    arg = parser.parse_args()

    main_inference(arg)
