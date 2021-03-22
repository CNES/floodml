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
import Common.demo_tools as dtool
from Common import FileSystem
from deep_learning.Imagery.Dataset import Dataset
from Common.GDalDatasetWrapper import GDalDatasetWrapper
from Common.ImageIO import transform_point
from Chain.DEM import get_copdem_codes


def main_inference(args):

    input_folder = args.input
    dir_output = args.Inf_ouput
    merit_dir = args.meritdir
    copdem_dir = args.copdemdir
    sat = args.sentinel
    db_path = args.db_path
    gsw_dir = args.gsw
    products = Dataset.get_available_products(root=input_folder, platforms=["s%s" % sat])
    tmp_in = args.tmp_dir
    FileSystem.create_directory(tmp_in)  # Create if not existing

    print("Number of products found:", len(products))
    [print(f) for f in products]

    # Initialise extent file
    FileSystem.create_directory(dir_output)

    # Select DEM based on provided paths
    dem_choice = "copernicus" if copdem_dir else "merit"

    extent_out = os.path.join(dir_output, "extents.csv")
    # Write extent file header
    with open(extent_out, 'a') as the_file:
        the_file.write('Flood extent in 10x10m^2\n')

    # Main loop
    for prod in products:

        # TMP folder
        tmp_dir = tempfile.mkdtemp(dir=tmp_in)

        tile = prod.tile
        print("Tile:", tile)

        if sat == 1:  # Sentinel-1 case
            filename = prod._vv
            date = prod.date.strftime("%Y-%m-%d")
            orbit = prod.base.split("_")[4]
            ds_in = GDalDatasetWrapper.from_file(filename)
            epsg = str(ds_in.epsg)
            extent_str = ds_in.extent(dtype=str)
            # MERIT topography file for corresponding tile (S1 case)
            if dem_choice == "copernicus":
                ul_lonlat = transform_point(ds_in.ul_lr[:2], old_epsg=ds_in.epsg, new_epsg=4326)
                lr_lonlat = transform_point(ds_in.ul_lr[-2:], old_epsg=ds_in.epsg, new_epsg=4326)
                topo_names = get_copdem_codes(copdem_dir, ul_lonlat, lr_lonlat)
            else:
                topo_names = [os.path.join(merit_dir, tile + ".tif")]
            print("\t\t MERIT_file: %s" % topo_names)
            slp_norm, _ = RDF_tools.slope_creator(tmp_dir, epsg, extent_str, topo_names)
            slp_norm[slp_norm <= 0] = 0.01  # To avoid planar over detection (slp=0 and nodata values set to 0.01)
            v_stack = RDF_tools.s1_inf_stack_builder(filename, slp_norm)
        elif sat == 2:  # Sentinel-2 case
            filename = prod.find_file(pattern=r"*B0?5(_20m)?.jp2$", depth=5)[0]
            ds_in = GDalDatasetWrapper.from_file(filename)
            date = prod.date.strftime("%Y-%m-%d")
            orbit = prod.rel_orbit
            v_stack = RDF_tools.s2_inf_stack_builder(prod)
        else:
            raise ValueError("Unknown Sentinel Satellite. Has to be 1 or 2.")

        exit(1)
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

        ds_extent = GDalDatasetWrapper.from_file(nexout)
        FileSystem.remove_directory(tmp_dir)

        # Write output file for current product
        with open(nexout.replace(".tif", ".txt"), 'a') as the_file:
            the_file.write('%s\n' % os.path.abspath(nexout))
            the_file.write('%s\n' % os.path.abspath(static_display_out))
            the_file.write('%s\n' % os.path.abspath(extent_out))

        # Write extent file
        with open(extent_out, 'a') as the_file:
            the_file.write('%s,%s\n' % (date, np.count_nonzero(ds_extent.array)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation scheduler')

    parser.add_argument('-i', '--input', help='Input EMSR folder', type=str, required=True)
    parser.add_argument('-o', '--Inf_ouput', help='Output folder', type=str, required=True)
    parser.add_argument('-m', '--meritdir', help='MERIT DEM folder.'
                                                 'Either this or --copdemdir has to be set for sentinel 1.',
                        type=str, required=False)
    parser.add_argument('-c', '--copdemdir', help='Copernicus DEM folder.'
                                                  'Either this or --meritdir has to be set for sentinel 1.',
                        type=str, required=False)
    parser.add_argument('--sentinel', help='S1 or S2', type=int, required=True, choices=[1, 2])
    parser.add_argument('-db', '--db_path', help='Learning database filepath', type=str, required=True)
    parser.add_argument('-tmp', '--tmp_dir', help='Global DB output folder ', type=str, required=False)
    parser.add_argument('-g', '--gsw', help='Tiled GSW folder', type=str, required=True)

    arg = parser.parse_args()

    main_inference(arg)
