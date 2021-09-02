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
from Chain.DEM import get_gswo_codes


def main_inference(args):

    input_folder = args.input
    dir_output = args.Inf_ouput
    merit_dir = args.meritdir
    copdem_dir = args.copdemdir
    sat = args.satellite
    db_path = args.db_path
    gsw_dir = args.gsw
    post = args.post
    rad = args.rad

    products = list(sorted(Dataset.get_available_products(root=input_folder, platforms=[sat])))

    tmp_in = args.tmp_dir
    FileSystem.create_directory(tmp_in)  # Create if not existing

    print("Number of products found:", len(products))

    if not products:
        print("No products found. Exiting...")
        return

    # Initialise extent file
    FileSystem.create_directory(dir_output)


    # Select DEM based on provided paths
    dem_choice = "copernicus" if copdem_dir else "merit"

    extent_out = os.path.join(dir_output, "extents_%s_%s_0.csv" % (products[0].date.strftime("%Y%m%d"),
                                                                   products[-1].date.strftime("%Y%m%d")))
    for i in range(100):
        extent_mod = extent_out.replace("_0.csv", "_%03d.csv" % i)
        if not os.path.exists(extent_mod):
            break
    # Write extent file header
    with open(extent_mod, 'a') as the_file:
        the_file.write('Flood extent in 10x10m^2\n')


    # Main loop
    for prod in products:

        # TMP folder
        print(prod)
        tmp_dir = tempfile.mkdtemp(dir=tmp_in)

        ## For each product determine the files to be processed
        filenames = []
        if sat == "s1":
            filenames.append(prod._vv)
            polar = prod.polarisations
        elif sat == "s2":
            filenames.append(prod.find_file(pattern=r"*B0?5(_20m)?.jp2$", depth=5)[0])
            polar = ""
        elif sat == "tsx":
            for f in range(len(prod.files)):
                filenames.append(os.path.join(input_folder, 'IMAGEDATA', prod.files[f]))
                                                            
        for filename in filenames:
        
            if sat == "s1":  # Sentinel-1 case
                orbit = prod.base.split("_")[4]
                ds_in = GDalDatasetWrapper.from_file(filename)
                epsg = str(ds_in.epsg)
                date = prod.date.strftime("%Y%m%dT%H%M%S")
                extent_str = ds_in.extent(dtype=str)

                #Topography file for corresponding tile (S1 case)
                if dem_choice == "copernicus":
                    ul_latlon = transform_point(ds_in.ul_lr[:2], old_epsg=ds_in.epsg, new_epsg=4326)
                    lr_latlon = transform_point(ds_in.ul_lr[-2:], old_epsg=ds_in.epsg, new_epsg=4326)
                    topo_names = get_copdem_codes(copdem_dir, ul_latlon, lr_latlon)
                else:
                    topo_names = [os.path.join(merit_dir, prod.tile + ".tif")]
                print("\tDEM file: %s" % topo_names)
                slp_norm, _ = RDF_tools.slope_creator(tmp_dir, epsg, extent_str, topo_names, res=[10, 10])
                slp_norm[slp_norm <= 0] = 0.01  # To avoid planar over detection (slp=0 and nodata values set to 0.01)
                v_stack = RDF_tools.s1_inf_stack_builder(filename, slp_norm)
                background = None

            elif sat == "s2":  # Sentinel-2 case
                ds_in = GDalDatasetWrapper.from_file(filename)
                date = prod.date.strftime("%Y%m%dT%H%M%S")
                orbit = prod.rel_orbit.replace("R", "")
                v_stack = RDF_tools.s2_inf_stack_builder(prod, tmp_dir)
                print(np.size(v_stack))
                background = prod.find_file(pattern=r"*TCI(_20m)?.jp2$", depth=5)[0]

            elif sat == "tsx":  # TSX
                polar = filename.split('/')[-1].split('_')[1]
                ds_in = GDalDatasetWrapper.from_file(filename)
                epsg = str(ds_in.epsg)
                orbit = prod.orbit
                date = prod.date.strftime("%Y%m%dT%H%M%S")
                extent_str = ds_in.extent(dtype=str)

                # Topography files for corresponding tile 
                if dem_choice == "copernicus":
                    ul_latlon = transform_point(ds_in.ul_lr[:2], old_epsg=ds_in.epsg, new_epsg=4326)
                    lr_latlon = transform_point(ds_in.ul_lr[-2:], old_epsg=ds_in.epsg, new_epsg=4326)
                    topo_names = get_copdem_codes(copdem_dir, ul_latlon, lr_latlon)
                else:
                    topo_names = [os.path.join(merit_dir, tile + ".tif")] ##Issue to be solved
                print("\tDEM file: %s" % topo_names)
                slp_norm, _ = RDF_tools.slope_creator(tmp_dir, epsg, extent_str, topo_names, prod.mnt_resolution)
                slp_norm[slp_norm <= 0] = 0.01  # To avoid planar over detection (slp=0 and nodata values set to 0.01)
                v_stack = RDF_tools.tsx_inf_stack_builder(filename, slp_norm, C=2500) #Calibration coefficient set manually here
                background = None
            else:
                raise ValueError("Unknown  Satellite. Has to be s1, s2 or tsx.")

            
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

            vec_out = np.concatenate(predictions).reshape(dim[0], dim[1])
            exout = np.array(vec_out, dtype=np.uint8)

            # Apply nodata
            exout[ds_in.array == 0] = 255

            if sat == "s2":
                scl_path = prod.find_file(pattern=r"\w+SCL_20m.jp2$", depth=5)[0]
                scl_img = GDalDatasetWrapper.from_file(scl_path).array
                # Add cloud, cloud shadow and snow layers
                cld_shadow = scl_img == 3
                cld = (scl_img >= 8) & (scl_img <= 10)
                snw = scl_img == 11
                exout[cld_shadow] = 2
                exout[cld] = 3
                exout[snw] = 4

            # Export
            FileSystem.create_directory(os.path.join(dir_output, prod.base))


            # Export raw inference
            if sat == "s1": 
                nexout = os.path.join(dir_output, prod.base, 'Inf_%s_%s_%s_%s_%s_%s.tif' % (sat.upper(), prod.level.upper(), 
                                                            prod.tile, date, orbit, 'RAW'))
            elif sat == "s2":
                nexout = os.path.join(dir_output, prod.base, 'Inf_%s_%s_%s_%s.tif' % (sat.upper(), orbit, date, 'RAW')) 
            elif sat == "tsx":
                basesplit = prod.base.replace('___','_').replace('__','_').split('_')
                nexout = os.path.join(dir_output, prod.base, 'Inf_%s_%s_%s_%s_%s_%s_%s.tif' % (sat.upper(), prod.type.upper(), 
                                                            polar, basesplit[7], basesplit[8], orbit, 'RAW'))

            ds_out = GDalDatasetWrapper(array=np.array(exout),
                                        projection=ds_filename.projection,
                                        geotransform=ds_filename.geotransform)
            ds_out.write(nexout, options=["COMPRESS=LZW"], nodata=255)
 
            # If post-treatment:
            if post==1:
                exout = RDF_tools.postreatment(exout, radius=rad)
                postsuf = 'POST_MAJr%s' % str(rad).zfill(2)
            
                if sat == "s1": 
                    nexoutpost = os.path.join(dir_output, prod.base, 'Inf_%s_%s_%s_%s_%s_%s.tif' % (sat.upper(), prod.level.upper(), 
                                                                prod.tile, date, orbit, postsuf))
                elif sat == "s2":
                    nexoutpost = os.path.join(dir_output, prod.base, 'Inf_%s_%s_%s_%s.tif' % (sat.upper(), orbit, date, postsuf)) 
                elif sat == "tsx":
                    basesplit = prod.base.replace('___','_').replace('__','_').split('_')
                    nexoutpost = os.path.join(dir_output, prod.base, 'Inf_%s_%s_%s_%s_%s_%s_%s.tif' % (sat.upper(), prod.type.upper(), 
                                                                polar, basesplit[7], basesplit[8], orbit, postsuf)) 

                ds_out = GDalDatasetWrapper(array=np.array(exout),
                                            projection=ds_filename.projection,
                                            geotransform=ds_filename.geotransform)
                ds_out.write(nexoutpost, options=["COMPRESS=LZW"], nodata=255)

            #### Rapid mapping map creation

            ## GSW overlay selection
            ul_latlon = transform_point(ds_in.ul_lr[:2], old_epsg=ds_in.epsg, new_epsg=4326)
            lr_latlon = transform_point(ds_in.ul_lr[-2:], old_epsg=ds_in.epsg, new_epsg=4326)
            gsw_files = get_gswo_codes(gsw_dir, ul_latlon, lr_latlon)
            print("\tGSWO file: %s" % gsw_files)

            # Raw inference
            static_display_out = nexout.replace(".tif", "_RapidMapping.png")
            dtool.static_display(nexout, tmp_dir, gsw_files,  prod.date.strftime("%Y-%m-%d %H:%M:%S"), polar, 
                                                            static_display_out, orbit, sat=sat, background=background) 
                                              
            # With post-processing
            if post==1:
                static_display_out = nexoutpost.replace(".tif", "_RapidMapping.png")
                dtool.static_display(nexoutpost, tmp_dir, gsw_files,  prod.date.strftime("%Y-%m-%d %H:%M:%S"), polar, 
                                                                static_display_out, orbit, sat=sat, background=background, post=post, rad=rad) 

            # Write output file for current product
            # with open(nexout.replace(".tif", ".txt"), 'a') as the_file:
            #     the_file.write('%s\n' % os.path.abspath(nexout))
            #     the_file.write('%s\n' % os.path.abspath(static_display_out))
            #    the_file.write('%s\n' % os.path.abspath(extent_mod))

            ### Write extent file
            #with open(extent_mod, 'a') as the_file:
            #    the_file.write('%s,%s,%s\n' % (prod.tile, date, np.count_nonzero(ds_extent.array)))
    FileSystem.remove_directory(tmp_dir)       
    FileSystem.remove_directory(tmp_in)
    print("FloodML finished ;)")


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
    parser.add_argument('--satellite', help='s1, s2 or tsx', type=str, required=True, choices=["s1", "s2", "tsx"])
    parser.add_argument('-db', '--db_path', help='Learning database filepath', type=str, required=True)
    parser.add_argument('-tmp', '--tmp_dir', help='Global DB output folder ', type=str, required=False, default="tmp")
    parser.add_argument('-g', '--gsw', help='Tiled GSW folder', type=str, required=True)
    parser.add_argument('-p', '--post', help='Post-treatment to be applied to the output', type=int, required=False)
    parser.add_argument('-r', '--rad', help='MAj filter radius', type=int, required=False)

    arg = parser.parse_args()

    main_inference(arg)
