#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
import joblib
import numpy as np
from datetime import datetime
import argparse
import tempfile
import Common.Rapid_mapper as rapid_mapper
from Common import RDF_tools
from Common import FileSystem
from Common.Imagery.Dataset import Dataset
from Common.GDalDatasetWrapper import GDalDatasetWrapper
from Common.ImageIO import transform_point
from Common.ImageTools import gdal_warp, gdal_buildvrt
from Common.Mosaicist import get_copdem_codes
from Common.Mosaicist import get_gswo_codes
from Common.Mosaicist import get_esawc_codes

def main_inference(args):

    input_folder = args.input
    dir_output = args.Inf_ouput
    merit_dir = args.meritdir
    copdem_dir = args.copdemdir
    sat = args.satellite
    db_path = args.db_path
    gsw_dir = args.gsw
    rad = args.rad
    wc_dir = args.wc_dir
    tmp_in = args.tmp_dir

    products = list(sorted(Dataset.get_available_products(root=input_folder, 
                                                          platforms=[sat])))

    print('Temporary directory: {}'.format(tmp_in))
    print("Number of products found:", len(products))

    if not products:
        print("No products found. Exiting...")
        return

    # Initialise extent file
    FileSystem.create_directory(dir_output)

    # Select DEM based on provided paths
    dem_choice = "copernicus" if copdem_dir else "merit"

    # Main loop
    for prod in products:

        # TMP folder
        print(prod)
        FileSystem.create_directory(tmp_in)  # Create if not existing
        tmp_dir = tempfile.mkdtemp(dir=tmp_in)
        print('Temporary directory created:', tmp_dir)


        ## For each product determine the files to be processed
        filenames = []
        if sat == "s1":
            filenames.append(prod._vv)
            polar = prod.polarisations
        elif sat == "s2":
            filenames.append(prod.find_file(pattern=r"*B0?4(_10m)?.jp2$", depth=5)[0])
            polar = ""
        elif sat == "tsx":
            for f in range(len(prod.files)):
                filenames.append(os.path.join(input_folder, 'IMAGEDATA', prod.files[f]))
        elif sat in ["l8", "l9"]:
            filenames.append(prod.find_file(pattern=r"*B2.TIF", depth=5)[0])
            polar=""

                                                            
        for filename in filenames:
            start=datetime.now()
            if sat == "s1":  # Sentinel-1 case
                orbit = prod.base.split("_")[4]
                ds_in = GDalDatasetWrapper.from_file(filename)
                epsg = str(ds_in.epsg)
                extent = list(ds_in.extent(dtype=float))
                date = prod.date.strftime("%Y%m%dT%H%M%S")
                extent_str = ds_in.extent(dtype=str)
                res = ds_in.resolution

                #Topography file for corresponding tile (S1 case)
                if dem_choice == "copernicus":
                    ul_latlon = transform_point(ds_in.ul_lr[:2], 
                                                old_epsg=ds_in.epsg, 
                                                new_epsg=4326)
                    lr_latlon = transform_point(ds_in.ul_lr[-2:], 
                                                old_epsg=ds_in.epsg,
                                                new_epsg=4326)
                    topo_names = get_copdem_codes(copdem_dir, 
                                                  ul_latlon, 
                                                  lr_latlon)
                else:
                    topo_names = [os.path.join(merit_dir, prod.tile + ".tif")]
                print("\tDEM file: %s" % topo_names)
                slp_norm, _ = RDF_tools.slope_creator(tmp_dir, 
                                                      epsg, 
                                                      extent_str, 
                                                      topo_names, 
                                                      res=[10, 10])
                # To avoid planar over detection (slp=0 and nodata values set to 0.01)
                slp_norm[slp_norm <= 0] = 0.01  
                v_stack = RDF_tools.s1_inf_stack_builder(filename, slp_norm)
                background = None

                #ESA world cover
                if dem_choice!="copernicus":
                    ul_latlon = transform_point(ds_in.ul_lr[:2], 
                                                old_epsg=ds_in.epsg, 
                                                new_epsg=4326)
                    lr_latlon = transform_point(ds_in.ul_lr[-2:], 
                                                old_epsg=ds_in.epsg, 
                                                new_epsg=4326)
                wc_files = get_esawc_codes(wc_dir, 
                                           ul_latlon, 
                                           lr_latlon)

            elif sat == "s2":  # Sentinel-2 case
                ds_in = GDalDatasetWrapper.from_file(filename)
                date = prod.date.strftime("%Y%m%dT%H%M%S")
                orbit = prod.rel_orbit.replace("R", "")
                epsg = str(ds_in.epsg)
                extent = list(ds_in.extent(dtype=float))
                # extent_str = ds_in.extent(dtype=str)
                res = ds_in.resolution
                v_stack = RDF_tools.s2_inf_stack_builder(prod, tmp_dir)
                background = prod.find_file(pattern=r"*TCI(_20m)?.jp2$", depth=5)[0]

                #ESA world cover
                ul_latlon = transform_point(ds_in.ul_lr[:2], 
                                            old_epsg=ds_in.epsg, 
                                            new_epsg=4326)
                lr_latlon = transform_point(ds_in.ul_lr[-2:], 
                                            old_epsg=ds_in.epsg, 
                                            new_epsg=4326)
                wc_files = get_esawc_codes(wc_dir, 
                                           ul_latlon, 
                                           lr_latlon)
            elif sat == "tsx":  # TSX
                polar = filename.split('/')[-1].split('_')[1]
                ds_in = GDalDatasetWrapper.from_file(filename)
                epsg = str(ds_in.epsg)
                extent = list(ds_in.extent(dtype=float))
                orbit = prod.orbit
                date = prod.date.strftime("%Y%m%dT%H%M%S")
                extent_str = ds_in.extent(dtype=str)
                res = ds_in.resolution
                basesplit = prod.base.replace('___','_').replace('__','_').split('_')

                # Topography files for corresponding tile 
                if dem_choice == "copernicus":
                    ul_latlon = transform_point(ds_in.ul_lr[:2], 
                                                old_epsg=ds_in.epsg, 
                                                new_epsg=4326)
                    lr_latlon = transform_point(ds_in.ul_lr[-2:], 
                                                old_epsg=ds_in.epsg, 
                                                new_epsg=4326)
                    topo_names = get_copdem_codes(copdem_dir, ul_latlon, lr_latlon)
                else:
                    ## NOT WORKING - Issue to be solved
                    topo_names = [os.path.join(merit_dir, tile + ".tif")] 
                print("\tDEM file: %s" % topo_names)
                slp_norm, _ = RDF_tools.slope_creator(tmp_dir, 
                                                      epsg, 
                                                      extent_str, 
                                                      topo_names, 
                                                      prod.mnt_resolution)
                # To avoid planar over detection (slp=0 and nodata values set to 0.01)
                slp_norm[slp_norm <= 0] = 0.01  
                #Calibration coefficient set manually here
                v_stack = RDF_tools.tsx_inf_stack_builder(filename, 
                                                          slp_norm, 
                                                          C=2500) 
                background = None
                
                #ESA world cover
                ul_latlon = transform_point(ds_in.ul_lr[:2], 
                                            old_epsg=ds_in.epsg, 
                                            new_epsg=4326)
                lr_latlon = transform_point(ds_in.ul_lr[-2:], 
                                            old_epsg=ds_in.epsg, 
                                            new_epsg=4326)
                wc_files = get_esawc_codes(wc_dir, 
                                           ul_latlon, 
                                           lr_latlon)
            elif sat == "l8" or sat =="l9":  # Landsat-8/9 case
                ds_in = GDalDatasetWrapper.from_file(filename)

                epsg = str(ds_in.epsg)
                extent = list(ds_in.extent(dtype=float))
                date = prod.date.strftime("%Y%m%dT%H%M%S")
                res = ds_in.resolution
                orbit = ""

                if extent[1]<=0 or extent[3]<=0: # If we are in the southern hemisphere
                    if epsg[2]=='6': # Turns northern hemisphere...
                        epsg = epsg[0:2]+'7'+epsg[3:]# into southern hemisphere
                    extent[1]+=10000000 # And extent corrected in latitude
                    extent[3]+=10000000 # And extent corrected in latitude
                    UL_LR = list(ds_in.ul_lr)
                    UL_LR[1]+=10000000
                    UL_LR[3]+=10000000
                    UL_LR = tuple(UL_LR)

                else:
                    UL_LR = ds_in.ul_lr

                v_stack = RDF_tools.ldt_inf_stack_builder(prod, tmp_dir)
                background = None

                #ESA world cover
                if dem_choice!="copernicus":
                    ul_latlon = transform_point(UL_LR[:2], 
                                                old_epsg=int(epsg), 
                                                new_epsg=4326)
                    lr_latlon = transform_point(UL_LR[-2:], 
                                                old_epsg=int(epsg), 
                                                new_epsg=4326)
                wc_files = get_esawc_codes(wc_dir, 
                                           ul_latlon, 
                                           lr_latlon)
            else:
                raise ValueError("Unknown  Satellite. Has to be s1, s2, l8, l9 or tsx.")

            n_divisions = 20
            windows = np.array_split(v_stack, n_divisions, axis=0)
            predictions = []

            # RANDOM FOREST
            print('\tLoading RDF model...')
            rdf = joblib.load(db_path)  # /path to be changed
            for idx in range(len(windows)):
                # Remove NaN & predict
                current = windows[idx]
                current[np.isnan(current)] = 0
                rdf_pred = rdf.predict(current)
                predictions.append(rdf_pred)

            ### Inference Output image reconstruction
            ds_filename = GDalDatasetWrapper.from_file(filename)
            dim = ds_filename.array.shape[:2]
            vec_out = np.concatenate(predictions).reshape(dim[0], dim[1])
            exout = np.array(vec_out, dtype=np.uint8)

            # Apply nodata
            exout[ds_in.array == 0] = 255

            ## adding clouds and shadows
            if sat == "s2":
                #Cloud detection using Sen2corSCL
                scl_path = prod.find_file(pattern=r"\w+SCL_20m.jp2$", depth=5)[0]
                scl_img = gdal_warp(scl_path, tr="10 10", r="cubic").array
                exout[scl_img == 8] = 6 # Cloud
                exout[scl_img == 9] = 6 # Cloud
                exout[scl_img == 10] = 6 # Cloud

                #Cloud shadow
                scl_path = prod.find_file(pattern=r"\w+SCL_20m.jp2$", depth=5)[0]
                scl_img = gdal_warp(scl_path, tr="10 10", r="cubic").array
                exout[scl_img == 3] = 7 # Cloud shadow

            elif sat == "l8" or sat == "l9":
                #Cloud detection using blue band
                blue = prod.find_file(pattern=r"\w+B2.TIF", depth=5)[0]
                #Landsat 8/9 values
                blue_img = np.multiply(GDalDatasetWrapper.from_file(blue).array, 2.75e-5)-0.2 
                cloud = blue_img >0.2 
                exout[cloud] = 6


            ### File export
            if sat in ["s1", "s2", "l8", "l9"]: 
                dirfile = "FloodMapping_{}_{}_{}_{}".format(
                                                            prod.tile, 
                                                            date, 
                                                            sat.upper(), 
                                                            orbit)
            elif sat in ["tsx"]:
                dirfile = "FloodMapping_{}_{}_{}_{}_{}".format(
                                                            sat.upper(), 
                                                            orbit, 
                                                            polar, 
                                                            basesplit[7], 
                                                            basesplit[8])
            FileSystem.create_directory(os.path.join(dir_output, dirfile))

            #####
            ### Export inference with post-processing
            outpost = RDF_tools.postreatment(exout, radius=rad) #Post-processed inference
            outpost[ds_in.array == 0]=255
            if sat in ["s1", "s2", "l8", "l9"]: 
                outifpost = os.path.join(dir_output, 
                                         dirfile, 
                                         'FM_{}_{}_{}_{}_POST.tif'.format(prod.tile, 
                                                                          date, 
                                                                          sat.upper(), 
                                                                          orbit))
            elif sat in ["tsx"]: 
                outifpost = os.path.join(dir_output, 
                                         dirfile, 
                                         'FM_{}_{}_{}_{}_{}_{}_POST.tif'.format(sat.upper(), 
                                                                                prod.type.upper(), 
                                                                                polar, 
                                                                                basesplit[7], 
                                                                                basesplit[8], 
                                                                                orbit))

            ds_out = GDalDatasetWrapper(array=np.array(outpost),
                                        projection=ds_filename.projection,
                                        geotransform=ds_filename.geotransform)
            ds_out.write(outifpost, options=["COMPRESS=LZW"], nodata=255)

            #####
            ### Rapid mapping map creation
            
            ## GSW overlay selection
            gsw_files = get_gswo_codes(gsw_dir, 
                                       ul_latlon, 
                                       lr_latlon)
            print("\tGSWO file: %s" % gsw_files)

            static_display_out = outifpost.replace(".tif", ".png")
            rapid_mapper.static_display(outifpost, 
                                        tmp_dir, 
                                        gsw_files,  
                                        prod.date.strftime("%Y-%m-%d %H:%M:%S"), 
                                        polar, 
                                        static_display_out, 
                                        orbit, 
                                        sat=sat, 
                                        background=background, 
                                        rad=rad)
            
            #### End rapid mapping map creation


            ## ESA WC mask
            # ESA worldcover retrieval and cropping
            wc_array = RDF_tools.wc_classifier(tmp_dir, 
                                               epsg, 
                                               extent, 
                                               wc_files, 
                                               res=[abs(res[0]), abs(res[1])])
            WCmask = wc_array.copy()
            WCmask[:]=0
            WCmask[wc_array==10]=2 #Forest
            WCmask[wc_array==50]=4 #Urban

            #####
            ### Export inference post-processed + OCS 3 classes
            outarray = outpost.copy(); 
            outarray[:]= 0

            # 1-Flood 2-Forest 3-Forest+Flood 4-Urban 5-Urban+Flood
            outarray = outpost + WCmask 
            if sat in ["s2", "l8", "l9"]: 
                outarray[outpost==6] = 6 # Clouds
                outarray[outpost==7] = 7 # Shadows

            outarray[ds_in.array == 0]=255
            if sat in ["s1", "s2", "l8", "l9"]: 
                outif = os.path.join(dir_output, 
                                     dirfile, 
                                     'FM_{}_{}_{}_{}_OCS.tif'.format(prod.tile, 
                                                                     date, 
                                                                     sat.upper(), 
                                                                     orbit))
            elif sat in ["tsx"]: 
                outif = os.path.join(dir_output, 
                                     dirfile, 
                                     'FM_{}_{}_{}_{}_{}_{}_OCS.tif'.format(sat.upper(), 
                                                                           prod.type.upper(),
                                                                           polar, 
                                                                           basesplit[7], 
                                                                           basesplit[8], 
                                                                           orbit))

            ds_out = GDalDatasetWrapper(array=np.array(outarray),
                                        projection=ds_filename.projection,
                                        geotransform=ds_filename.geotransform)
            ds_out.write(outif, options=["COMPRESS=LZW"], nodata=255) 
                      
            print(datetime.now()-start)
        
        FileSystem.remove_directory(tmp_dir)       

    print("Inference finished !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation scheduler')

    parser.add_argument('-i', '--input', help='Input folder', type=str, required=True)
    parser.add_argument('-o', '--Inf_ouput', help='Output folder', type=str, required=True)
    parser.add_argument('-m', '--meritdir', help='MERIT DEM folder.'
                                                 'Either this or --copdemdir has to be set for sentinel 1.',
                        type=str, required=False)
    parser.add_argument('-c', '--copdemdir', help='Copernicus DEM folder.'
                                                  'Either this or --meritdir has to be set for sentinel 1.',
                        type=str, required=False)
    parser.add_argument('-wc', '--wc_dir', help='ESA world cover directory', type=str, required=True)
    parser.add_argument('--satellite', help='s1, s2, l8, l9 or tsx', type=str, required=True, choices=["s1", "s2", "l8", "l9", "tsx"])
    parser.add_argument('-db', '--db_path', help='Learning database filepath', type=str, required=True)
    parser.add_argument('-tmp', '--tmp_dir', help='Global DB output folder ', type=str, required=False, default="tmp")
    parser.add_argument('-g', '--gsw', help='Tiled GSW folder', type=str, required=True)
    parser.add_argument('-r', '--rad', help='Post-process MAj filter radius', type=int, required=False)

    arg = parser.parse_args()

    main_inference(arg)
