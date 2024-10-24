#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""



import os
import glob
import numpy as np
import argparse
import tempfile
from Common import RDF_tools
from Common import FileSystem
from Common.GDalDatasetWrapper import GDalDatasetWrapper
from Common.ImageTools import gdal_warp
from Common.ImageIO import transform_point
from Common.Mosaicist import get_copdem_codes


def main_preparation(args):
    emsr_dir = args.input
    gsw_dir = args.gsw
    db_dirout = args.output
    merit_dir = args.meritdir
    copdem_dir = args.copdemdir
    sat = args.sentinel
    emsr_numbers = args.emsr_numbers
    tag = args.suffix

    # EMSR directories listing
    emsr_list = glob.glob(os.path.join(emsr_dir, "EMSR*"), recursive=False)

    # Select DEM based on provided paths
    dem_choice = "copernicus" if copdem_dir else "merit"

    # Create Temporary file-directory
    tmp_dir = tempfile.mkdtemp(dir=os.getcwd())

    os.environ["PATH"] = os.environ["PATH"].split(";")[-1]
    print(emsr_list)

    # EMSR parsing
    for emsr_id in emsr_numbers:

        # EMSR directories listing
        emsr_path = sorted(glob.glob(os.path.join(emsr_dir, "EMSR" + str(emsr_id)), recursive=False))

        if not emsr_path:
            print("Skipping EMSR %s. Cannot find folder." % emsr_id)
            continue
        emsr_path = emsr_path[0]

        vstack = np.array([], dtype=np.float32)
        rdn = np.array([], dtype=np.float32)

        print("\nEMSR considered: ", emsr_path)

        tiles = [f for f in os.listdir(emsr_path) if len(f) == 5]
        print("tiles : ", tiles)

        # Tile parsing in each EMSR case
        for tile in tiles:

            print("\n\t Tile: ", tile)

            if sat == 1:
                file_list = glob.glob(os.path.join(emsr_path, tile) + "/s*.tif",
                                      recursive=True)
            elif sat == 2:
                file_list = glob.glob(os.path.join(emsr_path, tile) + "/S2*/GRANULE/*/IMG_DATA/*/*B03*10m.jp2",
                                      recursive=True)
            else:
                raise ValueError("Unknown Sentinel Satellite. Has to be 1 or 2.")
            # Tile info (epsg and extent)
            ds_in = GDalDatasetWrapper.from_file(file_list[0])
            epsg = str(ds_in.epsg)
            extent_str = ds_in.extent(dtype=str)

            # Mask formation from GSWO
            gswo_name = os.path.join(gsw_dir, "%s.tif" % tile)

            print("\t\t GSWO_file:  ", gswo_name)
            ds_out = gdal_warp(gswo_name, tr="%s %s" % (ds_in.resolution[0], ds_in.resolution[1]),
                               s_srs="EPSG:4326",
                               t_srs="EPSG:%s" % epsg,
                               te=ds_in.extent(dtype=str),
                               r="cubic", ot="Int16")
            dim = ds_out.array.shape[:2]
            gswo = ds_out.array
            idx_reject_gswo = np.where(gswo == 255)  # index to reject
            mask_gswo = np.zeros(shape=(dim[1], dim[0]))
            mask_gswo[np.where((gswo > 90) & (gswo != 255))] = 1  # Threshold put to 90%

            imask_rdn = np.ravel(np.flatnonzero(gswo == 0))

            # Parsing for each file of each tile of each EMSR case
            if sat == 1:  # Sentinel-1 case
                # MERIT or Copernicus-DEM topography files for corresponding tile (S1 case)
                if dem_choice == "copernicus":
                    ul_latlon = transform_point(ds_in.ul_lr[:2], old_epsg=ds_in.epsg, new_epsg=4326)
                    lr_latlon = transform_point(ds_in.ul_lr[-2:], old_epsg=ds_in.epsg, new_epsg=4326)
                    topo_names = get_copdem_codes(copdem_dir, ul_latlon, lr_latlon)
                else:
                    topo_names = [os.path.join(merit_dir, tile + ".tif")]

                print("\t\t DEM files:  ", topo_names)
                slp_norm, idx_reject_slp = RDF_tools.slope_creator(tmp_dir, epsg, extent_str, topo_names)

                # Water proof areas (where water occurrence >90% and slopes <10°)
                imask_roi = np.ravel(np.flatnonzero(mask_gswo > 0))

                # S1 related file listing
                s1_vv = glob.glob(os.path.join(emsr_path, tile, "**/*", "*vv*.tif"), recursive=True)
                print("\n\t** ", len(s1_vv), "S1 files to consider")

                # S1 parsing and processing (VV & VH)
                vstack, rdn = RDF_tools.s1_prep_stack_builder(s1_vv, slp_norm, idx_reject_gswo, idx_reject_slp,
                                                              mask_gswo, imask_roi, imask_rdn, vstack, rdn)
            elif sat == 2:  # Sentinel-2 case

                imask_roi = np.ravel(np.flatnonzero((mask_gswo > 0)))
                file_list = glob.glob(os.path.join(emsr_path, tile) + "/S2*", recursive=False)

                # S2 parsing and processing (NDVI & MNDWI)
                vstack, rdn = RDF_tools.s2_prep_stack_builder(file_list, idx_reject_gswo,
                                                              mask_gswo, imask_roi, imask_rdn, vstack, rdn)

        # Save outputs for training
        vstack_out = vstack.transpose()
        rdn_out = rdn.transpose()
        vec_ok = np.ravel(np.flatnonzero(np.sum(vstack, axis=0)))

        vstack_out = vstack_out[vec_ok]
        rdn_out = rdn_out[vec_ok]
        emsr_id = emsr_path.split("/")[-1]

        FileSystem.create_directory(db_dirout)
        if tag is None:
            namo = os.path.join(db_dirout,  "DB_S%s_%s_WAT.npy" % (sat, emsr_id))
            namo2 = os.path.join(db_dirout, "DB_S%s_%s_RDN.npy" % (sat, emsr_id))
        else:
            namo = os.path.join(db_dirout,  "DB_S%s_%s_WAT_%s.npy" % (sat, emsr_id, tag))
            namo2 = os.path.join(db_dirout,  "DB_S%s_%s_RDN_%s.npy" % (sat, emsr_id, tag))

        np.save(namo, vstack_out, allow_pickle=False)
        np.save(namo2, rdn_out, allow_pickle=False)

        print("Water stack size: ", vstack_out.shape)
        print("RDN stack size: ", rdn_out.shape)

    FileSystem.remove_directory(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation scheduler')

    parser.add_argument('-i', '--input', help='Input folder (EMSR folder)', type=str, required=True)
    parser.add_argument('-g', '--gsw', help='Tiled GSW folder', type=str, required=True)
    parser.add_argument('--sentinel', help='S1 or S2', type=int, required=True, choices=[1, 2])
    parser.add_argument('-n', '--emsr_numbers', help='EMSR cases name', nargs='+', type=int)
    parser.add_argument('-o', '--output', help='Output folder (NPY folder)', type=str, required=True)
    parser.add_argument('-m', '--meritdir', help='MERIT DEM folder.'
                                                 'Either this or --copdemdir has to be set for sentinel 1.',
                        type=str, required=False)
    parser.add_argument('-c', '--copdemdir', help='Copernicus DEM folder.'
                                                  'Either this or --meritdir has to be set for sentinel 1.',
                        type=str, required=False)
    parser.add_argument('-s', '--suffix', help='Suffix tag', type=str, required=False)
    arg = parser.parse_args()

    main_preparation(arg)
