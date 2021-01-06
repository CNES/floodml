#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
import glob
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import subprocess

import aux_files.pykic.raster.pykic_gdal as rpg
import aux_files.pykic.miscellaneous.miscdate as mmd

print("-**=== We're in the code! ===**-")

# EMSR directory parsing
EMSR_dir = "/work/OT/floodml/data/deliveries/phase-1-cls/"
EMSR_list = glob.glob(os.path.join(EMSR_dir, "EMSR*"), recursive=False)


Output_dir = os.path.join(os.getcwd(), "Output_all_EMSR")
cmd = ["mkdir" + "-p" + Output_dir]
if not os.path.isdir("./"+Output_dir):
    subprocess.call(cmd, shell=True)
    print("Directory created: "+Output_dir)

db_dirout = "/work/scratch/fatrasc/NPY_EMSR/"

###====----- EMSR Parsing
os.environ["PATH"] = os.environ["PATH"].split(";")[-1]

print(EMSR_list)


for EMSR_path in EMSR_list:

    Vstack_S1 = np.array([], dtype=np.float32)
    RDN = np.array([], dtype=np.float32)

    print("\nEMSR considered: ", EMSR_path)

    Tiles = [ f for f in os.listdir(EMSR_path) if len(f)==5 ]
    print("Tiles : ", Tiles)

    for tile in Tiles:

        print("\n\t Tile: ", tile)
        s1files = glob.glob(os.path.join(EMSR_path,tile)+"/s*.tif", recursive=True)

        # Tile info "epsg and extent)
        epsg = rpg.geoinfo(s1files[0], onlyepsg=True)
        extent = rpg.getextent(s1files[0])
        extent_str = str(int(extent[0])) + " " + str(int(extent[2])) + " " + str(int(extent[1])) + " " + str(int(extent[3]))

        # MERIT topography file for corresponding tile
        topo_name = os.path.join(EMSR_dir, "MERIT_S2/") + tile + ".tif"
        print("\t\t MERIT_file:  ", topo_name, rpg.geoinfo(topo_name, onlyepsg=True))
        os.system("gdalwarp -q -overwrite -s_srs EPSG:4326 -t_srs EPSG:"+epsg+"  -ot Float32 " + topo_name + " "+Output_dir+"/Temp_32.tif")
        os.system("gdaldem slope -q -of GTiff "+Output_dir+"/Temp_32.tif "+Output_dir+"/Temp_slope.tif")
        os.system("gdalwarp -q -overwrite -t_srs EPSG:"+epsg+" -tr 10 10 -te "+extent_str+" -r near -ot Float32 "+Output_dir+"/Temp_slope.tif "+Output_dir+"/Temp_slope_tiled.tif")

        # Deleting temporary files
        os.system("rm -f "+Output_dir+"/Temp_32.tif "+Output_dir+"/Temp_slope.tif")
        slp, proj, dim, tr = rpg.gdal2array(Output_dir+"/Temp_slope_tiled.tif")
        slp_norm = slp / 90  # Normalization
        # print("\t\t Slp_norm size: ", slp_norm.shape)
        I_reject_slp = np.where(slp < 0)

        # Mask formation from GSWO
        gswo_name = os.path.join(EMSR_dir, "GSW_Tiled/") + tile + ".tif"
        os.system("cp " + gswo_name + " "+Output_dir+"/.")
        print("\t\t GSWO_file:  ", gswo_name)
        epsg_gswo = rpg.geoinfo(gswo_name, onlyepsg=True)
        os.system("gdalwarp -q -overwrite -s_srs EPSG:4326 -t_srs EPSG:" + epsg + " -tr 10 10 -te " + extent_str + " -r near -ot Int16 "+Output_dir+"/" + tile + ".tif "+Output_dir+"/GSWO_tiled_temp.tif")
        gswo, proj, dim, tr = rpg.gdal2array(Output_dir+"/GSWO_tiled_temp.tif")

        I_reject_gswo = np.where(gswo == 255)
        mask_gswo = np.zeros(shape=(dim[1], dim[0]))
        mask_gswo[np.where((gswo > 90) & (gswo != 255))] = 1
        os.system("rm -f "+Output_dir+"/GSWO_tiled_temp.tif " +Output_dir+gswo_name.split("/")[-1])

        imask_roi = np.ravel(np.flatnonzero((mask_gswo > 0) & (slp < 10)))  # Threshold put to 90%
        imask_rdn = np.ravel(np.flatnonzero(gswo == 0))

        # S1 related file listing
        S1_vv = glob.glob(os.path.join(EMSR_path, tile,"**/*" "*vv*.tif"), recursive=True)
        print("\n\t** ", len(S1_vv), "S1 files to consider")


        # S1 parsing and processing (VV & VH)
        # for s1 in [S1_vv[0]]:
        for j, s1 in enumerate(S1_vv):
            print("\t ("+str(j+1)+"/"+str(len(S1_vv))+") -- S1 filedate: ", mmd.datefromstr(
                s1.replace("/", "$$").replace("t", "$$").replace("_", "$$").replace("-", "$$").split("$$")[-3]))
            print("\t", s1)

            name_vh = s1.replace("vv", "vh")
            if os.path.exists(name_vh):
                print("\t", name_vh)

                # VV and VH array loading
                VV = np.array([], dtype=np.float32)
                VH = np.array([], dtype=np.float32)

                VV, proj, dim, tr = rpg.gdal2array(s1)
                VH, proj, dim, tr = rpg.gdal2array(s1.replace("vv", "vh"))

                VV[I_reject_gswo] = 0
                VH[I_reject_gswo] = 0
                VV[I_reject_slp] = 0
                VH[I_reject_slp] =0
                print("\t\t\tNumber of interest pixels: ",len(np.where(VV>0)[0]),"/", 10980*10980)

                if len(np.where(VV>0)[0])!=0:

                    VV = np.float32(VV)
                    VH = np.float32(VH)

                    # VV and VH array truncation
                    VV[VV >= 1] = 1
                    VH[VH >= 1] = 1

                    # S1 vertical Stack creation/updating
                    if not Vstack_S1.any():
                        Vstack_S1 = np.vstack((VV[np.unravel_index(imask_roi, mask_gswo.shape)],
                                               VH[np.unravel_index(imask_roi, mask_gswo.shape)]))
                        Vstack_S1 = np.vstack((Vstack_S1, slp_norm[np.unravel_index(imask_roi, mask_gswo.shape)]))
                    else:
                        Vstack_S1_tmp = np.vstack((VV[np.unravel_index(imask_roi, mask_gswo.shape)],
                                                   VH[np.unravel_index(imask_roi, mask_gswo.shape)]))
                        Vstack_tmp = np.vstack((Vstack_S1_tmp, slp_norm[np.unravel_index(imask_roi, mask_gswo.shape)]))
                        Vstack_S1 = np.hstack((Vstack_S1, Vstack_tmp))

                    # Random selection stack creation/updating (on array of same size of ROI)
                    nonzero = np.flatnonzero(VV[np.unravel_index(imask_rdn, mask_gswo.shape)]) * 1
                    if len(nonzero) > len(np.flatnonzero(mask_gswo)):
                        rdn = np.random.choice(len(nonzero), size=len(np.flatnonzero(mask_gswo)), replace=False)
                    elif len(nonzero) <= len(np.flatnonzero(mask_gswo)):
                        rdn = np.random.choice(len(nonzero), size=len(np.flatnonzero(mask_gswo)), replace=True)
                        print("\t\t\tBeware: probably more water than ground in the image")

                    if not RDN.any():
                        RDN = np.vstack((VV[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]],
                                         VH[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
                        RDN = np.vstack((RDN, slp_norm[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
                    else:
                        RDN_tmp1 = np.vstack((VV[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]],
                                              VH[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
                        RDN_tmp2 = np.vstack((RDN_tmp1, slp_norm[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
                        RDN = np.hstack((RDN, RDN_tmp2))

                    # Clean memory
                    temp = None;
                    nonzero = None;
                    rdn = None;
                    VV = None;
                    VH = None
                    gc.collect()


    # os.system("rm -f Temp_slope_tiled.tif "+Output_dir+"/"+tile+"*.tif")


    # Save S1 outputs for modeling
    Vstack_S1_out = Vstack_S1.transpose()
    RDN_out = RDN.transpose()

    vec_ok = np.ravel(np.flatnonzero(np.sum(Vstack_S1, axis=0)))

    Vstack_S1_out = Vstack_S1_out[vec_ok]
    RDN_out = RDN_out[vec_ok]
    EMSR = EMSR_path.split("/")[-1]
    namo = os.path.join(db_dirout,"DB_S1_"+EMSR+"_water.npy")
    namo2 = os.path.join(db_dirout,"DB_S1_"+EMSR+"_RDN_.npy")

    np.save(namo, Vstack_S1_out, allow_pickle=False)
    np.save(namo2, RDN_out, allow_pickle=False)

    print("Vstack_S1_out", np.size(Vstack_S1_out, 0), np.size(Vstack_S1_out, 1))
    print("RDN_out", np.size(RDN_out, 0), np.size(RDN_out, 1))


