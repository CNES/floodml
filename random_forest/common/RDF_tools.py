#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import numpy as np
import os
import gc
import glob
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from functools import reduce
from Common.GDalDatasetWrapper import GDalDatasetWrapper
from Common.ImageTools import gdal_warp
from Common import FileSystem
from Chain import Product


def s1_prep_stack_builder(s1_vv, slp_norm, idx_reject_gswo, idx_reject_slp, mask_gswo, imask_roi, imask_rdn,
                          vstack_s1, rdn_stack):
    """
    S1 parsing and processing (VV & VH) with GSWO (water ground truth) and MERIT (slope)

    :param s1_vv: S1 file list
    :param slp_norm: Normalized slope array
    :param idx_reject_gswo: Rejected GSWO mask values
    :param idx_reject_slp: Rejected MERIT slope values
    :param mask_gswo: Selected pixels (water pixels at >90% occurrence)
    :param imask_roi: index map of ROI
    :param imask_rdn: index of random pixels (random non-water)
    :param vstack_s1: Input S1 Water stack
    :param rdn_stack: Input S1 rdn stack
    :return: S1 Water and rdn_stack stacks
    """

    for j, s1 in enumerate(s1_vv):

        print("\t", s1)

        name_vh = s1.replace("vv", "vh")
        if os.path.exists(name_vh):
            print("\t", name_vh)

            # VV and VH array loading
            ds_vv = GDalDatasetWrapper.from_file(s1)
            ds_vh = GDalDatasetWrapper.from_file(s1.replace("vv", "vh"))

            raster_vv = ds_vv.array
            raster_vh = ds_vh.array
            # Filtering bloc here

            raster_vv[idx_reject_gswo] = 0
            raster_vh[idx_reject_gswo] = 0
            raster_vv[idx_reject_slp] = 0
            raster_vh[idx_reject_slp] = 0
            print("\t\t\tNumber of pixels used: %s/%s" %
                  (len(np.where(raster_vv > 0)[0]), reduce(lambda x, y: x*y, list(ds_vv.array.shape))))

            if len(np.where(raster_vv > 0)[0]) != 0:

                raster_vv = np.float32(raster_vv)
                raster_vh = np.float32(raster_vh)

                # VV and VH array truncation
                raster_vv[raster_vv >= 1] = 1
                raster_vh[raster_vh >= 1] = 1

                # S1 vertical Stack creation/updating
                if not vstack_s1.any():
                    vstack_s1 = np.vstack((raster_vv[np.unravel_index(imask_roi, mask_gswo.shape)],
                                           raster_vh[np.unravel_index(imask_roi, mask_gswo.shape)]))
                    vstack_s1 = np.vstack((vstack_s1, slp_norm[np.unravel_index(imask_roi, mask_gswo.shape)]))
                else:
                    vstack_s1_tmp = np.vstack((raster_vv[np.unravel_index(imask_roi, mask_gswo.shape)],
                                               raster_vh[np.unravel_index(imask_roi, mask_gswo.shape)]))
                    vstack_tmp = np.vstack(
                        (vstack_s1_tmp, slp_norm[np.unravel_index(imask_roi, mask_gswo.shape)]))
                    vstack_s1 = np.hstack((vstack_s1, vstack_tmp))

                # Random selection stack creation/updating (on array of same size of ROI)
                nonzero = np.flatnonzero(raster_vv[np.unravel_index(imask_rdn, mask_gswo.shape)]) * 1
                if len(nonzero) > len(np.flatnonzero(mask_gswo)):
                    rdn = np.random.choice(len(nonzero), size=len(np.flatnonzero(mask_gswo)), replace=False)
                else:
                    rdn = np.random.choice(len(nonzero), size=len(np.flatnonzero(mask_gswo)), replace=True)
                    print("\t\t\tBeware: probably more water than ground in the image")
                
                if not rdn_stack.any():
                    rdn_stack = np.vstack((raster_vv[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]],
                                           raster_vh[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
                    rdn_stack = np.vstack((rdn_stack,
                                           slp_norm[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
                else:
                    rdn_stack_tmp1 = np.vstack((raster_vv[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]],
                                                raster_vh[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
                    rdn_stack_tmp2 = np.vstack(
                        (rdn_stack_tmp1, slp_norm[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
                    rdn_stack = np.hstack((rdn_stack, rdn_stack_tmp2))
                    
    return vstack_s1, rdn_stack


def s2_prep_stack_builder(s2files, idx_reject_gswo,  mask_gswo, imask_roi, imask_rdn, vstack_s2, rdn_stack):
    """
    S2 parsing and processing (MNDWI & NDVI) with GSWO (water proof)

    :param s2files: S2 file list
    :param idx_reject_gswo: Rejected GSWO mask values
    :param mask_gswo: Selected pixels (water pixels at >90% occurrence)
    :param imask_roi: index map of ROI
    :param imask_rdn: index of random pixels (random non-water)
    :param vstack_s2: Input S2 Water stack
    :param rdn_stack: Input S2 rdn stack
    :return: Output S2 Water and rdn_stack stacks
    """

    for j, s2 in enumerate(s2files):
        prod = Product.MajaProduct.factory(s2)
        print("\t ("+str(j+1)+"/"+str(len(s2files))+")")
        print(prod)
        print(glob.glob(os.path.join(s2, "index", "*_MNDWI.tif")))

        # MNDWI and NDVI file loading:
        ds_mndwi = gdal_warp(prod.get_synthetic_band("mndwi"), tr="20 20")
        ds_ndvi = gdal_warp(prod.get_synthetic_band("ndvi"), tr="20 20")
        mndwi = ds_mndwi.array
        ndvi = ds_ndvi.array

        mndwi[idx_reject_gswo] = -10000
        ndvi[idx_reject_gswo] = -10000

        nonzero = np.flatnonzero(ndvi[np.unravel_index(imask_rdn, mask_gswo.shape)] != -10000) * 1
        if (len(np.where(ndvi > -10000)[0]) != 0) & (len(nonzero) != 0):
            mndwi = mndwi / 5000
            ndvi = ndvi / 5000

            # Final
            if not vstack_s2.any():
                vstack_s2 = np.vstack((ndvi[np.unravel_index(imask_roi, mask_gswo.shape)],
                                       mndwi[np.unravel_index(imask_roi, mask_gswo.shape)]))
            else:
                vstack_s2_tmp = np.vstack((ndvi[np.unravel_index(imask_roi, mask_gswo.shape)],
                                           mndwi[np.unravel_index(imask_roi, mask_gswo.shape)]))
                vstack_s2 = np.hstack((vstack_s2, vstack_s2_tmp))

            # Random selection stack creation/updating (on array of same size of ROI)
            print("Length nonzero:", len(nonzero), len(np.flatnonzero(mask_gswo)))
            if len(nonzero) > len(np.flatnonzero(mask_gswo)):
                rdn = np.random.choice(len(nonzero), size=len(np.flatnonzero(mask_gswo)), replace=False)
            else:
                rdn = np.random.choice(len(nonzero), size=len(np.flatnonzero(mask_gswo)), replace=True)
                print("\t\t\tBeware: probably more water than ground in the image")

            if not rdn_stack.any():
                rdn_stack = np.vstack((ndvi[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]],
                                       mndwi[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
            else:
                rdn_stack_tmp1 = np.vstack((ndvi[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]],
                                            mndwi[np.unravel_index(imask_rdn, mask_gswo.shape)][nonzero[rdn]]))
                rdn_stack = np.hstack((rdn_stack, rdn_stack_tmp1))

    return vstack_s2, rdn_stack


def slope_creator(tmpdir, epsg, extent_str, topo_name):
    """
    :param tmpdir: temporary folder
    :param epsg: epsg tile number
    :param extent_str: tile extent
    :param topo_name: DEM filename from which SLP calculation will be made
    :return: normalized slope tile & index of pixels to be rejected
    """
    # Conversion to float32 format

    tmpwarp = os.path.join(tmpdir, "Temp_32.tif")
    tmpslope = os.path.join(tmpdir, "Temp_slope.tif")

    gdal_warp(topo_name, tmpwarp, s_srs="EPSG:4326", t_srs="EPSG:%s" % epsg)

    # Slope formation
    os.system("gdaldem slope -q -of GTiff %s %s" % (tmpwarp, tmpslope))

    # Slope formatting (crop and resampling)
    ds_final = gdal_warp(tmpslope, t_srs="EPSG:%s" % epsg, tr="10 10",
                         te=extent_str, r="bilinear", ot="Float32")
    # Deleting temporary files
    FileSystem.remove_file(tmpslope)
    FileSystem.remove_file(tmpwarp)

    slp = ds_final.array
    slp_norm = slp / 90  # Normalization
    idx_reject_slp = np.where(slp < 0)  # index to reject

    return slp_norm, idx_reject_slp


def lee_filter(img, size):
    """
    :param img: Image array to be filtered
    :param size: Size of the filter box (must be odd number)
    :return: filtered array
    """
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def s1_inf_stack_builder(filename, slp_norm):
    """
    Stack builder for Sentinel-1 files for inference purposes
    :param filename:  Sentinel-1 path and filename for inference
    :param slp_norm: normalized slope array from MERIT
    :return: Stack array for inference
    """

    print("\tFile on which inference will be done: ", filename)
    name_vh = filename.replace("vv", "vh")
    assert os.path.exists(name_vh), "Cannot find VH Image: %s" % name_vh
    ds_vv = GDalDatasetWrapper.from_file(filename)
    ds_vh = GDalDatasetWrapper.from_file(filename.replace("vv", "vh"))

    s1_vv = np.array(ds_vv.array, dtype=np.float32)
    s1_vh = np.array(ds_vh.array, dtype=np.float32)
    s1_vv[s1_vv == 0] = np.nan
    s1_vh[s1_vh == 0] = np.nan

    stacked = np.hstack((np.reshape(s1_vv, (-1, 1)), np.reshape(s1_vh, (-1, 1))))
    vstack = np.hstack((stacked, np.reshape(slp_norm, (-1, 1))))
    return vstack


def s2_inf_stack_builder(product):

    """
    Stack builder for Sentinel-2 files for inference purposes

    :param product:  Sentinel-2 L2A product
    :return: Stack array for inference
    """

    # MNDWI and NDVI file loading:
    ds_mndwi = gdal_warp(product.get_synthetic_band("mndwi"), tr="20 20")
    ds_ndvi = gdal_warp(product.get_synthetic_band("ndvi"), tr="20 20")
    mndwi = ds_mndwi.array
    ndvi = ds_ndvi.array

    mndwi = mndwi / 5000
    ndvi = ndvi / 5000

    mndwi[mndwi == -2] = np.nan
    ndvi[ndvi == -2] = np.nan

    vstack_s2 = np.hstack((np.reshape(mndwi, (-1, 1)), np.reshape(ndvi, (-1, 1))))

    gc.collect()
    return vstack_s2
