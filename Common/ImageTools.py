#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""



from PIL.Image import BICUBIC
import os
import numpy as np
import uuid
from osgeo import gdal
from Common import FileSystem
from Common.GDal import buildvrt, translate, warp, rasterize
from Common.GDalDatasetWrapper import GDalDatasetWrapper


def resize(img, input_res, output_res, resample=BICUBIC):
    """
    Resize a numpy array using PIL
    :param img: The input image of ndim == 2
    :param input_res: The input resolution
    :param output_res: The output resolution
    :param resample: The resampling mode. Default is BICUBIC.
    :return: The numpy array of different resolution.
    """
    from PIL import Image
    assert img.ndim == 2
    output_size = tuple(int(i * input_res / output_res) for i in img.shape)
    out = Image.fromarray(img).resize(output_size, resample)
    return np.array(out)


def extract_class(mask, class_to_extract, value_type):
    """
    Extract either bits or values from an image.
    :param mask: Numpy array
    :param value_type: The type of the value to be extracted. Can be one of 'Bit', 'Value' or 'Threshold'.
    :param class_to_extract: The items to be extracted.
    In the case of 'Bit' and 'Value' an array of items to be extracted is expected; In the case of threshold a single
    integer.
    :return: Numpy array of the same size as the input with the desired classes extracted.
    """
    value_type = str(value_type).lower()
    if value_type == "bit":
        return extract_bits(mask, class_to_extract)
    elif value_type == "value":
        return extract_values(mask, class_to_extract)
    elif value_type == "threshold":
        # TODO Add double thresholding
        try:
            class_to_extract = int(class_to_extract)
        except ValueError as e:
            # TODO Add workaround for error
            raise e
        out = np.zeros_like(mask, dtype=np.uint8)
        out[mask >= int(class_to_extract)] = 1
        return out

    raise ValueError("Unknown value type encountered: %s" % value_type)


def extract_values(img, values):
    """
    Extract values from a given image
    :param img: The input image
    :param values: The list of values to be extracted
    :return:
    """
    if len(values) < 1:
        raise ValueError("No list of values given")
    full_img = [img == val for val in values]
    return np.bitwise_or.reduce(full_img)


def extract_bits(img, bits):
    """
    Extract bits from a given image
    :param img: The input image
    :param bits: The list of bit positions to be extracted.
    E.g. bit position 5 --> Value 32 (2**5) will be extracteed
    :return:
    """
    if len(bits) < 1:
        raise ValueError("No list of bits given")
    full_img = [np.bitwise_and(img, 2**bit) > 0 for bit in bits]
    return np.bitwise_or.reduce(full_img)


def normalize(img, value_range_out, value_range_in, clip=False, dtype=None):
    """
    Normalize an image to a fixed range of values

    :param img: The input image
    :param value_range_out: The range of output values
    :param value_range_in: The range of output values
    :param clip: Clip the min max values
    :param dtype: The output array dtype
    :return: A numpy array of the same size as the input with values scaled in the desired range
    """
    new_value_min, new_value_max = value_range_out
    old_value_min, old_value_max = value_range_in

    # Cast to float array if needed:
    temp_img = np.array(img, dtype=float)
    if clip:
        temp_img[temp_img < old_value_min] = old_value_min
        temp_img[temp_img > old_value_max] = old_value_max

    # Need to catch the following case separately as it produces nan otherwise:
    if abs(old_value_max - old_value_min) < np.finfo(float).eps:
        scale_factor = 1.
    else:
        scale_factor = 1. * (new_value_max - new_value_min) / (old_value_max - old_value_min)
    if not dtype:
        dtype = float
    return np.array(scale_factor * (temp_img - old_value_max) + new_value_max, dtype=dtype)


def gdal_buildvrt(*inputs, dst=None, **options):
    """
    Build a gdalvrt using in memory bindings.

    :param inputs: The list of input filenames or datasets.
    :param dst: If specified, the vrt will be writen to
    the given path on a physical disk.
    Note: This overwrites a previous vrt at the same location.
    :return: VRT of the given inputs, by default as an in-memory file.
    :type: `osgeo.gdal.dataset` or .vrt on disk (see param ``dst``).
    """
    return buildvrt.gdal_buildvrt(*[i for i in inputs], dst=dst, **options)


def gdal_translate(src, dst=None, **options):
    """
    Translate a dataset.

    :param src: The input filename or dataset.
    :param dst: If specified, the output will be writen to
    the given path on a physical disk.
    Note: This overwrites a previous file at the same location.
    :return: A :class:`Common.GDalDatasetWrapper.GDalDatasetWrapper` object
    :rtype: `osgeo.gdal.dataset` or file on disk (see parameter ``dst``).
    """
    return translate.gdal_translate(src, dst=dst, **options)


def gdal_warp(src, dst=None, **options):
    """
    Warp a dataset.

    :param src: The input filename or dataset.
    :param dst: If specified, the output will be writen to
    the given path on a physical disk.
    Note: This overwrites a previous file at the same location.
    :return: A :class:`Common.GDalDatasetWrapper.GDalDatasetWrapper` object
    :rtype: `osgeo.gdal.dataset` or file on disk (see parameter ``dst``).
    """
    return warp.gdal_warp(src, dst=dst, **options)


def gdal_merge(*src, dst=None, **options):
    """
    Merge a dataset using a modified gdal_merge.

    :param src: The list of input filenames or datasets.
    :param dst: If specified, the output will be writen to
    the given path on a physical disk.
    :param options: Optional keyword arguments
    :return:
    """
    gdal_common_params = ["optfile"]
    options_list = [""]
    for k, v in options.items():
        if k in gdal_common_params:
            options_list += ["--%s" % k, "%s" % v]
        elif type(v) is not bool:
            options_list += ["-%s" % k, "%s" % v]
        elif type(v) is bool and v is True:
            options_list.append("-%s" % k)
        else:
            pass
    src = [i.get_ds() if type(i) == GDalDatasetWrapper else i for i in src]

    raise NotImplementedError


def gdal_retile(src, dst, **options):
    """
    Tile a raster and write the individual files to the given dst folder.
    :param dst: The destination directory
    :param src: The source filename or dataset
    :type src: :class:`Gdal.Dataset or str
    :param options: Optional arguments
    :return: The list of files created
    :rtype: list of str
    """
    if not os.path.isdir(dst):
        os.makedirs(dst)
    if type(src) == GDalDatasetWrapper:
        src = src.get_ds()
    raise NotImplementedError


def gdal_rasterize(src, dst=None, **options):
    """
    Rasterize a dataset.

    :param src: The input filename or dataset.
    :param dst: If specified, the output will be writen to
    the given path on a physical disk.
    Note: This overwrites a previous file at the same location.
    :return: A :class:`Common.GDalDatasetWrapper.GDalDatasetWrapper` object
    :rtype: `osgeo.gdal.dataset` or file on disk (see parameter ``dst``).
    """
    return rasterize.gdal_rasterize(src, dst=dst, **options)

