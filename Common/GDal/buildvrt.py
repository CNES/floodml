#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


from osgeo import gdal
import uuid
from Common import FileSystem
from Common.GDalDatasetWrapper import GDalDatasetWrapper


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
    gdal_common_params = ["optfile"]
    options_list = []
    for k, v in options.items():
        if k in gdal_common_params:
            options_list += ["--%s" % k, "%s" % v]
        elif type(v) is not bool:
            options_list += ["-%s" % k, "%s" % v]
        elif type(v) is bool and v is True:
            options_list.append("-%s" % k)
        else:
            pass
    options_list = " ".join(options_list)

    inputs = [i.get_ds() if type(i) == GDalDatasetWrapper else i for i in inputs]
    # Remove previous existing file if writing to disk is enabled:
    if dst:
        FileSystem.remove_file(dst)
        # Note: De-allocation before file is actually written to disk
        # cf. https://gdal.org/api/python_gotchas.html
        _ = gdal.BuildVRT(dst, [i for i in inputs], options=options_list)
        _ = None
        ds_out = gdal.Open(dst)
    else:
        # Write directly into memory if no path specified (faster):
        dst = "/vsimem/" + uuid.uuid4().hex
        ds_out = gdal.BuildVRT(dst, [i for i in inputs], options=options_list)
    return GDalDatasetWrapper(ds=ds_out)
