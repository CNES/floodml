#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


def getDatetimeShort(bdate):
    """
    Parse a datetime in byte literal to string and format it to YYYYMMDD
    """
    return str(bdate.decode('UTF-8').split(" ")[0].replace(".",""))

def getDatetimeInterval(strdate):
    """
    Parse a date and return the previous and the succeeding day as string
    """
    from datetime import datetime,timedelta
    ddate=datetime.strptime(strdate, "%Y%m%d")
    date_start=datetime.strftime(ddate - timedelta(days=1), "%Y-%m-%d")
    date_end=datetime.strftime(ddate + timedelta(days=1), "%Y-%m-%d")
    return (date_start, date_end)

def getH5File(filename):
    """
    Read a HDF5 File and its keys
    """
    import h5py
    f = h5py.File(filename)
    keys = list(f.keys())
    return f,keys

def createDownloadList(filename):
    """
    Create the list of downloadable L1C and L2A products from the HDF5 File
    """
    import numpy as np
    f, keys = getH5File(filename)
    dates = 'dates'
    granule = 'granule_id'
    d = np.array(f[dates])
    g = np.array(f[granule])
    total = list(zip(d,g))
    filtered = list(set(total))
    for i, item in enumerate(filtered):
        filtered[i] = (getDatetimeShort(item[0]), "T" + str(item[1].decode('utf-8')))
    return filtered

if __name__ == "__main__":
    import sys
    assert sys.version_info[:2] == (2,7)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",help="The Input filename", required=True, type=str)
    parser.add_argument("-o", "--output",help="The output dir", required=True, type=str)
    args = parser.parse_args()
    assert args.input[-3:] == ".h5"
    l = createDownloadList(args.input)
    import S2Download as dl
    import os
    wdir = os.path.realpath(args.output)
    if(os.path.exists(wdir)):
        if(not os.path.isdir(wdir)):
            raise OSError("File existing for directory-creation of {0}".format(args.output))
    else:
        try:
            os.mkdirs(wdir)
        except:
            raise OSError("Cannot create directory {0}".format(wdir))
    
    for date, tile in l:
        start, end = getDatetimeInterval(date)
        print(start)
        print(end)
        print(tile)
        print(date)
        #d = dl.Downloader(tile, start, None, end, None, args.output)
        #d.run()
        from theia_download.theia_download import TheiaDownloader
        t = TheiaDownloader()
        t.downloadSelected(tile, [date], args.output, None, None)

