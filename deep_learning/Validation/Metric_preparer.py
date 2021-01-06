#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import argparse
import os
import itertools
from datetime import datetime
from datetime import timedelta
import numpy as np
from functools import reduce
import pandas as pd
from Common import FileSystem
from prepare_mnt.mnt.SiteInfo import Site
from Common.GDalDatasetWrapper import GDalDatasetWrapper
from Common.ImageTools import gdal_merge, gdal_warp


def ogr2ogr(src, dst, **options):
    """

    Reprojection of an OGR layer using a subprocess.

    :param src: OGR shapefile filepath
    :param dst: The destination filepath
    :return: Returncode of subprocess.

    """
    gdal_common_params = ["optfile", "config", "debug"]
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
    options_list += [dst, src]

    return FileSystem.run_external_app("ogr2ogr", options_list)


def gdal_rasterize(src, dst, **options):
    """
    Rasterize a given ogr-file.

    :param src: Source shp path
    :param dst: Destination raster path
    :return: A .tif file
    """
    gdal_common_params = ["optfile", "config", "debug"]
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
    options_list += [src, dst]

    return FileSystem.run_external_app("gdal_rasterize", options_list)


def reproject_n_rasterize(output, epsg, date_str, rasters, burn):
    """
    Reproj and rasterize a list of rasters.

    :param rasters: The list of rasters
    :param output: The output basename
    :param epsg: The epsg code
    :param date_str: The date as str
    :param burn: The value to burn
    :return:
    """
    temp_rasters = []
    for raster in rasters:
        bname = os.path.splitext(os.path.basename(raster))[0]
        n_reproj = os.path.join(output, "reproj_%s_epsg%s_%s.shp" % (bname, epsg, date_str))
        ogr2ogr(raster, n_reproj, t_srs="EPSG:%s" % epsg, lco="ENCODING=UTF-8")
        n_raster = os.path.join(output, "raster_%s_epsg%s_%s.tif" % (bname, epsg, date_str))
        gdal_rasterize(n_reproj, n_raster, of="GTiff", ot="Byte", tr="10 10", burn=burn, a_nodata=0, q=True)
        temp_rasters.append(n_raster)
        FileSystem.remove_file(n_reproj)
        FileSystem.remove_file(n_reproj.replace("shp", "shx"))
        FileSystem.remove_file(n_reproj.replace("shp", "dbf"))
        FileSystem.remove_file(n_reproj.replace("shp", "prj"))
        FileSystem.remove_file(n_reproj.replace("shp", "cpg"))
    return temp_rasters


def get_platform(src):
    """
    Get platform file from a given folder

    :param src: The source folder
    :return: The platform as string, e.g. 'sentinel1' or 'radarsat2'
    """

    known_platforms = ["sentinel1", "sentinel2", "cosmoskymed",
                       "radarsat2", "terrasarx", "pleiades",
                       "worldview3", "geoeye1", "spot6", "spot7",
                       "landsat8"]
    platform_found = []
    for p in known_platforms:
        try:
            platform_found.append(os.path.basename(FileSystem.find_single(r"*%s*" % p, src, ftype="file")))
        except ValueError:
            continue
    if not platform_found:
        print("WARNING: Cannot find platform in %s" % src)
        return ""
    elif len(platform_found) > 1:
        print("WARNING: More than one platform found for %s" % src)
        return ""
    return platform_found[0]


def get_activation_shp_files(emsr_path, platform):
    """
    Get the list of shp files for each DELINEATION in the latest version.

    :param emsr_path: The path to the emsr folder containing all delineations
    :param platform: Extract only events which were observed by the given platforms, e.g. ['sentinel1']
    :return: A list. For each activation, a dict with keys "hydro" (list of shps) and "aoi" (one shp).
    """

    # Get list of delineation folders
    delineations = FileSystem.find(pattern="(*DELINEATION|DEL)", path=emsr_path, ftype="folder")
    map_monit_folders = []
    for dln in delineations:
        try:
            map_monit_folders.append(FileSystem.find("(MAP|MONIT*|PRODUCT)", dln, depth=2, ftype="folder"))
        except ValueError:
            continue
    # flatten
    map_monit_folders = list(itertools.chain(*map_monit_folders))
    # Find v0, v1..n folders and get the 'newest' (last element in sorted list):
    version_folders = [sorted(FileSystem.find(r"v\d", m, depth=1, ftype="folder"))[-1] for m in map_monit_folders]
    shapefiles = []
    for vs in version_folders:
        platform_found = get_platform(vs)
        if platform_found != platform:
            continue
        try:
            date = os.path.basename(FileSystem.find_single(r"\d{8}T\d{6}", vs, ftype="file"))
        except ValueError:
            print("WARNING: Cannot find date-file in %s" % vs)
            continue
        try:
            shapefiles.append(
                {"name": vs.split(os.sep)[-4],
                 "platform": platform_found,
                 "date": datetime.strptime(date, "%Y%m%dT%H%M%S"),
                 "aoi": FileSystem.find_single("*area*of*interest*.shp$", vs),
                 "hydro": FileSystem.find("*hydrography*.shp$", vs, ftype="file")
                          + FileSystem.find("(*observed*event*|*crisis*information*).shp$", vs, ftype='file')})
        except ValueError as e:
            # Usually should happen only if *observed_event* not found. If so, then skip.
            print(e)
    return shapefiles


def get_tile_infos(input_path, resolution):
    """
    Get the list of tiles, their extent and their epsg code for all tiles related to an emsr event.

    :param input_path: The path to the emsr input tiles
    :param resolution: Output resolution in m as int
    :return: A list of site: :class:`prepare_mnt.mnt.SiteInfo.Site` objects.
    """
    tile_folders = FileSystem.find(pattern=r"^T?\d{2}[A-Z]{3}$", path=input_path, ftype="folder")
    sites = []
    for tile in tile_folders:
        tif_img = FileSystem.find_single(r"^s1*tif$", tile, ftype="file")
        site_name = os.path.basename(tile)
        site = Site.from_raster(site_name, tif_img)
        site.res_x = resolution
        site.res_y = resolution
        sites.append(site)
    unique_epsg_codes = list(set([s.epsg for s in sites]))
    return sites, unique_epsg_codes


def regroup_dates(dates, td=2):
    """
    Bin dates in equal sizes of x Hours.

    :param dates: The list of dates
    :param td: The bin size
    :return: The list of datetimes corresponding to the bins which have at least 1 input date.
    """
    dates = sorted(dates)
    bins = pd.date_range(start=dates[0] - timedelta(days=1), end=dates[-1] + timedelta(days=1), freq="%sH" % td)
    df = pd.DataFrame({"date": dates, "count": [1] * len(dates)})
    df["binned"] = pd.cut(df.date, bins)
    dates_regrouped = sorted(list(set([df.at[i, "binned"].mid.to_pydatetime() for i in range(len(df))])))
    return dates_regrouped


def get_gsw_for_site(site, gsw_path, gsw_threshold):
    """
    Get a given GSW Occurrence raster for a given site. E.g. 31TCJ.tif

    :param site: :class:`prepare_mnt.mnt.SiteInfo.Site` object.
    :param gsw_path: The path to the GSW files
    :param gsw_threshold: The threshold to be applied to the file
    :return: A reprojected, binary GSW raster
    """
    ds_gsw = GDalDatasetWrapper.from_file(os.path.join(gsw_path, "%s.tif" % site.nom))
    gsw_thres = (ds_gsw.array > gsw_threshold) * 254
    ds_gsw_thres = GDalDatasetWrapper(ds=ds_gsw.get_ds(), array=gsw_thres, nodata=0)
    return gdal_warp(ds_gsw_thres, t_srs="EPSG:%s" % site.epsg, te=site.te_str, tr=site.tr_str)


def create_bitmask(gsw, emsr):
    """
    Create a combined bitmask from GSW, Flood and AOI datasets.
    Note: The datasets need to have the same transform and projection

    :param gsw: The GSW occurrence dataset
    :param emsr: The Flood dataset: 1 == AOI, 255 == Flood
    :return: A bitmask combining the three inputs with aoi = bit2, flood = bit3, aoi = bit4
    """

    msk_gsw = gsw.array > 0
    msk_aoi = emsr.array > 0
    msk_flood = emsr.array == 255

    assert msk_gsw.shape == msk_aoi.shape == msk_flood.shape

    bitmask = np.zeros_like(msk_gsw, dtype=np.uint8)
    for bit, msk in enumerate([msk_aoi, msk_flood, msk_gsw]):
        bitmask = np.bitwise_or(bitmask, (2**(bit + 1)) * msk)
    return GDalDatasetWrapper(ds=gsw.get_ds(), array=bitmask, nodata_value=0)


def main(args):
    """
    Run the EMSR rasterisation pipeline

    :param args:
    :return:
    """

    emsr_number = os.path.basename(os.path.normpath(args.emsr))
    output = os.path.join(args.output, emsr_number)
    FileSystem.create_directory(output)
    # Get all shps from the EMSR folder:
    shps = [get_activation_shp_files(args.emsr, p) for p in args.platforms]
    # flatten
    shps = [i for sub in shps for i in sub]
    if not shps:
        raise FileNotFoundError("Cannot find shapefiles for case %s" % emsr_number)
    # Get list of tiles and epsg from input/ folder:
    tiles, epsg_codes = get_tile_infos(args.product_dir, args.resolution)
    # Group dates in blocks of x hours:
    dates = regroup_dates([aoi["date"] for aoi in shps], args.timedelta)
    rasters_created = []
    for date in dates:
        # Get AOIs for given date:
        aois = [aoi for aoi in shps if abs(aoi["date"] - date) < timedelta(hours=args.timedelta)]
        if not aois:
            continue
        elif all(aois[0]["date"] == a["date"] for a in aois):
            date_str = aois[0]["date"].strftime("%Y%m%dT%H%M%S")
        else:
            date_str = date.strftime("%Y%m%dT%H%M%S")
        print(date_str, ":")
        for a in aois:
            print(a["name"], a["platform"], a["date"])
        for epsg in epsg_codes:
            tif_hydro = []
            for aoi in aois:
                # Reproj and rasterize:
                temp_rasters = reproject_n_rasterize(output, epsg, date_str, [aoi["aoi"]], burn=1)
                tif_hydro.append(gdal_merge(*temp_rasters, n=0, q=True))
                [FileSystem.remove_file(f) for f in temp_rasters]
                temp_rasters = reproject_n_rasterize(output, epsg, date_str, aoi["hydro"], burn=255)
                tif_hydro.append(gdal_merge(*temp_rasters, n=0, q=True))
                [FileSystem.remove_file(f) for f in temp_rasters]
            ds_hydro_merged = gdal_merge(*tif_hydro, n=0, a_nodata=0, q=True)
            selected_sites = [t for t in tiles if t.epsg == epsg]
            for site in selected_sites:
                print("Creating raster for %s for date %s" % (site.nom, date_str))
                flood_name = os.path.join(output, "%s_%s.tif" % (site.nom, date_str))
                ds_flood = gdal_warp(ds_hydro_merged, te=site.te_str, dstnodata=0, tr=site.tr_str)
                # Count number of empty pixels
                n_pix = reduce(lambda x, y: x * y, list(ds_flood.array.shape))
                n_nonzero = np.count_nonzero(ds_flood.array)
                n_pix_nodata = 100. * (1 - (n_nonzero / n_pix))
                # Skip empty ones
                if n_pix_nodata > args.maxpix:
                    print("Skipping empty raster. Nodata: {:.2f}%%".format(n_pix_nodata))
                    continue
                # If we arrived here: Combine GSW and Hydro file and write to disk
                ds_gsw = get_gsw_for_site(site, args.gsw, args.gsw_threshold)
                ds_bitmask = create_bitmask(ds_gsw, ds_flood)
                ds_bitmask.write(flood_name, options=["COMPRESS=DEFLATE"])
                rasters_created.append(flood_name)
    if not rasters_created:
        print("Did not create any rasters.")
        FileSystem.remove_directory(output)
    return 0


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='for i in `ls /projects/floodml/inputs_all/validation/`; do'
                                                 'python Metric_preparer.py -e /projects/floodml/dbEMS_filtered/$i/'
                                                 '-p /projects/floodml/inputs_all/validation/$i/'
                                                 '-g /projects/floodml/tiled_gsw/'
                                                 '-o tiled_emsr_10m -r 10 ;done')
    parser.add_argument('-e', '--emsr', action="store", type=str, required=True,
                        help="The directory containing all EMSR shapefiles that should be rasterized")
    parser.add_argument('-p', '--product_dir', action="store", type=str, required=True,
                        help="The directory containing all products for the EMSR event.")
    parser.add_argument('-g', '--gsw', action="store", type=str, required=True,
                        help="The directory containing all GSW occurrence files.")
    parser.add_argument('--gsw_threshold', action="store", type=str,
                        help="Threshold at which GSW will be considered 'water'. Default is 90.", default=90)
    parser.add_argument('-m', '--maxpix', action="store", type=float,
                        help="Max nodata percentage. Default is 98.", default=99.5)
    parser.add_argument('-t', '--timedelta', action="store", type=int,
                        help="Maximum timedelta between two separate shapefiles to be considered 'of-the-same-event'."
                             "Default is 24 [hr].", default=2)
    parser.add_argument('--platforms', action='store', nargs="+",
                        help="Only extract these platforms. By default, extract all known ones.",
                        default=["sentinel1", "sentinel2"])
    parser.add_argument('-r', '--resolution', action="store", type=int,
                        help="Output raster resolution in x and y direction"
                             "Default is 30[m].", default=30)
    parser.add_argument('-o', '--output', action="store",
                        help="Output directory for rasterized and tiled shapefiles.",
                        default=os.getcwd())
    args_cmd = parser.parse_args()
    main(args_cmd)
