#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
from os import path as p
import numpy as np
import sys


class FloodExtentEstimator(object):
    """
    Estimate flood extents using S1 or S2 imagery. A basic demonstrator using eodag, SWM and Snap's methods to output
    binary masks and the time-series plot of the water extent for both platforms.
    """

    _eodag_file_prefix = "file://"

    def __init__(self, config_file):
        """
        Init the extent estimation
        :param config_file: The path to the config file containing the site-specific parameters.
        """
        from eodag.api.core import EODataAccessGateway
        from eodag.utils.logging import setup_logging
        import configparser
        from dateutil import parser
        import json

        assert p.isfile(config_file)

        self.cfg_file = configparser.ConfigParser()
        self.cfg_file.read(config_file)

        self.workspace = self.cfg_file.get("PATH", "Workspace")
        self.__downloaded = p.join(self.workspace, ".downloaded")
        self.resolution = self.cfg_file.getfloat("PROCESSING", "Resolution")
        self.start_date = parser.parse(self.cfg_file.get("PROCESSING", "Start_date"))
        self.end_date = parser.parse(self.cfg_file.get("PROCESSING", "End_date"))
        self.max_nodata = self.cfg_file.getfloat("PROCESSING", "Maximum_Nodata_Percent") / 100.
        self.crs = "EPSG:4326"  # Set default value
        lonmin = self.cfg_file.getfloat("ROI", "Longitude_Min")
        lonmax = self.cfg_file.getfloat("ROI", "Longitude_Max")
        latmin = self.cfg_file.getfloat("ROI", "Latitude_Min")
        latmax = self.cfg_file.getfloat("ROI", "Latitude_Max")
        self.extent_json = {
            'lonmin': lonmin,
            'lonmax': lonmax,
            'latmin': latmin,
            'latmax': latmax
        }
        # Note the ordering convention is different for each library:
        self.extent_lst = [str(lonmin), str(latmin), str(lonmax), str(latmax)]
        self.extent_tup = [lonmin, lonmax, latmin, latmax]
        self.extent_center = ((lonmax + lonmin) / 2, (latmax + latmin) / 2)
        self.site_id = self.cfg_file.get("ROI", "Site_ID")
        self.site_name = self.cfg_file.get("ROI", "Site_Name")

        self.swm_threshold = self.cfg_file.getfloat("SENTINEL2", "SWM_Threshold")
        self.nir_code = self.cfg_file.get("SENTINEL2", "NIR")
        self.swir_code = self.cfg_file.get("SENTINEL2", "SWIR")
        self.blue_code = self.cfg_file.get("SENTINEL2", "Blue")
        self.green_code = self.cfg_file.get("SENTINEL2", "Green")
        self.cloudmask_code = self.cfg_file.get("SENTINEL2", "Cloud_Mask")

        self._band_list = json.loads(self.cfg_file.get("SENTINEL2", "BAND_LIST"))
        self._band_res = json.loads(self.cfg_file.get("SENTINEL2", "BAND_RES_M"))
        assert len(self._band_list) == len(self._band_res)
        self._used_bands = json.loads(self.cfg_file.get("SENTINEL2", "USED_BANDS"))
        # Get the resolutions in the used bands:
        self._used_res = [res for res, band in zip(self._band_res, self._band_list)
                          if band in self._used_bands]
        self._s2_product_type = json.loads(self.cfg_file.get("SENTINEL2", "Product_Type"))
        self.max_cloud_cover = self.cfg_file.getint("SENTINEL2", "Max_Cloud")

        self.snap_threshold = self.cfg_file.getfloat("SENTINEL1", "SNAP_Threshold")
        self._s1_product_type = json.loads(self.cfg_file.get("SENTINEL1", "Product_Type"))
        if not os.path.isdir(self.workspace):
            os.mkdir(self.workspace)

        # To have some basic feedback on what eodag is doing, we configure logging to output minimum information
        setup_logging(verbose=1)

        # Load the eodag config, which always has to have the name 'eodag_conf.yml'
        conf_path = os.path.join(self.workspace, 'eodag_conf.yml')
        self.dag = EODataAccessGateway(user_conf_file_path=conf_path)
        self.dag.set_preferred_provider(u'theia')

    @staticmethod
    def get_s2_flood_mask_swm(img_blue, img_green, img_swir, img_nir, threshold=1.5):
        """
        Get the flood mask using the Sentinel-2 Water Mask (Milczarek et al.).
        All rasters have to be the same size/resolution and in reflectance TOA!
        :param img_blue: The blue band image (B02)
        :param img_green: The green band image (B03)
        :param img_swir: The SWIR band image (B11)
        :param img_nir: The NIR band image (B08)
        :param threshold: The detection threshold.
        :return: The binary SWM mask at input resolution
        """

        if img_blue.shape != img_green.shape != img_swir.shape != img_nir.shape:
            raise ValueError("Input images are of unequal size!")

        return (img_blue + img_green) / (img_swir + img_nir) > threshold

    @staticmethod
    def get_s1_flood_mask_snap(img_vv, nodata_threshold=5, upper_threshold=150):
        """
        Get the Sentinel-1 flood mask (snap-like) using a provided threshold
        :param img_vv: The VV-polarized image
        :param upper_threshold: The upper detection threshold
        :param nodata_threshold: The nodata threshold. Everything below is disregarded
        :return: The binary water mask at input resolution
        """
        # Mask the remaining no-data parts
        img_vv[img_vv < nodata_threshold] = np.iinfo(img_vv.dtype).max
        return img_vv < upper_threshold

    def get_products(self, start_date, end_date, product_type="S2_MSI_L2A", **kwargs):
        """
        Get a list of eodag-products for a specific roi and date window
        :param product_type: The eodag product type. Default is S2_MSI_L1C for peps
        :param start_date: The start date as date object
        :param end_date: The end date as date object
        :return: The list of products as well as the amount of estimated products found.
        """

        cloud_cover = kwargs.get("cloudCover", None)
        products, estimated_total_nbr_of_results = self.dag.search(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            box=self.extent_json,
            productType=product_type,
            cloudCover=cloud_cover,
            items_per_page=200
        )
        return products, estimated_total_nbr_of_results

    def get_product_data(self, product, band, resolution=10, crs="epsg:4326"):
        """
        Get a specific band from a specific product
        :param product: The EOProduct object
        :param band: The band ID. E.g. 'B03'
        :param resolution: The resolution. Default is 10[m].
        :param crs: The coordinate reference system code. Default is EPSG:4325
        :return: The file as xarray
        """
        assert crs.split(":")[0].lower() == "epsg"
        return product.get_data(crs=crs, resolution=resolution, band=band, extent=self.extent_tup)

    @staticmethod
    def get_product_date(product):
        """
        Get the date of an EOProduct as date object.
        :param product:
        :return: date object of the acquisition date
        """
        from dateutil import parser
        return parser.parse(product.as_dict()['properties']['creationDate'])

    @staticmethod
    def get_ifile_path(product_path, product_type, band_code, **kwargs):
        """
        Get the images with trigram and file type inside the product directory
        :param product_path: The root product folder
        :param product_type: The EOProduct type
        :param band_code: The band code, e.g. 'B03'
        :return: The first result's full path to the selected file. OSError if not.
        """
        import re
        file_type = kwargs.get("file_type", "(tif|tiff)")
        if band_code[0] == "B":
            # Remove leading zero if existing
            band_number = "".join(["B", band_code[2:] if band_code[1] == "0" else band_code[1:]])
        else:
            band_number = band_code
        regex_by_ptype = {"S2_MSI_L1C": r"\w+_%s\.%s" % (band_code, file_type),
                          "S1_SAR_GRD": r"[\w-]+%s[\w-]+\.%s" % (band_code, file_type),
                          "S2_MSI_L2A": r"\w+_%s\w*\.%s" % (band_number, file_type)}

        reg = regex_by_ptype[product_type]
        for path, dirs, files in os.walk(product_path, followlinks=True):
            for fl in files:
                if re.search(reg, fl):
                    return p.join(path, fl)

        raise FileNotFoundError("Cannot find any filename in %s with: band %s; file type %s"
                                % (product_path, band_code, file_type))

    def reproject_to_crs(self, img_in, crs, extent):
        """
        Reproject an image to a defined CRS using gdalwarp as a subprocess
        :param img_in: The full path to the image
        :param crs: The desired output CRS
        :param extent: The extent in lat/lon
        :return: The filename to the newly created image
        """
        import subprocess
        env = {**os.environ}
        # Conda prepends a windows path inside linux's PATH variable. Need to force the use of v-env executable:
        # TODO Find a way to remove the windows path by default
        env["PATH"] = str(env["PATH"]).split(";")[-1]
        img_out = p.join(self.workspace, "".join([p.basename(img_in).split(".")[0], "_proj.tif"]))
        cmd = [
            "gdalwarp",
            "-q",
            "-overwrite",
            "-r",
            "bilinear",
            "-tr",
            "10",
            "10",
            "-t_srs",
            crs,
            "-te_srs",
            "EPSG:4326",
            "-te"] + extent + [img_in, img_out]
        process = subprocess.Popen(cmd, shell=False, env=env)
        process.communicate()
        assert process.returncode == 0

        return img_out

    def preprocess_roi(self, product_root, product_type, used_bands):
        """
        Open and resize an ROI of a selected product
        :param product_root:
        :param product_type:
        :param used_bands:
        :return:
        """
        from Common import ImageIO
        file_endings = {"S2_MSI_L1C": "jp2",
                        "S1_SAR_GRD": "tiff",
                        "S2_MSI_L2A": "tif"}
        file_type = file_endings.get(product_type, "tif")
        bands = [self.get_ifile_path(product_root,
                                     product_type,
                                     b,
                                     file_type=file_type) for b in used_bands]
        reprojected_bands = [self.reproject_to_crs(b, self.crs, self.extent_lst) for b in bands]
        rasters = [ImageIO.tiffToArray(b)[0] for b in reprojected_bands]
        driver = ImageIO.openTiff(reprojected_bands[0])
        return rasters, driver

    @staticmethod
    def determine_nodata(raster, nodata_threshold=5):
        """
        Determine the percentage of nodata-pixels inside a raster
        :param raster: The 2D numpy array
        :param nodata_threshold: The nodata threshold under which everything is disregarded
        :return: A value between 0 and 1, with 1 meaning the complete image contains only nodata values
        """
        assert raster.ndim == 2
        # TODO Find a way to compare to a fixed value instead of a threshold
        return np.count_nonzero(raster <= nodata_threshold) / (raster.shape[0] * raster.shape[1])

    def determine_epsg(self, s2_product):
        """
        Determine the EPSG code of the underlying s2 product
        An example can be found here:

        https://gis.stackexchange.com/questions/267321/extracting-epsg-from-a-raster-using-gdal-bindings-in-python

        :param s2_product: The root of the S2 product
        :return: The EPSG code of the product as str, e.g. "EPSG:4326"
        """
        # TODO Do not have this as cfg param but deduct it from the s2 product instead
        return self.cfg_file.get("PROCESSING", "Reference_System")

    def get_cloud_mask(self, s2_product):
        """
        Get the path to a cloud mask if existing
        :param s2_product: The root of the S2_product
        :return: The full path to the cloud mask if existing, None if not
        """
        try:
            return self.get_ifile_path(s2_product, self._s2_product_type, self.cloudmask_code, file_type="tif")
        except OSError:
            return None

    def process_cloud_mask(self, product, rasters, no_data_value=-10000):
        """
        Read and apply the cloud mask to all raster
        :param product: The EOProduct
        :param rasters: The raster in uniform resolution already read
        :param no_data_value: The no data value to put. Default is -10000.
        :return: The same raster but with the cloud mask applied
        """

        cloud_masks, _ = self.preprocess_roi(product, self._s2_product_type, [self.cloudmask_code])
        # Mask rasters
        for raster in rasters:
            # Check that sizes of mask an raster are consistent
            assert cloud_masks[0].shape == raster.shape
            raster[cloud_masks[0] > 0] = no_data_value
        return rasters

    def write_hash_files(self, products):
        """
        Write the eodag hashfile in advance to allow external downloads
        :param products: The eodag product list
        :return:
        """
        import hashlib
        for product in products:
            url = product.as_dict()['properties']['downloadLink']
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
            record_filename = p.join(self.__downloaded, url_hash)
            if not p.isfile(record_filename):
                with open(record_filename, 'w') as fh:
                    fh.write(url)
        return

    def acquire_s1(self):
        """
        Download an s1 time series from the given provider
        :return: The EOProducts (cf. eodag doc) and the path to the root of each product
        """
        s1_products, _ = self.get_products(start_date=self.start_date,
                                           end_date=self.end_date,
                                           product_type=self._s1_product_type)
        self.write_hash_files(s1_products)
        s1_product_paths = []
        for product in s1_products:
            # Need to catch eventual eodag download-errors here:
            try:
                s1_product_paths.append(self.dag.download(product))
            except NotADirectoryError:
                continue

        # Remove the eodag file prefix if necessary
        s1_product_paths = [path[len(self._eodag_file_prefix):] if path.startswith(self._eodag_file_prefix) else path
                            for path in s1_product_paths]
        return s1_products, s1_product_paths

    def process_s1(self, s1_products, s1_product_paths):
        """
        Process a fixed time series for s1:
         - Download products from the given provider
         Then for each:
         - Preprocess the rasters
         - Determine the water extent
         - Save the flood extent and acquisition date in two separate arrays
        :return: The water extent for each acquired s1 product
        """
        from Common import ImageIO

        s1_dates, s1_extent = [], []
        # Main loop for Sentinel-1:
        for idx, product in enumerate(s1_product_paths):
            # Need to catch eventual eodag extraction-errors here:
            try:
                rasters, driver = self.preprocess_roi(product, self._s1_product_type, ["vv"])
            except FileNotFoundError:
                continue
            if self.determine_nodata(rasters[0], nodata_threshold=5) >= self.max_nodata:
                print("Product contains too much nodata. Skipping...")
                continue
            mask_radar_bin = self.get_s1_flood_mask_snap(rasters[0], upper_threshold=self.snap_threshold)
            mask_radar = np.array(mask_radar_bin * 255, np.uint8)
            flood_count = np.count_nonzero(mask_radar_bin)
            acq_date = self.get_product_date(s1_products[idx])
            s1_dates.append(acq_date)
            s1_extent.append(flood_count)
            file_name = p.join(self.workspace, "%s_radar_%s.tif" % (self.site_id, acq_date.strftime("%Y%m%d_%H%M")))
            ImageIO.writeGTiffExisting(mask_radar, driver, file_name)
        s1_sorted = sorted(zip(s1_dates, s1_extent))
        s1_dates, s1_extent = [t[0] for t in s1_sorted], [t[1] for t in s1_sorted]
        return s1_dates, s1_extent

    def acquire_s2(self):
        """
        Download an s2 time series from the given provider
        :return: The EOProducts (cf. eodag doc) and the path to the root of each product
        """
        s2_products, _ = self.get_products(start_date=self.start_date,
                                           end_date=self.end_date,
                                           product_type=self._s2_product_type,
                                           cloudCover=self.max_cloud_cover)
        self.write_hash_files(s2_products)
        s2_product_paths = []
        for product in s2_products:
            print(product.as_dict()['id'])
            # Need to catch eventual eodag download-errors here:
            try:
                s2_product_paths.append(self.dag.download(product))
            except NotADirectoryError:
                continue

        # Remove the eodag file prefix if necessary
        s2_product_paths = [path[len(self._eodag_file_prefix):] if path.startswith(self._eodag_file_prefix) else path
                            for path in s2_product_paths]
        return s2_products, s2_product_paths

    def process_s2(self, s2_products, s2_product_paths):
        """
        Process an acquired time series for s2:
        For each:
         - Pre process each raster
         - Determine the water extent
         - Save the flood extent and acquisition date in two separate arrays
        :param s2_products: The EOProduct list
        :param s2_product_paths: The path to the EOProduct root folder
        :return: The water extent for each acquired s2 product
        """
        from Common import ImageIO

        s2_dates, s2_extent = [], []
        # Main loop for Sentinel-2:
        for idx, product in enumerate(s2_product_paths):
            # Need to catch eventual eodag extraction-errors here:
            try:
                rasters, driver = self.preprocess_roi(product, self._s2_product_type, self._used_bands)
            except FileNotFoundError:
                continue
            clm_path = self.get_cloud_mask(product)
            if clm_path:
                # Apply the cloud mask
                rasters = self.process_cloud_mask(product, rasters)
            if self.determine_nodata(rasters[0], nodata_threshold=1) >= self.max_nodata:
                print("Product contains too much nodata. Skipping...")
                continue
            img_nir = rasters[self._used_bands.index(self.nir_code)]
            img_swir = rasters[self._used_bands.index(self.swir_code)]
            img_blue = rasters[self._used_bands.index(self.blue_code)]
            img_green = rasters[self._used_bands.index(self.green_code)]
            mask_optical_bin = self.get_s2_flood_mask_swm(img_blue=img_blue,
                                                          img_green=img_green,
                                                          img_swir=img_swir,
                                                          img_nir=img_nir)
            mask_optical = np.array(mask_optical_bin * 255, np.uint8)
            flood_count = np.count_nonzero(mask_optical_bin)
            acq_date = self.get_product_date(s2_products[idx])
            s2_dates.append(acq_date)
            s2_extent.append(flood_count)
            file_name = p.join(self.workspace, "%s_optical_%s.tif" % (self.site_id, acq_date.strftime("%Y%m%d_%H%M")))
            ImageIO.writeGTiffExisting(mask_optical, driver, file_name)
        s2_sorted = sorted(zip(s2_dates, s2_extent))
        s2_dates, s2_extent = [t[0] for t in s2_sorted], [t[1] for t in s2_sorted]
        return s2_dates, s2_extent

    def get_max_extent(self, extents, threshold=0.1):
        """
        Get the maximum extent of the extent by thresholding.
        :param extents: The extent per date
        :param threshold: The threshold above which the maximum extent is saved.
        :return:
        """
        import numpy as np
        max_extent = np.max(extents) * (1 - threshold)
        max_extent_reached = ["Flood" if extent > max_extent else "Normal" for extent in extents]
        return max_extent_reached

    def plot_extent(self, *plots):
        """
        Plots the water-extent for a time-series
        :param plots: The time series to be plotted with x- and y- data in two separate lists of same length
        :return: A png file in the current directory will be created
        """
        from matplotlib import pyplot as plt
        import matplotlib.ticker as mticker
        import matplotlib.dates as mdates

        plt.plot(*plots, '-', linewidth=1)
        plt.title("Location: Lat %.3f | Lon %.3f" % self.extent_center)
        plt.legend(["Sentinel-1", "Sentinel-2"])
        plt.ylabel("Water extent [10x10m^2]")
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.gcf().subplots_adjust(bottom=.2, left=.2)  # Fixes the correct display of legends and axis-labels
        plt.savefig("./plot_%s.png" % self.site_id)

    def save_as_csv(self, product_type, dates, extents, extent_indicator):
        """
        Save the extent per day in a csv file.
        The header contains the following information:
        - Extent: The center lon/lat of the bounding box used
        - Satellite name
        - Water body code (USGS/Hydroweb)
        Then the each line corresponds to:
        - The acquisition date as utc
        - The measured extent in 10x10m^2
        :param product_type: The associated product type (S1 or S2)
        :param dates: The list of acquisition dates
        :param extents: The list of extents
        :return: A csv file in the current directory will be written
        """
        file_name = "_".join(["extent", self.site_id, product_type + ".csv"])
        with open(file_name, "w") as csv_file:
            header = ",".join(["Time", "Lat", "Lon", "SW_Extent", "Label", "Type", "Source", "Name", "Notes"])
            csv_file.write(header)
            csv_file.write("\n")
            for d, e, i in zip(dates, extents, extent_indicator):
                date_as_utc = d.strftime("%Y-%m-%d %H:%M:%S")
                csv_file.write(",".join([date_as_utc,
                                         str(self.extent_center[1]), str(self.extent_center[0]),
                                         str(e), str(i), "Satellite", "PEPS", self.site_name, "-"]))
                csv_file.write("\n")
        return

    def run(self):
        """
        Run the whole artillery ;)
        :return:
        """

        s1_products, s1_product_paths = self.acquire_s1()
        s2_products, s2_product_paths = self.acquire_s2()
        self.crs = self.determine_epsg(s2_product_paths[0])
        s2_dates, s2_extent = self.process_s2(s2_products, s2_product_paths)
        s2_extent_indicator = self.get_max_extent(s2_extent)
        s1_dates, s1_extent = self.process_s1(s1_products, s1_product_paths)
        s1_extent_indicator = self.get_max_extent(s1_extent)
        self.plot_extent(s1_dates, s1_extent, s2_dates, s2_extent)
        self.save_as_csv(self._s1_product_type, s1_dates, s1_extent, s1_extent_indicator)
        self.save_as_csv(self._s2_product_type, s2_dates, s2_extent, s2_extent_indicator)


if __name__ == '__main__':

    import argparse
    # Check if python version is above or equal 3.5
    assert sys.version_info >= (3, 5)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The config file. Required.", required=True, type=str)
    args = parser.parse_args()

    f = FloodExtentEstimator(args.config)
    f.run()
