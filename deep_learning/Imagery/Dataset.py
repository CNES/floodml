#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
import re
import numpy as np
from pandas import DataFrame
import collections
from tqdm import tqdm
import json
import gc
from functools import reduce
from datetime import datetime, timedelta
from Common.GDalDatasetWrapper import GDalDatasetWrapper
from Chain import Product
from Common import ImageTools, FileSystem


class Dataset(object):
    """
    Sets up the gsw-o dataset from a set of L1C products and L2 masks
    """

    max_timedelta = timedelta(hours=1)

    reg_masks = {"gsw": r"^\d{2}[A-Z]{3}.tif$",
                 "ems": r"\d{2}[A-Z]{3}_\d{8}T\d{6}.tif$"}
    max_nodata = 90

    def __init__(self, cfg_file, mode, backup_json, pickup_json):
        self.cfg_file = cfg_file
        assert mode in ["training", "validation"], "Mode needs to be 'training' or 'validation': %s" % mode
        self.mode = mode
        # Parsing configuration file
        with open(cfg_file, 'r') as json_file:
            self.args = json.load(json_file)
        train_dict, val_dict = [[{"mask": algo.lower(),
                                  "class": c,
                                  "path": self.args["path"][algo.lower()],
                                  "vtype": self.args["mask_types"][algo.lower()][c]["type"],
                                  "value": self.args["mask_types"][algo.lower()][c]["value"]}
                                 for c in self.args["ground_truth"]["classes"]
                                 for algo in self.args["ground_truth"][i]] for i in ["train", "val"]]
        if self.mode == "training":
            self.minfo = train_dict
        else:
            self.minfo = val_dict
        self.backup_json = backup_json
        self.prod_pairs = pickup_json

    @staticmethod
    def get_available_products(root, **kwargs):
        """
        Parse the products from the constructed L1- or L2- directories
        :param root: The root folder to be searched from
        :keyword tiles_excluded: The tileID's to be excluded from the search ['T31TCJ', 'SUDOUE-5', ...]
        :return: A list of MajaProducts available in the given directory
        """
        platforms = kwargs.get("platforms", [])
        tiles_excluded = kwargs.get("tiles_excluded", [])
        avail_files = [os.path.join(r, f) for r, dirs, files in os.walk(root, followlinks=True) for f in dirs + files]
        avail_files.append(root)
        avail_products = [Product.MajaProduct.factory(f) for f in avail_files]
        # Remove the ones that didn't work:
        avail_products = [prod for prod in avail_products if prod is not None]
        # Remove the excluded tiles, product levels and platforms:
        prods_filtered = [prod for prod in avail_products if
                          prod.level in kwargs.get("levels", ["l1c", "l2a"]) and
                          prod.tile not in tiles_excluded]
        prods_filtered = sorted(list(prods_filtered))
        if platforms:
            return [prod for prod in prods_filtered if prod.short_name in platforms]
        return prods_filtered

    @staticmethod
    def __get_mask_file_info(path):
        """
        Get the tile and datetime info from a file path, if it can be found inside the path.
        :param path: The path to the given mask file
        :return: The datetime and tile info
        """
        date_pattern = [r"\d{8}T\d{6}", r"\d{8}-\d{6}-"]
        tile_pattern = r"\d{2}[A-Z]{3}"
        date_raw = []
        for pat in date_pattern:
            date_raw += re.findall(pat, path)
        if date_raw:
            date_str = date_raw[0].replace("T", "").replace("-", "")
            date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
        else:
            date = "all"
        tile = str(re.findall(tile_pattern, path)[-1])
        return date, tile

    def _get_available_masks(self, algorithm):
        """
        Get the available masks for a given algorithm in the specified folder
        :param algorithm: The algorithm e.g. GSW or EMS
        :return: A list of dict for each mask with the following items:
                 - tile
                 - date
                 - path
        """
        reg_algo = self.reg_masks[algorithm["mask"]]
        try:
            avail_masks = FileSystem.find(pattern=reg_algo, path=algorithm["path"], case_sensitive=False)
        except ValueError:
            avail_masks = []
        mask_products = []
        for m in avail_masks:
            date, tile = self.__get_mask_file_info(m)
            mask_products.append({"tile": tile, "date": date, "type": algorithm["mask"], "path": m})
        return mask_products

    @staticmethod
    def _get_mnt(mnt_path, tile, **kwargs):
        """
        Scan an mnt available for the current tile
        :param tile: The tile id
        :param mnt_path: The folder where the DEM/MNT's are stored.
        :return: The path to the following mnt files: _ASP_, _ALT_ and _SLP_
        """
        mnt_band = kwargs.get("mnt_band", None)
        mnt_origin, mnt_type = mnt_band.split("_")
        if mnt_origin == "srtm":
            version = "(1001|0001)"
        elif mnt_origin == "merit":
            version = "2001"
        else:
            raise ValueError("Unknown mnt type: %s" % mnt_band)
        reg_mnt = {"ALT": r"\w+_AUX_REF\w+_T?%s_%s_(ALT|ALT_R1).TIF$",
                   "ASP": r"\w+_AUX_REF\w+_T?%s_%s_(ASP|ASP_R1).TIF$",
                   "SLP": r"\w+_AUX_REF\w+_T?%s_%s_(SLP|SLP_R1).TIF$",
                   "WAT": r"\w+_AUX_REF\w+_T?%s_%s_(MSK|MSK_R1).TIF$"}

        files = [os.path.join(r, file) for r, _, f in os.walk(mnt_path, followlinks=True) for file in f]
        reg_file = reg_mnt[mnt_type.upper()] % (tile, version)
        matching_files = [f for f in files if re.search(reg_file, f)]
        if not matching_files:
            raise FileNotFoundError("Cannot find Mask %s for tile %s and type %s in %s" % (mnt_band, tile, mnt_type,
                                                                                           mnt_path))
        if len(matching_files) > 1:
            print("WARNING: More than one match found for file %s: %s" % (mnt_band, matching_files))
        mnt_file = matching_files[0]
        return {"type": mnt_band,
                "path": mnt_file}

    @staticmethod
    def _get_additional_band(band, path, **kwargs):
        """
        Get an additional band by its name.

        :param band: A band specifier, e.g. "HAND" or "OSM"
        :param path: The path where the files of the given band type are located.
        :return: A type-path dict
        """

        if band.lower() == "hand":
            tile = kwargs["tile"]
            band_found = FileSystem.find_single(pattern="%s.tif" % tile, path=path)
            return {"type": band, "path": band_found}

        raise ValueError("Unknown band encountered: %s" % band)

    @staticmethod
    def create_img_dict(args, product):
        """
        Create a dict with the following items:
        - tile,
        - date,
        - rasters, a list of dicts:
            - type
            - path
        :param args: The config file arguments `Common.Arguments.Arguments`
        :param product: A `Chain.Product.MajaProduct`
        :return: The dict for the given product
        """
        rasters = []
        for band in args["inputs"]["bands_used"]:
            short_name, band_idx = band.split("_")
            if product.short_name != short_name:
                continue
            band_path = product.find_file(pattern="*" + band_idx + "*.(tif|jp2)$")[0]
            rasters.append({"type": band_idx,
                            "path": band_path})
        for band in args["inputs"]["synthetic_bands"]:
            short_name, band_idx = band.split("_")
            if product.short_name != short_name:
                continue
            band_path = product.get_synthetic_band(band_idx, wdir=args["path"]["wdir"])
            rasters.append({"type": band_idx,
                            "path": band_path})

        # If no rasters present, then don't return anything:
        if not rasters:
            return {}

        mnt = []
        for mtype in args["inputs"]["mnt_used"]:
            mnt_origin = mtype.split("_")[0]
            try:
                mnt.append(Dataset._get_mnt(args["path"]["mnt"], product.tile, mnt_band=mtype))
            except FileNotFoundError:
                product.get_mnt(dem_dir=args["path"]["mnt"], type_dem=mnt_origin, full_res_only=True,
                                raw_dem=args["path"][mnt_origin])
                mnt.append(Dataset._get_mnt(args["path"]["mnt"], product.tile, mnt_band=mtype))

        additional_bands = []
        for band_dict in args["inputs"]["additional_bands"]:
            additional_bands.append(Dataset._get_additional_band(band=band_dict["band"], path=band_dict["path"],
                                                                 tile=product.tile))

        # Please note:
        # The order of the rasters has to stay the same for training and prediction!
        img_dict = {"tile": product.tile,
                    "date": product.date.strftime("%Y%m%dT%H%M%S"),
                    "nodata": product.nodata,
                    "rasters": rasters + mnt + additional_bands,
                    }
        return img_dict

    def get_all_product_pairs(self, root, algorithms, *tiles_excluded):
        """
        Get all product pairs
        :param root: The path to the masks and products (training or validation)
        :param algorithms: A pandas dataframe
        Uses:
        - The list of algorithms to be used
        - The list of synthetic bands
        - Indicator for usage of MNT
        - The list of tiles to be excluded
        :return: A dict with the following items:
                 - tile,
                 - date,
                 - algo,
                 - rasters, a list of dicts:
                    - type
                    - path
                - mask path
        """

        platforms = list(set(b.split("_")[0] for b in
                             self.args["inputs"]["bands_used"] +
                             self.args["inputs"]["synthetic_bands"]))
        images = self.get_available_products(root, platforms=platforms)
        masks = [self._get_available_masks(algo) for algo in algorithms]
        # Flatten the 2D array
        masks = [m for algo in masks for m in algo]
        image_and_mask_pairs = []
        # Get all available tiles
        available_tiles = [prod["tile"] for prod in masks] + [prod.tile for prod in images]
        # Filter out the excluded tiles
        available_tiles = [t for t in available_tiles if t not in tiles_excluded]
        if len(images) < 1 or len(masks) < 1:
            print("No images/masks available for %s masks" % (len(available_tiles)))
            return image_and_mask_pairs
        # The goal is to merge the two 'images' and 'masks' arrays and finding the common dates and tiles
        # There are two possibilities: Take each mask and search for its image or vice versa.
        # We chose the first option:
        print("Found %s masks. Creating image/mask pairs" % len(masks))
        for m in tqdm(masks):
            tile, date, algo, mask_path = m["tile"], m["date"], m["type"], m["path"]
            if tile not in available_tiles:
                continue
            is_single_date_mask = True if date is not "all" else False
            # Find one or multiple corresponding products while including any date for the global GSW masks
            corresponding_prods = []
            for i in images:
                if i.tile == tile:
                    if date == "all":
                        corresponding_prods.append(i)
                    elif abs(i.date - date) <= self.max_timedelta:
                        corresponding_prods.append(i)
                    elif i.date.strftime("%H%M%S") == "000000" and \
                            i.date.strftime("%Y%m%d") == date.strftime("%Y%m%d"):
                        # This case corresponds to a product which does not have any info about the acquisition hour
                        corresponding_prods.append(i)
                    else:
                        # Do not use this product and continue
                        pass
            # Filter by band. The end result corresponds to a list of dicts with the bands
            # in the same order as in the config file.
            if len(corresponding_prods) == 0:
                continue
            elif len(corresponding_prods) > 1 and is_single_date_mask:
                print("WARNING: More than one corresponding product found for tile %s, date %s and mask type %s" %
                      (tile, date, algo))
                # If more than one, take the youngest product
                corresponding_prods = [sorted(corresponding_prods)[0]]
            for prod in corresponding_prods:
                # Create the img dict
                img_dict = self.create_img_dict(self.args, prod)
                # If no rasters present, then skip:
                if not img_dict:
                    continue
                img_dict["algo"] = algo
                img_dict["mask"] = mask_path
                image_and_mask_pairs.append(img_dict)
        return image_and_mask_pairs

    @staticmethod
    def process_rasters(args, img_mask_pair, epsg, extent, retile=True, **kwargs):
        """
        Process a set of raster files.
        First resizes the input to the desired resolution. Then optionally normalizes it to [-1, 1].
        Finally, it creats overlapping image patches using `Common.ImageIO.gdal_retile`
        :param args: The config file arguments `Common.Arguments.Arguments`
        :param img_mask_pair: A dict with the following items:
                 - tile,
                 - date,
                 - algo,
                 - rasters, a list of dicts:
                    - type
                    - path
                - mask path
        :param epsg: The epsg code to be used for the reprojection
        :param extent: The reprojection extent in format [xmin ymin xmax ymax]
        :param retile: If True: Returned patches of images (written to disk); otherwise return full image as ds
        :return: If retile: The list paths to the resized, normalized and tiled sub-rasters,
                 else: A :class:`Common.GDalDatasetWrapper.GDalDatasetWrapper` object containing the full image array
                 of size (x, y, n_bands)
        :keyword output_dir: The directory to write the tiled sub-rasters to.
        """
        nodata_mask = kwargs.get("nodata_mask", None)
        rasters = img_mask_pair["rasters"]
        resized_datasets = []
        bands = []
        tile, date, algo = img_mask_pair["tile"], img_mask_pair["date"], img_mask_pair["algo"]
        for raster in rasters:
            band = raster["type"]
            bands.append(band)
            path = raster["path"]
            tr_str = "%s %s" % (args["preprocessing"]["coarse_res_m"], args["preprocessing"]["coarse_res_m"])
            ds_resized = ImageTools.gdal_warp(path,
                                              t_srs="EPSG:%s" % epsg,
                                              tr=tr_str,
                                              te=extent,
                                              r="bilinear",
                                              multi=True,
                                              wdir=args["path"]["wdir"])
            if nodata_mask is not None:
                img_cut = np.where(nodata_mask > 0, ds_resized.array, 0)
                resized_datasets.append(GDalDatasetWrapper(ds=ds_resized.get_ds(), array=img_cut))
            else:
                resized_datasets.append(ds_resized)

        ds_combined = ImageTools.gdal_merge(*resized_datasets, separate=True, q=True)
        if retile:
            output_dir = kwargs.get("output_dir", None)
            if not output_dir:
                raise KeyError("Must provide parameter 'output_dir' for retile to work.")
            tile_size, overlap = args["preprocessing"]["tile_size"], args["preprocessing"]["overlap"]
            patches = ImageTools.gdal_retile(ds_combined, output_dir,
                                             filename_base="img_%s_%s_%s" % (tile, date, algo),
                                             TileWidth=tile_size,
                                             TileHeight=tile_size,
                                             Overlap=overlap,
                                             AddPadding=True,
                                             CreateOptions=["COMPRESS=DEFLATE"],
                                             Quiet=True)
            return patches
        return ds_combined

    def process_mask(self, img_mask_pair, output_dir, algorithms, selected_classes, epsg, extent, nodata_mask):
        """
        Process a single mask file.
        First resizes the input to the desired resolution. Then extracts the values of the selected class.
        Finally, it creats overlapping image patches using :func:`Common.ImageTools.gdal_retile`

        :param img_mask_pair: A dict with the following items:
                 - tile,
                 - date,
                 - algo,
                 - rasters, a list of dicts:
                    - type
                    - path
                - mask path
        :param algorithms: The training or validation algorithm dict
        :param output_dir: The directory to write the tiled sub-rasters to.
        :param selected_classes: The list of names of selected classes, e.g. [Water] or [Cloud, Shadow]
        :param epsg: The epsg code to be used for the reprojection
        :param extent: The reprojection extent in format [xmin ymin xmax ymax]
        :param nodata_mask: A GDalDataset containing a binary nodata mask, where True==Data, False==Nodata
        :return: The paths to the resized, normalized and tiled sub-rasters
        """
        msk_path = img_mask_pair["mask"]
        tile, date = img_mask_pair["tile"], img_mask_pair["date"]
        algo = img_mask_pair["algo"]
        resized_datasets = []
        for selected in selected_classes:
            selected_algorithm = [a for a in algorithms if a["mask"] == algo and a["class"] == selected]
            assert len(selected_algorithm) == 1, "Error while finding mask info for class %s and algorithm %s" \
                                                 % (selected_algorithm, algo)
            selected_algorithm = selected_algorithm[0]
            tr_str = "%s %s" % (self.args["preprocessing"]["coarse_res_m"], self.args["preprocessing"]["coarse_res_m"])
            ds_resized = ImageTools.gdal_warp(msk_path,
                                              t_srs="EPSG:%s" % epsg,
                                              tr=tr_str,
                                              te=extent,
                                              r="near",
                                              multi=True,
                                              wdir=self.args["path"]["wdir"]
                                              )
            msk_extracted = ImageTools.extract_class(ds_resized.array,
                                                     selected_algorithm["value"],
                                                     selected_algorithm["vtype"])

            # Cut mask where no data in original image:
            msk_cut = np.where(nodata_mask > 0, msk_extracted, 0)
            resized_datasets.append(GDalDatasetWrapper(ds=ds_resized.get_ds(), array=msk_cut))

        ds_combined = ImageTools.gdal_merge(*resized_datasets, separate=True, q=True)
        tile_size, overlap = self.args["preprocessing"]["tile_size"], self.args["preprocessing"]["overlap"]
        patches = ImageTools.gdal_retile(ds_combined, output_dir,
                                         filename_base="msk_%s_%s_%s" % (tile, date, algo),
                                         TileWidth=tile_size,
                                         TileHeight=tile_size,
                                         Overlap=overlap,
                                         AddPadding=True,
                                         CreateOptions=["COMPRESS=DEFLATE"],
                                         Quiet=True)
        return patches

    def _check_imgs_processed(self, tile, date, tiles_dates_processed):
        """
        Check if a given date and tile were already processed
        :param tile: The tile ID
        :param date: The datetime
        :param tiles_dates_processed: Dict of lists with tiles as key
        :return:
        """
        tiles = tiles_dates_processed.keys()
        if tile not in tiles:
            return False
        dates = tiles_dates_processed[tile]
        for dt in dates:
            if abs(date - dt) <= self.max_timedelta:
                return True
        return False

    def build_set(self, dst, img_mask_pair, algorithms, selected_classes):
        """
        Reads a single img-msk-pair and calls the pre-processing
        sub-routines `preprocess_rasters` and `preprocess_mask`.
        The fixed-tile size images are written into root_img and root_msk with
        optionalcleaning of the directories before.

        In some cases, an img-msk-pair already has its image processed.
        To gain some time, this img pre-processing is then skipped and the resulting filepaths are simply appended.
        :param img_mask_pair: A dict with the following items:
                 - tile,
                 - date,
                 - algo,
                 - rasters, a list of dicts:
                    - type
                    - path
                - mask path
        :param dst: The folder to write the images to.
        :param algorithms: The full dict of algorithms used for training/validation:
                            - algorithm name (e.g. maja, fmask):
                              - value type: bit or value
                              - class-name-1: List of integers
                              - class-name-np: List of integers
        :param selected_classes: The selected classes for extraction. E.g. [Water] or [Cloud, Shadow]
        :return: Writes a set of fixed-size geo-referenced .tif-tiles to the `tiled_path` directory.
        """
        date, tile, mtype = img_mask_pair["date"], img_mask_pair["tile"], img_mask_pair["algo"]
        root_rasters = os.path.join(dst, "_".join([tile, date]))
        FileSystem.create_directory(root_rasters)
        ds = GDalDatasetWrapper.from_file(img_mask_pair["rasters"][0]["path"])
        uniform_epsg = ds.epsg
        extent = ds.extent()
        uniform_extent_str = " ".join([str(ex) for ex in extent])
        tr_str = "%s %s" % (self.args["preprocessing"]["coarse_res_m"], self.args["preprocessing"]["coarse_res_m"])
        # In case of EMS, use nodata mask from gt:
        if mtype == "ems":
            ds_nodata = ImageTools.gdal_warp(img_mask_pair["mask"], r="near", tr=tr_str, te=uniform_extent_str,
                                             dstnodata=img_mask_pair["nodata"],
                                             wdir=self.args["path"]["wdir"]
                                             )
            nodata_mask = np.bitwise_and(ds_nodata.array, 2)
        else:
            ds_nodata = ImageTools.gdal_warp(ds, tr=tr_str, te=uniform_extent_str,
                                             r="cubic", dstnodata=img_mask_pair["nodata"],
                                             wdir=self.args["path"]["wdir"]
                                             )
            nodata_mask = ds_nodata.nodata_mask
        # Open, resize, extract values and split all masks:
        msks = self.process_mask(img_mask_pair,
                                 output_dir=root_rasters,
                                 selected_classes=selected_classes,
                                 algorithms=algorithms,
                                 epsg=uniform_epsg,
                                 extent=uniform_extent_str,
                                 nodata_mask=nodata_mask)
        # Open, Resize, normalize and split all bands:
        imgs = self.process_rasters(self.args,
                                    img_mask_pair,
                                    epsg=uniform_epsg,
                                    extent=uniform_extent_str,
                                    retile=True,
                                    output_dir=root_rasters,
                                    nodata_mask=nodata_mask)
        gc.collect()
        # The two lengths should be equal
        assert len(msks) == len(imgs)
        return imgs, msks

    def remove_empty_patches(self, df):
        """
        Remove patches that are fully or almost empty

        :param df: The dataframe containing in each row the path to an image and a mask
        :return:
        """

        for index, row in df.iterrows():
            try:
                ds_img = GDalDatasetWrapper.from_file(row["img"])
            except NameError:
                df.drop([index], inplace=True)
                continue
            # Count number of empty pixels
            n_pix = reduce(lambda x, y: x * y, list(ds_img.array.shape))
            n_nonzero = np.count_nonzero(ds_img.array)
            n_pix_nodata = 100. * (1 - (n_nonzero / n_pix))
            # Skip empty ones
            if n_pix_nodata > self.max_nodata:
                print("Skipping empty raster {}. Nodata: {:.2f}%%".format(row["img"], n_pix_nodata))
                FileSystem.remove_file(row["img"])
                FileSystem.remove_file(row["masks"])
                df.drop([index], inplace=True)
        return df

    def write_data_frame(self, dst, df, output_type):
        """
        Writes a csv file containing all img and msk paths to the "Results/csv" directory of the given root.
        :param dst: The destination directory
        :param df: The dataframe containing the filenames for imgs masks in each column
        :param output_type: 'training' or 'validation'
        :return: The path to the csv that was written
        """
        classes = self.args["ground_truth"]["classes"]
        filename = "_".join([output_type, datetime.now().strftime('%Y%m%d_%H%M%S')] + classes) + ".csv"
        full_path = os.path.join(dst, filename)
        df.to_csv(full_path, index=False)
        return full_path

    def run(self):
        """
        Run the whole dataset generation
        :return: Creates the dataset to the specified cfg paths
        """

        # Create working-directory
        FileSystem.create_directory(self.args["path"]["wdir"])

        root = self.args["path"][self.mode]
        assert os.path.exists(self.args["path"][self.mode]), "Cannot find path %s" % self.args["path"][self.mode]
        if self.prod_pairs:
            with open(self.prod_pairs, "r") as tmpjson:
                prod_pairs = json.load(tmpjson)
        else:
            print("Collecting images for %s" % self.mode)
            prod_pairs = self.get_all_product_pairs(root, self.minfo)
        if self.backup_json:
            bckp_path = os.path.join(os.getcwd(), "%s_%s.json" % (self.mode, datetime.now().strftime("%Y%m%dT%H%M%S")))
            with open(bckp_path, "w") as testjson:
                json.dump(prod_pairs, testjson, indent=4)
        print("Found %s pairs for %s" % (len(prod_pairs), self.mode))
        n_items = list(set([len(pair) for pair in prod_pairs]))
        if not n_items:
            print("WARNING: No products found.")
            return

        if len(n_items) > 1:
            raise ValueError("More than one configuration found for data items in list: %s" % n_items)
        print("Pre-processing files for %s" % self.mode)
        column_names = ["img", "masks"]
        all_imgs = {name.lower(): [] for name in column_names}

        # Create the folders to write the tiles in; delete old one:
        root = os.path.join(self.args["path"]["tiles"], self.mode)
        FileSystem.remove_directory(root)
        FileSystem.create_directory(root)

        for item in tqdm(range(len(prod_pairs))):
            imgs, msks = self.build_set(root, prod_pairs[item],
                                        self.minfo, self.args["ground_truth"]["classes"])
            # Append all rasters to dict
            for name, rasters in zip(column_names, [imgs, msks]):
                for raster in rasters:
                    all_imgs[name.lower()].append(raster)
        img_df = DataFrame(all_imgs)
        print("Created %s images." % (len(img_df)))

        # Remove empty patches:
        img_df = self.remove_empty_patches(img_df)

        csv_path = self.write_data_frame(root, img_df, self.mode)
        # Write config file
        config_path = os.path.join(root, "config.json")
        with open(os.path.join(root, "config.json"), "w") as cfgfile:
            new_cfg = self.args.copy()
            new_cfg.pop("path", None)
            new_cfg["ground_truth"].pop("train" if self.mode == "training" else "val", None)
            new_cfg["path"] = {"wdir": self.args["path"]["wdir"],
                               "csv": csv_path,
                               "results": os.path.abspath(os.path.join(root, "..", "results"))}
            od = collections.OrderedDict(sorted(new_cfg.items()))
            json.dump(od, cfgfile, indent=4)
        print("Created %s images. Config: %s" % (len(img_df), config_path))
