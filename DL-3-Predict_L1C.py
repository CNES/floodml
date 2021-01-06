#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import tempfile
import pandas
import tensorflow as tf
import argparse
import json
from tqdm import tqdm
from tensorflow.keras.models import load_model
from deep_learning.Imagery.Dataset import Dataset
from deep_learning.Imagery.Predict import Predict
from deep_learning.Imagery.EOSequence import EOSequence
from Common import ImageTools, FileSystem
from Common.GDalDatasetWrapper import GDalDatasetWrapper


class PredictL1C(object):
    """
    Predict on a set of L1C products by first preprocessing all available products
    """

    def __init__(self, cfg_file, product_dir, output_dir, **kwargs):
        self.cfg_file = cfg_file
        self.product_dir = product_dir
        self.output_dir = output_dir
        # Parsing configuration file
        with open(cfg_file, 'r') as json_file:
            self.args = json.load(json_file)
        # Get model file:
        model_cfg = kwargs.get("model", None)
        if not model_cfg:
            results = os.path.join(self.args["path"]["tiles"], "results")
            model_cfg = Predict.get_latest_model(os.path.join(results, "model"))[0]

        with open(model_cfg, 'r') as json_file:
            model_info = json.load(json_file)
            self.model = model_info["path"]
            self.selected_bands = model_info["selected_bands"]

        self.remove_temp = not kwargs.get("keep_temp", False)
        self.full_mask = kwargs.get("full_res_mask", False)
        self.full_rgb = kwargs.get("full_rgb", False)
        self.inputs = self.args["inputs"]["bands_used"] + \
            self.args["inputs"]["synthetic_bands"] + \
            self.args["inputs"]["mnt_used"] + \
            [b["band"] for b in self.args["inputs"]["additional_bands"]]

        # Data verification
        assert os.path.exists(self.model)
        assert os.path.exists(self.args["path"]["wdir"])

    def create_rgb(self, prod, rgb_path):
        """
        Create an RGB image using the provided RGB bands

        :param prod: The product
        :type prod: :class:`Chain.Product.MajaProduct`
        :param rgb_path: The output path
        :return:
        """
        bands_rgb = []

        rgb_bands, rgb_scaling = prod.rgb_values
        for b in rgb_bands:
            try:
                bands_rgb.append(prod.find_file(pattern=r"*%s*.(jp2|tif)$" % b)[0])
            except ValueError:
                bands_rgb.append(prod.get_synthetic_band(b, wdir=self.args["path"]["wdir"]))
        ds = ImageTools.gdal_merge(*bands_rgb, separate=True, q=True)
        # Normalize to [0, 255] for matplotlib:
        return ImageTools.gdal_translate(ds, dst=rgb_path, **rgb_scaling, ot="Byte", q=True)

    def create_color_image(self, *predictions, **kwargs):
        """
        Create a color coded image highlighting predicted classes

        :param predictions: A dict of the predicted datasets:
                            - ds: A :class:`Common.GDalDatasetWrapper.GDalDatasetWrapper`.
                                 Each with an array (of the same size) of values between 0 and 255.
                            - class_name: The name of the class as str (e.g. "Water")
        """
        if not predictions:
            raise ValueError("No predictions given.")
        alpha = kwargs.get("alpha", .6)
        png_path = kwargs.get("filepath")
        colors = [np.array([0, 0, 1, alpha]),  # Blue
                  np.array([0, .4, 0, alpha]),  # Dark green
                  np.array([0, 1, 1, alpha])]  # Yellow

        initial_shape = predictions[0]["ds"].array.shape
        combined_img = np.zeros((initial_shape[0],
                                 initial_shape[1],
                                 4))
        legend_items = []
        for pred, color in zip(predictions, colors):
            combined_img += pred["ds"].array[..., np.newaxis] * color
            legend_items.append(mpatches.Patch(color=color, label=pred["class_name"]))

        # Cast to unit8:
        combined_img = np.array(combined_img, dtype=np.uint8)

        # If path given, write the png to the given location superposing the combined_img over an rgb
        if png_path:
            title = kwargs.get("title", "-")
            rgb_path = kwargs["rgb"]

            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.legend(handles=legend_items, title=title, fontsize=8)
            ds_rgb = ImageTools.gdal_translate(rgb_path,
                                               tr="%s %s" % (self.args["preprocessing"]["coarse_res_m"],
                                                             self.args["preprocessing"]["coarse_res_m"]),
                                               q=True)
            rgb = ds_rgb.array
            plt.imshow(rgb)
            plt.imshow(combined_img)
            plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
        # Leave out alpha when returning, as it's useless for tif images:
        return combined_img[..., :3]

    @staticmethod
    def predict_and_reshape(img, model, n_classes):
        """
        Predict and reshape back to input shape

        :param img: A numpy array of shape (batch_size, x, y, n_bands)
        :type img: :class:`np.ndarray`
        :param model: A keras model instance
        :param n_classes: The number of classes predicted
        :return: The predicted image in shape (batch_size, x, y, n_classes)
        :rtype: :class:`np.ndarray`
        """
        batch_size, tile_size_x, tile_size_y, _ = img.shape
        prediction = model.predict(img)
        return np.reshape(prediction, (batch_size, tile_size_x, tile_size_y, n_classes))

    @staticmethod
    def predict_full_img(img, model, cfg_options, band_names, selected_bands, pad_length=33):
        """
        Predict and full-sized image by doing the following steps:
        * Pad full-sized image
        * Cut image into sub-tiles and pad the edges with overlapping
        * Predict each subtile
        * Unpad the sub-tiles
        * Interpolate the overlaps and combine into a full image
        * Return the interpolated image, which now has the same size as the input

        :param img: A numpy array of shape (x, y, n_bands)
        :param model: A keras model instance
        :param cfg_options: The args parsed from a cfg-file as dict.
        :type cfg_options: :class:`Common.Arguments.Arguments`
        :param pad_length: Add a padding of shape ((pad_length, pad_length), (pad_length, pad_length), (0, 0)) to img
        :type pad_length: int
        :param selected_bands: List of selected bands. All bands have to be present in ``band_names`` as well.
        :param band_names:  List of bands present. Has to be the same length as ``img.shape[-1]``
        :return: A numpy array of shape (x, y, n_classes) of values (0...1) with individual patches interpolated
        :rtype: :class:`np.ndarray`
        """
        from aux_files.smooth_tiled_predictions import predict_img_with_smooth_windowing
        cfg = cfg_options
        # CNN's receptive field's border size: size of patches
        window_size = cfg["preprocessing"]["tile_size"]
        # Amount of categories predicted per pixels.
        nb_classes = len(cfg["ground_truth"]["classes"])

        # Load an image.
        # Convention is channel_last, such as having an input_img.shape of: (x, y, nb_channels)
        x_extracted = EOSequence.extract_bands(img, selected_bands=selected_bands, band_names=band_names)
        x_scaled = [ImageTools.normalize(x_extracted[..., i],
                                         value_range_in=(cfg["band_values"][b]["min"],
                                                         cfg["band_values"][b]["max"]),
                                         value_range_out=(0, 1),
                                         clip=True)
                    for i, b in zip(range(x_extracted.shape[-1]), selected_bands)]
        x_scaled = np.moveaxis(np.array(x_scaled), 0, -1)
        nodata_mask = x_extracted[..., 0] == 0
        # Add additional padding (predict_img_with_smooth_windowing also does it) due to 'faded borders' appearing
        pad_size = ((pad_length, pad_length), (pad_length, pad_length), (0, 0))
        x_padded = np.pad(x_scaled, pad_size, mode="reflect")
        # Use the algorithm.
        # The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap,
        # called once with all those image as a batch outer dimension.
        # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
        predictions_smooth = predict_img_with_smooth_windowing(
            x_padded,
            window_size=window_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=nb_classes,
            pred_func=(
                lambda img_batch_subdiv: PredictL1C.predict_and_reshape(img_batch_subdiv, model, nb_classes))
            )

        # Remove the initial padding
        predictions_unpadded = predictions_smooth[pad_length:-pad_length, pad_length:-pad_length, :]
        # Set predictions outside of nodata mask to 0
        predictions_unpadded[nodata_mask] = 0

        return predictions_unpadded

    def run_smooth_prediction(self):
        """
        Run the prediction pipeline. Applies an interpolation to the full image in order to get rid of the
        edge effects.
        In order to use it, please clone https://github.com/Vooban/Smoothly-Blend-Image-Patches
        And add it to your PYTHONPATH.

        :return: For each product detected in the `self.product_dir`, this creates the following files:
                 * An RGB image of the product
                 * A Colored PNG image superposing the predicted classes
                 * Optional: A full-resolution 8-bit mask of each predicted class
         """
        print("Using model %s" % self.model)
        platforms = list(set(b.split("_")[0] for b in
                             self.args["inputs"]["bands_used"] +
                             self.args["inputs"]["synthetic_bands"]))
        prods = Dataset.get_available_products(self.product_dir, platforms=platforms,
                                               level="l1c")
        print("Predicting %s products" % len(prods))

        # Load the model
        model = load_model(self.model, compile=False)

        color_imgs_written = []
        for prod in prods:
            prod_dict = Dataset.create_img_dict(self.args, prod)
            prod_dict["algo"] = "infer"  # Need to add this information manually
            # Get extent and epsg from first band and apply it to all other rasters:
            ds = GDalDatasetWrapper.from_file(prod_dict["rasters"][0]["path"])
            uniform_epsg = ds.epsg
            uniform_extent_str = " ".join(map(str, ds.extent()))
            tr_str = "%s %s" % (self.args["preprocessing"]["coarse_res_m"], self.args["preprocessing"]["coarse_res_m"])
            ds_nodata = ImageTools.gdal_warp(ds, tr=tr_str, te=uniform_extent_str,
                                             r="cubic", dstnodata=prod_dict["nodata"],
                                             wdir=self.args["path"]["wdir"]
                                             )
            # Create separate tiles for each:
            ds_in = Dataset.process_rasters(self.args, prod_dict, uniform_epsg, uniform_extent_str,
                                            nodata_mask=ds_nodata.nodata_mask, retile=False)
            # Create results folder
            FileSystem.create_directory(self.output_dir)

            # At this point, the ds contains the preprocessed full-size image
            predictions = self.predict_full_img(img=ds_in.array,
                                                model=model,
                                                cfg_options=self.args,
                                                band_names=self.inputs,
                                                selected_bands=self.selected_bands)
            # Cast to uint8
            predictions_uint8 = np.array(predictions * 255, dtype=np.uint8)

            # Separate each predicted class and parse to GDalDatasetWrapper
            coarse_predictions = [{"ds": GDalDatasetWrapper(ds=ds_in.get_ds(), array=predictions_uint8[..., i]),
                                   "class_name": cname}
                                  for cname, i in zip(self.args["ground_truth"]["classes"],
                                                      range(predictions_uint8.shape[-1]))]
            # If desired, write each predicted class as full-resolution uint8 mask:
            if self.full_mask:
                for pred in coarse_predictions:
                    full_mask = os.path.join(self.output_dir, "%s_%s.tif" % (prod.base, pred["class_name"]))
                    ImageTools.gdal_translate(pred["ds"],
                                              dst=full_mask,
                                              tr=" ".join(map(str, prod.mnt_resolution)),
                                              q=True)

            # Create RGB image for representation
            full_rgb = os.path.join(self.output_dir, "%s_RGB.tif" % prod.base)
            self.create_rgb(prod, full_rgb)
            # Create PNG overlaying predictions in color (with alpha) over RGB:
            full_color_png = os.path.join(self.output_dir, "%s_COLOR.png" % prod.base)
            self.create_color_image(*coarse_predictions,
                                    rgb=full_rgb,
                                    title="\n".join(["Sat:  " + prod.platform.upper(),
                                                     "Site: " + prod.tile,
                                                     "Date: " + prod.date.strftime("%Y/%m/%d")]),
                                    filepath=full_color_png)
            color_imgs_written.append(full_color_png)
        print("Wrote pngs:\n%s" % " ".join(color_imgs_written))
        return 0

    @staticmethod
    def unpad_stack(prediction, tile_size, overlap):
        """
        Unpad a stack of tiled predictions.
        Does not unpad the left and upper edges for the first row/column in order to avoid 'blank' parts.

        :param prediction: The filename of one of the tiled predictions.
        :type prediction: str
        :param overlap: The overlap between two tiles in both x- and y- direction.
        :type overlap: int
        :param tile_size: The square tile_size
        :type tile_size: int
        :return: A wrapper class containing both the numpy array as well as the geo information.
        :rtype: :class:`Common.GDalDatasetWrapper.GDalDatasetWrapper`
        """

        row_number, col_number = map(int, os.path.basename(prediction).split("_")[4:6])
        # Full image size:
        # s_(x|y) = (tile_size * n_tiles) - (n_tiles - 1) * overlap
        # Unpad, but only the lower and right edges for the first row.
        # The upper and left edges need to stay, otherwise the detections in that part are cut.
        offset = overlap // 2
        if col_number == 1:
            xoff = 0
            xsize = tile_size - offset
        else:
            xoff = offset
            xsize = tile_size - overlap

        if row_number == 1:
            yoff = 0
            ysize = tile_size - offset
        else:
            yoff = offset
            ysize = tile_size - overlap
        window = "%s %s %s %s" % (xoff, yoff, xsize, ysize)
        return ImageTools.gdal_translate(prediction,
                                         srcwin=window,
                                         q=True)

    def run_direct_prediction(self):
        """
        Run the prediction pipeline and concantenate directly.
        Warning: This method can contain artifacts on the edges of the predictions.

        :return: For each product detected in the `self.product_dir`, this creates the following files:
                 * An RGB image of the product
                 * A Colored PNG image superposing the predicted classes
                 * Optional: A full-resolution 8-bit mask of each predicted class
        """
        print("Using model %s" % self.model)
        platforms = list(set(b.split("_")[0] for b in
                             self.args["inputs"]["bands_used"] +
                             self.args["inputs"]["synthetic_bands"]))
        prods = Dataset.get_available_products(self.product_dir, platforms=platforms,
                                               level="l1c")
        root_tiles = tempfile.mkdtemp(dir=self.args["path"]["wdir"], prefix="Tiles_")
        print("Predicting %s products" % len(prods))
        color_imgs_written = []
        for prod in tqdm(prods):
            prod_dict = Dataset.create_img_dict(self.args, prod)
            prod_dict["algo"] = "infer"  # Need to add this information manually
            # Get extent and epsg from first band and apply it to all other rasters:
            ds = GDalDatasetWrapper.from_file(prod_dict["rasters"][0]["path"])
            uniform_epsg = ds.epsg
            uniform_extent_str = " ".join(map(str, ds.extent()))
            uniform_ullr_str = " ".join(map(str, ds.ul_lr))
            tr_str = "%s %s" % (self.args["preprocessing"]["coarse_res_m"], self.args["preprocessing"]["coarse_res_m"])
            ds_nodata = ImageTools.gdal_warp(ds, tr=tr_str, te=uniform_extent_str,
                                             r="cubic", dstnodata=prod_dict["nodata"],
                                             wdir=self.args["path"]["wdir"]
                                             )
            # Create separate tiles for each:
            imgs = Dataset.process_rasters(self.args, prod_dict, uniform_epsg, uniform_extent_str,
                                           nodata_mask=ds_nodata.nodata_mask, retile=True, output_dir=root_tiles)
            df = pandas.DataFrame({"img": imgs})
            # Create results folder
            FileSystem.create_directory(self.output_dir)
            # At this point, the df contains the preprocessed tiles
            predictions = Predict.predict_from_df(df=df,
                                                  cfg=self.args,
                                                  model_file=self.model,
                                                  out_dir=self.args["path"]["wdir"],
                                                  ground_truth=False,
                                                  band_names=self.inputs,
                                                  selected_bands=self.selected_bands,
                                                  write_png=False)
            # For each class, unpad and stitch back together (cf. arXiv:1805.12219)
            coarse_predictions = []
            for index, class_name in enumerate(self.args["ground_truth"]["classes"]):
                # Get only the predictions for the current class:
                class_predictions = [p[index] for p in predictions]
                # Unpad individual tiles:
                ds_array_predictions = [self.unpad_stack(pred,
                                                         self.args["preprocessing"]["tile_size"],
                                                         self.args["preprocessing"]["overlap"])
                                        for pred in class_predictions]
                ds_combined = ImageTools.gdal_merge(*ds_array_predictions, q=True)
                ds_unpadded = ImageTools.gdal_translate(ds_combined,
                                                        projwin=uniform_ullr_str,
                                                        projwin_srs="EPSG:%s" % uniform_epsg,
                                                        r="nearest",
                                                        q=True)
                # Remove padding from full image:
                if self.full_mask:
                    full_mask = os.path.join(self.output_dir, "%s_%s.tif" % (prod.base, class_name.upper()))
                    ImageTools.gdal_translate(ds_unpadded,
                                              dst=full_mask,
                                              tr=" ".join(map(str, prod.mnt_resolution)),
                                              q=True)
                coarse_predictions.append({"ds": ds_unpadded, "class_name": class_name})
            # Create RGB image for representation
            full_rgb = os.path.join(self.output_dir, "%s_RGB.tif" % prod.base)
            self.create_rgb(prod, full_rgb)
            # Create PNG overlaying predictions in color (with alpha) over RGB:
            full_color_png = os.path.join(self.output_dir, "%s_COLOR.png" % prod.base)
            self.create_color_image(*coarse_predictions,
                                    rgb=full_rgb,
                                    title="\n".join(["Sat:  " + prod.platform.upper(),
                                                     "Site: " + prod.tile,
                                                     "Date: " + prod.date.strftime("%Y/%m/%d")]),
                                    filepath=full_color_png)
            color_imgs_written.append(full_color_png)
            if self.remove_temp:
                FileSystem.remove_directory(os.path.dirname(predictions[0][0]))
        if self.remove_temp:
            FileSystem.remove_directory(root_tiles)
        print("Wrote pngs:\n%s" % " ".join(color_imgs_written))
        return 0


if __name__ == "__main__":
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
    except Exception as e:
        print(e)
    # Argument parsing
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('cfg_file', action="store", type=str, help="Configuration json")
    parser.add_argument('-p', '--product_dir', action="store", type=str,
                        help="The directory containing all products that should be predicted")
    parser.add_argument('-o', '--output_dir', action="store", type=str,
                        help="The directory containing all predictions. Default is the current directory.",
                        default=os.getcwd())
    parser.add_argument('--model', '-m', action="store", type=str, help="Path to a model.h5 file to be used. If none is"
                                                                        "given, the newest available one will be used.")
    parser.add_argument('-k', '--keep_temp', action="store_true",
                        help="Keep the individual tiled predictions after processing. Default is False", default=False)
    parser.add_argument('--full_mask', action="store_true",
                        help="Produce a full-resolution mask for each product")
    parser.add_argument('--full_rgb', action="store_true",
                        help="Produce a full-resolution RGB for each product")
    parser.add_argument('-s', '--smooth', action="store_true",
                        help="Smoothen the predicted sub-tiles using a spline-interpolation. Default is False.",
                        default=False)
    parser.add_argument('--cpu', action="store_true", help="Use CPU for prediction instead of GPU."
                                                           "Default is False.", default=False)
    args_cmd = parser.parse_args()
    if args_cmd.cpu:
        tf.config.experimental.set_visible_devices([], 'GPU')
    # main
    predict = PredictL1C(args_cmd.cfg_file,
                         product_dir=args_cmd.product_dir,
                         output_dir=args_cmd.output_dir,
                         model=args_cmd.model,
                         keep_temp=args_cmd.keep_temp,
                         full_res_mask=args_cmd.full_mask,
                         full_rgb=args_cmd.full_rgb)
    if args_cmd.smooth:
        predict.run_smooth_prediction()
    else:
        predict.run_direct_prediction()
