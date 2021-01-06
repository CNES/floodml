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
from imageio import imwrite
from datetime import datetime
import re
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas
import json
from deep_learning.Validation.validationTools import calculate_fscore_2
from Common import ImageIO, FileSystem
from deep_learning.Imagery.EOSequence import EOSequence


class Predict(object):
    """
    Predict on a set of preprocessed .tif files
    """

    def __init__(self, cfg_file, **kwargs):
        self.cfg_file = cfg_file
        # Parsing configuration file
        with open(cfg_file, 'r') as json_file:
            self.args = json.load(json_file)

        prob_threshold = kwargs.get("prob_threshold", None)
        if prob_threshold:
            self.prob_threshold = np.array([prob_threshold])
        else:
            self.prob_threshold = np.linspace(0, 1, 10, endpoint=False)
        # Get model file:
        model_cfg = kwargs.get("model", None)
        if not model_cfg:
            results = self.args["path"]["results"]
            model_cfg = self.get_latest_model(os.path.join(results, "model"))[0]

        with open(model_cfg, 'r') as json_file:
            model_info = json.load(json_file)
            self.model = model_info["path"]
            self.selected_bands = model_info["selected_bands"]

        try:
            self.inputs = self.args["inputs"]["bands_used"] + \
                          self.args["inputs"]["synthetic_bands"] + \
                          self.args["inputs"]["mnt_used"] + \
                          [b["band"] for b in self.args["inputs"]["additional_bands"]]
        except KeyError:
            self.inputs = self.args["inputs"]
        # Data verification
        assert os.path.exists(self.model), "Cannot find model %s" % self.model
        assert os.path.exists(self.args["path"]["wdir"]), "Cannot find wdir %s" % self.args["path"]["wdir"]

    @staticmethod
    def get_latest_model(model_dir):
        """
        Get the latest model file from the results/model/ directory.

        :param model_dir: The Model directory
        :return: The path to the latest model file or OSError if not.
        """

        assert os.path.isdir(model_dir)
        datetimes = []
        regex = r"^\d{8}_\d{6}_model.json"
        for f in os.listdir(model_dir):
            if re.search(regex, f):
                datetime_raw = "_".join(f.split("_")[0:2])
                datetimes.append([f, datetime.strptime(datetime_raw, '%Y%m%d_%H%M%S')])
        if not datetimes:
            raise OSError("Cannot find model file in directory %s" % model_dir)
        datetimes.sort(key=lambda tup: tup[1])
        model_filename = datetimes[-1][0]
        date_vector = model_filename.split('_')
        date_time = "_".join(date_vector)
        return os.path.join(model_dir, datetimes[-1][0]), date_time

    @staticmethod
    def plot_histogram(fscores, filepath, n_bins=20):
        """
        Plots a histogram of fscores
        :param fscores: The list of fscores
        :param filepath: The path to write the file to
        :param n_bins: The number of bins for the histogram
        :return:
        """
        plt.figure()
        plt.hist(fscores, n_bins, facecolor='blue', alpha=0.5, range=(0, 1))
        plt.xlabel('Fscores')
        plt.ylabel('Occurences')
        plt.title(r'Histogram of Fscores')
        plt.savefig(filepath, dpi=300)

    @staticmethod
    def plot_scores(thresholds, fscores, dst, maximum):
        """
        Plot scores over probability thresholds and highlights the maximum.

        :param thresholds: The list of thresholds used for calculating the scores.
        :param fscores: The numpy array of scores for the different thresholds
        :param dst: The destination for the plot to be saved in.
        :param maximum: The (x,y) of the maximum of the plot to be highlighted as red dot and in the title.
        :return:
        """
        plt.figure()
        plt.plot(thresholds, fscores)
        plt.plot(maximum[0], maximum[1], "ro")
        plt.xlabel('Probability threshold')
        plt.ylabel('Fscore')
        plt.title("Maximum: {0:.5f}".format(maximum[1]))
        plt.grid()
        plt.savefig(dst, dpi=300)

    @staticmethod
    def predict_from_df(df, cfg, model_file, band_names, **kwargs):
        """
        Predict a set of images and parse them to .tif and .png for easier visualisation.
        Calculates fscore
        wrt the ground truth masks.
        :param df: The dataframe of available .tif files for prediction.
        :param cfg: The dict containing the configuration params used during training
        :param band_names: The list of names for the available bands for the image.
        :param model_file: file containing the model and its weights after training
        Predict a set of preprocessed images.
        Note: This function requires to run Dataset.py on the desired L1 products.
        :return:
        """
        selected_bands = kwargs.get("selected_bands", None)
        ground_truth = kwargs.get("ground_truth", False)
        output_dir = kwargs.get("out_dir", os.getcwd())
        write_png = kwargs.get("write_png", False)
        prob_threshold = kwargs.get("prob_threshold", [.5])
        model = load_model(model_file, compile=False)
        png_folder = os.path.join(output_dir, "png")
        tif_folder = os.path.join(output_dir, "tif")
        FileSystem.create_directory(tif_folder)
        if not selected_bands:
            selected_bands = band_names
        prediction_paths, fscores = [], []
        if write_png:
            FileSystem.create_directory(png_folder)
            print(".png: %s" % png_folder)
        try:
            classes = list(set([m["class"] for m in cfg["masks_used"]]))
        except KeyError:
            classes = cfg["ground_truth"]["classes"]
        fscores = {c: [0] * len(prob_threshold) for c in classes}

        generator = EOSequence(df, batch_size=1, selected_bands=selected_bands, mode="inference",
                               band_names=band_names, band_values=cfg["band_values"], do_augmentation=False)
        for idx in range(df.shape[0]):
            # Load an input image
            img_path = df["img"][idx]
            img, drv = ImageIO.tiff_to_array(img_path, array_only=False)
            output_basename = os.path.basename(img_path).split(".")[0]
            prediction = model.predict_on_batch(generator.__getitem__(idx))
            if ground_truth:
                msk_path = df["masks"][idx]
                y_gt = ImageIO.tiff_to_array(msk_path)[..., np.newaxis]
                y_gt = EOSequence.fill_nodata_zones(img, y_gt)
            else:
                y_gt = None
            coarse_prediction_paths = {}
            for index, class_name in enumerate(classes):
                # Extract selected class
                y_pred_coarse = np.array(prediction[0, ..., index])
                if ground_truth:
                    y_gt_coarse = np.asarray(y_gt[..., index],
                                             dtype=np.bool)
                    # Calculate the F-score of the couple of above matrices and create a confusion matrix image
                    fscores_new = [calculate_fscore_2(y_pred_coarse > thres, y_gt_coarse)
                                   for thres in prob_threshold]
                    # Add fscores element-wise
                    fscores[class_name] = [sum(x) for x in zip(fscores[class_name], fscores_new)]
                    # Transform to uint8:
                    y_gt_coarse = np.array(y_gt_coarse * 255, dtype=np.uint8)
                    gt_tif = os.path.join(tif_folder, "%s_%s_gt.tif" % (
                        output_basename, class_name))
                    gt_png = os.path.join(png_folder, "%s_%s_gt.png" % (
                        output_basename, class_name))
                    if write_png:
                        imwrite(gt_png, y_gt_coarse)
                    ImageIO.write_geotiff_existing(y_gt_coarse, gt_tif, drv)

                # Transform to uint8:
                y_pred_coarse = np.array(y_pred_coarse * 255, dtype=np.uint8)
                pred_tif = os.path.join(tif_folder, "%s_%s_pred.tif" % (
                                        output_basename, class_name))
                pred_png = os.path.join(png_folder, "%s_%s_pred.png" % (
                                        output_basename, class_name))
                ImageIO.write_geotiff_existing(y_pred_coarse, pred_tif, drv)
                if write_png:
                    imwrite(pred_png, y_pred_coarse)
                coarse_prediction_paths[index] = pred_tif
            prediction_paths.append(coarse_prediction_paths)

        if ground_truth:
            for class_name in classes:
                # Take the mean
                fscores[class_name] = np.array(fscores[class_name]) / len(df)
                max_y = np.max(fscores[class_name])
                max_x = prob_threshold[np.array(np.where(fscores[class_name] == max_y)[0])[0]]
                maximum = (max_x, max_y)
                Predict.plot_scores(prob_threshold, fscores[class_name], "fscores_thresholds_%s.png" % class_name,
                                    maximum)
                print("Total F2-Score for '{0}': {1:.5}".format(class_name, max_y))
                hist_path = os.path.join(os.getcwd(), "fscores_hist_%s.png" % class_name)
                Predict.plot_histogram(fscores[class_name], hist_path)
                print("Distribution of fscores for %s: %s" % (class_name, hist_path))
        return prediction_paths

    def run(self):
        """
        Run the prediction pipeline
        :return:
        """
        print("Using model %s" % self.model)
        model_date = os.path.basename(self.model).split("_model")[0]
        df = pandas.read_csv(self.args["path"]["csv"])
        assert len(list(df.columns)) == 2
        columns = list(df.columns)[1:]
        for idx, row in df.iterrows():
            for col in columns:
                if not os.path.exists(row[col]):
                    print("WARNING Image does not exist: %s" % row[col])

        now = datetime.now()
        prediction_bname = "_".join(['predictions', now.strftime("%Y%m%d_%Hh%Mm"), 'with_the', model_date, 'model'])
        predicted_masks_path = os.path.abspath(os.path.join(self.args["path"]["results"],
                                                            'Predictions',
                                                            prediction_bname))
        pred_path = os.path.join(predicted_masks_path)
        os.makedirs(predicted_masks_path, exist_ok=True)
        os.makedirs(pred_path, exist_ok=True)
        self.predict_from_df(df=df,
                             cfg=self.args,
                             model_file=self.model,
                             out_dir=predicted_masks_path,
                             cfg_options=self.args,
                             ground_truth=True,
                             prob_threshold=self.prob_threshold,
                             band_names=self.inputs,
                             selected_bands=self.selected_bands,
                             write_png=True)
