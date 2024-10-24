#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""



import numpy as np
from tensorflow.keras.utils import Sequence
from imageio import imwrite
from Common import ImageIO, ImageTools


class EOSequence(Sequence):
    """
    Load a sequence of images into memory
    Source: https://keras.io/utils/#sequence
    """
    def __init__(self, x_set, batch_size, selected_bands, band_names, band_values, **kwargs):
        """
        Init the generator
        :param x_set: The pandas dataframe containing image and mask paths in each row
        :param batch_size: The batch size
        :param band_values: The input values for each band as dict.
        :param band_names: The names of the bands used in the same order they appear inside the image file
        :keyword augment: An `albumentations` generator
        """
        self.x = x_set
        self.batch_size = batch_size
        self.selected_bands = selected_bands
        self.band_names = band_names
        self.band_values = band_values
        self.true_weight = 5
        self.false_weight = 1
        self.augment = kwargs.get("augment", None)
        self.do_augmentation = kwargs.get("do_augmentation", True)
        self.write_debug = kwargs.get("write_debug", False)
        self.mode = kwargs.get("mode", "training")

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        current_batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_arr, y_arr = [], []
        for idx, row in current_batch.iterrows():
            img_path = row["img"]
            x_unscaled, drv = ImageIO.tiff_to_array(img_path, array_only=False)
            x_extracted = self.extract_bands(x_unscaled, self.selected_bands, self.band_names)
            x_normalized = self.normalize(x_extracted)
            img = np.array(x_normalized, dtype=np.float32)
            if self.mode == "training":
                msk_path = row["masks"]
                mask = ImageIO.tiff_to_array(msk_path)[..., np.newaxis]
                mask_nodata_filled = self.fill_nodata_zones(img, mask)
                if self.do_augmentation:
                    augmented = self.augment(image=img, mask=mask_nodata_filled)
                    mask = augmented["mask"]
                    img = augmented["image"]
                if self.write_debug:
                    imwrite("./batch_%s_y.png" % idx, np.array(mask * 255, dtype=np.uint8))
                y_arr.append(mask)
            if self.write_debug:
                imwrite("./batch_%s_x.png" % idx, img[..., 0])
            x_arr.append(img)
        # New shape: N, x, y, n_bands
        x_comb = np.stack(x_arr)
        if self.mode == "training":
            # New shape: N, x, y, n_classes
            y_comb = np.stack(y_arr)
            return x_comb, y_comb
        return x_comb

    @staticmethod
    def fill_nodata_zones(img, msk):
        """
        Fill the nodata zones of img in msk
        :param img: The image
        :param msk: Its associated mask
        :return: A new mask of same size with nodata == 1
        """
        msk_out = msk.copy()
        msk_out[img[..., 0] == 0] = 1
        return msk_out

    @staticmethod
    def extract_bands(x, selected_bands, band_names):
        """
        Extract a list of selected bands from a numpy array.

        :param x: Numpy array of shape [x, y, n_bands]
        :param selected_bands: List of selected bands. All bands have to be present in band_names as well.
        :param band_names:  List of bands present. Has to be the same length as the number of bands present in ``x``.
        :return: Numpy array with shape [x, y, n_selected_bands]
        """
        # Don't do extraction if all bands were selected:
        if set(selected_bands) == set(band_names):
            return x
        x_extracted = [x[..., band_names.index(b)] for b in selected_bands]
        return np.stack(x_extracted, axis=-1)

    def normalize(self, img):
        """
        Normalize an image in the given input range to [-1,1]
        :param img: The image of shape (x, y, n_bands)
        :return: The normalized image
        """
        img_normalized = [ImageTools.normalize(img[..., i],
                                               value_range_out=(0, 1),
                                               value_range_in=(self.band_values[b]["min"], self.band_values[b]["max"]),
                                               clip=True)
                          for i, b in zip(range(img.shape[-1]), self.selected_bands)]
        return np.moveaxis(np.array(img_normalized), 0, -1)
