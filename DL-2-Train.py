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
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import pandas
import argparse
from albumentations import Compose, RandomCrop, HorizontalFlip, VerticalFlip, RandomRotate90, Transpose
from albumentations import ShiftScaleRotate, RandomBrightnessContrast, RandomGamma, IAAEmboss, Blur
import segmentation_models as sm
from deep_learning.Imagery.Unet import dice_coef, dice_coef_loss
from Common import FileSystem
from deep_learning.Imagery.EOSequence import EOSequence


def train_model(results, df_train, df_valid, input_size, num_epochs,
                model_fn, classes, selected_bands, band_names, band_values, batch_size=32, multi_gpu=False):
    """
    Train the model. Save the weights of the trained model into a file named $model_file and the intermadiatte weights
    for each epoch of the training in a folder named checkpoints. It also runs tensorboard visualisation tool
    :param df_train: The list of available .npy files for training
    :param df_valid: The list of available .npy files for validation
    :param results: path to folder containing the results
    :param input_size: size of the input images
    :param num_epochs: number of epochs
    :param model_fn: name of the file that will be created to store the final weights of the model
    :param classes: Classes to be segmented.
    :param selected_bands: The list of bands selected. All bands need to be existing in ``band_names``.
    :param band_names: The list of bands used. e.g. [B01, ALT, ASP, SLP]
    :param band_values: The dict of input value range for each band e.g. {'B01': (0, 16384)}
    :param batch_size: The batch size
    :param multi_gpu: Allow distributed training strategy
    :return trained model and its history
    """
    # Create necessary folders:
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = p.join(results, "model")
    FileSystem.create_directory(model_path)
    tboard_logs = p.join(results, "tensorboard_logs")
    FileSystem.create_directory(tboard_logs)
    tboard_current_run = p.join(tboard_logs, now)
    chk_path = p.join(results, "checkpoints")
    FileSystem.create_directory(chk_path)
    chk_run = p.join(chk_path, now)
    FileSystem.create_directory(chk_run)

    crop_prob = 0
    augment_train = Compose([
        RandomCrop(width=input_size[0], height=input_size[1], p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        IAAEmboss(p=0.25),
        Blur(p=0.01, blur_limit=3)], p=1)

    augment_valid = Compose([
        RandomCrop(width=input_size[0], height=input_size[1], p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        IAAEmboss(p=0.25),
        Blur(p=0.01, blur_limit=3)], p=1)

    x_train_gen = EOSequence(df_train, batch_size=batch_size, selected_bands=selected_bands,
                             band_names=band_names, band_values=band_values, augment=augment_train)
    x_validation_gen = EOSequence(df_valid, batch_size=batch_size, selected_bands=selected_bands,
                                  band_names=band_names, band_values=band_values, augment=augment_valid)
    file_writer = tf.summary.create_file_writer(tboard_current_run + "/metrics")
    file_writer.set_as_default()

    # Save the weights and the loss of the model after every epoch
    model_checkpoint = ModelCheckpoint(os.path.join(chk_run, 'weights.{epoch:02d}-{val_loss:.5f}.hdf5'),
                                       monitor='val_loss', save_best_only=True, verbose=1)

    # Define callback to reduce learning rate when learning stagnates
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_delta=0.002, cooldown=2, verbose=1)

    tensorboard = TensorBoard(log_dir=tboard_current_run, write_graph=True)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=15)

    loss = dice_coef_loss
    metrics = [dice_coef, "accuracy"]
    optimizer = Adam(lr=0.001)
    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

    with strategy.scope():
        model = sm.Unet(backbone_name='resnet152',
                        encoder_weights=None,
                        encoder_freeze=False,
                        input_shape=(input_size + (len(selected_bands),)),
                        classes=len(classes)
                        )
        # Parametrize the model
        print(model.summary())
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    # Train the model
    model.fit(x_train_gen, epochs=num_epochs, workers=24,
              validation_data=x_validation_gen, use_multiprocessing=False,
              validation_steps=len(df_valid) // batch_size,
              callbacks=[model_checkpoint, reduce_lr, tensorboard, early_stopping], shuffle=True)
    # Export model
    model_path = p.join(model_path, "_".join([now, model_fn]))
    model.save(model_path)
    model_info_path = "%s.json" % p.splitext(model_path)[0]
    with open(model_info_path, "w") as modelcfg:
        cfg = {"path": model_path,
               "selected_bands": selected_bands}
        json.dump(cfg, modelcfg, indent=4)
    print("Successfully saved model-config at %s" % model_info_path)
    FileSystem.remove_directory(chk_run)
    return model


def main(cfg_train, cfg_valid, bands, multi_gpu, results=None):
    """
    Run the training pipeline
    :param cfg_train: Path to the config file containing the training images
    :param cfg_valid: Path to the config file containing the validation/testing images
    :param bands: List of bands e.g. [s1_vv, s2_b2 ...] or 'all'
    :param multi_gpu: Bool to indicate whether multi-gpu strategy is applied or not.
    :param results: Optional path to the results folder
    :return:
    """

    # Parsing configuration file
    with open(cfg_train, 'r') as json_file:
        args = json.load(json_file)

    with open(cfg_valid, 'r') as json_file:
        _args_val = json.load(json_file)

    df_train = pandas.read_csv(args["path"]["csv"])
    df_valid = pandas.read_csv(_args_val["path"]["csv"])
    assert list(df_train.columns) == list(df_valid.columns)
    for df in [df_train, df_valid]:
        for idx, row in df.iterrows():
            if not p.exists(df.at[idx, "img"]):
                df.drop([idx], inplace=True)
                print("Cannot find %s" % df.at[idx, "img"])

    # df_valid.to_csv(_args_val["path"]["csv"], index=False)
    # df_valid.to_csv(args["path"]["csv"], index=False)

    classes = args["ground_truth"]["classes"]
    if not results:
        results = args["path"]["results"]
    inputs = args["inputs"]["bands_used"] + \
        args["inputs"]["synthetic_bands"] + \
        args["inputs"]["mnt_used"] + \
        [b["band"] for b in args["inputs"]["additional_bands"]]
    if bands == ["all"]:
        selected_bands = inputs
    else:
        selected_bands = bands

    FileSystem.create_directory(results)
    # #######################################
    # STEP 2: Launch learning on U-net model
    train_model(results=results,
                df_train=df_train,
                df_valid=df_valid,
                input_size=(args["preprocessing"]["tile_size"], args["preprocessing"]["tile_size"]),
                num_epochs=args["hyperparams"]["epochs"],
                model_fn="model.h5",
                classes=classes,
                batch_size=args["hyperparams"]["batch_size"],
                selected_bands=selected_bands,
                band_names=inputs,
                band_values=args["band_values"],
                multi_gpu=multi_gpu)


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--cfg_train', action="store", type=str, help="Configuration file created by Dataset.py",
                        required=True)
    parser.add_argument('--cfg_valid', action="store", type=str, help="Valid config file created by Dataset.py",
                        required=True)
    parser.add_argument('--results', action="store", type=str, help="Optional path to the results folder",
                        required=False)
    parser.add_argument('--bands', action="store", type=str, help="List of bands to use as input. Default is 'all'.",
                        default=["all"], nargs="+")
    parser.add_argument('--multi_gpu', action="store_true", help="Allow distributed training",
                        required=False, default=False)
    arg = parser.parse_args()

    # main
    main(cfg_train=arg.cfg_train, cfg_valid=arg.cfg_valid,
         results=arg.results, bands=arg.bands, multi_gpu=arg.multi_gpu)
