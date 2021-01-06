#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


from tensorflow.keras import backend as k
import tensorflow as tf


def focal_loss(gamma=2., alpha=.25):
    """
    Focal loss using fixed values for gamma and alpha.

    :param gamma:
    :param alpha:
    :return:
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.compat.v1.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.compat.v1.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -k.mean(alpha * k.pow(1. - pt_1, gamma) * k.log(pt_1)) - k.mean((1 - alpha) * k.pow(pt_0, gamma) * k.log(1. - pt_0))
    return focal_loss_fixed


def weighted_xentropy_dice(y_true, y_pred):
    """
    Loss = BCE + 1 - DCE with BCE being weighted.

    :param y_true:
    :param y_pred:
    :return:
    """
    return w_binary_crossentropy(y_true, y_pred) + 1 - dice_coef(y_true, y_pred)


def w_binary_crossentropy(y_true, y_pred):
    """
    Calculate the binary xentropy.

    :param y_true:
    :param y_pred:
    :return:
    """
    return k.mean(k.binary_crossentropy(y_true, y_pred))


def dice_coef(y_true, y_pred):
    """
    Computes Dice loss.

    :param y_true:
    :param y_pred:
    :return:
    """
    smooth = 1.

    dtype = tf.float32
    y_true_f = tf.cast(k.flatten(y_true), dtype)
    y_pred_f = tf.cast(k.flatten(y_pred), dtype)
    intersection = k.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    Loss function using the dice coefficient.

    :param y_true:
    :param y_pred:
    :return:
    """
    return -1. * dice_coef(y_true, y_pred)
