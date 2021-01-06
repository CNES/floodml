#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import numpy as np


def calculate_fscore_2(predicted, actual, beta=2):
    if predicted.shape != actual.shape:
        raise ValueError("Arrays have to be then same shape: {0} != {1}".format(predicted.shape, actual.shape))
    from sklearn.metrics import fbeta_score
    return fbeta_score(actual.flatten(), predicted.flatten(), beta=beta, average='weighted')


def calculate_fscore(predicted, actual, beta=2):
    """
    Calculate the f-score for a predicted and actual binary-array of the same size
    :param predicted: The predicted binary image
    :param actual: The ground truth binary image
    :param beta: Weigh recall using this parameter. Default: 2
    """
    if predicted.shape != actual.shape:
        raise ValueError("Arrays have to be then same shape: {0} != {1}".format(predicted.shape, actual.shape))
    if predicted.dtype != actual.dtype or predicted.dtype != np.bool:
        print("WARNING: Forcing type %s" % np.bool)
    import tensorflow as tf

    sess = tf.Session()

    # Calculate TP, FP, FN:
    tp = tf.count_nonzero(predicted * actual)
    fp = tf.count_nonzero(predicted * np.logical_not(actual))
    fn = tf.count_nonzero(np.logical_not(predicted) * actual)
    precision = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    if tp.eval(session=sess) == 0:
        recall = tf.constant(1)
        if fp.eval(session=sess) > 0:
            precision = tf.constant(0)
        else:
            precision = tf.constant(1)
    f_score = (((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)).eval(session=sess)
    sess.close()
    return f_score


def calculate_fscore_np(predicted, actual, beta=2):
    """
    Calculate the f-score for a predicted and actual binary-array of the same size
    :param predicted: The predicted binary image
    :param actual: The ground truth binary image
    :param beta: Weigh recall using this parameter. Default: 2
    """
    if predicted.shape != actual.shape:
        raise ValueError("Arrays have to be then same shape: {0} != {1}".format(predicted.shape, actual.shape))
    if predicted.dtype != actual.dtype or predicted.dtype != np.bool:
        print("WARNING: Forcing type %s" % np.bool)
    # Calculate TP, FP, FN:
    tp = np.count_nonzero(predicted * actual)
    fp = np.count_nonzero(predicted * np.logical_not(actual))
    fn = np.count_nonzero(np.logical_not(predicted) * actual)
    print(tp)
    print(fp)
    print(fn)
    precision = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    if tp == 0:
        recall = 1
        if fp > 0:
            precision = 0
        else:
            precision = 0
    f_score = (((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall))
    return f_score


if __name__ == "__main__":
    pass
