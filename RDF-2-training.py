#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
import gc
import numpy as np
import joblib
import argparse
from Common.validationTools import calculate_fscore_2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Common import FileSystem


def main_training(args):

    npy_dir = args.NPY_dir
    db_output = args.db_output
    sat = args.sentinel
    emsr_numbers = list(set(args.EMSR_numbers))  # Only get unique EMSR numbers - Do not train twice on the same
    intag = args.suffix_in
    outag = args.suffix_out

    if args.gpu:
        from cuml import RandomForestClassifier
    else:
        from sklearn.ensemble import RandomForestClassifier

    all_gt, all_train = list(), list()
    for emsr in emsr_numbers:

        emsr = str(emsr)
        print("EMSR considered:", emsr)

        if intag is None:
            ground_truth = os.path.join(npy_dir,  "DB_S%s_EMSR%s_WAT.npy" % (sat, emsr))
            trained = os.path.join(npy_dir, "DB_S%s_EMSR%s_RDN.npy" % (sat, emsr))
        else:
            ground_truth = os.path.join(npy_dir,  "DB_S%s_EMSR%s_WAT_%s.npy" % (sat, emsr, intag))
            trained = os.path.join(npy_dir,  "DB_S%s_EMSR%s_RDN_%s.npy" % (sat, emsr, intag))

        try:
            data_vt = np.load(ground_truth)
            data_rdn = np.load(trained)
        except FileNotFoundError as e:
            print(e)
            continue

        # NPY duplicated rows reduction  for WATER#
        # Perform lex sort and get sorted data
        sorted_idx = np.lexsort(data_vt.T)
        sorted_data = data_vt[sorted_idx, :]
        # Get unique row mask
        row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
        # Get unique rows
        data_vt = sorted_data[row_mask]

        # NPY duplicated rows reduction  for RDN
        # Perform lex sort and get sorted data
        sorted_idx = np.lexsort(data_rdn.T)
        sorted_data = data_rdn[sorted_idx, :]
        # Get unique row mask
        row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
        # Get unique rows
        data_rdn = sorted_data[row_mask]

        all_gt.append(data_vt)
        all_train.append(data_rdn)

    gt_concat = np.concatenate(all_gt, axis=0)
    train_concat = np.concatenate(all_train, axis=0)

    # Y vector ######################################
    yvt = np.ones((gt_concat.shape[0], 1), dtype=np.float32)
    yrdn = np.zeros((train_concat.shape[0], 1), dtype=np.float32)

    # Concat ########################################
    xb = np.array(np.vstack((gt_concat, train_concat)), dtype=np.float32)
    yb = np.vstack((yvt, yrdn))

    # Classif ######################################
    # Split into train and test set
    x_train, x_test, y_train, y_test = train_test_split(xb, yb, test_size=0.33)

    # Random Forest
    print("\n###    Random forest training    ###")
    if args.gpu:
        parameters = {"n_estimators": 100}
    else:
        parameters = {"n_estimators": 100, "n_jobs": -1}

    rdf = RandomForestClassifier(**parameters)
    rdf.fit(x_train, y_train)

    # Export cuML RF model as Treelite checkpoint for CPU computing
    checkpoint_path = os.path.join(db_output, "DB_S%s_CPU_%s.sav" % (sat, outag)) 
    rdf.convert_to_treelite_model().to_treelite_checkpoint(checkpoint_path)

    rdf_pred = rdf.predict(x_test)
    rdf_score = accuracy_score(rdf_pred, y_test)
    print("Accuracy: {:.5f}".format(rdf_score))
    print("FScore:   {:.5f}".format(calculate_fscore_2(rdf_pred, y_test[..., 0])))
    gc.collect()

    FileSystem.create_directory(db_output)
    if outag is None:
        joblib.dump(rdf, os.path.join(db_output, "DB_RDF_global_S%s.sav" % sat))
    else:  # adds output tag
        joblib.dump(rdf, os.path.join(db_output, "DB_RDF_global_S%s_%s.sav" % (sat, outag)))

    print("Successfully finished training step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation scheduler')

    parser.add_argument('-i', '--NPY_dir', help='Input folder (NPY folder)', type=str, required=True)
    parser.add_argument('-n', '--EMSR_numbers', help='EMSR cases name', nargs='+', type=int, required=True)
    parser.add_argument('--sentinel', help='S1 or S2', type=int, required=True, choices=[1, 2])
    parser.add_argument('-o', '--db_output', help='Global DB output folder ', type=str, required=True)
    parser.add_argument('-si', '--suffix_in', help='Input suffix tag ', type=str, required=False)
    parser.add_argument('-so', '--suffix_out', help='Output suffix tag ', type=str, required=False)
    parser.add_argument("--gpu", help="Use GPU for training. Requires cuML to be installed.",
                        default=False, action="store_true")
    arg = parser.parse_args()

    main_training(arg)
