#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from fiona.crs import from_epsg
from deep_learning.Validation.validationTools import calculate_fscore_2

import aux_files.pykic.raster.pykic_gdal as rpg

# source /home/eh/fatrasc/scratch/Bib_annexes/OTB-7.0/otbenv.profile

# FOR EACH EMSR SITE:
# tile_extent = '499980 6090240 609780 6200040'; tile='34UEG'; EMSR_n = '267'; oEPSG = 32634;  AOI_n = 1; Monit_vers = 0 #20180202 DES 051
# tile_extent = '499980 4290240 609780 4400040'; tile='34SEJ'; EMSR_n = '271'; oEPSG = 32634;  AOI_n = 5; Monit_vers = 1 # 20180228 DES 080
# tile_extent = '600000 4590240 709800 4700040'; tile='30TXM'; EMSR_n = '279'; oEPSG = 32630; AOI_n = 4; Monit_vers = 0 # 20180413 ASC 030
# tile_extent = '600000 4590240 709800 4700040'; tile='30TXM'; EMSR_n = '279'; oEPSG = 32630; AOI_n = 5; Monit_vers = 0 # 20180413 ASC 030
# tile_extent = '199980 3390240 309780 3500040'; tile='39RTQ'; EMSR_n = '352'; oEPSG = 32639;  AOI_n = 1; Monit_vers = 0 # 20190408 DES 108 # MARCHE PAS
EMSR_n = '267'; tile='34UEG'; oEPSG = 32634;  AOI_n = 1; Monit_vers = 0; date='20180202'; orb = 'DES_051'
# EMSR_n = '271'; tile='34SEJ'; oEPSG = 32634;  AOI_n = 5; Monit_vers = 1; date ='20180228'; orb = 'DES_080'
# EMSR_n = '279'; tile='30TXM'; oEPSG = 32630; AOI_n = 4; Monit_vers = 0; date ='20180413'; orb = 'ASC030'

print('\n')
EMSR_dir = '/work/OT/floodml/data/deliveries/phase-1-cls/'
Metric_dir= glob.glob(os.path.join(EMSR_dir, 'EMSR'+str(EMSR_n), 'Metric'), recursive=False)[0]

ROI_EMSR_dir = os.path.join(Metric_dir, 'ROI_EMSR'+str(EMSR_n)+'_'+tile+'_'+date+'_AOI'+str(AOI_n).zfill(2)+'.tif')
OBS_EMSR_dir = os.path.join(Metric_dir, 'OBS_EMSR'+str(EMSR_n)+'_'+tile+'_'+date+'_AOI'+str(AOI_n).zfill(2)+'.tif')

ROI_EMSR, proj, dim, tr = rpg.gdal2array(ROI_EMSR_dir)
OBS_EMSR, proj, dim, tr = rpg.gdal2array(OBS_EMSR_dir)

# For checking purposes
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(ROI_EMSR)
#
# plt.subplot(1,2,2)
# plt.imshow(OBS_EMSR)
# plt.show()

Inf_file = os.path.join('/work/scratch/fatrasc/Predictions/', 'Inference_RDF_S1_EMSR'+EMSR_n+'_T'+tile+'_'+date+'_'+orb+'.tif')
Inference, proj, dim, tr = rpg.gdal2array(Inf_file)

Inference = np.float32(Inference)
OBS_EMSR = np.float32(OBS_EMSR)

Out_ROI = np.where(ROI_EMSR==0)
InROI = np.where(ROI_EMSR==1)

# Inference[Out_ROI] = 0
# OBS_EMSR[Out_ROI] = np.nan
# f_score = calculate_fscore_2(Inference, OBS_EMSR, beta=2)

Inf_ROI =Inference[InROI].flatten()
OBS_ROI =OBS_EMSR[InROI].flatten()

print(np.size(Inf_ROI,0))
print(np.size(OBS_ROI,0))

f_score = calculate_fscore_2(Inf_ROI, OBS_ROI, beta=2)


print('FSCORE: ', f_score)


# plt.figure(figsize=(12,6))
# plt.subplot(1, 2, 1)
# plt.imshow(OBS_EMSR)
# plt.title('Observed EMSR')
#
# plt.subplot(1, 2, 2)
# plt.imshow(Inference)
# plt.title('Inference')
#
# plt.suptitle('Fscore: '+str(f_score))
# # namout = 'EMSR271_AOI5_OBS_INF.png'
# plt.show()
