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
import sys
import json
import configparser
from osgeo import ogr
import argparse

sys.path.append('/home/akynos/AnacondaProjects36/floodml')
import aux_files.pykic.vector.pykic_ogr as vpo
from aux_files.S1Tiling.S1Processor import mainproc as s1p

###====----- Initialization
os.environ["PATH"] = os.environ["PATH"].split(";")[-1]
# Workpath
workdir = '/work/scratch/fatrasc/EMSR352/'

# os.system("source /home/eh/fatrasc/scratch/Bib_annexes/OTB-7.0/otbenv.profile")

def applyconf(jsonfile):
    with open(jsonfile, 'r') as f:
        return json.load(f)


def modifcfgs1(file, tiles, input, srtm, tmpdir):
    cfg = configparser.ConfigParser()
    cfg.read(file)
    cfg.set('DEFAULT', 'region', '')
    cfg.set('Paths', 'Output', 'Input')
    cfg.set('Paths', 'S1Images', input)
    cfg.set('Paths', 'SRTM', srtm)
    cfg.set('Paths', 'GeoidFile', os.path.join(os.path.dirname(file), 'Geoid/egm96.grd'))
    cfg.set('Paths', 'tmp', tmpdir)
    cfg.set('PEPS', 'Download', 'False')

    cfg.set('Processing', 'TilesShapefile', os.path.join(os.path.dirname(file), 'shapefile/Features.shp'))
    cfg.set('Processing', 'SRTMShapefile', os.path.join(os.path.dirname(file), 'shapefile/srtm.shp'))
    cfg.set('Processing', 'tiles', tiles)
    cfg.set('Filtering', 'filtering_activated', 'False')

    output = os.path.join(tmpdir, 'S1Processor_tmp.cfg')
    cfg.write(open(output, 'w'))
    return output



## Check Tiles ####################
pfoot = glob.glob(os.path.join(workdir, '*area_of_interest*.shp'))[0] # Senegal 28QDD - Landes 30TXP - Der 31UFP
ps2t = os.path.join('/work/scratch/fatrasc/SHP', 'Sentinel2_tiles.shp')
print('pfoot',pfoot)


chkp, proj_foot, proj_tile = vpo.checkproj(pfoot, ps2t)
oEPSG = int(proj_tile.split(':')[1])
print(oEPSG)
if not chkp:
    _, pfoot_new = vpo.ogreproj(pfoot, oEPSG, write=True)
    layer_int = vpo.sprocessing(pfoot_new, ps2t, method='intersects')
    ogrl = ogr.Open(pfoot_new)
else:
    layer_int = vpo.sprocessing(pfoot, ps2t, method='intersects')
    ogrl = ogr.Open(pfoot)

tiles = layer_int['Name']  # Sometimes Name_left, usually 'Name'
tiles = list(dict.fromkeys(tiles))
print(tiles)

# Extent (must be in WGS84)
if oEPSG == 4326:
    x_min, x_max, y_min, y_max = ogrl.GetLayer().GetExtent()
elif int(proj_foot.split(':')[1]) == 4326:
    ogrl = ogr.Open(pfoot)
    x_min, x_max, y_min, y_max = ogrl.GetLayer().GetExtent()
elif int(proj_foot.split(':')[1]) != 4326:
    _, _ = vpo.ogreproj(pfoot, 4326, write=True)
    pfoot_tmp = pfoot.replace('.shp', '_{0:d}_tmp.shp'.format(4326))
    ogrl = ogr.Open(pfoot_tmp)
    x_min, x_max, y_min, y_max = ogrl.GetLayer().GetExtent()
    shptmp = glob.glob(pfoot_tmp.replace('.shp', '*'))
    for sp in shptmp:
        os.remove(sp)



# ### initialisations
s1tmp = os.path.join(workdir, 'TMP')
srtmp = '/datalake/static_aux/MNT/SRTM_30_hgt'

"""
dag = EODataAccessGateway(user_conf_file_path='S1_tiler/eodag.yml')
dag.set_preferred_provider(u'peps')

product_type = 'S1_SAR_GRD'
date_d = '2019-04-08'
date_f = '2019-04-09'
extent = {'lonmin': x_min,
          'lonmax': x_max,
          'latmin': y_min,
          'latmax': y_max}
print("dag.search")
dags_S1_GRD, _ = dag.search(productType=product_type,
                            items_per_page=500,
                            box=extent,
                            start=date_d,
                            end=date_f)
print(dags_S1_GRD)


print("\nDownloading...")
if len(dags_S1_GRD) == 0:
    print('\nNo match.')
else:
    print('\nData exists!')

    for f in dags_S1_GRD:
        print(f)
        os.system("cd EMSR352")
        path = dag.download(f)
        os.system("cd ..")
        print('\n PATH: ', path)
        # os.system('unzip ' + f + ' -d ./RAW/')
        # os.system('rm -f ' + path)
"""
print("Dezip")
# files_RAW = glob.glob(os.path.join(workdir, '*.zip'))
# for raw in files_RAW:
#     os.system('unzip -f ' + raw + ' -d .')
#     # os.system('rm -f ' + path)


files = glob.glob(os.path.join(workdir,'S1dir', '*.SAFE'))
print("Files to consider: ",files)
# S1 Tiling ######################
conf = applyconf(os.path.join('/work/scratch/fatrasc/S1_tiler', 'config.json'))

for s1f in files:
    for tile in tiles:

        # file, tiles, input, srtm, tmpdir
        cfg_s1 = modifcfgs1(os.path.join('/work/scratch/fatrasc/Bib_annexes/S1tiling/', 'S1Processor.cfg'),
                            tile,
                            os.path.join(workdir, 'S1dir'),
                            srtmp,
                            os.path.join(workdir,'TMP'))
        print('\ncfg_s1')
        print(cfg_s1)
        ins1 = argparse.ArgumentParser()
        ins1.add_argument('--input', type=str)
        ins1.add_argument('--zip', type=bool)
        args1 = ins1.parse_args(['--input', cfg_s1, '--zip', False])

        print("\n\nTiling beginning")
        s1p.mainproc(args1)

