#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""

import os, glob, sys, logging, argparse, shutil, gc, joblib
from distutils.dir_util import copy_tree
from osgeo import ogr
import lxml.etree as ET
import numpy as np
from tqdm import tqdm

import FML_disp as fml
from process_functions import *


bypass = True

# Tuiles S2
pwd = os.getcwd()
ps2t = os.path.join(pwd, 's2tiles', 'Sentinel2_tiles.shp')

def mainproc(args):
    ## Parse inputs ###################
    pconf = args.config
    input = args.input
    if input[-1] == '/': input = input[0:-1]
    output = args.output
    if output[-1] == '/': output = output[0:-1]
    if args.workdir:
        workdir = args.workdir
    else:
        workdir = os.path.join(os.getcwd(), 'tmp')
        if not os.path.isdir(workdir):
            os.mkdir(workdir)
    pfoot = args.footprint
    reso = args.resolution
    sdate = args.startDate
    edate = args.endDate


    ## Import config ##################
    outyml_eodag = modifeodag('eodag.yml', workdir)

    conf = applyconf(os.path.join(pwd, pconf))
    srtmp = conf['srtm']
    # EODAG
    sys.path.append(conf['eodag'])
    from eodag import EODataAccessGateway
    # SEN2COR
    bin_s2c = conf['bin_sen2cor']
    # S1TILING
    sys.path.append(conf['s1tiling'])
    import S1Processor as s1p
    # PYKIC
    sys.path.append(conf['pykic'])
    import vector.pykic_ogr as vpo
    import raster.pykic_gdal as rpg
    import raster.resafilter as rrf
    import miscellaneous.vi as mvi
    import miscellaneous.miscdate as mmd


    ## Check Tiles ####################
    chkp, proj_foot, proj_tile = vpo.checkproj(pfoot, ps2t)
    oEPSG = int(proj_tile.split(':')[1])
    if not chkp:
        _, pfoot_new = vpo.ogreproj(pfoot, oEPSG, write=True)
        layer_int = vpo.sprocessing(pfoot_new, ps2t, method='intersects')
        ogrl = ogr.Open(pfoot_new)
    else:
        layer_int = vpo.sprocessing(pfoot, ps2t, method='intersects')
        ogrl = ogr.Open(pfoot)
    tiles = layer_int['Name']

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

    
    if not bypass:
        ## EO Data collection #############
        sdate_ts = mmd.dconvert(sdate, fmtin='%Y%m%d', fmtout='ts')
        edate_ts = mmd.dconvert(edate, fmtin='%Y%m%d', fmtout='ts')
    
        # EODAG
        extent = {'lonmin': x_min,
                  'lonmax': x_max,
                  'latmin': y_min,
                  'latmax': y_max}
        dag = EODataAccessGateway(user_conf_file_path=outyml_eodag)
        dag.set_preferred_provider(u'peps')
        product_type = 'S2_MSI_L2A' # ATTENTION IMPOSSIBLE DE CHOISIR PROVIDER
        dags_L2A, _ = dag.search(productType=product_type,
                                 items_per_page=500,
                                 box=extent,
                                 start=mmd.dconvert(sdate, fmtin='%Y%m%d', fmtout='%Y-%m-%d'),
                                 end=mmd.dconvert(edate, fmtin='%Y%m%d', fmtout='%Y-%m-%d'))
        product_type = 'S2_MSI_L1C'
        dags_L1C, _ = dag.search(productType=product_type,
                                 items_per_page=500,
                                 box=extent,
                                 start=mmd.dconvert(sdate, fmtin='%Y%m%d', fmtout='%Y-%m-%d'),
                                 end=mmd.dconvert(edate, fmtin='%Y%m%d', fmtout='%Y-%m-%d'))
        product_type = 'S1_SAR_GRD'
        dags_S1_GRD, _ = dag.search(productType=product_type,
                                 items_per_page=500,
                                 box=extent,
                                 start=mmd.dconvert(sdate, fmtin='%Y%m%d', fmtout='%Y-%m-%d'),
                                 end=mmd.dconvert(edate, fmtin='%Y%m%d', fmtout='%Y-%m-%d'))
    
        # SENTINEL2
        for r in tiles:
            # Download L2A
            data01 = glob.glob(os.path.join(input, '**/*', 'MTD_MSIL2A.xml'), recursive=True)
            date01 = []
            tmp = data01.copy()
            for fi in tmp:
                xmlt = ET.parse(fi).findall('.//PRODUCT_START_TIME')[0]
                str_prod = str(ET.tostring(xmlt)).replace('>', '$$').replace('</', '$$').replace('.', '$$').replace('T', '$$').replace('-', '').split('$$')
                for da in str_prod:
                    test = mmd.datefromstr(da)
                    if test:
                        cur_date_ts = mmd.dconvert(str(test), fmtin='%Y%m%d', fmtout='ts')
                        break
    
                xmlt = ET.parse(fi).findall('.//PRODUCT_URI')[0]
                uri = str(ET.tostring(xmlt))
                if uri.find(r) < 0 or cur_date_ts < sdate_ts or cur_date_ts > edate_ts:
                    data01.remove(fi)
                else:
                    date01.append(test)
    
            im_dl = []
            date02 = []
            for d in dags_L2A:
                if data01:
                    if mmd.datefromstr(str(d)) not in date01 and str(d).find(r) >= 0:
                        date02.append(mmd.datefromstr(str(d)))
                        # path = dag.download(d) # Download
                        # if path.find('.zip') < 0:
                        #     continue
                        # im_dl.append(path)
                        # logging.warning(' Downloaded : {}'.format(path))
                else:
                    if str(d).find(r) >= 0:
                        date02.append(mmd.datefromstr(str(d)))
                        # path = dag.download(d) # Download
                        # if path.find('.zip') < 0:
                        #     continue
                        # im_dl.append(path)
                        # logging.warning(' Downloaded : {}'.format(path))
    
            # Move
            if im_dl:
                L2A_collection = moveandzip(im_dl, input, delzip=True)
    
            # Download L1C
            date01 = date01 + date02
            data02 = glob.glob(os.path.join(input, '**/*', 'MTD_MSIL1C.xml'), recursive=True)
            date02 = []
            tmp = data02.copy()
            for fi in tmp:
                xmlt = ET.parse(fi).findall('.//PRODUCT_START_TIME')[0]
                str_prod = str(ET.tostring(xmlt)).replace('>', '$$').replace('</', '$$').replace('.', '$$').replace('T', '$$').replace('-', '').split('$$')
                for da in str_prod:
                    test = mmd.datefromstr(da)
                    if test:
                        cur_date_ts = mmd.dconvert(str(test), fmtin='%Y%m%d', fmtout='ts')
                        break
    
                xmlt = ET.parse(fi).findall('.//PRODUCT_URI')[0]
                uri = str(ET.tostring(xmlt))
                if uri.find(r) < 0 or cur_date_ts < sdate_ts or cur_date_ts > edate_ts:
                    data02.remove(fi)
                else:
                    if test in date01:
                        data02.remove(fi)
                    else:
                        date02.append(test)
    
            im2corr = data02.copy()
            im2zip = []
            date03 = date01 + date02
            for d in dags_L1C:
                if mmd.datefromstr(str(d)) not in date03 and str(d).find(r) >= 0:
                    date03.append(mmd.datefromstr(str(d)))
                    path = dag.download(d) # Download
                    if path.find('.zip') < 0:
                        continue
                    im2zip.append(path)
                    logging.warning(' Downloaded : {}'.format(path))
    
            # Move
            if im2zip:
                new_im2corr = moveandzip(im2zip, workdir, delzip=True)
                im2corr.append(new_im2corr)
    
            # Sen2Cor Atmospheric correction
            if os.path.isdir(bin_s2c):
                for im in im2corr:
                    if im.find('xml'):
                        im = os.path.dirname(im)
                    cmd = 'bash {0:s} --output_dir {1:s} {2:s}'.format(os.path.join(bin_s2c, 'L2A_Process'), input, im)
                    os.system(cmd)
                    shutil.rmtree(im)
            else:
                logging.error(' Sen2cor binaries does not exist or wrong path !')
    
            # MAJA
    
        # Download SENTINEL1
        # Check data from S1Tiling ?
        data05 = []
        date05 = []
        tmp = glob.glob(os.path.join(input, '**/*', 'manifest.safe'), recursive=True)
        for fi in tmp:
            if str(ET.tostring(ET.parse(fi))).find('<safe:familyName>SENTINEL-1</safe:familyName>') >= 0: # Sentinel1
                xmlt = str(ET.tostring(ET.parse(fi))).replace('<safe:startTime>', '$$').replace('</safe:startTime>', '$$').split('$$')[1]
                test = mmd.dconvert(xmlt.split('T')[0], fmtin='%Y-%m-%d', fmtout='%Y%m%d')
                cur_date_ts = mmd.dconvert(test, fmtin='%Y%m%d', fmtout='ts')
    
                if cur_date_ts >= sdate_ts or cur_date_ts <= edate_ts:
                    date05.append(test)
                    data05.append(os.path.dirname(fi))
    
        im_dl = []
        for d in dags_S1_GRD:
            if data05:
                if str(mmd.datefromstr(str(d))) not in date05:
                    date05.append(mmd.datefromstr(str(d)))
                    path = dag.download(d) # Download
                    if path.find('.zip') < 0:
                        continue
                    im_dl.append(path)
                    logging.warning(' Downloaded : {}'.format(path))
            else:
                date05.append(mmd.datefromstr(str(d)))
                path = dag.download(d) # Download
                if path.find('.zip') < 0:
                    continue
                im_dl.append(path)
                logging.warning(' Downloaded : {}'.format(path))
    
        # Move
        s1tmp = os.path.join(workdir, 'tmps1_work')
        if os.path.isdir(s1tmp):
            shutil.rmtree(s1tmp)
        os.mkdir(s1tmp)
        if im_dl:
            new_im_dl = moveandzip(im_dl, s1tmp, delzip=True)
        if data05:
            for m in data05:
                tmpout = os.path.join(s1tmp, os.path.basename(m))
                os.mkdir(tmpout)
                copy_tree(m, tmpout)
    
    
        ## S1 Tiling ######################
        for ti in tiles:
            cfg_s1 = modifcfgs1(os.path.join(conf['s1tiling'], 'S1Processor.cfg'),
                                ti,
                                s1tmp,
                                srtmp,
                                workdir)
    
            ins1 = argparse.ArgumentParser()
            ins1.add_argument('--input', type=str)
            ins1.add_argument('--zip', type=bool)
            args1 = ins1.parse_args(['--input', cfg_s1, '--zip', False])
            s1p.mainproc(args1)
    
        # Move
        s1t = glob.glob(os.path.join(s1tmp, '**/*', '.tif'), recursive=True)
        s1move = os.path.join(input, 's1tiling')
        if not os.path.isdir(s1move):
            os.mkdir(s1move)
        for s in s1t:
            shutil.copy2(s, s1move)
    
        # Clean
        shutil.rmtree(s1tmp)
        os.remove(outyml_eodag)
        os.remove(cfg_s1)
    
    # Graphic (Footprint)
    foot_fig = 'Footprint.png'
    fml.footprint_displayer(input, os.path.join(input, 'Footprints.png'), pfoot)


    ## Synthetic bands ################
    data_s = glob.glob(os.path.join(input, '**/*', 'MTD_MSIL2A.xml'), recursive=True)
    for sb in tqdm(data_s, desc='Synthetic bands in progress', total=len(data_s)):
        pdir = os.path.dirname(sb)
        out_index = os.path.join(pdir, 'index')
        if not os.path.isdir(out_index):
            os.mkdir(out_index)
        swirpath = glob.glob(os.path.join(pdir, '**/*', '*B11_20m.jp2'), recursive=True)

        # Import
        img, proj, dim, tr = rpg.gdal2array(pdir, sensor='SEN2COR')
        swir, _, _, _ = rpg.gdal2array(swirpath[0])
        swir = rrf.resample_2d(swir, dim, method='nearest')

        # NDVI
        name = os.path.basename(swirpath[0].replace('B11_20m.jp2', 'NDVI.tif'))
        ndvi = mvi.ndvi(img[:,:,3], img[:,:,2])
        ndvi[ndvi < 0] = 0
        ndvi = np.uint8(ndvi * 254)
        ndvi[img[:,:,-1] > 0] = 255 # Cloud / NoData mask
        rpg.array2tif(os.path.join(out_index, name), ndvi, proj, dim, tr, format='uint8')
        fml.Synthetic_displayer(os.path.join(out_index, name), os.path.join(out_index, 'Display_NDVI.png'), cmap='BrBG')

        # MNDWI
        name = os.path.basename(swirpath[0].replace('B11_20m.jp2', 'MNDWI.tif'))
        ndwi = mvi.ndwi(swir, img[:,:,1])
        ndwi = np.int16(ndwi * 254)
        ndwi[img[:,:,-1] > 0] = 255 # Cloud / NoData mask
        rpg.array2tif(os.path.join(out_index, name), ndwi, proj, dim, tr, format='int16')
        fml.Synthetic_displayer(os.path.join(out_index, name), os.path.join(out_index, 'Display_MNDWI.png'), cmap='RdBu')

        img = None; swir = None; ndwi = None; ndvi = None
        gc.collect()


    ## Apply model ####################
    # Load model
    rdf = joblib.load('rdf_model.sav')

    # Extract data
    d_index = sorted(glob.glob(os.path.join(input, '**/*', '*NDVI.tif'), recursive=True))
    yb = np.array([])
    for sa in d_index:
        pndwi = sa.replace('NDVI.tif', 'MNDWI.tif')
        ndvi, proj, dim, tr = rpg.gdal2array(sa)
        ndwi, _, _, _ = rpg.gdal2array(pndwi)

        ndvi = np.float32(ndvi)
        ndvi[ndvi == 255] = np.nan

        ndwi = np.float32(ndwi)
        ndwi[ndwi == 255] = np.nan

        if yb.size == 0:
            yb = np.hstack((np.reshape(ndvi/254, (-1, 1)), np.reshape(ndwi/254, (-1, 1))))
        else:
            yb = np.hstack((yb, np.hstack((np.reshape(ndvi/254, (-1, 1)), np.reshape(ndwi/254, (-1, 1))))))
        ndvi = None; ndwi = None
        gc.collect()

    # Remove NaN & predict
    vec_ok = np.ravel(np.flatnonzero(np.sum(np.isnan(yb) * 1, axis=1) == 0))
    yb = yb[vec_ok]
    rdf_pred = rdf.predict(yb)

    # Output image
    exout = np.zeros((dim[1], dim[0]), dtype=np.uint8)
    exout[np.unravel_index(vec_ok, (dim[1], dim[0]))] = rdf_pred

    # Export
    nexout = os.path.join(input, 'export_flood.tif')
    rpg.array2tif(nexout, exout, proj, dim, tr)
    
    # Export
    fml.Flood_displayer(nexout, os.path.join(output, 'Flood_display.tif'))
    
    
    

    # TEST
    svm = joblib.load('svm_model.sav')
    svm_pred = svm.predict(yb)
    exout3 = np.zeros((dim[1], dim[0]), dtype=np.uint8)
    exout3[np.unravel_index(vec_ok, (dim[1], dim[0]))] = svm_pred

    knn = joblib.load('knn_model.sav')
    knn_pred = knn.predict(yb)
    exout2 = np.zeros((dim[1], dim[0]), dtype=np.uint8)
    exout2[np.unravel_index(vec_ok, (dim[1], dim[0]))] = knn_pred

    mlp = joblib.load('mlp_model.sav')
    mlp_pred = mlp.predict(yb)


## Main Call ####################################
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Ordonnanceur AI4FM')
    parser.add_argument('-c', '--config', help='Config file', type=str, required=True)
    parser.add_argument('-i', '--input', help='Input folder', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output folder', type=str, required=True)
    parser.add_argument('-w', '--workdir', help='Temporary folder (OPTIONAL)', type=str)
    parser.add_argument('-f', '--footprint', help='Footprint', type=str, required=True)
    parser.add_argument('-r', '--resolution', help='Resolution', type=int, required=True)
    parser.add_argument('-s', '--startDate', help='Start date - format YYYYMMDD', type=str, required=True)
    parser.add_argument('-e', '--endDate', help='End date - format YYYYMMDD', type=str, required=True)
    args = parser.parse_args()

    mainproc(args)
