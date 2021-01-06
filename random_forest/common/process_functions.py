#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES, CLS, SIRS - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES
"""


def applyconf(jsonfile):
    with open(jsonfile, 'r') as f:
        return json.load(f)

def modifeodag(ymlfile, input):
    with open(ymlfile, 'r') as f:
        data = yaml.safe_load(f)
    for key, value in data.items():
        for key2, value2 in value.items():
            if value2:
                for key3, value3 in value2.items():
                    if key3 =='outputs_prefix':
                        data[key][key2][key3] = input

    output = os.path.join(input, 'eodag_tmp.yml')
    with open(output, 'w') as f:
        yaml.safe_dump(data, f)
    return output

def modifcfgs1(file, tiles, input, srtm, tmpdir):
    cfg = configparser.ConfigParser()
    cfg.read(file)
    cfg.set('DEFAULT', 'region', '')
    cfg.set('Paths', 'Output', input)
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

def moveandzip(imtmp, workdir, delzip=True):
    output = []
    for im in imtmp:
        new_im = im.replace('file://', '')
        if os.path.dirname(new_im) != workdir:
            shutil.move(new_im, workdir)
            new_im = os.path.join(workdir, os.path.basename(im))
        if new_im.find('.zip'):
            # Unzip file
            with ZipFile(new_im, mode='r') as zipObj:
                zipObj.extractall(os.path.dirname(new_im))
            if delzip == True:
                os.remove(new_im)
            new_im = new_im.replace('.zip', '.SAFE')
        output.append(new_im)
    return output