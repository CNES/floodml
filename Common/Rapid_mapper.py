#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Project:        FloodML, CNES

Creates an output image displaying the flooded area
Indicates the EPSG projection and the water extent estimation date
The background is a Googlemap tile
"""

from osgeo import osr
import os
import sys
import ssl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from datetime import datetime
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from PIL import Image
from Common.ImageTools import gdal_warp
from Common.ImageIO import transform_point
from Common.GDalDatasetWrapper import GDalDatasetWrapper
from Common import RDF_tools


def draw_scale_bar(ax, central_lat, central_lon, length=20, unit="km"):
    """
    Draw a scale bar on the given ax-handle with fixed length

    :param ax: Matplotlib ax-handle
    :param central_lat: Central latitude expressed in PlateCarree() projection
    :param central_lon: Central longitude expressed in PlateCarree() projection
    :param length: Length expressed in the given unit
    :param unit: Unit as string
    :return:
    """
    merc = ccrs.TransverseMercator(central_lon, central_lat, approx=True)
    x0, x1, y0, y1 = ax.get_extent(merc)
    sbx = x0 + (x1 - x0) * .1  # 10% from the left
    sby = y0 + (y1 - y0) * .05  # 5% from the bottom
    bar_xs = [sbx - length * 500, sbx + length * 500]  # Convert to meters (Scale factor 500)
    ax.plot(bar_xs, [sby, sby],
            transform=merc,
            color='k',
            linewidth=2.5, zorder=10)
    ax.text(sbx, sby, '%s %s' % (length, unit),
            transform=merc,
            horizontalalignment='center',
            verticalalignment='bottom', zorder=10)


def draw_legend(ax3, sat):
    """
    Draw legend in a seperate subplot

    :param ax3: Ax-Handle
    :param sentinel: 1 or 2 (s1 or s2)
    :return:
    """
    ax3.set_title("Legend", loc="left")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    if sat in ['s2', 'l8',"l9"]:
        flood_sq = patches.Rectangle((0, .8), .1, 0.1, linewidth=0, edgecolor='black', facecolor='#AA0000', alpha=1)
        gsw_sq = patches.Rectangle((0., .6), .1, 0.1, linewidth=0, edgecolor='black', facecolor='#222E50', alpha=1)
        cldsh_sq = patches.Rectangle((0., .4), .1, 0.1, linewidth=0, edgecolor='black', facecolor='#439A86', alpha=1)
        cld_sq = patches.Rectangle((0., .2), .1, 0.1, linewidth=0, edgecolor='black', facecolor='#E9D985', alpha=1)
        nodata_sq = patches.Rectangle((0., .0), .1, 0.1, linewidth=0, edgecolor='black', facecolor='#BCB6B3', alpha=1)
        ax3.add_patch(flood_sq)
        ax3.add_patch(gsw_sq)
        ax3.add_patch(cldsh_sq)
        ax3.add_patch(cld_sq)
        ax3.add_patch(nodata_sq)
        ax3.text(0.15, 0.8, "Estimated flooded area", fontsize=9)
        ax3.text(0.15, 0.6, "Permanent water (occurrence >50%)", fontsize=9)
        ax3.text(0.15, 0.4, "Cloud Shadows", fontsize=9)
        ax3.text(0.15, 0.2, "Clouds", fontsize=9)
        ax3.text(0.15, 0.0, "No data", fontsize=9)
    else:
        flood_sq = patches.Rectangle((0, .6), .1, 0.2, linewidth=0, edgecolor='black', facecolor='#AA0000', alpha=1)
        gsw_sq = patches.Rectangle((0., .3), .1, 0.2, linewidth=0, edgecolor='black', facecolor='#222E50', alpha=1)
        nodata_sq = patches.Rectangle((0., .0), .1, 0.2, linewidth=0, edgecolor='black', facecolor='#BCB6B3', alpha=1)
        ax3.add_patch(flood_sq)
        ax3.add_patch(gsw_sq)
        ax3.add_patch(nodata_sq)
        ax3.text(0.15, 0.65, "Estimated flooded areas", fontsize=9)
        ax3.text(0.15, 0.4, "Permanent water (occurrence >50%)", fontsize=9, va='center')
        ax3.text(0.15, 0.1, "No data", fontsize=9)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.axis('off')


def draw_data_source(ax4, **kwargs):
    """
    Draw data source in a separate subplot

    :param ax4: Ax-Handle
    :param kwargs: Dict to display
    :return:
    """
    ax4.set_title("Data source", loc="left")
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.axis('off')

    # TABLE
    proj = kwargs.get("projection", "Unknown")
    source = kwargs.get("sat", "Unknown")
    rad = kwargs.get("rad", "Unknown")
    orbit = kwargs.get("orbit", "Unknown")
    date = kwargs.get("date", "Unknown")
    pol = kwargs.get("pol", "Unknown")

    if source=='s1': dispsat='Sentinel-1'
    elif source=='s2': dispsat='Sentinel-2'
    elif source=='l8': dispsat='Landsat-8'
    elif source=='l9': dispsat='Landsat-9'
    elif source=='tsx': dispsat='TerraSAR-X/TanDEM-X'
    else: dispsat='Unknown source'

    if source in ['s1','tsx']:
        table_vals = [['Map Projection', str(proj)],
                  ['Data source', str(dispsat)],
                  ['Relative orbit', str(orbit)],
                  ['Acq. date (UTC)', str(date)],
                  ['Polarization', str(pol)],
                  ['Map production date', str(datetime.now())]]            
    elif source in ['s2', 'l8', 'l9']:
        table_vals = [['Map Projection', str(proj)],
                  ['Data source', str(dispsat)],
                  ['Relative orbit', str(orbit)],
                  ['Acq. date (UTC)', str(date)],
                  ['Map production date', str(datetime.now())]]

    n_lines = len(table_vals)
    # if post==1:
    table_vals = np.vstack((table_vals, ['Post-processing', 'Majority filter r=%s' % rad]))
    n_lines+=1

    line_width = 1/n_lines # line relative thickness
    ax4.table(cellText=table_vals, cellLoc='left',
                  bbox=[0., 1 - n_lines * line_width, 1, n_lines * line_width])


def draw_disclaimer(ax6, add=""):
    """
    Draw disclaimer in separate subplot

    :param ax6: Ax-Handle
    :param add: Append string to disclaimer
    :return:
    """

    credit = r"""
- This map is derived automatically using the FloodDAM Rapid-Mapping (FloodML) tool.
  More info: https://www.spaceclimateobservatory.org/flooddam-garonne
  
- How to cite this map: FloodDAM Rapid-Mapping (© CNES-CLS-CS, 2019-2021)
- Surface Water Occurrence (GSW) data: Jean-Francois Pekel, Andrew Cottam, Noel Gorelick, Alan S. Belward.
  High-resolution mapping of global surface water and its long-term changes. Nature 540, 418-422 (2016). (doi:10.1038/nature20584)
""" + add
    ax6.text(.01, 1, "Disclaimer", fontsize=10, wrap=True, ha='left', linespacing=1,
             verticalalignment='top',)
    ax6.text(0, .9, credit, fontsize=6, wrap=True, ha='left', linespacing=1.1,
             verticalalignment='top')
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.axis('off')


def static_display(infile, tmp_dir, gsw_files, date, pol, outfile, orbit, sat, background=None, rad=None):
    """
    Create a static display map using the binary inference mask.
    Overlays the mask over a google-maps/OSM background - Needs internet in order to request the data.

    :param infile: Path to inference mask (Binary)
    :param tile: Tile number e.g. 30TXM
    :param date: Date as string
    :param pol: Polarisation #HH or VV
    :param outfile: Path where the image shall be written to
    :param gswo_dir: GSW directory containing tiled gsw data in the format TILEID.tif e.g. 30TXM.tif
    :param sat: S1, S2 or TSX indicating whether the original image comes from S1, S2 or TSX.
    :param background: Optional filepath to image to override the WMTS background.
    :return:
    """

    # SSL workaround for urllib certificate path
    ssl._create_default_https_conext = ssl._create_unverified_context

    ds_in = GDalDatasetWrapper.from_file(infile)
    data = ds_in.array
    gt = ds_in.geotransform
    proj = ds_in.projection
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)
    projcs = inproj.GetAuthorityCode('PROJCS')
    epsg = str(ds_in.epsg)

    # Extent
    extent_ax1 = list(ds_in.extent(order="lonmin-lonmax", dtype=int))
    extent = list(ds_in.extent(order="lonmin-lonmax", dtype=float)) #xmin xmax ymin ymax

    if sat in ["l8","l9"] and (extent[2]<=0 or extent[3]<=0):
        if extent[2]<=0 or extent[3]<=0: # If we are in the southern hemisphere
            if projcs[2]=='6': #Turns northern hemisphere...
                projcs=projcs[0:2]+'7'+projcs[3:] # into southern hemisphere
                epsg = epsg[0:2]+'7'+epsg[3:]# into southern hemisphere

            extent[2]+=10000000 #  extent corrected in latitude
            extent[3]+=10000000 #  extent corrected in latitude
            extent_ax1 = list(extent)

    #  Permanent water mask
    gswo_projected = RDF_tools.gsw_cutter(tmp_dir, epsg, extent, gsw_files, res=[abs(gt[1]), abs(gt[-1])])

    #  Display the data
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 in inches

    heights = [1, .25, .45, .25]
    widths = [1, 1, 1]
    spec = gridspec.GridSpec(ncols=3, nrows=4, wspace=0.025, hspace=0.2, height_ratios=heights, width_ratios=widths)
    ax1 = fig.add_subplot(spec[:-1, :-1], projection=ccrs.epsg(projcs), anchor="NW")  # Main map
    ax2 = fig.add_subplot(spec[0, -1], projection=ccrs.PlateCarree(), anchor="NW")  # World localisation
    ax3 = fig.add_subplot(spec[1, -1], anchor="NW")  # Legend
    ax4 = fig.add_subplot(spec[2, -1], anchor="NW")  # Data description
    ax5 = fig.add_subplot(spec[-1, :-1], anchor="NW")  # Disclaimer
    ax6 = fig.add_subplot(spec[-1, 1:], anchor="SE")  # Logos
    # Cartopy 0.18 bug - No interpolation option available: https://github.com/SciTools/cartopy/issues/1563

    ax1.set_extent(extent, crs=ccrs.epsg(projcs))

    # Main Background image and gridlines
    if not background:
        print("Using WMTS background.")

        bg_map = cimgt.GoogleTiles(url="http://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png") ##Initial, marche plus
        ax1.add_image(bg_map, 11, interpolation="spline36", regrid_shape=2000)
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=.3, color='gray', alpha=0.8, zorder=9)
        disclaimer_add =\
            "- Fond de carte par Yohan Boniface & Humanitarian OpenStreetMap Team sous licence domaine public CC0"
    else:
        bg = GDalDatasetWrapper.from_file(background)
        visu = np.moveaxis(bg.array, 0, -1)
        ax1.imshow(visu, extent=extent, transform=ccrs.epsg(projcs),  origin='upper', interpolation="bicubic")
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=.3, color='black', alpha=1, zorder=9)
        disclaimer_add = ""

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}

    # Scale Bar
    lonmin, lonmax, latmin, latmax = ax1.get_extent(ccrs.PlateCarree())
    lon_center = lonmin + (lonmax - lonmin) * .1
    lat_center = latmin + (latmax - latmin) * .05
    draw_scale_bar(ax1, central_lat=lat_center, central_lon=lon_center)

    # Flooded area display (in red), permanent water in blue
    masked_data = np.ma.masked_where(data != 1, data)
    masked_gsw = np.ma.masked_where(gswo_projected.array < 50, gswo_projected.array)
    masked_gsw = np.ma.masked_where(data > 1, masked_gsw)

    cmap2 = matplotlib.colors.ListedColormap(["#222E50"], name='from_list', N=None)  # Color for perma areas
    img2 = ax1.imshow(masked_gsw, extent=extent, transform=ccrs.epsg(projcs),  origin='upper', cmap=plt.get_cmap(cmap2),
                      alpha=1, interpolation="nearest")
    img2.set_zorder(4)

    cmap = matplotlib.colors.ListedColormap(["#AA0000"], name='from_list', N=None)  # Color for flooded areas 
    img = ax1.imshow(masked_data, extent=extent, transform=ccrs.epsg(projcs), origin='upper', cmap=plt.get_cmap(cmap),
                     alpha=1, interpolation="nearest")
    img.set_zorder(3)

    if sat in ["s2","l8","l9"]:
        #Cloud shadow
        masked_cld_shadow = np.ma.masked_where(data != 7, data)
        cmap3 = matplotlib.colors.ListedColormap(["#439A86"], name='from_list', N=None)  # Color for cloud shadow
        img3 = ax1.imshow(masked_cld_shadow, extent=extent, transform=ccrs.epsg(projcs), origin='upper',
                          cmap=plt.get_cmap(cmap3),
                          alpha=.7, interpolation="nearest")
        img3.set_zorder(3)

        #Cloud
        masked_cld = np.ma.masked_where(data != 6, data)
        cmap4 = matplotlib.colors.ListedColormap(["#E9D985"], name='from_list', N=None)  # Color for cloud
        img4 = ax1.imshow(masked_cld, extent=extent, transform=ccrs.epsg(projcs), origin='upper',
                          cmap=plt.get_cmap(cmap4),
                          alpha=.7, interpolation="nearest")
        img4.set_zorder(2)

    masked_nodata = np.ma.masked_where(data != 255, data)
    cmap5 = matplotlib.colors.ListedColormap(["#BCB6B3"], name='from_list', N=None)  # Color for perma areas
    img5 = ax1.imshow(masked_nodata, extent=extent, transform=ccrs.epsg(projcs), origin='upper',
                      cmap=plt.get_cmap(cmap5),
                      alpha=.9, interpolation="nearest")
    img5.set_zorder(5)

    # AX2 - Location map
    lat_mean, lon_mean = transform_point((float(np.mean(extent[0:2])), float(np.mean(extent[2:4]))),
                                         old_epsg=int(epsg), new_epsg=4326)

    # This should be ratio 3:2:
    ax2.set_extent([lon_mean-15, lon_mean+15, lat_mean-15, lat_mean+15])  # lon1 lon2 latmin1 lat2
    ax2.set_xticks([])
    ax2.set_yticks([])
    qkl_map = cimgt.GoogleTiles(url="http://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png") ##Initial, marche plus
    ax2.add_image(qkl_map, 6, interpolation="spline36") #5
    
    pts_aoi = list()
    y, x = transform_point((extent_ax1[0], extent_ax1[2]), old_epsg=int(epsg), new_epsg=4326)
    pts_aoi.append([x, y])
    y, x = transform_point((extent_ax1[1], extent_ax1[2]), old_epsg=int(epsg), new_epsg=4326)
    pts_aoi.append([x, y])
    y, x = transform_point((extent_ax1[1], extent_ax1[3]), old_epsg=int(epsg), new_epsg=4326)
    pts_aoi.append([x, y])
    y, x = transform_point((extent_ax1[0], extent_ax1[3]), old_epsg=int(epsg), new_epsg=4326)
    pts_aoi.append([x, y])
    y, x = transform_point((extent_ax1[0], extent_ax1[2]), old_epsg=int(epsg), new_epsg=4326)
    pts_aoi.append([x, y])

    for lin in range(len(pts_aoi) - 1):
        xs = [pts_aoi[lin][0], pts_aoi[lin+1][0]]
        ys = [pts_aoi[lin][1], pts_aoi[lin+1][1]]
        ax2.plot(xs, ys, lw=1, color='red')

    # AX3 - Legend
    draw_legend(ax3, sat=sat)
    
    # AX4 - Data information
    draw_data_source(ax4, projection="EPSG:%s" % ds_in.epsg, sat=sat, orbit=orbit, date=date,
                     pol=pol, rad=rad)

    # AX5 - Disclaimer
    draw_disclaimer(ax5, add=disclaimer_add)

    # AX6 - Logos
    ax6.axis("off")
    im = Image.open(os.path.join(sys.path[0], 'flooddam.png'))
    ax6.imshow(im, aspect='equal')

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=.99, bottom=0, right=.99, left=.06,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(outfile, dpi=300)

    plt.close()
    return plt
