#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Sea Level Utilities

Created on Sun Mar 14 21:51:31 2021

@author: gliu
"""

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import pygmt
from tqdm import tqdm

import glob
import time
import cmocean
import time
#import tqdm
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz
import yo_box as ybx
import tbx
from scipy.signal import butter, lfilter, freqz, filtfilt, detrend

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pylab import cm


from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform


# ----------------
# %% Preprocessing
# ----------------

def remove_GMSL(ssh,lat,lon,times,tol=1e-10,viz=False,testpoint=[330,50]):
    """
    Parameters
    ----------
    ssh : ARRAY [time x lat x lon]
        Sea Surface Height Anomalies to process
    lat : ARRAY [lat]
        Latitude values
    lon : ARRAY [lon]
        Longitude values 
    times : ARRAY [time]
        Time values to plot (years)
    tol : FLOAT, optional
        Tolerance to check if GMSL is zero. The default is 1e-10.
    viz : BOOL, optional
        Visualize GMSL removal. The default is False.
    testpoint : LIST [lon,lat]
        Longitude and latitude points to visualize removal at. The default is [330,50].

    Returns
    -------
    ssh : ARRAY [time x lat x lon]
        Sea surface height anomalies with GMSL removed
    gmslrem: ARRAY [time]
        Time series that was removed
    
    Additional outputs for viz == True:
        
        fig,ax that was visualized
        
    Note: Add latitude weights for future update...
    """
    # Calculate GMSL (Not Area Weighted)
    gmslrem = np.nanmean(ssh,(1,2))
    
    if np.any(gmslrem>tol):
        # Remove GMSL
        ssh_ori = ssh.copy()
        ssh     = ssh - gmslrem[:,None,None]
        
        # Plot Results
        if viz:
            lonf,latf = testpoint
            klon,klat = proc.find_latlon(lonf,latf,lon,lat)
            
            fig,ax = plt.subplots(1,1)
            ax.set_xticks(np.arange(0,240,12))
            ax.set_xticklabels(times[::12],rotation = 45)
            ax.grid(True,ls='dotted')
            
            ax.plot(ssh_ori[:,klat,klon],label="Original",color='k')
            ax.plot(ssh[:,klat,klon],label="Post-Removal")
            ax.plot(gmslrem,label="GMSL")
            
            ax.legend()
            ax.set_title("GMSL Removal at Lon %.2f Lat %.2f (%s to %s)" % (lon[klon],lat[klat],times[0],times[-1]))
            ax.set_ylabel("SSH (m)")
            return ssh,gmslrem,fig,ax
    else:
        print("GMSL has already been removed, largest value is %e" % (gmslrem.max()))
    return ssh,gmslrem


def lp_butter(varmon,cutofftime,order):
    """
    Design and apply a low-pass filter (butterworth)

    Parameters
    ----------
    varmon : 
        Input variable to filter (monthly resolution)
    cutofftime : INT
        Cutoff value in months
    order : INT
        Order of the butterworth filter

    Returns
    -------
    varfilt : ARRAY [time,lat,lon]
        Filtered variable

    """
    # Input variable is assumed to be monthy with the following dimensions:
    flag1d=False
    if len(varmon.shape) > 1:
        nmon,nlat,nlon = varmon.shape
    else:
        flag1d = True
        nmon = varmon.shape[0]
    
    # Design Butterworth Lowpass Filter
    filtfreq = nmon/cutofftime
    nyquist  = nmon/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
    # Reshape input
    if ~flag1d: # For 3d inputs, loop thru each point
        varmon = varmon.reshape(nmon,nlat*nlon)
        # Loop
        varfilt = np.zeros((nmon,nlat*nlon)) * np.nan
        for i in tqdm(range(nlon*nlat)):
            varfilt[:,i] = filtfilt(b,a,varmon[:,i])
        varfilt=varfilt.reshape(nmon,nlat,nlon)
    else: # 1d input
        varfilt = filtfilt(b,a,varmon)
    return varfilt


# -----------------
# %% Visualization
# -----------------

def add_coast_grid(ax,bbox=[-180,180,-90,90],proj=None):
    """
    Add Coastlines, grid, and set extent for geoaxes
    
    Parameters
    ----------
    ax : matplotlib geoaxes
        Axes to plot on 
    bbox : [LonW,LonE,LatS,LatN], optional
        Bounding box for plotting. The default is [-180,180,-90,90].
    proj : cartopy.crs, optional
        Projection. The default is None.

    Returns
    -------
    ax : matplotlib geoaxes
        Axes with setup
    """
    if proj is None:
        proj = ccrs.PlateCarree()
    ax.add_feature(cfeature.COASTLINE,color='black',lw=0.75)
    ax.set_extent(bbox)
    gl = ax.gridlines(crs=proj, draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle="dotted",lw=0.75)
    gl.xlabels_top = False
    gl.ylabels_right = False
    return ax