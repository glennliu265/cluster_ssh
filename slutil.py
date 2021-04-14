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
            #ax.set_xticks(np.arange(0,len(times)+1,12))
            ax.set_xticks(np.arange(0,len(times),12))
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


# -------------
# %% Clustering
# -------------

def calc_matrices(invar,lon,lat,return_all=False):
    """
    Calculate correlation, covariance, and distance matrices in preparation
    for clustering.

    Parameters
    ----------
    invar : ARRAY (Time x Lat x Lon)
        Input variable
    lon : ARRAY (Lon)
        Longitudes
    lat : ARRAY (Lat)
        Latitudes
    return_all : BOOL, optional
        Set to true to return non-nan points, indices, and coordinates. The default is False.

    Returns
    -------
    srho: ARRAY [npts x npts]
        Correlation Matrix
    scov: ARRAY [npts x npts]
        Covariance Matrix
    sdist: ARRAY [npts x npts]
        Distance Matrix

    """
    
    # ---------------------
    # Remove All NaN Points
    # ---------------------
    ntime,nlat,nlon = invar.shape
    varrs = invar.reshape(ntime,nlat*nlon)
    okdata,knan,okpts = proc.find_nan(varrs,0)
    npts = okdata.shape[1]
    
    # ---------------------------------------------
    # Calculate Correlation and Covariance Matrices
    # ---------------------------------------------
    srho = np.corrcoef(okdata.T,okdata.T)
    scov = np.cov(okdata.T,okdata.T)
    srho = srho[:npts,:npts]
    scov = scov[:npts,:npts]
    
    # --------------------------
    # Calculate Distance Matrix
    # --------------------------
    lonmesh,latmesh = np.meshgrid(lon,lat)
    coords  = np.vstack([lonmesh.flatten(),latmesh.flatten()]).T
    coords  = coords[okpts,:]
    coords1 = coords.copy()
    coords2 = np.zeros(coords1.shape)
    coords2[:,0] = np.radians(coords1[:,1]) # First point is latitude
    coords2[:,1] = np.radians(coords1[:,0]) # Second Point is Longitude
    sdist = haversine_distances(coords2,coords2) * 6371
    
    if return_all:
        return srho,scov,sdist,okdata,okpts,coords2
    return srho,scov,sdist
    
def make_distmat(srho,sdist,distthres=3000,rhowgt=1,distwgt=1):
    """
    Make distance matrix, using output from calc_matrices
    
    dist = 1 - exp(-dist/(2a^2)) * corr

    Parameters
    ----------
    srho : ARRAY [npts x npts]
        DESCRIPTION.
    sdist : ARRAY [npts x npts]
        DESCRIPTION.
    distthres : INT, optional
        Point at which exponential term is 0.5. The default is 3000.
    rhowgt : FLOAT, optional
        Amount to weight the correlation matrix. The default is 1.
    distwgt : TYPE, optional
        Amount to weight the exponential (distance) term. The default is 1.

    Returns
    -------
    distance_matrix : ARRAY [npts x npts]
        Distance matrix for input to the clustering algorithm

    """
    
    # Calculate exponential term
    a_fac = np.sqrt(-distthres/(2*np.log(0.5))) # Calcuate so exp=0.5 when distance is distthres
    expterm = np.exp(-sdist/(2*a_fac**2))
    
    # Apply weighting to distance and correlation matrix
    # So that as wgt --> 0, value --> 1
    # and as wgt --> 1, value --> itself
    #expterm *= (expterm-1)*distwgt + 1
    #srho    *= ((np.abs(srho)-1)*rhowgt + 1)  * np.sign(srho)
    
    # Calculate 
    distance_matrix = 1-expterm*srho
    
    return distance_matrix
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
    gl.right_labels = False
    gl.top_labels = False
    return ax

def check_lpfilter(rawdata,lpdata,chkval,M,tw,dt=24*3600*30):
    """
    

    Parameters
    ----------
    rawdata : ARRAY [time x space]
        Raw Dataset (no NaNs)
    lpdata : ARRAY [time x space]
        Low-pass (LP) filtered dataset (no NaNs)
    chkval : INT
        Period (1/f) to check filter transfer function at (in units of dt)
    M : INT
        Bands to average over
    tw : INT
        Cutoff time low-pass filter was performed at (for plotting)
    dt : INT, optional
        Timestep size in seconds. The default is 24*3600*30.

    Returns
    -------
    lpspec : ARRAY [freq x space]
        Power spectra for LP filtered timeseries
    rawspec : ARRAY [freq x space]
        Power spectra for raw filtered timeseries
    p24 : ARRAY [space]
        Power spectra value for each point at chkval
    filtxfer : ARRAY [freq x space]
        Filter transfer function
    fig : mpl figure
    ax : mpl axes

    """
    # Set x-tick parameters
    xtk = [1/(10*12*dt),1/(24*dt),1/(12*dt),1/(3*dt),1/dt]
    xtkl = ['decade','2-yr','year','season','month']
    
    # Get number of points
    npts5 = lpdata.shape[1]
    
    # Compute power spectra for each point
    lpspec  = []
    rawspec = []
    for i in tqdm(range(npts5)):
        X_spec,freq,_=tbx.bandavg_autospec(rawdata[:,i],dt,M,.05)
        X_lpspec,_,_ =tbx.bandavg_autospec(lpdata[:,i],dt,M,.05)
        lpspec.append(X_lpspec)
        rawspec.append(X_spec)
    lpspec   = np.real(np.array(lpspec))
    rawspec  = np.real(np.array(rawspec))
    
    # Calculate filter transfer function
    filtxfer = lpspec/rawspec
    
    # Get index for the frequency of interest [k24mon]
    k24mon = np.argmin(np.abs(freq-chkval)) # Get index for 24 mon (chkval)
    if freq[k24mon] < chkval: # less than 24 months
        ids = [k24mon,k24mon+1]
    else:
        ids = [k24mon-1,k24mon]
    
    # Linearly interpolate to obtain value of spectrum at k24mon
    p24 = np.zeros(npts5)
    for it in tqdm(range(npts5)):
        p24[it] = np.interp(chkval,freq[ids],filtxfer[it,ids])
    
    # Plot results
    plotnum=npts5
    fig,axs= plt.subplots(2,1)
    ax = axs[0]
    ax.plot(freq,rawspec[:plotnum,:].T,label="",color='gray',alpha=0.25)
    ax.plot(freq,lpspec[:plotnum,:].T,label="",color='red',alpha=0.15)
    ax.set_xscale('log')
    ax.set_xticks(xtk)
    ax.set_xticklabels(xtkl)
    ax.set_title("Raw (gray) and Filtered (red) spectra")
    ax.grid(True,ls='dotted')
    
    ax = axs[1]
    plotp24 = p24.mean()#np.interp(chkval,freq[ids],filtxfer[:plotnum,:].mean(0)[ids]) #p24[:plotnum].mean()
    ax.plot(freq,filtxfer[:plotnum,:].T,label="",color='b',alpha=0.05,zorder=-1)
    ax.plot(freq,filtxfer[:plotnum,:].mean(0),label="",color='k',alpha=1)
    ax.scatter(chkval,[plotp24],s=100,marker="x",color='k',zorder=1)
    ax.set_ylim([0,1])
    ax.set_xscale('log')
    ax.set_xticks(xtk)
    ax.set_xticklabels(xtkl)
    ax.set_title("Filter Transfer Function (Filtered/Raw), %.3f" % (plotp24*100) +"%  Cutoff at " + "%i months" %(tw))
    ax.grid(True,ls='dotted')
    plt.suptitle("AVISO 5deg SSH Timeseries, %i-Band Average"% (M))
    plt.tight_layout()
    
    return lpspec,rawspec,p24,filtxfer,fig,ax