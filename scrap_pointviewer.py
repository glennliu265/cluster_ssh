#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Point Analysis

Created on Wed Jun  9 20:04:13 2021

@author: gliu
"""


from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import silhouette_score,silhouette_samples

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xarray as xr
import numpy as np

import pygmt
from tqdm import tqdm

import os

import glob
import time
import cmocean

from scipy.signal import butter, lfilter, freqz, filtfilt, detrend
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pylab import cm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

# Custom Toolboxes
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/03_Scripts/cluster_ssh/")
from amv import proc,viz
import slutil
import yo_box as ybx
import tbx
#%%


datpath = 


#%%

# Load low-pass filtered AVISO Datasets
datpath   = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
start     = '1993-01'
end       = '2013-01'
rem_gmsl  = 1
order     = 5
tw        = 15
datname   = "AVISO_%s_to_%s_remGMSL%i" % (start,end,rem_gmsl)
outname   = "%sSSHA_LP_%s_order%i_cutoff%i.npz" % (datpath,datname,order,tw)
ld = np.load(outname,allow_pickle=True)
aviso_sla = ld['sla_lp']
lat5      = ld['lat']
lon5      = ld['lon']
atimes    = ld['times']

# Load CESM1 Dataset
order     = 5
tw        = 18
datname   = "CESM_PIC_remGMSL%i" % (rem_gmsl)
outname   = "%sSSHA_LP_%s_order%i_cutoff%i.npz" % (datpath,datname,order,tw)
ld = np.load(outname,allow_pickle=True)
cesm_sla  = ld['sla_lp']
ctimes    = ld['times']

#%% Make a plot comparing their differences

proj = ccrs.PlateCarree(central_longitude=180)
fig,axs = plt.subplots(1,2,figsize=(10,3.5),subplot_kw={'projection':proj})
vm = .1
cmap = 'copper'

ax   = axs[0]
ax = slutil.add_coast_grid(ax=ax,proj=proj)
pcm  = ax.pcolormesh(lon5,lat5,aviso_sla.std(0),vmin=0,vmax=vm,transform=ccrs.PlateCarree(),cmap=cmap)
#fig.colorbar(pcm,ax=ax)
ax.set_title("AVISO (1993-2013)")

ax   = axs[1]
ax = slutil.add_coast_grid(ax=ax,proj=proj)
pcm  = ax.pcolormesh(lon5,lat5,cesm_sla.std(0),vmin=0,vmax=vm,transform=ccrs.PlateCarree(),cmap=cmap)
ax.set_title("CESM1 Preindustrial Control (400-2200)")

fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='horizontal',shrink=0.45,pad=0.08)

plt.suptitle("$1\sigma$ SSH Anomalies (m)",y=.95)
plt.savefig("%sAVISO-CESMPIC-Compare.png"%(outfigpath),dpi=200,transparent=True)




#%%







lonf  = 330
latf  = 50

# Requires the following
#okpts = 
#invar = 
#lon5  
#lat5  





def get_linid(lon5,lat5,lonf,latf,okpts=None):
    # Make Grid, Flatten
    lonmesh,latmesh = np.meshgrid(lon5,lat5)
    coords = np.vstack([lonmesh.flatten(),latmesh.flatten()]).T
    
    # Get linear indices
    if okpts is not None: # Get non-NaN values
        coords = coords[okpts,:]
    kpt = np.where((coords[:,0] == lonf) * (coords[:,1] == latf))[0][0]
    print("Found %.2f Lon %.2f Lat" % (coords[kpt,0],coords[kpt,1]))
    return kpt

def retrieve_point(invar,kpt,nlat5,nlon5,okpts):
    """
    Retrieve values for a point  [kpt] given a distance or correlation matrix [var],
    and the indices of nonNaN values [okpts], and the lat/lon sizes [nlat5,nlon5]
    
    """
    # Get Value
    vrow = invar[kpt,:]
    
    # Place into variable
    mappt = np.zeros(nlat5*nlon5)*np.nan
    mappt[okpts] = vrow
    mappt = mappt.reshape(nlat5,nlon5)
    return mappt


# Calculate matrices
srho,scov,sdist,okdata,okpts,coords2=slutil.calc_matrices(varin,lon5,lat5,return_all=True)

# Indicate Point to find
lonf = 40#-70+360
latf = -35
kpt  = get_linid(lon5,lat5,lonf,latf,okpts=okpts)

kpt = np.where((coords[:,0] == lonf) * (coords[:,1] == latf))[0][0]
loctitle = "Lon %.1f Lat %.1f" % (lonf,latf)
locfn = "Lon%i_Lat%i" % (lonf,latf)
print("Found %.2f Lon %.2f Lat" % (coords[kpt,0],coords[kpt,1]))

# Plot Distance
distpt = retrieve_point(sdist,kpt,nlat5,nlon5,okpts)
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
ax.set_title("Distance (km) from %s" % (loctitle))
pcm = ax.pcolormesh(lon5,lat5,distpt,cmap='Blues')
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sDistance_%s.png"%(outfigpath,locfn),dpi=200,transparent=True)

# Plot Exponential Term
distpt = retrieve_point(expterm,kpt,nlat5,nlon5,okpts)
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
ax.set_title("${exp(-(Distance) / (2a^{2})})$ (km) from %s" % (loctitle))
pcm = ax.pcolormesh(lon5,lat5,distpt,cmap='Greens')
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sExpTerm_%s.png"%(outfigpath,locfn),dpi=200,transparent=True)

# Plot Correlation
distpt = retrieve_point(srho,kpt,nlat5,nlon5,okpts)
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
ax.set_title("Correlation with %s" % (loctitle))
pcm = ax.pcolormesh(lon5,lat5,distpt,cmap=cmocean.cm.balance,vmin=-1,vmax=1)
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sCorrelation_%s.png"%(outfigpath,locfn),dpi=200,transparent=True)

# Plot Covariance
distpt = retrieve_point(scov,kpt,nlat5,nlon5,okpts)
vm = np.nanmax(np.abs(distpt))
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
ax.set_title("Covariance with %s" % (loctitle))
pcm = ax.pcolormesh(lon5,lat5,distpt,cmap=cmocean.cm.balance,vmin=-vm,vmax=vm)
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sCovariance_%s.png"%(outfigpath,locfn),dpi=200,transparent=True)



# Plot Final Distance Matrix
distpt = retrieve_point(distance_matrix,kpt,nlat5,nlon5,okpts)
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
ax.set_title("Distance Matrix with %s" % (loctitle))
pcm = ax.pcolormesh(lon5,lat5,distpt,cmap=cmocean.cm.dense)
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sDistanceMatrixFinal_%s.png"%(outfigpath,locfn),dpi=200,transparent=True)
