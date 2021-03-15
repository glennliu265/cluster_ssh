#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:00:32 2021

@author: gliu
"""
from sklearn.metrics.pairwise import haversine_distances

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

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

#%% Used edits

datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210309/"
start      = '1993-01'
end        = '2013-01'
rem_gmsl   = True

debug      = True

cmbal      = cmocean.cm.balance
#%% Functions
def add_coast_grid(ax,bbox=[-180,180,-90,90],proj=None):
    if proj is None:
        proj = ccrs.PlateCarree()
    ax.add_feature(cfeature.COASTLINE,color='black',lw=0.75)
    ax.set_extent(bbox)
    gl = ax.gridlines(crs=proj, draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle="dotted",lw=0.75)
    gl.xlabels_top   = False
    gl.ylabels_right = False
    return ax


#%% Load CESM Data (Comparing Unfiltered Results)

e      = 0
ensnum = e+1


# Load SSH field from CESM
st = time.time()
ds = xr.open_dataset("%sSSH_coarse_ens%02d.nc"%(datpath,ensnum))
ssh_cesm = ds.SSH.values/100 # Convert to meters
lat5 = ds.lat.values
lon5 = ds.lon.values
times = ds.time.values
ntime,nlat5,nlon5 = ssh_cesm.shape
print("Loaded data in %.2fs"%(time.time()-st))
ssh_cesm = ssh_cesm - ssh_cesm.mean(0)[None,:,:]


fze = ssh_cesm.copy()
fze = fze.reshape(ntime,nlat5*nlon5)

plt.plot(fze)



izero = (fze.sum(0)==0)
fze[:,np.where(izero)] = np.nan
ssh_cesm = fze.reshape(ntime,nlat5,nlon5)
zeropts = izero.reshape(nlat5,nlon5)






# Load SSH field from AVISO
ld1 = np.load(datpath+"SSHA_AVISO_1993-01to2013-01.npz",allow_pickle=True)
ssh_avi = ld1['sla_5deg']

# Calculate variance of each plot and plot..
sshs  = [ssh_avi,ssh_cesm]
names = ["AVISO","CESM1_ENS%02d"%ensnum] 


# Plot the standard deviation of each
vrg = [0,0.2]
fig,axs = plt.subplots(2,1,figsize=(8,8),subplot_kw={"projection":ccrs.PlateCarree(central_longitude=180)})
for i in range(2):
    ax = axs[i]
    ax = add_coast_grid(ax)
    pcm=ax.pcolormesh(lon5,lat5,sshs[i].std(0),transform=ccrs.PlateCarree(),cmap='bone',vmin=vrg[0],vmax=vrg[-1])
    fig.colorbar(pcm,ax=ax,fraction=0.046)
    ax.set_title(names[i] + " " + r"$1 \sigma_{SSH}$"+" (m)")
plt.suptitle("1 Standard Deviation of SSH (1993-2013)")
plt.tight_layout()
plt.savefig("%sSSHA_Stdev_comparison.png"%(outfigpath),dpi=200)


# Plot the difference in standard deviation
fig,ax = plt.subplots(1,1,subplot_kw={"projection":ccrs.PlateCarree(central_longitude=180)})
ax = add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,sshs[0].std(0) - sshs[1].std(0),transform=ccrs.PlateCarree(),cmap=cmbal,vmin=-.2,vmax=.2)
ax.scatter(nlat5,nlon5,zeropts)
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title(r"$1 \sigma_{SSH}$"+" (m); 1993 to 2013, AVISO - CESM")
#plt.tight_layout()
plt.savefig("%sSSHA_Stdev_difference.png"%(outfigpath),dpi=200)

#%% Compare Low Pass Filtered GMSL


ld1 = "SSHA_AVISO_1993-01to2013-01_LowPassFilter_order4_cutoff15.npz"
ld2 = "SSHA_ens01_1993-01to2013-01_LowPassFilter_order5_cutoff15_filteragain0.npz"

sshlp = []
lds = [ld1,ld2]
for i in range(2):
    ld = np.load(datpath+lds[i],allow_pickle=True)
    sshlp.append(ld['sla_lp'])
    times = ld['times']
    

# Quickly Conv Timeseries
timesmon = np.array(["%04d-%02d"%(t.year,t.month) for t in times])
idstart  = np.where(timesmon==start)[0][0]
idend    = np.where(timesmon==end)[0][0]
timesyr  = np.array(["%04d"%(t.year) for t in times])[idstart:idend]

# GMSL
sstord = [sshs[0],sshlp[0],sshs[1],sshlp[1]]
gmsls  = []
for i in range(4):
    gmsls.append(np.nanmean(sstord[i],(1,2)))


# Make Plot

fig,ax = plt.subplots(1,1)
ax.set_title("Global Mean Sea Level (1993-2013)")
ax.set_xticks(np.arange(0,240,12))
ax.set_xticklabels(timesyr[::12],rotation = 45)
ax.set_ylabel("SSHA (m)")
ax.set_xlabel("Time (Years)")
ax.grid(True,ls='dotted')

ax.plot(gmsls[0],label="AVISO Unfiltered",color='gray')
ax.plot(gmsls[1],label="AVISO Filtered",color='k',ls='dashdot')
ax.plot(gmsls[2][idstart:idend],label="CESM Unfiltered",color='blue',alpha=0.5)
ax.plot(gmsls[3],label="CESM Filtered",color='blue',ls='dashdot')

ax.legend()
plt.savefig(outfigpath+"GMSL_comparison.png",dpi=200)
