#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test smoothing order

Examine difference sif one smoothes, then anomalizes and vice versa

Created on Mon May 10 18:29:09 2021

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pygmt
from tqdm import tqdm
import time

import cartopy.crs as ccrs
import cmocean
#%% User Edits

datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
outpath    = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210510/"

ssh = xr.open_dataset(datpath+'SSH_coarse_PIC.nc')
ssha = xr.open_dataset(datpath+'SSHA_coarse_PIC.nc')

ssha_smoothfirst = ssh - ssh.SSH.mean('time')

diff = ssha - ssha_smoothfirst

diff.SSH.max('time').plot(),plt.title("Anomalize First - Smooth First, SSH Difference")

diffout = diff.SSH.values
lat5 = diff.lat.values
lon5 = diff.lon.values

vstd = np.nanstd(diffout.max(0))
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
pcm=ax.pcolormesh(lon5,lat5,diffout.max(0),vmin=-2*vstd,vmax=2*vstd,cmap=cmocean.cm.balance)
fig.colorbar(pcm,ax=ax,fraction=0.026)
ax.set_title("Anomalize First - Smooth First, SSH Difference,\n Max = %.2e"%(np.nanmax(diffout.max(0))))
plt.savefig("%sSmoothing_Anomalizing_Order_Difference.png"%(outpath),dpi=200)

