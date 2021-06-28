#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:43:44 2021

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def findzero_2d(lon,lat,invar):
    """
    

    Parameters
    ----------
    lon : TYPE
        DESCRIPTION.
    lat : TYPE
        DESCRIPTION.
    invar : ARRAY [time x lat x lon]
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ntime,nlat,nlon = invar.shape
    rvar = invar.reshape(ntime,nlat*nlon)
    
    boolmap = rvar.sum(0) == 0
    print("Found %i all zero points" % np.nansum(boolmap))
    boolmap = boolmap.reshape(nlat,nlon)

    return boolmap


datpath = "/stormtrack/data3/glliu/01_Data/03_SeaLevelProject/SSH/"
#stage ="Regridded/"
#fn = "SSH_HTR_ens01.nc"

stage = "Smoothed/"
fn = "SSH_smoothed_ens01.nc"

stage = "Coarsened/"
fn = "SSH_coarse_ens01.nc"

ds = xr.open_dataset(datpath+stage+fn)

ssh   = ds.SSH.values/100 # Convert to meters
lat5  = ds.lat.values
lon5  = ds.lon.values
times = ds.time.values