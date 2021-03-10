#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preproc CESM SSH
Created on Tue Mar  9 18:26:03 2021

@author: gliu
"""
import glob
import xarray as xr
import numpy as np
import xesmf as xe
import matplotlib.pyplot as plt
from tqdm import tqdm

datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/SSH/"
outpath = "/stormtrack/data3/glliu/01_Data/03_SeaLevelProject/SSH/Regridded/"
mconfig = "HTR"
mnum    = np.hstack([np.arange(1,36,1),np.arange(101,108,1)])

latlonf = "/home/glliu/01_Data/cesm_latlon360.npz"
ld = np.load(latlonf,allow_pickle=True)
lat = ld['lat']
lon = ld['lon']

# Historical
#b.e11.B20TRC5CNBDRD.f09_g16.001.pop.h.SSH.185001-200512.nc 

# RCP85
#b.e11.BRCP85C5CNBDRD.f09_g16.001.pop.h.SSH.200601-208012.nc

varkeep  = ['SSH','time','TLAT','TLONG'] 
debug    = False
#%% Functions

def preprocess(ds,varlist=varkeep):
    """"preprocess dataarray [ds],dropping variables not in [varlist] and 
    selecting surface variables at [lev=-1]"""
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in varlist]
    ds = ds.drop(remvar)
    return ds

def fix_febstart(ds):
    if ds.time.values[0].month != 1:
        print("Warning, first month is %s"% ds.time.values[0])
        # Get starting year, must be "YYYY"
        startyr = str(ds.time.values[0].year)
        while len(startyr) < 4:
            startyr = '0' + startyr
        nmon = ds.time.shape[0] # Get number of months
        # Corrected Time
        correctedtime = xr.cftime_range(start=startyr,periods=nmon,freq="MS",calendar="noleap")
        ds = ds.assign_coords(time=correctedtime) 
    return ds

#%%
# Get NC Files
ncstring = "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h.SSH.*.nc"
nclist = glob.glob(datpath+ncstring)
nclist.sort()
nclist = [fn for fn in nclist if "OIC" not in fn]
nfiles = len(nclist)
print("Found %i files" % (nfiles))

# Set up lat/lon target grid
ds_out = xr.Dataset({'lat': (['lat'], lat), 'lon': (['lon'], lon) })
dy = lat[1]-lat[0]
dx = lon[1]-lon[0]
dx,dy
#ds_out = xe.util.grid_global(dx,dy)

for n in tqdm(range(nfiles)):
    # Open and Load Dataset
    ncname = nclist[n]
    ds = xr.open_dataset(ncname)
    ds = preprocess(ds,varlist=varkeep)
    ds = fix_febstart(ds)
    
    # Visualize first timestep and grid
    if debug:
        ds.SSH.isel(time=0).plot()
        plt.show()
        
        plt.scatter(ds['TLONG'], ds['TLAT'], s=0.01)
        plt.show()
    
    # Rename lat/lon
    ds = ds.rename({'TLONG': 'lon', 'TLAT': 'lat'})
    
    # Make regridder  and Regrid (set periodic to true to get rid of strip along lon=0)
    regridder = xe.Regridder(ds, ds_out, 'bilinear',periodic=True)
    ds_regrid = regridder(ds.SSH)
    # Visualize result
    if debug:
        ds_regrid.isel(time=0).plot()
        plt.show()
    
    # Save the output
    outname = outpath + "SSH_HTR_ens%02d.nc" % (n+1)
    ds_regrid.to_netcdf(outname,encoding={'SSH': {'zlib': True}})   


