#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given CESM SSH data processed by preproc_CESM, apply spatial smoothing
and regrid to 5 x 5 degree

@author: gliu
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pygmt
from tqdm import tqdm
import time

#%% User Edits

datpath    = "/stormtrack/data3/glliu/01_Data/03_SeaLevelProject/SSH/Regridded/"

savesmooth = True
smoothdir  = "/stormtrack/data3/glliu/01_Data/03_SeaLevelProject/SSH/Smoothed/"
coarsedir  = "/stormtrack/data3/glliu/01_Data/03_SeaLevelProject/SSH/Coarsened/"
#%% Functions

def load_cesm(datpath,ensnum):
    
    # Open both datasets
    HTRname = "%sSSH_HTR_ens%02d.nc" % (datpath,ensnum)
    R85name = "%sSSH_R85_ens%02d.nc" % (datpath,ensnum)
    hds = xr.open_dataset(HTRname)
    rds = xr.open_dataset(R85name)
    
    # Concatenate
    ds = xr.concat([hds,rds],dim='time')
    return ds

def coarsen_byavg(invar,lat,lon,deg,tol,latweight=True,verbose=True,ignorenan=False):
        """
        Coarsen an input variable to specified resolution [deg]
        by averaging values within a search tolerance for each new grid box.
        
        Dependencies: numpy as np
    
        Parameters
        ----------
        invar : ARRAY [TIME x LAT x LON]
            Input variable to regrid
        lat : ARRAY [LAT]
            Latitude values of input
        lon : ARRAY [LON]
            Longitude values of input
        deg : INT
            Resolution of the new grid (in degrees)
        tol : TYPE
            Search tolerance (pulls all lat/lon +/- tol)
        
        OPTIONAL ---
        latweight : BOOL
            Set to true to apply latitude weighted-average
        verbose : BOOL
            Set to true to print status
        
    
        Returns
        -------
        outvar : ARRAY [TIME x LAT x LON]
            Regridded variable       
        lat5 : ARRAY [LAT]
            New Latitude values of input
        lon5 : ARRAY [LON]
            New Longitude values of input
    
        """
    
        # Make new Arrays
        lon5 = np.arange(0,360+deg,deg)
        lat5 = np.arange(-90,90+deg,deg)
        
        
        # Set up latitude weights
        if latweight:
            _,Y = np.meshgrid(lon,lat)
            wgt = np.cos(np.radians(Y)) # [lat x lon]
            invar *= wgt[None,:,:] # Multiply by latitude weight
        
        # Get time dimension and preallocate
        nt = invar.shape[0]
        outvar = np.zeros((nt,len(lat5),len(lon5)))
        
        # Loop and regrid
        i=0
        for o in range(len(lon5)):
            for a in range(len(lat5)):
                lonf = lon5[o]
                latf = lat5[a]
                
                lons = np.where((lon >= lonf-tol) & (lon <= lonf+tol))[0]
                lats = np.where((lat >= latf-tol) & (lat <= latf+tol))[0]
                
                varf = invar[:,lats[:,None],lons[None,:]]
                
                if latweight:
                    wgtbox = wgt[lats[:,None],lons[None,:]]
                    if ignorenan:
                        varf = np.nansum(varf/np.nansum(wgtbox,(0,1)),(1,2)) # Divide by the total weight for the box
                    else:
                        varf = np.sum(varf/np.sum(wgtbox,(0,1)),(1,2)) # Divide by the total weight for the box
                    
                    
                else:
                    if ignorenan:   
                        varf = np.nanmean(varf,axis=(1,2))
                    else:
                        varf = varf.mean((1,2))
                    
                outvar[:,a,o] = varf.copy()
                i+= 1
                msg="\rCompleted %i of %i"% (i,len(lon5)*len(lat5))
                print(msg,end="\r",flush=True)
        return outvar,lat5,lon5
#%% Load data and smooth


for e in tqdm(range(40)):

    ensnum = e+1

    # Load in data for a particular ensemble member
    st = time.time()
    ds = load_cesm(datpath,ensnum)
    sla = ds.SSH.values
    times = ds.time.values
    lat = ds.lat.values
    lon = ds.lon.values
    ntime,nlat,nlon = sla.shape
    print("Loaded data in %.2fs"%(time.time()-st))
    
    # Apply Smoothing
    slasmooth = np.zeros((ntime,nlat,nlon))
    for i in tqdm(range(ntime)):
        da = xr.DataArray(sla[i,:,:].astype('float32'),
                        coords={'lat':lat,'lon':lon},
                        dims={'lat':lat,'lon':lon},
                        name='sla')
        timestamp = times[i]
        smooth_field = pygmt.grdfilter(grid=da, filter="g500", distance="4",nans="i")
        slasmooth[i,:,:] = smooth_field.values
    
    
    # Reapply Mask to correct for smoothed edges
    mask = sla.sum(0)
    mask[~np.isnan(mask)] = 1
    sla_filt = slasmooth * mask[None,:,:]
    
    
    # Save Smoothing (If option is set)
    if savesmooth:
        ds_smooth = xr.DataArray(sla_filt,
                    coords={'time':times,'lat':lat,'lon':lon},
                    dims={'time':times,'lat':lat,'lon':lon},
                    name='SSH')
        outname = smoothdir + "SSH_smoothed_ens%02d.nc" % ensnum
        ds_smooth.to_netcdf(outname,encoding={'SSH': {'zlib': True}})

    
    # if debug:
    #     fig,ax = plt.subplots(1,1)
    #     pcm = ax.pcolormesh(lon,lat,slasmooth[0,:,:],vmin=-0.4,vmax=0.4,cmap=cmap)
    #     fig.colorbar(pcm,ax=ax)
    
    #%% Coarsen the dataset
    
    # Apply Regridding (coarse averaging for now)
    tol  = 0.75
    deg  = 5
    sla_5deg,lat5,lon5 = coarsen_byavg(sla_filt,lat,lon,deg,tol)
    
    
    # Save result
    ds_coarse = xr.DataArray(sla_5deg,
                coords={'time':times,'lat':lat5,'lon':lon5},
                dims={'time':times,'lat':lat5,'lon':lon5},
                name='SSH')
    outname = coarsedir + "SSH_coarse_ens%02d.nc" % ensnum
    ds_coarse.to_netcdf(outname,encoding={'SSH': {'zlib': True}})
