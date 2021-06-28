#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:30:38 2021

@author: gliu
"""

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


#%% Function

def lp_butter(varmon,cutofftime,order):
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
    if flag1d is False: # For 3d inputs, loop thru each point
        varmon = varmon.reshape(nmon,nlat*nlon)
        # Loop
        varfilt = np.zeros((nmon,nlat*nlon)) * np.nan
        for i in tqdm(range(nlon*nlat)):
            varfilt[:,i] = filtfilt(b,a,varmon[:,i])
        
        varfilt=varfilt.reshape(nmon,nlat,nlon)
    else: # 1d input
        varfilt = filtfilt(b,a,varmon)
    return varfilt



#%%
# Set filter parameters

order = 5          # Order of butterworth filter
M     = 5          # Bands to average over 
tw    = 24         # Lowpass Cut-off (in units of dt)
nitr  = 10000      # Number of wn time series to generate
dt    = 24*3600*30 # Timestep (in seconds)
tlen  = 240        # Length of timeseries

plotdir = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210301/"

xtk  = [1/(10*12*dt),1/(24*dt),1/(12*dt),1/(3*dt),1/dt]
xtkl = ['decade','2-yr','year','season','month']

#%%



#Preallocate
wn_ts  = np.zeros((nitr,tlen))
wn_lp  = np.zeros(wn_ts.shape)

wn_spec = []#np.zeros(wn_ts.shape)
lp_spec = []#np.zeros(wn_ts.shape)


for it in tqdm(range(nitr)):
    wn_ts[it,:] = np.random.normal(0,1,tlen)
    wn_lp[it,:] = lp_butter(wn_ts[it,:],tw,5)

    X_spec,freq,[lower,upper]=tbx.bandavg_autospec(wn_ts[it,:],dt,M,0.05)
    X_lpspec,_,_=tbx.bandavg_autospec(wn_lp[it,:],dt,M,0.05)
    
    wn_spec.append(X_spec)
    lp_spec.append(X_lpspec)

wn_spec = np.array(wn_spec)
lp_spec = np.array(lp_spec)
#%% Plot first 100 spectra



filtxfer = np.real(lp_spec)/np.real(wn_spec)


# Need to interpolate linearly
k24mon = np.argmin(np.abs(freq-xtk[1])) # Get index for 24 mon
if freq[k24mon] < xtk[1]: # less than 24 months
    ids = [k24mon,k24mon+1]
else:
    ids = [k24mon-1,k24mon]
p24 = np.zeros(nitr)
for it in tqdm(range(nitr)):
    p24[it] = np.interp(xtk[1],freq[ids],filtxfer[it,ids])




plotnum = 10000
fig,axs= plt.subplots(2,1)
ax = axs[0]
ax.plot(freq,wn_spec[:plotnum,:].T,label="",color='gray',alpha=0.25)
ax.plot(freq,lp_spec[:plotnum,:].T,label="",color='red',alpha=0.15)
ax.set_xscale('log')
ax.set_xticks(xtk)
ax.set_xticklabels(xtkl)
ax.set_title("Raw (gray) and Filtered (red) spectra")
ax.grid(True,ls='dotted')

ax = axs[1]
plotp24 = np.interp(xtk[1],freq[ids],filtxfer[:plotnum,:].mean(0)[ids]) #p24[:plotnum].mean()
ax.plot(freq,filtxfer[:plotnum,:].T,label="",color='b',alpha=0.05,zorder=-1)
ax.plot(freq,filtxfer[:plotnum,:].mean(0),label="",color='k',alpha=1)
ax.scatter(xtk[1],[plotp24],s=100,marker="x",color='k',zorder=1)
#ax.scatter(freq[k24mon+1],p24,s=100,marker="x",color='k',)
ax.set_ylim([0,1])
ax.set_xscale('log')
ax.set_xticks(xtk)
ax.set_xticklabels(xtkl)
ax.set_title("Filter Transfer Function (Filtered/Raw), %.3f" % (plotp24*100) +"%  at 24 months")
ax.grid(True,ls='dotted')
plt.suptitle("%i White Noise Timeseries, %i-Band Average"% (plotnum,M))

plt.tight_layout()
plt.savefig("%sFilter_Transfer_%imonLP_%ibandavg_plot%ix"%(plotdir,tw,M,plotnum),dpi=200)

#%%

k24mon = np.argmin(np.abs(freq-xtk[1])) # Get index for 24 mon
p24 = np.real((filtxfer))[:,k24mon].mean()

fig,ax=plt.subplots(1,1)
#ax.plot(freq,X_unfilt)
#ax.plot(freq,X_filt)
ax.plot(freq,X_filt/X_unfilt,label='Filtered/Unfiltered Spectrum',color='b')
ax.scatter(freq[k24mon],p24,marker="x")
ax.set_xscale('log')
ax.set_xticks(xtk)
ax.set_xticklabels(xtkl)
ax.set_title("%i-Band Averaged, Passing %f at 24 months"% (M,p24))
ax.legend()


