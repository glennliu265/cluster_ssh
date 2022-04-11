#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 01:27:02 2021

@author: gliu
"""


from sklearn.metrics.pairwise import haversine_distances

import matplotlib.pyplot as plt
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
from scipy import stats,signal
# Custom Toolboxes
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/03_Scripts/cluster_ssh/")
from amv import proc,viz
import slutil
import yo_box as ybx
import tbx

from statsmodels.regression import linear_model

#%% Load Aviso Data

# Set Paths
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210610/"

# Experiment Names
start       = '1993-01'
end         = '2013-01'
#start      = '1850-01'
#end        = '2100-12'
nclusters   = 6
rem_gmsl    = True
rem_seas    = True
dt_point    = True
maxiter     = 5  # Number of iterations for elimiting points
minpts      = 30 # Minimum points per cluster

# Other Toggles
debug       = True
savesteps   = True  # Save Intermediate Variables
filteragain = False # Smooth variable again after coarsening 

datname = "AVISO_%s_to_%s_remGMSL%i" % (start,end,rem_gmsl)
expname = "Synthetic_%iclusters_minpts%i_maxiters%i" % (nclusters,minpts,maxiter)
print(datname)
print(expname)

#Make Folders
#expdir = outfigpath+expname +"/"
expdir = outfigpath+"../20210818/"
checkdir = os.path.isdir(expdir)
if not checkdir:
    print(expdir + " Not Found!")
    os.makedirs(expdir)
else:
    print(expdir+" was found!")


#%%

def calc_Wk(sla_in,clusterrng):
    """
    sla_in [time x lon x lat]: input sea lvl anomalies
    clusterrng : array of cluster thresholds to try
    
    stripped down section from elim_points function
    
    """
    
    
    # Some other clustering params
    distthres = 3000
    absmode = 0
    distmode = 0
    uncertmode = 0
    printmsg = False
    calcsil=False
    sigtest=False

    ntime,nlon,nlat = sla_in.shape

    # Preallocate
    allWk = []
    for c in tqdm(range(len(clusterng))):
        nclusters = clusterng[c]
        
        # Perform Clustering
        clustoutput = slutil.cluster_ssh(sla_in,lat,lon,nclusters,distthres=distthres,
                                                     absmode=absmode,distmode=distmode,uncertmode=uncertmode,
                                                     printmsg=printmsg,calcsil=calcsil,sigtest=sigtest)
        
        clustered,uncert,uncertsig,cluster_count,Wk = clustoutput
        allWk.append(Wk.sum())
    return allWk

#%% Load some synthetic timeseries

niter = 100
fn    = "%sNoiseMaps_AVISO_niter%i.npz"%(datpath,niter)
ld    = np.load(fn,allow_pickle=True)
wnout = ld['wnout'] # [iter x time x lon x lat]
rnout = ld['rnout'] # [iter x time x lon x lat]
lon   = ld['lon']
lat   = ld['lat']

#%% Calculate the gap statistic (scrap)


clusterrng = np.arange(1,11,1) # Let's start by trying 10 clusters...
sla_in     = wnout[0,:,:,:]

Wk_test = np.zeros((2,niter,len(clusterrng)))*np.nan
for i,noisetype in enumerate([wnout,rnout]):
    for n in tqdm(range(niter)):
        sla_in = noisetype[n,:,:,:]
        allWk = calc_Wk(sla_in,clusterrng)
        
        Wk_test[i,n,:] = allWk
        
np.save("ScrapSave_Wk_test.npy",Wk_test)
#%% Do the same for AVISO Dataset

enames = ["AVISO",
          "AVISO (GMSL removed)"]
fns = ["SSHA_LP_AVISO_1993-01_to_2013-01_remGMSL0_order5_cutoff15_remGMSL0.npz",
       "SSHA_LP_AVISO_1993-01_to_2013-01_remGMSL1_order5_cutoff15_remGMSL1.npz"]
ecolors = ["k","gray"]

e_slas = []
e_Wks = []
for f in fns:
    ld = np.load(datpath+f,allow_pickle=True)
    e_slas.append(ld['sla_lp'])
    Wk = calc_Wk(ld['sla_lp'],clusterrng)
    e_Wks.append(Wk)
    



#%% Plot Gap Statistic Results

noisenames = ["White Noise","Red Noise"]
noisecolor = ['b','r']

fig,ax = plt.subplots(1,1)
for i in range(2):
    #for n in tqdm(range(niter)):
        #ax.plot(clusterrng,Wk_test[i,n,...],alpha=0.25,color=noisecolor[i],label="")
    ax.plot(clusterrng,np.log(Wk_test[i,...].mean(0)),color=noisecolor[i],label=noisenames[i])

for i in range(2):
    ax.plot(clusterrng,np.log(e_Wks[i]),color=ecolors[i],label=enames[i])

ax.legend()
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("log($W_k)$")
ax.grid(True,ls='dotted')
ax.set_xlim([1,10])
ax.set_title("Total Within Cluster Distance ($W_k$) vs. Number of Clusters")
#ax.set_ylim([])
plt.savefig("Wk_test_plot.png",dpi=200)

    


#%% Try to actually calculate gap statistic
plt.plot(np.log(Wk_test[0,...].mean(0))-np.log(e_Wks[1]))







#%% Streamlined gap statistic calculation





