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

#%% Load some synthetic timeseries

niter = 100
fn = "%sNoiseMaps_AVISO_niter%i.npz"%(datpath,niter)
ld = np.load(fn,allow_pickle=True)
wnout = ld['wnout'] # [iter x time x lon x lat]
rnout = ld['rnout'] # [iter x time x lon x lat]
lon   = ld['lon']
lat   = ld['lat']

#%% Calculate the gap statistic (scrap)


clusterng = np.arange(1,11,1) # Let's start by trying 10 clusters...


# Some other clustering params
distthres = 3000
absmode = 0
distmode = 0
uncertmode = 0
printmsg = False
calcsil=False
sigtest=False

sla_in = wnout[0,:,:,:]

nlat = len(lat)
nlon = len(lon)

# Preallocate
# allclusters  = []
# alluncert    = []
# alluncertsig = []
# allcount     = []
allWk = []
# if calcsil:
#     alls           = []
#     alls_byclust = []
# rempts      = np.zeros((nlat*nlon))*np.nan

for c in tqdm(range(len(clusterng))):
    nclusters = clusterng[c]
    
    
    # Perform Clustering
    clustoutput = slutil.cluster_ssh(sla_in,lat,lon,nclusters,distthres=distthres,
                                                 absmode=absmode,distmode=distmode,uncertmode=uncertmode,
                                                 printmsg=printmsg,calcsil=calcsil,sigtest=sigtest)
    
    
    if calcsil:
        clustered,uncert,uncertsig,cluster_count,Wk,s,s_byclust = clustoutput
        alls.append(s)
        alls_byclust.append(s_byclust)
    else:
        clustered,uncert,uncertsig,cluster_count,Wk = clustoutput
    
    # Save results
    #allclusters.append(clustered)
    #alluncert.append(uncert)
    #alluncertsig.append(uncertsig)
    #allcount.append(cluster_count)
    allWk.append(Wk.mean())

#%% Streamlined gap statistic calculation





