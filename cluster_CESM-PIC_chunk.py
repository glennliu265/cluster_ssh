#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Cluster CESM-PIC Data in moving chunks

ported over from cluster-CESM-PIC
Created on Mon May 24 11:25:47 2021

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

#%% Used edits

# Set Paths
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210519/"
proc.makedir(outfigpath)

# Experiment Names
#start       = '1993-01'
#end         = '2013-01'
#start      = '1850-01'
#end        = '2100-12'
nclusters   = 6
rem_gmsl    = False
e           = 0  # Ensemble index (ensnum-1), remove after loop is developed
maxiter     = 5  # Number of iterations for elimiting points
minpts      = 30 # Minimum points per cluster

# Other Toggles
debug       = True
savesteps   = True  # Save Intermediate Variables
filteragain = False # Smooth variable again after coarsening 

add_gmsl    = False # Add AVISO GMSL
if add_gmsl:
    rem_gmsl=0

ensnum  = e+1
datname = "CESM_PIC_remGMSL%i" % (rem_gmsl)
expname = "%s_%iclusters_minpts%i_maxiters%i" % (datname,nclusters,minpts,maxiter)
print(datname)
print(expname)

# Make Directory for Experiment
expdir = outfigpath+expname +"/"
checkdir = os.path.isdir(expdir)
if not checkdir:
    print(expdir + " Not Found!")
    os.makedirs(expdir)
else:
    print(expdir+" was found!")

#%% Load in the dataset

# Load data (preproc, then anomalized)
st = time.time()
ds = xr.open_dataset("%sSSHA_coarse_PIC.nc"%(datpath))
ssh = ds.SSH.values/100 # Convert to meters
lat5 = ds.lat.values
lon5 = ds.lon.values
times = ds.time.values
ntime,nlat5,nlon5 = ssh.shape
print("Loaded data in %.2fs"%(time.time()-st))

# Convert Datestrings
timesmon = np.array(["%04d-%02d"%(t.year,t.month) for t in times])

# Plotting utilities
cmbal = cmocean.cm.balance

# ------------------------------
# Apply land ice mask from aviso
# ------------------------------
mask = np.load(datpath+"AVISO_landice_mask_5deg.npy")
ssha = ssh * mask[None,:,:]

# ------------------
# Remove GMSL Signal
# ------------------
lonf = 330
latf = 50
timesyr = np.arange(0,int(len(times)/12)) 
if rem_gmsl>0:
    print("Removing GMSL")
    out1 = slutil.remove_GMSL(ssha,lat5,lon5,timesyr,viz=True,testpoint=[lonf,latf])
    
    if len(out1)>2:
        ssha,gmslrem,fig,ax = out1
        plt.savefig(expdir+"GMSL_Removal_CESM_ens%i_testpoint_lon%i_lat%i.png"%(ensnum,lonf,latf),dpi=200)
    else:
        ssha,gmsl=out1
        
    if np.all(np.abs(gmslrem)>(1e-10)):
        print("Saving GMSL")
        #np.save(datpath+"CESM1_ens%i_GMSL_%s_%s.npy"%(ensnum,start,end),gmslrem)
else:
    print("GMSL Not Removed")

# ---------------------
# Add in the Aviso GMSL
# ---------------------
if add_gmsl:
    gmslav = np.load(datpath+"AVISO_GMSL_1993-01_2013-01.npy")
    ssh_ori = ssha.copy()
    ssha += gmslav[:,None,None]
    
    fig,ax = plt.subplots(1,1)
    ax.plot(gmslav,label="GMSL")
    ax.plot()
    
    
    
    klon,klat = proc.find_latlon(lonf,latf,lon5,lat5)
    fig,ax = plt.subplots(1,1)
    #ax.set_xticks(np.arange(0,len(times)+1,12))
    ax.set_xticks(np.arange(0,len(timesyr),12))
    ax.set_xticklabels(timesyr[::12],rotation = 45)
    ax.grid(True,ls='dotted')
    
    ax.plot(ssh_ori[:,klat,klon],label="Original",color='k')
    ax.plot(ssha[:,klat,klon],label="After Addition")
    ax.plot(gmslav,label="AVISO-GMSL")
    
    ax.legend()
    ax.set_title("GMSL Addition at Lon %.2f Lat %.2f (%s to %s)" % (lon5[klon],lat5[klat],timesyr[0],timesyr[-1]))
    ax.set_ylabel("SSH (m)")
    plt.savefig(expdir+"GMSL_Addition.png",dpi=150)
else:
    print("No GMSL Added!")



# ----------------------
#%% Design Low Pass Filter
# ----------------------
ntimer   = ssha.shape[0]

# ---
# Apply LP Filter
# ---
# Filter Parameters and Additional plotting options
dt   = 24*3600*30
M    = 5
xtk  = [1/(10*12*dt),1/(24*dt),1/(12*dt),1/(3*dt),1/dt]
xtkl = ['decade','2-yr','year','season','month']
order  = 5
tw     = 18 # filter size for time dim
sla_lp = slutil.lp_butter(ssha,tw,order)

#% Remove NaN points and Examine Low pass filter
slars = sla_lp.reshape(ntimer,nlat5*nlon5)

# ---
# Locate points where values are all zero
# ---
tsum     = slars.sum(0)
zero_pts = np.where(tsum==0)[0]
ptmap = np.array(tsum==0)
slars[:,zero_pts] = np.nan
ptmap = ptmap.reshape(nlat5,nlon5)
# Map removed points
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree(central_longitude=0)})
ax     = slutil.add_coast_grid(ax)
pcm = ax.pcolormesh(lon5,lat5,ptmap,cmap='bone',transform=ccrs.PlateCarree(),alpha=0.88)
fig.colorbar(pcm,ax=ax)
ax.set_title("Removed Zero Points")

# # ---
# # Visualize Filter Transfer Function
# # ---
# okdata,knan,okpts = proc.find_nan(slars,0)
# npts5 = okdata.shape[1]
# lpdata  = okdata.copy()
# rawdata = ssha.reshape(ntimer,nlat5*nlon5)[:,okpts]
# lpspec,rawspec,p24,filtxfer,fig,ax=slutil.check_lpfilter(rawdata,lpdata,xtk[1],M,tw,dt=24*3600*30)
# plt.savefig("%sFilter_Transfer_%imonLP_%ibandavg_%s.png"%(expdir,tw,M,expname),dpi=200)

# ---
# Save results
# ---
if savesteps: # Save low-pass-filtered result, right before clustering
    outname = "%sSSHA_LP_%s_order%i_cutoff%i.npz" % (datpath,datname,order,tw)
    print("Saved to: %s"%outname)
    np.savez(outname,**{
        'sla_lp':sla_lp,
        'lon':lon5,
        'lat':lat5,
        'times':times
        })

#%% Cluster in moving windows

sla_in = sla_lp[:,:,:]
maxiter = 6

# Get ranges for sliding window
winsize = 240
#rngs = []
clusters   = []
uncerts    = []
counts     = []
rempts     = []
Wks        = []
s_all      = []
s_by_clust = []
rngs       = []
for i in tqdm(range(ntime-240)):
    rng = np.arange(i,i+winsize+1)
    sla_in = sla_lp[rng,:,:]
    
    allclusters,alluncert,allcount,rempt,allWk,alls,alls_byclust = slutil.elim_points(sla_in,lat5,lon5,nclusters,minpts,maxiter,expdir,
                                                             viz=False,printmsg=False,calcsil=True)
    
    
    clusters.append(np.array(allclusters[-1])) # Save, and just take the final result # [niter x nlat x nlon]
    uncerts.append(np.array(alluncert[-1]))    # [niter x nlat x nlon]
    counts.append(np.array(allcount[-1]))      # [niter x nclusters]
    rempts.append(np.array(rempt))             # [lat x lon]
    Wks.append(np.array(allWk[-1]))            # [niter x nclusters]
    s_all.append(np.array(alls[-1]))           # [niter x nanpts]
    s_by_clust.append(alls_byclust[-1])        # [niter x ncluster]
    rngs.append(times[rng])
    # End loop


# Make everything to arrays
clusters = np.array(clusters)  # [time,]
uncerts  = np.array(uncerts)   
counts   = np.array(counts)
rempts  = np.array(rempts)
Wks = np.array(Wks)
s_all = np.array(s_all)
s_by_clust = np.array(s_by_clust)
rngs = np.array(rngs)

np.savez("%s%s_Results_winsize%i.npz"%(datpath,expname,winsize),**{
    'lon':lon5,
    'lat':lat5,
    'clusters':clusters,
    'uncert':uncerts,
    'count':counts,
    'rempts':rempts,
    'Wks':Wks,
    's_all':s_all,
    's_by_clust':s_by_clust,
    'rngs':rngs},allow_pickle=True)
