#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Monte Carlo Uncertainty Tests, Visualizations (AVISO)

Created on Tue Aug 17 21:12:45 2021

@author: gliu
"""

from sklearn.metrics.pairwise import haversine_distances

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib as mpl

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
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210818/"

# Experiment Names
start       = '1993-01'
end         = '2013-01'
#start      = '1850-01'
#end        = '2100-12'
nclusters   = 6
rem_gmsl    = False
maxiter     = 5  # Number of iterations for elimiting points
minpts      = 30 # Minimum points per cluster

# Other Toggles
debug       = True
savesteps   = True  # Save Intermediate Variables
filteragain = False # Smooth variable again after coarsening 

datname = "AVISO_%s_to_%s_remGMSL%i" % (start,end,rem_gmsl)
expname = "%s_%iclusters_minpts%i_maxiters%i" % (datname,nclusters,minpts,maxiter)
print(datname)
print(expname)

#Make Folders
expdir = outfigpath+expname +"/"
checkdir = os.path.isdir(expdir)
if not checkdir:
    print(expdir + " Not Found!")
    os.makedirs(expdir)
else:
    print(expdir+" was found!")

#%% Load in the dataset

# Load data (preproc, then anomalized)
st       = time.time()
ld       = np.load("%sSSHA_AVISO_%sto%s.npz" % (datpath,start,end),allow_pickle=True)
sla_5deg = ld['sla_5deg']
lon5     = ld['lon']
lat5     = ld['lat']
times    = ld['times']
print("Loaded data in %.2fs"%(time.time()-st))

# Plotting utilities
cmbal = cmocean.cm.balance

#%% Additional Preprocessing Steps

# -----------------------------------------
# Limit to particular period (CESM Version)
# -----------------------------------------
# Convert Datestrings
timesmon = np.datetime_as_string(times,unit="M")

# Find indices
idstart  = np.where(timesmon==start)[0][0]
idend    = np.where(timesmon==end)[0][0]

# Restrict Data
ssha = sla_5deg[idstart:idend,:,:]
timeslim = timesmon[idstart:idend]
timesyr  = np.datetime_as_string(times,unit="Y")[idstart:idend]
ntimer,nlat5,nlon5   = ssha.shape

# ------------------
# Remove GMSL Signal
# ------------------
lonf = 330
latf = 50
if rem_gmsl:
    print("Removing GMSL")
    out1 = slutil.remove_GMSL(ssha,lat5,lon5,timesyr,viz=True,testpoint=[lonf,latf],awgt=True)
    
    if len(out1)>2:
        ssha,gmslrem,fig,ax = out1
        plt.savefig(expdir+"GMSL_Removal_AVISO_testpoint_lon%i_lat%i.png"%(lonf,latf),dpi=200)
    else:
        ssha,gmsl=out1
        
    if np.all(np.abs(gmslrem)>(1e-10)):
        print("Saving GMSL")
        np.save(datpath+"AVISO_GMSL_%s_%s.npy"%(start,end),gmslrem)
else:
    print("GMSL Not Removed")
    

# ----------------------
#%% Design Low Pass Filter
# ----------------------

# ---
# Apply LP Filter
# ---
# Filter Parameters and Additional plotting options
dt   = 24*3600*30
M    = 5
xtk  = [1/(10*12*dt),1/(24*dt),1/(12*dt),1/(3*dt),1/dt]
xtkl = ['decade','2-yr','year','season','month']
order  = 5
tw     = 15 # filter size for time dim
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

# ---
# Visualize Filter Transfer Function
# ---
okdata,knan,okpts = proc.find_nan(slars,0)
npts5 = okdata.shape[1]
lpdata  = okdata.copy()
rawdata = ssha.reshape(ntimer,nlat5*nlon5)[:,okpts]
lpspec,rawspec,p24,filtxfer,fig,ax=slutil.check_lpfilter(rawdata,lpdata,xtk[1],M,tw,dt=24*3600*30)
#plt.savefig("%sFilter_Transfer_%imonLP_%ibandavg_%s.png"%(expdir,tw,M,expname),dpi=200)

#%% Calculate covariance matrix

sla = sla_lp
lon = lon5
lat = lat5


absmode   = 0
distthres = 3000
distmode  = 0
uncertmode = 0
# --------------------------------------------------------
# Calculate Correlation, Covariance, and Distance Matrices
# --------------------------------------------------------
ntime,nlat,nlon = sla.shape
srho,scov,sdist,okdata,okpts,coords2=slutil.calc_matrices(sla,lon,lat,return_all=True)


# -------------------------------
# Apply corrections based on mode
# -------------------------------
if absmode == 1: # Take Absolute Value of Correlation/Covariance
    scov = np.abs(scov)
    srho = np.abs(srho)
elif absmode == 2: # Use Anticorrelation, etc
    scov *= -1
    srho *= -1

# --------------------------
# Combine the Matrices
# --------------------------
a_fac = np.sqrt(-distthres/(2*np.log(0.5))) # Calcuate so exp=0.5 when distance is 3000km
expterm = np.exp(-sdist/(2*a_fac**2))

if distmode == 0: # Include distance and correlation
    distance_matrix = 1-expterm*srho
elif distmode == 1: # Just Include distance
    distance_matrix = 1-expterm
elif distmode == 2: # Just Include correlation
    distance_matrix = 1-srho

# --------------------------
# Do Clustering (scipy)
# --------------------------
cdist      = squareform(distance_matrix,checks=False)
linked     = linkage(cdist,'weighted')
clusterout = fcluster(linked, nclusters,criterion='maxclust')

# -------------------------
# Calculate the uncertainty
# -------------------------
uncertout = np.zeros(clusterout.shape)
uncertsig  = np.zeros(clusterout.shape)
mcdistr = [] # Distributions
mcbnds  = [] # Lower and upper bounds
mcidxs  = []
for i in tqdm(range(len(clusterout))):
    covpt     = scov[i,:]     # 
    cid       = clusterout[i] #
    covin     = covpt[np.where(clusterout==cid)]
    covout    = covpt[np.where(clusterout!=cid)]
    if uncertmode == 0:
        uncertpt  = np.mean(covin)/np.mean(covout)
    elif uncertmode == 1:
        uncertout[i] = np.median(covin)/np.median(covout)
    uncertout[i] = uncertpt
    
    # --------------------------------------------
    # Monte-Carlo Analysis to compute significance
    # --------------------------------------------
    result = slutil.monte_carlo_cluster(uncertpt,covpt,len(covin),uncertmode,
                                       mciter=1000,p=0.05,tails=2,
                                       return_values=True)
    sigpt,mcvalues,mcidx,bnds=result
    uncertsig[i] = sigpt
    mcdistr.append(mcvalues)
    bnds.append(mcbnds)
    mcidxs.append(mcidx)


# -----------------------
# Replace into full array
# -----------------------
clustered = np.zeros(nlat*nlon)*np.nan
clustered[okpts] = clusterout

clustered = clustered.reshape(nlat,nlon)


cluster_count = []
for i in range(nclusters):
    cid = i+1
    cnt = (clustered==cid).sum()
    cluster_count.append(cnt)
uncert         = np.zeros(nlat*nlon)*np.nan
uncertsig_full = uncert.copy()

uncertsig_full[okpts] = uncertsig
uncert[okpts] = uncertout
uncert            = uncert.reshape(nlat,nlon)
uncertsig_full     = uncertsig_full.reshape(nlat,nlon)

# --------------
# Plot all ECDFs
# -------------- 
ccolors = np.array(
                [[233,51,35],
                [73,161,68],
                [154,219,232],
                [251,237,79],
                [81,135,195],
                [138,39,113],
                ])/255

#ccolors = ["red","blue","cyan","magenta","yellow","black"]
fig,ax = plt.subplots(1,1)
for pt in range(len(covpt)):
    cid = clusterout[pt]-1

    ax.plot(mcdistr[pt],np.linspace(0,1,1000),alpha=0.25,label="",color=ccolors[cid])
ax.set_xlabel("Uncertainty ($<\sigma^2_{x,in}>/<\sigma^2_{x,out}>$)")
ax.set_ylabel("Cumulative Probability (%)")
ax.set_title("Empirical CDF of Uncertainty ($u_x$), All Points \n %i Monte Carlo Simulations"% (1000))
ax.grid(True,ls='dotted')
ax.set_xlim([-200,200])
plt.savefig("%sEmpiricalCDF_MCTest.png"%expdir,dpi=200)

# --------------------------
# Plot all ECDFs separatenly
# --------------------------
fig,axs = plt.subplots(2,3,figsize=(10,6))
for pt in range(len(covpt)):
    if uncertsig[pt] == 0:
        plotcolor="k"
    else:
        plotcolor=ccolors[cid]
    cid = clusterout[pt]-1
    ax = axs.flatten()[cid]
    ax.plot(mcdistr[pt],np.linspace(0,1,1000),alpha=0.5,label="",color=plotcolor)
for cid in range(nclusters):
    ax = axs.flatten()[cid]
    ax.set_title("Cluster %i"%(cid+1))
    ax.set_xlim([-100,100])
    ax.grid(True,ls='dotted')

plt.suptitle("Empirical CDF of Uncertainty ($u_x$), All Points \n %i Monte Carlo Simulations"% (1000))
#ax.set_xlim([-200,200])
plt.savefig("%sEmpiricalCDF_MCTest_bycluster.png"%expdir,dpi=200)

# ----------------------
# Plot Clustering Result
# ----------------------
proj  = ccrs.PlateCarree(central_longitude=180)
cmapn = (mpl.colors.ListedColormap(ccolors))

fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = slutil.add_coast_grid(ax)
pcm    = ax.pcolormesh(lon5,lat5,clustered,cmap=cmapn,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
for o in range(nlon5):
    for a in range(nlat5):
        pt  = uncertsig_full[a,o]
        if pt == 1:
            pt = np.nan

        if np.isnan(pt):
            continue
        else: # Plot the removed points
            ax.scatter(lon5[o],lat5[a],s=10,marker="x",color="k",transform=ccrs.PlateCarree())

ax.set_title("AVISO Clusters (%s to %s)"%(start,end))
plt.savefig("%s%s_ClustersMap_iter%i.png"%(expdir,expname,0),dpi=200,bbox_inches='tight')




    



