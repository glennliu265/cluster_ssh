#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Perform Clustering Analysis on CESM Data
Created on Wed Mar 10 10:10:37 2021

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
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210309/"

# Experiment Names
start       = '1993-01'
end         = '2013-01'
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

ensnum  = e+1
datname = "CESM_ens%i_%s_to_%s_remGMSL%i" % (ensnum,start,end,rem_gmsl)
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

#%% Functions

def cluster_ssh(sla,lat,lon,nclusters,distthres=3000,
                returnall=False):
    # Remove All NaN Points
    ntime,nlat,nlon = sla.shape
    slars = sla.reshape(ntime,nlat*nlon)
    okdata,knan,okpts = proc.find_nan(slars,0)
    npts = okdata.shape[1]
    
    # ---------------------------------------------
    # Calculate Correlation and Covariance Matrices
    # ---------------------------------------------
    srho = np.corrcoef(okdata.T,okdata.T)
    scov = np.cov(okdata.T,okdata.T)
    srho = srho[:npts,:npts]
    scov = scov[:npts,:npts]
    
    # --------------------------
    # Calculate Distance Matrix
    # --------------------------
    lonmesh,latmesh = np.meshgrid(lon,lat)
    coords  = np.vstack([lonmesh.flatten(),latmesh.flatten()]).T
    coords  = coords[okpts,:]
    coords1 = coords.copy()
    coords2 = np.zeros(coords1.shape)
    coords2[:,0] = np.radians(coords1[:,1]) # First point is latitude
    coords2[:,1] = np.radians(coords1[:,0]) # Second Point is Longitude
    sdist = haversine_distances(coords2,coords2) * 6371
    
    # --------------------------
    # Combine the Matrices
    # --------------------------
    a_fac = np.sqrt(-distthres/(2*np.log(0.5))) # Calcuate so exp=0.5 when distance is 3000km
    expterm = np.exp(-sdist/(2*a_fac**2))
    distance_matrix = 1-expterm*srho
    
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
    for i in range(len(clusterout)):
        covpt     = scov[i,:]     # 
        cid       = clusterout[i] #
        covin     = covpt[np.where(clusterout==cid)]
        covout    = covpt[np.where(clusterout!=cid)]
        uncertout[i] = np.mean(covin)/np.mean(covout)

    # Apply rules from Thompson and Merrifield (Do this later)
    # if uncert > 2, set to 2
    # if uncert <0.5, set to 0
    #uncertout[uncertout>2]   = 2
    #uncertout[uncertout<0.5] = 0 
    
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
        print("Found %i points in cluster %i" % (cnt,cid))
    uncert = np.zeros(nlat*nlon)*np.nan
    uncert[okpts] = uncertout
    uncert = uncert.reshape(nlat,nlon)
    
    if returnall:
        return clustered,uncert,cluster_count,srho,scov,sdist,distance_matrix
    return clustered,uncert,cluster_count



def plot_results(clustered,uncert,expname,lat5,lon5,outfigpath):
    
    # Set some defaults
    
    ucolors = ('Blues','Purples','Greys','Blues','Reds','Oranges','Greens')
    proj = ccrs.PlateCarree(central_longitude=180)
    cmap = cm.get_cmap("jet",nclusters)
    
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
    ax = slutil.add_coast_grid(ax)
    gl = ax.gridlines(ccrs.PlateCarree(central_longitude=0),draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle="dotted",lw=0.75)
    gl.xlabels_top = False
    gl.ylabels_right = False
    pcm = ax.pcolormesh(lon5,lat5,clustered,cmap=cmap,transform=ccrs.PlateCarree())#,cmap='Accent')#@,cmap='Accent')
    plt.colorbar(pcm,ax=ax,orientation='horizontal')
    ax.set_title("Clustering Results \n nclusters=%i %s" % (nclusters,expname))
    plt.savefig("%sCluster_results_n%i_%s.png"%(outfigpath,nclusters,expname),dpi=200,transparent=True)
    
    # Plot raw uncertainty
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
    ax     = slutil.add_coast_grid(ax)
    pcm    = plt.pcolormesh(lon5,lat5,uncert,cmap='copper',transform=ccrs.PlateCarree())
    ax.set_title(r"Uncertainty $(<\sigma^{2}_{out,x}>/<\sigma^{2}_{in,x}>)$")
    fig.colorbar(pcm,ax=ax,fraction=0.02)
    plt.savefig(outfigpath+"Uncertainty.png",dpi=200)
    
    
    # Apply Thompson and Merrifield thresholds
    uncert[uncert>2] = 2
    uncert[uncert<0.5]=0
    
    # Plot Cluster Uncertainty
    fig1,ax1 = plt.subplots(1,1,subplot_kw={'projection':proj})
    ax1 = slutil.add_coast_grid(ax1)
    for i in range(nclusters):
        cid = i+1
        if (i+1) > len(ucolors):
            ci=i%len(ucolors)
        else:
            ci=i
        cuncert = uncert.copy()
        cuncert[clustered!=cid] *= np.nan
        ax1.pcolormesh(lon5,lat5,cuncert,vmin=0,vmax=2,cmap=ucolors[ci],transform=ccrs.PlateCarree())
        #fig.colorbar(pcm,ax=ax)
    ax1.set_title("Clustering Output (nclusters=%i) %s "% (nclusters,expname))
    plt.savefig(outfigpath+"Cluster_with_Shaded_uncertainties_%s.png" % expname,dpi=200)
    
    

    
    return fig,ax,fig1,ax1
    

def elim_points(sla,lat,lon,nclusters,minpts,maxiter,outfigpath,distthres=3000):
    
    ntime,nlat,nlon = sla.shape
    slain = sla.copy()
    
    # Preallocate
    allclusters = []
    alluncert   = []
    allcount    = []
    rempts      = np.zeros((nlat*nlon))*np.nan
    
    # Loop
    flag = True
    it   = 0
    while flag or it < maxiter:
        
        expname = "iteration%02i" % (it+1)
        print("Iteration %i ========================="%it)
        
        # Perform Clustering
        clustered,uncert,cluster_count = cluster_ssh(slain,lat,lon,nclusters,distthres=distthres)
        
        # Visualize Results
        fig,ax,fig1,ax1 = plot_results(clustered,uncert,expname,lat,lon,outfigpath)
        
        # Save results
        allclusters.append(clustered)
        alluncert.append(uncert)
        allcount.append(cluster_count)
        
        # Check cluster counts
        for i in range(nclusters):
            cid = i+1
            
            flag = False
            if cluster_count[i] < minpts:
                flag = True # Set flag to continue running
                print("\tCluster %i (count=%i) will be removed" % (cid,cluster_count[i]))
                
                clusteredrs = clustered.reshape(nlat*nlon)
                slainrs = slain.reshape(ntime,nlat*nlon)
                
                
                slainrs[:,clusteredrs==cid] = np.nan # Assign NaN Values
                rempts[clusteredrs==cid] = it # Record iteration of removal
                
                slain = slainrs.reshape(ntime,nlat,nlon)
        it += 1
    
    print("COMPLETE after %i iterations"%it)
    rempts = rempts.reshape(nlat,nlon)
    return allclusters,alluncert,allcount,rempts



#%% Load in the dataset

# Load data (preproc, then anomalized)
st = time.time()
ds = xr.open_dataset("%sSSH_coarse_ens%02d.nc"%(datpath,ensnum))
ssh = ds.SSH.values/100 # Convert to meters
lat5 = ds.lat.values
lon5 = ds.lon.values
times = ds.time.values
ntime,nlat5,nlon5 = ssh.shape
print("Loaded data in %.2fs"%(time.time()-st))

# Plotting utilities
cmbal = cmocean.cm.balance

#%% Additional Preprocessing Steps

# -----------------------------------------
# Limit to particular period (CESM Version)
# -----------------------------------------
# Convert Datestrings
timesmon = np.array(["%04d-%02d"%(t.year,t.month) for t in times])

# Find indices
idstart  = np.where(timesmon==start)[0][0]
idend    = np.where(timesmon==end)[0][0] + 1 

# Restrict Data to period
ssh     = ssh[idstart:idend,:,:]
timeslim = timesmon[idstart:idend]
timesyr  = np.array(["%04d"%(t.year) for t in times])[idstart:idend]
ntimer   = ssh.shape[0]

# -------------------------
# Remove the Long Term Mean
# -------------------------
ssha = ssh - ssh.mean(0)[None,:,:]
if debug: # Plot Results of mean removal
    fig,axs = plt.subplots(2,1,figsize=(8,8))
    ax = axs[0]
    pcm = ax.pcolormesh(lon5,lat5,ssh[0,:,:])
    fig.colorbar(pcm,ax=ax)
    ax.set_title("SSH")
    
    ax = axs[1] 
    pcm = ax.pcolormesh(lon5,lat5,ssha[0,:,:],cmap=cmbal)
    fig.colorbar(pcm,ax=ax)
    ax.set_title("SSH Anomaly (Long Term Mean Removed")



# ------------------------------
# Filter Again, If Option is Set
# ------------------------------
if filteragain:
    slasmooth = np.zeros((ntimer,nlat5,nlon5))
    for i in tqdm(range(ntimer)):
        da = xr.DataArray(ssha[i,:,:].astype('float32'),
                        coords={'lat':lat5,'lon':lon5},
                        dims={'lat':lat5,'lon':lon5},
                        name='sla')
        timestamp = times[i]
        smooth_field = pygmt.grdfilter(grid=da, filter="g500", distance="4",nans="i")
        slasmooth[i,:,:] = smooth_field.values
    
    
    # Reapply Mask to correct for smoothed edges
    mask = ssha.sum(0)
    mask[~np.isnan(mask)] = 1
    sla_filt = slasmooth * mask[None,:,:]


    if debug:
        fig,axs = plt.subplots(2,1,figsize=(8,8))
        ax = axs[0]
        pcm = ax.pcolormesh(lon5,lat5,ssha[0,:,:],cmap=cmbal,vmin=-30,vmax=30)
        fig.colorbar(pcm,ax=ax)
        ax.set_title("SSHA Before Filtering")
        
        ax = axs[1] 
        pcm = ax.pcolormesh(lon5,lat5,sla_filt[0,:,:],cmap=cmbal,vmin=-30,vmax=30)
        fig.colorbar(pcm,ax=ax)
        ax.set_title("SSHA After Filtering")
        
        

# ------------------------------
# Apply land ice mask from aviso
# ------------------------------
mask = np.load(datpath+"AVISO_landice_mask_5deg.npy")
ssha = ssha * mask[None,:,:]

# ------------------
# Remove GMSL Signal
# ------------------
lonf = 330
latf = 50
if rem_gmsl:
    print("Removing GMSL")
    out1 = slutil.remove_GMSL(ssha,lat5,lon5,timesyr,viz=True,testpoint=[lonf,latf])
    
    if len(out1)>2:
        ssha,gmslrem,fig,ax = out1
        plt.savefig(expdir+"GMSL_Removal_CESM_ens%i_testpoint_lon%i_lat%i.png"%(ensnum,lonf,latf),dpi=200)
    else:
        ssha,gmsl=out1
        
    if np.all(np.abs(gmslrem)>(1e-10)):
        print("Saving GMSL")
        np.save(datpath+"CESM1_ens%i_GMSL_%s_%s.npy"%(ensnum,start,end),gmslrem)
else:
    print("GMSL Not Removed")
    

# --------------------------------------------------------
# %% Compare with data that was anomalized, then smoothed
# --------------------------------------------------------

# ds2 = xr.open_dataset(datpath+"SSHA_coarse_ens01.nc")

# ssha2 = ds2.SSH.values/100*mask[None,:,:]


# fig,axs = plt.subplots(3,1,figsize=(6,12))

# ax = axs[0]
# pcm = ax.pcolormesh(lon5,lat5,ssha[0,:,:],cmap=cmbal,vmin=-.3,vmax=.3)
# ax.set_title("SSHA Preprocess, then Anomalize")
# fig.colorbar(pcm,ax=ax)

# ax = axs[1]
# pcm = ax.pcolormesh(lon5,lat5,ssha2[0,:,:],cmap=cmbal,vmin=-.3,vmax=.3)
# ax.set_title("SSHA Anomalize, then Preprocess")
# fig.colorbar(pcm,ax=ax)

# ax = axs[2]
# pcm = ax.pcolormesh(lon5,lat5,np.nanmax(ssha[:,:,:]-ssha2[:,:,:],0),cmap=cmbal,vmin=-.3,vmax=.3)
# ax.set_title("Plot(1) - Plot (2)")
# fig.colorbar(pcm,ax=ax)

# print(" Max Difference is %e" % (np.nanmax(np.nanmax(ssha[:,:,:]-ssha2[:,:,:],0).flatten())) )

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
# Map removed points
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree(central_longitude=0)})
ax     = slutil.add_coast_grid(ax)
pcm = ax.pcolormesh(lon5,lat5,ptmap,cmap='bone',transform=ccrs.PlateCarree(),alpha=0.88)
fig.colorbar(pcm,ax=ax)
ax.set_title("Removed Zero Points")

# ---
# Visualize Filter Transfer Function
# ---
okdata,knan,okpts = proc.find_nan(slars,0)
npts5 = okdata.shape[1]
lpdata  = okdata.copy()
rawdata = ssha.reshape(ntimer,nlat5*nlon5)[:,okpts]
lpspec,rawspec,p24,filtxfer,fig,ax=slutil.check_lpfilter(rawdata,lpdata,xtk[1],M,tw,dt=24*3600*30)
plt.savefig("%sFilter_Transfer_%imonLP_%ibandavg_%s.png"%(expdir,tw,M,expname),dpi=200)

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
#%% Perform Clustering

allclusters,alluncert,allcount,rempts = elim_points(sla_lp,lat5,lon5,nclusters,minpts,maxiter,expdir)

np.savez("%s%s_Results.npz"%(datpath,expname),**{
    'lon':lon5,
    'lat':lat5,
    'sla':sla_lp,
    'clusters':allclusters,
    'uncert':alluncert,
    'count':allcount,
    'rempts':rempts},allow_pickle=True)

cmap2  = cm.get_cmap("jet",len(allcount)+1)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
ax     = slutil.add_coast_grid(ax)
pcm    = ax.pcolormesh(lon5,lat5,rempts,cmap=cmap2,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax)
ax.set_title("Removed Points")
plt.savefig("%sRemovedPoints_by_Iteration.png" % (expdir),dpi=200)
plt.pcolormesh(lon5,lat5,rempts)
