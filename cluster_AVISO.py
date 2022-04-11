#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Perform Clustering Analysis on AVISO data

Uses coarsened/preprocessed data from preproc_AVISO

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
#%% Functions

def monte_carlo_cluster(uncertpt,covpt,N_in,mciter=1000,p=0.05,tails=2):
    """
    
    result = monte_carlo_cluster(uncertpt,covpt,N_in,mciter=1000,p=0.05,tails=2)
    
    Perform monte carlo significance test on the uncertainty metric
    [uncertpt] for a clustering output. 
    
    Repeats uncertainty calculation of N_in randomly selected points and
    checks to yield a distribution. Uses the significance level [p] (two-tailed)
    to see if the computed uncertainty [uncertpt] occured by chance.

    Parameters
    ----------
    uncertpt : FLOAT
        Uncertainty value for target point [mean(cov_in)/mean(cov_out)]
    covpt : ARRAY [number of pts]
        Covariance of the point with all other points
    N_in : INT
        Points within the cluster
    mciter : INT, optional
        Number of iterations. The default is 1000.
    p : FLOAT, optional
        Significance level. The default is 0.05.

    Returns
    -------
    1 or 0 : BOOL
        1: uncert value is significant (outside randomly generated distr.)
        0: uncert value is not significant (within randomly generated distr.)

    """
    
    # Assumes 2-tailed distribution (splits sig level to top/bottom of distr.)
    ptilde   = p/2
    
    # Get total point count
    N_tot   = len(covpt) # Total Points
    
    # Bootstrapping section
    mcuncert = np.zeros(mciter) # Output distribution
    for m in range(mciter):
        
        # Create index and shuffle
        shuffidx = np.arange(0,N_tot)
        np.random.shuffle(shuffidx) # Shuffles in place
        
        # Get first N_in last N_out points 
        pts_in  = covpt[shuffidx[:N_in]]
        pts_out = covpt[shuffidx[N_in:]]
        
        # Compute uncertainty ratio
        mcuncert[m] = np.mean(pts_in)/np.mean(pts_out)
    
    # Sort data, and find the significance thresholds (conservative)
    mcuncert.sort()
    id_lower = int(np.ceil(mciter*ptilde))
    id_upper = int(np.floor(mciter*(1-ptilde)))
    lowerbnd = mcuncert[id_lower]
    upperbnd = mcuncert[id_upper]
    
    # Check for significance
    if (uncertpt>lowerbnd) and (uncertpt<upperbnd):
        return 0 # Point is within randomly generated distribution
    else: # Point is outside randomly generated distribution
        return 1

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
    uncertsig  = np.zeros(clusterout.shape)
    for i in range(len(clusterout)):
        covpt     = scov[i,:]     # 
        cid       = clusterout[i] #
        covin     = covpt[np.where(clusterout==cid)]
        covout    = covpt[np.where(clusterout!=cid)]
        uncertpt  = np.mean(covin)/np.mean(covout)
        uncertout[i] = uncertpt
        
        # --------------------------------------------
        # Monte-Carlo Analysis to compute significance
        # --------------------------------------------
        sigpt = monte_carlo_cluster(uncertpt,covpt,len(covin),mciter=1000,p=0.05,tails=2)
        uncertsig[i] = sigpt
        
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
        return clustered,uncert,uncertsig,cluster_count,srho,scov,sdist,distance_matrix
    return clustered,uncert,uncertsig,cluster_count

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
    uncertraw = uncert.copy()
    
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
        
        # Save results
        allclusters.append(clustered)
        alluncert.append(uncert)
        allcount.append(cluster_count)
        
        # Visualize Results
        fig,ax,fig1,ax1 = plot_results(clustered,uncert,expname,lat,lon,outfigpath)
        

        
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
    outname = "%sSSHA_LP_%s_order%i_cutoff%i_remGMSL%i.npz" % (datpath,datname,order,tw,rem_gmsl)
    print("Saved to: %s"%outname)
    np.savez(outname,**{
        'sla_lp':sla_lp,
        'lon':lon5,
        'lat':lat5,
        'times':times
        })
#%% Perform Clustering

maxiter = 1
#minpts  = 30

# allclusters,alluncert,alluncertsig,allcount,remptsall,Wk,alls,alls_byclust = slutil.elim_points(sla_lp,lat5,lon5,
#                                                                                     nclusters,minpts,maxiter,expdir,calcsil=True)

allclusters,alluncert,alluncertsig,allcount,remptsall,Wk,alls,alls_byclust = slutil.elim_points_mc(sla_lp,lat5,lon5,
                                                                                    nclusters,maxiter,expdir,calcsil=True)

#%% Make Clustering Map

# Dictionary of Bounding Boxes to search thru
it        = 0
# Inputs
clusterin = allclusters[it]
uncertin  = alluncert[it]
uncertsig = alluncertsig[it]
rempts    = remptsall
vlm       = [-10,10]
nclusters = 6
plotsig   = True
sameplot  = False

# Make Region Colors
cmapn,regiondict = slutil.get_regions()

# rempts = rempts.flatten()
# rempts[~np.isnan(rempts)] = 1
# rempts = rempts.reshape(nlat5,nlon5)

proj = ccrs.PlateCarree(central_longitude=180)

# Rearrange clustering number
clusternew,remapdict = slutil.remapcluster(clusterin,lat5,lon5,regiondict,returnremap=True)


# -------------
# Plot Clusters
# -------------
if sameplot:
    fig,axs = plt.subplots(1,2,subplot_kw={'projection':proj},figsize=(12,4))
    ax = axs[0]
else:
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = slutil.add_coast_grid(ax)
pcm    = ax.pcolormesh(lon5,lat5,clusternew,cmap=cmapn,transform=ccrs.PlateCarree())
#ax.pcolor(lon5,lat5,rempts,cmap='Greys',transform=ccrs.PlateCarree(),hatch=":")
for o in range(nlon5):
    for a in range(nlat5):
        if plotsig:
            pt  = uncertsig[a,o]
            if pt == 1:
                pt = np.nan
        else: # Just plot removed points
            pt = rempts[a,o]
        
        
        if np.isnan(pt):
            continue
        else: # Plot the removed points
            ax.scatter(lon5[o],lat5[a],s=10,marker="x",color="k",transform=ccrs.PlateCarree())
            
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("AVISO Clusters (%s to %s)"%(start,end))
if sameplot:
    ax = axs[1]
else:
    plt.savefig("%s%s_ClustersMap_iter%i.png"%(expdir,expname,it),dpi=200,bbox_inches='tight')
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
    

# ------------------
# Plot Uncertainties
# ------------------
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,uncertin,vmin=vlm[0],vmax=vlm[-1],cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
#cl = ax.contour(lon5,lat5,clusternew,levels=np.arange(0,nclusters+2),colors='k',transform=ccrs.PlateCarree())

fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title(r"AVISO Cluster Uncertainty $(<\sigma^{2}_{x,in}>/<\sigma^{2}_{x,out}>)$")

if sameplot:
    plt.savefig("%s%s_Cluster_and_Uncert.png"%(expdir,expname),dpi=200,bbox_inches='tight')
else:
    plt.savefig("%s%s_ClustersUncert.png"%(expdir,expname),dpi=200,bbox_inches='tight')

#%% Clustering mape with silhouette

sigval = 0

# Dictionary of Bounding Boxes to search thru
# Inputs
clusterin = allclusters[-1]
uncertin = alluncert[-1]
s_in = alls[-1]
rempts = remptsall
vlm = [-10,10]
nclusters = 6

sameplot=True

# Make Region Colors
cmapn,regiondict = slutil.get_regions()

# rempts = rempts.flatten()
# rempts[~np.isnan(rempts)] = 1
# rempts = rempts.reshape(nlat5,nlon5)

proj = ccrs.PlateCarree(central_longitude=180)

# Rearrange clustering number
clusternew,remapdict = slutil.remapcluster(clusterin,lat5,lon5,regiondict,returnremap=True)


clusterout,knan,okpts = proc.find_nan((clusternew).flatten(),0)

# -------------
# Plot Clusters
# -------------
if sameplot:
    fig,axs = plt.subplots(1,2,subplot_kw={'projection':proj},figsize=(12,4))
    ax = axs[0]
else:
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = viz.add_coast_grid(ax)
pcm    = ax.pcolormesh(lon5,lat5,clusternew,cmap=cmapn,transform=ccrs.PlateCarree())
#ax.pcolor(lon5,lat5,rempts,cmap='Greys',transform=ccrs.PlateCarree(),hatch=":")
for o in range(nlon5):
    for a in range(nlat5):
        pt = rempts[a,o]
        if np.isnan(pt):
            continue
        else:
            ax.scatter(lon5[o],lat5[a],s=10,marker="x",color="k",transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("AVISO Clusters (%s to %s)"%(start,end))
if sameplot:
    ax = axs[1]
else:
    plt.savefig("%s%s_ClustersMap.png"%(expdir,expname),dpi=200,bbox_inches='tight')
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
    
    

# Plot 1: the silhoutte value map
cmap="RdBu_r"
silmap = np.zeros(nlat5*nlon5)*np.nan
silmap[okpts] = s_in
silmap = silmap.reshape(nlat5,nlon5)

proj = ccrs.PlateCarree(central_longitude=180)
#fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = slutil.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,silmap,vmin=-.25,vmax=.25,cmap=cmap,transform=ccrs.PlateCarree())
ax.contour(lon5,lat5,silmap,levels=[sigval],colors='k',linewidths=0.75,linestyles=":",transform=ccrs.PlateCarree())
ax.pcolormesh(lon5,lat5,silmap,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("Silhouette Map ($s_{avg}=%.2e$)"%(s_in.mean()))



if sameplot:
    plt.savefig("%s%s_Cluster_and_Silmap.png"%(expdir,expname),dpi=200,bbox_inches='tight')
else:
    plt.savefig("%s%s_ClustersUncert.png"%(expdir,expname),dpi=200,bbox_inches='tight')





#%%

#
# Plot Silhouette Map
#
silmap = np.zeros(nlat5*nlon5)*np.nan
silmap[okpts] = s_in
silmap = silmap.reshape(nlat5,nlon5)
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,silmap,vmin=-.5,vmax=.5,cmap=cmap,transform=ccrs.PlateCarree())
ax.contour(lon5,lat5,silmap,levels=[0],colors='k',linewidths=0.75,linestyles=":",transform=ccrs.PlateCarree())
#ax.pcolormesh(lon5,lat5,silmap,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("Silhouette Values for CESM-PiC Clusters (Year %s) \n $s_{avg}$ = %.3f" % (yearstr,s_in.mean()))
plt.savefig("%sCESM1PIC_%s_SilhouetteMap_k%s%i_%s_gmslnew.png"%(outfigpath,expname,kmode,ii,yearstr),dpi=200,bbox_inches='tight')



#%%

# # Set some defaults
# ucolors = ('Reds','Greys','Blues','Reds','Oranges','Greens')
# proj = ccrs.PlateCarree(central_longitude=180)
# cmap = cm.get_cmap("jet",nclusters)


#  # Plot Cluster Uncertainty
# fig1,ax1 = plt.subplots(1,1,subplot_kw={'projection':proj})
# ax1 = slutil.add_coast_grid(ax1)
# for i in range(nclusters):
#     cid = i+1
#     if (i+1) > len(ucolors):
#         ci=i%len(ucolors)
#     else:
#         ci=i
#     cuncert = uncert.copy()
#     cuncert[clusternew!=cid] *= np.nan
#     ax1.pcolormesh(lon5,lat5,cuncert,vmin=0,vmax=2,cmap=ucolors[ci],transform=ccrs.PlateCarree())
#     #fig.colorbar(pcm,ax=ax)
# ax1.set_title("Clustering Output (nclusters=%i) "% (nclusters))
# plt.savefig(outfigpath+"Cluster_with_Shaded_uncertainties_%s.png" % expname,dpi=200)

#plt.pcolormesh(allclusters[0])


# np.savez("%s%s_Results.npz"%(datpath,expname),**{
#     'lon':lon5,
#     'lat':lat5,
#     'sla':sla_lp,
#     'clusters':allclusters,
#     'uncert':alluncert,
#     'count':allcount,
#     'rempts':rempts},allow_pickle=True)

# cmap2  = cm.get_cmap("jet",len(allcount)+1)
# fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
# ax     = slutil.add_coast_grid(ax)
# pcm    = ax.pcolormesh(lon5,lat5,rempts,cmap=cmap2,transform=ccrs.PlateCarree())
# fig.colorbar(pcm,ax=ax)
# ax.set_title("Removed Points")
# plt.savefig("%sRemovedPoints_by_Iteration.png" % (expdir),dpi=200)
# plt.pcolormesh(lon5,lat5,rempts)
