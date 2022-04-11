#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Perform Clustering Analysis on CESM Data (Pre-Industrial Control)
Created on Wed Mar 10 10:10:37 2021

@author: gliu
"""

from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import silhouette_score,silhouette_samples

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xarray as xr
import numpy as np

#import pygmt
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
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210610/"
proc.makedir(outfigpath)

# Experiment Names
#start       = '1993-01'
#end         = '2013-01'
#start      = '1850-01'
#end        = '2100-12'
nclusters   = 6
rem_gmsl    = True
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

#%% Functions

def cluster_ssh(sla,lat,lon,nclusters,distthres=3000,
                returnall=False,absmode=0,distmode=0,uncertmode=0,printmsg=True,
                calcsil=False):
    
    # ---------------------------------------------
    # Calculate Correlation, Covariance, and Distance Matrices
    # ---------------------------------------------
    ntime,nlat,nlon = sla.shape
    srho,scov,sdist,okdata,okpts,coords2=slutil.calc_matrices(sla,lon5,lat5,return_all=True)
    #npts = okdata.shape[1]
    
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
    
    
    # --------------------
    # Calculate Silhouette
    # --------------------
    if calcsil:
        s_score,s,s_bycluster = slutil.calc_silhouette(distance_matrix,clusterout,nclusters)
    # fig,ax = plt.subplots(1,1)
    # ax = slutil.plot_silhouette(clusterout,nclusters,s,ax1=ax)
    
    # -------------------------
    # Calculate the uncertainty
    # -------------------------
    uncertout = np.zeros(clusterout.shape)
    for i in range(len(clusterout)):
        covpt     = scov[i,:]     # 
        cid       = clusterout[i] #
        covin     = covpt[np.where(clusterout==cid)]
        covout    = covpt[np.where(clusterout!=cid)]
        
        if uncertmode == 0:
            uncertout[i] = np.mean(covin)/np.mean(covout)
        elif uncertmode == 1:
            uncertout[i] = np.median(covin)/np.median(covout)

    # Apply rules from Thompson and Merrifield (Do this later)
    # if uncert > 2, set to 2
    # if uncert <0.5, set to 0
    #uncertout[uncertout>2]   = 2
    #uncertout[uncertout<0.5] = 0 
    
    # ------------------------------
    # Calculate Wk for gap statistic
    # ------------------------------
    Wk = np.zeros(nclusters)
    for i in range(nclusters):
        
        cid = i+1
        ids = np.where(clusterout==cid)[0]
        dist_in = distance_matrix[ids[:,None],ids[None,:]] # Get Pairwise Distances within cluster
        dist_in = dist_in.sum()/2 # Sum and divide by 2 (since pairs are replicated)
        
        Wk[i] = dist_in
    
    
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
        if printmsg:
            print("Found %i points in cluster %i" % (cnt,cid))
    uncert = np.zeros(nlat*nlon)*np.nan
    uncert[okpts] = uncertout
    uncert = uncert.reshape(nlat,nlon)
    
    if calcsil: # Return silhouette values
        return clustered,uncert,cluster_count,Wk,s,s_bycluster
    if returnall:
        return clustered,uncert,cluster_count,Wk,srho,scov,sdist,distance_matrix
    return clustered,uncert,cluster_count,Wk

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
    

def elim_points(sla,lat,lon,nclusters,minpts,maxiter,outfigpath,distthres=3000,
                absmode=0,distmode=0,uncertmode=0,viz=True,printmsg=True,
                calcsil=False):
    
    ntime,nlat,nlon = sla.shape
    slain = sla.copy()
    
    # Preallocate
    allclusters = []
    alluncert   = []
    allcount    = []
    allWk = []
    if calcsil:
        alls           = []
        alls_byclust = []
    rempts      = np.zeros((nlat*nlon))*np.nan
    
    # Loop
    flag = True
    it   = 0
    while flag and it < maxiter:
        
        if printmsg:
            print("Iteration %i ========================="%it)
        expname = "iteration%02i" % (it+1)
        #print("Iteration %i ========================="%it)
        
        # Perform Clustering
        clustoutput = cluster_ssh(slain,lat,lon,nclusters,distthres=distthres,
                                                     absmode=absmode,distmode=distmode,uncertmode=uncertmode,
                                                     printmsg=printmsg,calcsil=calcsil)
        
        if calcsil:
            clustered,uncert,cluster_count,Wk,s,s_byclust = clustoutput
            alls.append(s)
            alls_byclust.append(s_byclust)
        else:
            clustered,uncert,cluster_count,Wk = clustoutput
        
        # Save results
        allclusters.append(clustered)
        alluncert.append(uncert)
        allcount.append(cluster_count)
        allWk.append(Wk)
        
        if viz:
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
        # if removeflag:
        #     flag = True
        # else:
        #     flag = False
        it += 1
    if printmsg:
        print("COMPLETE after %i iterations"%it)
    rempts = rempts.reshape(nlat,nlon)
    if calcsil:
        return allclusters,alluncert,allcount,rempts,allWk,alls,alls_byclust
    return allclusters,alluncert,allcount,rempts,allWk



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

# Set up time array
timesmon = np.array(["%04d-%02d"%(t.year,t.month) for t in times])

# Plotting utilities
cmbal = cmocean.cm.balance

#%% Work 

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
    out1 = slutil.remove_GMSL(ssha,lat5,lon5,timesyr,viz=True,testpoint=[lonf,latf],awgt=True)
    
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
    

#%% Making the Regiondict
from scipy import stats
import itertools
import matplotlib as mpl

ldzo = np.load(datpath+"CESM_PIC_remGMSL0_6clusters_minpts30_maxiters5_Results_winsize240.npz",allow_pickle=True)
test = ldzo["clusters"]

# Set Region IDs
regionids = ("1: Northwest Pacific",
             "2: ST N. Atl",
             "3: SP N. Atl",
             "4: East Pacific",
             "5: Indian-South Pacific Ocean",
             "6: South Atlantic"
             )

# Make Region Colors
regioncolors = np.array(
                [[233,51,35],
                [73,161,68],
                [154,219,232],
                [251,237,79],
                [81,135,195],
                [138,39,113],
                ])/255
cmapn = (mpl.colors.ListedColormap(regioncolors))

cmap = cm.get_cmap("jet",nclusters)

testmap = np.array(test[0])[0,:,:]

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,testmap[:,:],cmap=cmap)
ax     = viz.plot_box([150,179,5,50],ax=ax,leglab="1: Northwest Pacific")
ax     = viz.plot_box([280-360,350-360,20,45],ax=ax,leglab="2: ST N. Atl")
ax     = viz.plot_box([300-360,360-360,50,75],ax=ax,leglab="3: SP N. Atl")
ax     = viz.plot_box([200-360,250-360,0,35],ax=ax,leglab="4: East Pacific")
ax     = viz.plot_box([50,105,-30,15],ax=ax,leglab="5: Indian-South Pacific Ocean")
ax     = viz.plot_box([280-330,360-360,-50,-20],ax=ax,leglab="6: South Atlantic")
fig.colorbar(pcm,ax=ax)


#ax.legend()
# Dictionary of Bounding Boxes to search thru
regiondict = {1:[150,180,5,50],
             2:[280-360,350-360,20,45],
             3:[300-360,360-360,50,75],
             4:[200-360,250-360,0,35],
             5:[50,105,-30,15],
             6:[280-330,360-360,-50,-20]
             }



#%% Some New Tools


def remapcluster(inclust,lat5,lon5,regiondict,printmsg=True,returnremap=False):
    
    # Remap an input cluster [inclust] according
    # to a regiondict.
    # Searches within each region and assigns
    # value to most frequent class in a given region
    
    nlat,nlon = inclust.shape
    clusternew = inclust.copy()
    clusternewflat = clusternew.flatten()
    clusteroldflat = inclust.flatten()
    assigned = []
    remapdict = {}
    for r in regiondict.keys():
        #print(r)
        # Get Region
        bbox = regiondict[r].copy()
        for i in range(2): # Just check Longitudes
            if bbox[i] < 0:
                bbox[i]+=360
        varr,lonr,latr,=proc.sel_region(inclust.T,lon5,lat5,bbox,warn=printmsg)
        
        
        # Get rid of NaNs
        varrok = varr.flatten().copy()
        varrok = varrok[~np.isnan(varrok)]
        
        # Get unique elements and counts, sort by count
        eles,freqs = np.unique(varrok,return_counts=True)
        sortid = np.argsort(freqs)[::-1]
        eles = eles[sortid]
        done=False
        for ele in eles:
            if done: # Skip if already assigned
                continue
            if ele in assigned: # Skip if class has already be reassigned
                continue
            
            # Assign new cluster
            clusternewflat[clusteroldflat==ele] = r
            if printmsg:
                print("Reassigned Class %i to %i" % (ele,r))
            assigned.append(int(ele))
            remapdict[int(ele)] = r
            done=True
        
        if done is False: # When no cluster is assigned...
            # Get unassigned regions, and assign first one
            unassigned = np.setdiff1d(list(regiondict.keys()),assigned)
            ele = unassigned[0]
            clusternewflat[clusteroldflat==ele] = r
            assigned.append(int(ele))
            remapdict[int(ele)] = r
            if printmsg:
                print("Reassigned (Leftover) Class %i to %i because nothing was found" % (ele,r))
    clusternew = clusternewflat.reshape(nlat,nlon)
    if returnremap:
        return clusternew,remapdict
    return clusternew
        
def patterncorr(map1,map2):
    # From Taylor 2001,Eqn. 1, Ignore Area Weights
    
    # Get Non NaN values, Flatten, Array Size
    map1ok = map1.copy()
    map1ok = map1ok[~np.isnan(map1ok)].flatten()
    map2ok = map2.copy()
    map2ok = map2ok[~np.isnan(map2ok)].flatten()
    N = len(map1ok)
    
    # Anomalize
    map1a = map1ok - map1ok.mean()
    map2a = map2ok - map2ok.mean()
    std1  = np.std(map1ok)
    std2  = np.std(map2ok)
    
    # calculate
    R = 1/N*np.sum(map1a*map2a)/(std1*std2)
    return R

def make_mapdict(oldclass,newclass):
    mapdict = {oldclass[i] : newclass[i] for i in range(len(oldclass))}
    return mapdict

def reassign_classes(inclust,mapdict,printmsg=True):
    nlat,nlon = inclust.shape
    clusternew = inclust.copy()
    clusternewflat = clusternew.flatten()
    clusteroldflat = inclust.flatten()
    
    for i in mapdict.keys():
        newclass = mapdict[i]
        clusternewflat[clusteroldflat==i] = newclass
        if printmsg:
            print("Reassigned Class %i to %i "%(i,newclass))
    return clusternewflat.reshape(nlat,nlon)
        
        
        
def calc_cluster_patcorr(inclust,evalclust,oldclass=None,returnmax=True):
    if oldclass is None:
        oldclass = [1,2,3,4,5,6]
    
    
    # Make all possible permutations of classes
    pms = list(itertools.permutations(oldclass))
    
    # Loop through each permutation
    patcor = []
    for newclass in tqdm(pms):
        # Make Remapping Dictionary
        mapdict  = make_mapdict(oldclass,newclass)
        # Remap the Target Cluster
        remapclust = reassign_classes(evalclust,mapdict,printmsg=False)
        # Calculate Pattern Correlation and save
        pc = patterncorr(remapclust,inclust)
        patcor.append(pc)
    patcor = np.array(patcor)
    if returnmax:
        return np.nanmax(patcor)
    return patcor

#%% Testing out the functions
inclust = testmap


# Remapping Clusters
clusternew = remapcluster(testmap,lat5,lon5,regiondict)


# Plot Remapped Result
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,clusternew,cmap=cmapn)
for i in np.arange(1,7):
    ax,ls = viz.plot_box(regiondict[i],ax=ax,return_line=True,leglab=regionids[i-1])#,color=regioncolors[i-1])
#ax     = viz.plot_box(regiondict[1],ax=ax,leglab="1: Northwest Pacific")
#ax     = viz.plot_box([280-360,350-360,20,45],ax=ax,leglab="2: ST N. Atl")
#ax     = viz.plot_box([300-360,360-360,50,75],ax=ax,leglab="3: SP N. Atl")
#ax     = viz.plot_box([200-360,250-360,0,35],ax=ax,leglab="4: East Pacific")
#ax     = viz.plot_box([50,105,-30,15],ax=ax,leglab="5: Indian-South Pacific Ocean")
#ax     = viz.plot_box([280-330,360-360,-50,-20],ax=ax,leglab="6: South Atlantic")
fig.colorbar(pcm,ax=ax)


# Example here of testing out patterncorr script
map1 = testmap
map2 = clusternew
patterncorr(map2,map1)

# Testing patcorr iteratively 
patcorr = calc_cluster_patcorr(inclust,clusternew,returnmax=True)

"""
Some Notes

allclusters,alluncert = [nclusters,nlat,nlon]
allcount,allWk   = [niter,nclasses]
rempt     = [nlat,nlon]


"""


#
# %% First, Cluster for the whole PIC
#

sla_in = sla_lp[:,:,:]

# Do Clustering
allclusters,alluncert,allcount,rempt,allWk,alls,alls_byclust= slutil.elim_points(sla_in,lat5,lon5,nclusters,minpts,maxiter,expdir,
                                                             viz=False,printmsg=True,calcsil=True)


#%% Plot the results



# Dictionary of Bounding Boxes to search thru
# Inputs
clusterin = allclusters[-1]
uncertin = alluncert[-1]
rempts = rempt
vlm = [-10,10]
nclusters = 6


start = '400-01'
end = '2200-01'
sameplot=True

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
ax.set_title("CESM1 Clusters (%s to %s)"%(start,end))
if sameplot:
    ax = axs[1]
else:
    plt.savefig("%s%s_ClustersMap.png"%(expdir,expname),dpi=200,bbox_inches='tight')
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
# ------------------
# Plot Uncertainties
# ------------------
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,uncertin,vmin=vlm[0],vmax=vlm[-1],cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
#cl = ax.contour(lon5,lat5,clusternew,levels=np.arange(0,nclusters+2),colors='k',transform=ccrs.PlateCarree())

fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title(r"CESM1 Cluster Uncertainty $(<\sigma^{2}_{x,in}>/<\sigma^{2}_{x,out}>)$")

if sameplot:
    plt.savefig("%s%s_Cluster_and_Uncert.png"%(expdir,expname),dpi=200,bbox_inches='tight')
else:
    plt.savefig("%s%s_ClustersUncert.png"%(expdir,expname),dpi=200,bbox_inches='tight')




# inclust = np.array(allclusters[0])
# inuncert = np.array(alluncert[0])

# # Adjust classes
# clusterPIC = remapcluster(inclust,lat5,lon5,regiondict)

# # Plot some results (Clusters Themselves)
# proj = ccrs.PlateCarree(central_longitude=180)
# fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
# ax     = viz.add_coast_grid(ax)
# pcm=ax.pcolormesh(lon5,lat5,clusterPIC,cmap=cmapn,transform=ccrs.PlateCarree())
# fig.colorbar(pcm,ax=ax,fraction=0.025)
# ax.set_title("CESM-PiC Clusters (Year 400 to 2200)")
# plt.savefig("%sCESM1PIC_%s_Clusters_all.png"%(outfigpath,expname),dpi=200,bbox_inches='tight')

# # Now Plot the Uncertainties
# vlm = [-10,10]
# proj = ccrs.PlateCarree(central_longitude=180)
# fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
# ax     = viz.add_coast_grid(ax)
# pcm=ax.pcolormesh(lon5,lat5,inuncert,vmin=vlm[0],vmax=vlm[-1],cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
# fig.colorbar(pcm,ax=ax,fraction=0.025)
# ax.set_title(r"CESM-PIC Cluster Uncertainty $(<\sigma^{2}_{in,x}>/<\sigma^{2}_{out,x}>)$"+" \n (Year 400 to 2200) ")
# plt.savefig("%sCESM1PIC_%s_Uncert_all.png"%(outfigpath,expname),dpi=200,bbox_inches='tight')

#%% Plot results again, but this time with the silhouette metric


sigval = 0 #4.115e-3 # Significance Value (Greater than Red-Noise Null Hypothesis)

# Dictionary of Bounding Boxes to search thru
# Inputs
clusterin = allclusters[-1]
uncertin = alluncert[-1]
s_in = alls[-1]
rempts = rempt
vlm = [-10,10]
nclusters = 6


start = '400-01'
end = '2200-01'
sameplot=True

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
ax.set_title("CESM1 Clusters (%s to %s)"%(start,end))
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
# for o in range(nlon5):
#     for a in range(nlat5):
#         pt = silmap[a,o]
#         if pt > sigval:
#             continue
#         else:
#             ax.scatter(lon5[o],lat5[a],s=10,marker="x",color="k",transform=ccrs.PlateCarree())
ax.contour(lon5,lat5,silmap,levels=[sigval],colors='k',linewidths=0.75,linestyles=":",transform=ccrs.PlateCarree())
ax.pcolormesh(lon5,lat5,silmap,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("Silhouette Map ($s_{avg}=%.2e$)"%(s_in.mean()))


if sameplot:
    plt.savefig("%s%s_Cluster_and_Sil.png"%(expdir,expname),dpi=200,bbox_inches='tight')
else:
    plt.savefig("%s%s_ClustersUncert.png"%(expdir,expname),dpi=200,bbox_inches='tight')






#%% WTF is happening with uncertainty. Lets take a look

varin = sla_lp[:,:,:]

# distmode
# 0: Default (Distance and Corr)
# 1: Distance Only
# 2: Corr Only
# 3: Red Noise Dist

# uncertmode
# 0: Default (E(Cov_in) / E(Cov_out))
# 1: Median  (Med(Cov_in) / M(Cov_out))

# absmode
# 0: Default: Correlation and Covariances, no modification
# 1: Absolute Values: Take abs(corr) and abs(cov)
# 2: Anti: Take -1*corr, -1*cov

distmode   = 0
absmode    = 0
uncertmode = 0
chardist   = 3000

# ------------------
# Calculate Matrices
# ------------------
ntime,nlat,nlon = varin.shape
srho,scov,sdist,okdata,okpts,coords2=slutil.calc_matrices(varin,lon5,lat5,return_all=True)
if absmode == 1:
    scov = np.abs(scov)
    srho = np.abs(srho)
elif absmode == 2:
    scov *= -1
    srho *= -1

# --------------------------
# Combine the Matrices
# --------------------------
distthres=3000

# Default Distance Calculation
a_fac = np.sqrt(-distthres/(2*np.log(0.5)))
expterm = np.exp(-sdist/(2*a_fac**2))
if distmode == 0:
    distance_matrix = 1-expterm*srho
elif distmode == 1:
    distance_matrix = 1-expterm
elif distmode == 2:
    distance_matrix = 1-srho
elif distmode == 3:
    distance_matrix = 1-expterm*np.exp(-distthres/chardist)
    

# --------------------------
# Do Clustering (scipy)
# --------------------------
cdist      = squareform(distance_matrix,checks=False)
linked     = linkage(cdist,'weighted')
clusterout = fcluster(linked, nclusters,criterion='maxclust')


# ----------------------------
# Calculate Silhouette Metrics
# ----------------------------
s_score,s,s_byclust=slutil.calc_silhouette(distance_matrix,clusterout,nclusters)


# --------------------------
# Replace into pull matrix
# --------------------------
clustered = np.zeros(nlat*nlon)*np.nan
silmap    = clustered.copy()
clustered[okpts] = clusterout
silmap[okpts] = s
clustered = clustered.reshape(nlat,nlon)
silmap = silmap.reshape(nlat,nlon)

# ---------------------
# Calculate Uncertainty
# ---------------------
uncertout = np.zeros(clusterout.shape)
covins = []
covouts = []
for i in range(len(clusterout)):
    
    covpt     = scov[i,:]     #
    cid       = clusterout[i] #
    covin     = covpt[np.where(clusterout==cid)]
    covout    = covpt[np.where(clusterout!=cid)]
    covins.append(covin)
    covouts.append(covout)
    
    if uncertmode == 0:
        uncertout[i] = np.mean(covin)/np.mean(covout)
    elif uncertmode == 1:
        uncertout[i] = np.median(covin)/np.median(covout)
uncert = np.zeros(nlat*nlon)*np.nan
uncert[okpts] = uncertout
uncert = uncert.reshape(nlat,nlon)








# # Reassign to another Map
# clusterPICALL = clusterPIC.copy()
#
# %% Next, Cluster for some specific time period
#

start = '1750-02'
end   = '2200-12'
#end   = '1300-01'

# start = '1300-02'
# end   = '2200-12'


# Convert Datestrings
timesmon = np.array(["%04d-%02d"%(t.year,t.month) for t in times])

# Find indices
idstart  = np.where(timesmon==start)[0][0]
idend    = np.where(timesmon==end)[0][0]

# Restrict Data to period
sla_in     = sla_lp[idstart:idend,:,:]
timeslim = timesmon[idstart:idend]
timesyr  = np.array(["%04d"%(t.year) for t in times])[idstart:idend]
ntimer   = sla_in.shape[0]

timestr = "%s_to_%s" % (start,end)
timestrtitle = "%s to %s" % (start[:4],end[:4])

# Do Clustering
allclusters,alluncert,allcount,rempt,allWk = elim_points(sla_in,lat5,lon5,nclusters,minpts,maxiter,expdir,
                                                             viz=False,printmsg=False)

inclust = np.array(allclusters[-1])
inuncert = np.array(alluncert[-1])

# Adjust classes
clusterPIC = inclust
#clusterPIC = remapcluster(inclust,lat5,lon5,regiondict)
patcorr = calc_cluster_patcorr(clusterPIC,clusterPICALL,returnmax=True)

# Plot some results (Clusters Themselves)
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,clusterPIC,cmap=cmapn,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("CESM-PiC Clusters (Year %s) \n Pattern Correlation = %.3f" % (timestrtitle,patcorr))
plt.savefig("%sCESM1PIC_%s_Clusters_%s.png"%(outfigpath,expname,timestr),dpi=200,bbox_inches='tight')


# Now Plot the Uncertainties
vlm = [-10,10]
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,inuncert,vmin=vlm[0],vmax=vlm[-1],cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title(r"CESM-PIC Cluster Uncertainty $(<\sigma^{2}_{in,x}>/<\sigma^{2}_{out,x}>)$"+" \n (Year %s) " % (timestrtitle))
plt.savefig("%sCESM1PIC_%s_Uncert_%s.png"%(outfigpath,expname,timestr),dpi=200,bbox_inches='tight')




#%% Calculate Pattern Correlation for each moving 20-year window

npers = len(test)

remapclusts = np.zeros((npers,nlat5,nlon5))*np.nan

for i in tqdm(range(npers)):
    inclust    = np.array(test[i])[-1,:,:]
    clusterPIC = remapcluster(inclust,lat5,lon5,regiondict,printmsg=False)
    remapclusts[i,:,:] = clusterPIC.copy()
    


#%% Make an animation

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Animation parameters
frames = 12 #Indicate number of frames
figsize = (8,6)
vm = [-5,5]
interval = 0.1
bbox = [-80,0,0,80]
fps= 10
savetype="mp4"
dpi=100

yrstrs = []
for i in tqdm(range(ntime-240)):
    rng = np.arange(i,i+winsize+1)
    yrstr = "%s to %s" % (timesmon[rng[0]][:4],timesmon[rng[-1]][:4])
    yrstrs.append(yrstr)


lon180,remap180 = proc.lon360to180(lon5,remapclusts.transpose(2,1,0))

invar = remap180.transpose(1,0,2)
#invar = remapclusts.transpose(1,2,0) # [lat x lon x time]

def make_figure():
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    ax     = viz.add_coast_grid(ax)
    return fig,ax

start = time.time()
fig,ax = make_figure() # Make the basemap
pcm = ax.pcolormesh(lon180,lat5,invar[...,i],cmap=cmapn)
fig.colorbar(pcm,orientation='horizontal',fraction=0.046,pad=0.05)

pcm.set_array(invar[:,:,i].ravel())


def animate(i):
     pcm.set_array(invar[...,i].flatten())
     ax.set_title("Years %s" % (yrstrs[i]))
     print("\rCompleted frame %i"%i,end="\r",flush=True)
     
anim = FuncAnimation(
    fig, animate, interval=interval, frames=frames, blit=False,)

#anim.save('%sForcingAnim.mp4'%outfigpath, writer=animation.FFMpegWriter(fps=fps),dpi=dpi)
anim.save('%ssst_test.gif'%outfigpath, writer='imagemagick',fps=fps,dpi=dpi)

# Pass figure animator and draw on it
# blit = True, redraw only parts that have changed
print("Animation completed in %.2fs"%(time.time()-start))



# ------------------------
#%% Try Silhouette Metric
# -----------------------


start = '0400-01'
end   = '2200-12'

# Convert Datestrings
timesmon = np.array(["%04d-%02d"%(t.year,t.month) for t in times])

# Find indices
idstart  = np.where(timesmon==start)[0][0]
idend    = np.where(timesmon==end)[0][0]

# Restrict Data to period
sla_in     = sla_lp[idstart:idend,:,:]
timeslim = timesmon[idstart:idend]
timesyr  = np.array(["%04d"%(t.year) for t in times])[idstart:idend]
ntimer   = sla_in.shape[0]

timestr = "%s_to_%s" % (start,end)
timestrtitle = "%s to %s" % (start[:4],end[:4])

# Do Clustering
clustered,uncert,cluster_count,Wk,s,s_byclust = cluster_ssh(sla_in,lat5,lon5,6,returnall=True,calcsil=True)


# Set input data
inclust = np.array(clustered)
inuncert = np.array(uncert)

# Adjust classes
clusterPIC = inclust
clusterPIC,remapdict = remapcluster(inclust,lat5,lon5,regiondict,returnremap=True)
#patcorr = calc_cluster_patcorr(clusterPIC,clusterPICALL,returnmax=True)
new_sbyclust = np.zeros(nclusters)
for k in remapdict.keys():
    newclass = remapdict[k] # Class that k was remapped to
    new_sbyclust[newclass-1] = s_byclust[k-1] # Reassign
    print("Reassigned new class %i"%newclass)

# Recover clusterout for silhouette plotting
clusterout,knan,okpts = proc.find_nan(clusterPIC.flatten(),0)

# Plot the silhouette
fig,ax = plt.subplots(1,1)
ax = slutil.plot_silhouette(clusterout,nclusters,s,ax1=ax,cmap=regioncolors)
ax.grid(True,ls='dotted')
ax.set_title("Silhouette Plot for CESM-PiC Clusters (Year %s) \n Mean Silhouette Coefficient = %.3f" % (timestrtitle,s.mean()))
# Add dummy legend
for i in range(nclusters):
    cid = i+1
    ax.axvline([-100],lw=5,color=regioncolors[i],label="Cluster %i, s = %.3f"%(cid,new_sbyclust[i]))
ax.legend(fontsize=10)
ax.set_xticks(np.arange(-.2,.6,.1))
ax.set_xlim([-.25,.6])
plt.savefig("%sCESM1PIC_%s_SilhouettePlot_%s.png"%(outfigpath,expname,timestr),dpi=200,bbox_inches='tight')

# Replace silhouette into full map
silmap = np.zeros(nlat5*nlon5)*np.nan
silmap[okpts] = s
silmap = silmap.reshape(nlat5,nlon5)
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,silmap,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
ax.contour(lon5,lat5,silmap,levels=[0],colors='k',linewidths=0.75,transform=ccrs.PlateCarree())
#ax.pcolormesh(lon5,lat5,silmap,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("Silhouette Values for CESM-PiC Clusters (Year %s) \n $s_{avg}$ = %.3f" % (timestrtitle,s.mean()))
plt.savefig("%sCESM1PIC_%s_Silhouette_%s.png"%(outfigpath,expname,timestr),dpi=200,bbox_inches='tight')



# Plot some results (Clusters Themselves)
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,clusterPIC,cmap=cmapn,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("CESM-PiC Clusters (Year %s)" % (timestrtitle))
plt.savefig("%sCESM1PIC_%s_Clusters_%s.png"%(outfigpath,expname,timestr),dpi=200,bbox_inches='tight')


# Now Plot the Uncertainties
vlm = [-10,10]
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,inuncert,vmin=vlm[0],vmax=vlm[-1],cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title(r"CESM-PIC Cluster Uncertainty $(<\sigma^{2}_{in,x}>/<\sigma^{2}_{out,x}>)$"+" \n (Year %s) " % (timestrtitle))
plt.savefig("%sCESM1PIC_%s_Uncert_%s_stest.png"%(outfigpath,expname,timestr),dpi=200,bbox_inches='tight')





#%% Sections Below are old/under construction
