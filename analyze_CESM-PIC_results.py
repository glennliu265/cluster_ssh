#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze CESM-PIC-window results
Created on Mon May 24 14:05:27 2021

Sript to Analyze output for the CESM-PIC sliding window results generated by
cluster_CESM-PIC_chunk


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
import itertools
import matplotlib as mpl

#%%


# Set Paths
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210525/"
proc.makedir(outfigpath)

# Experiment Names
#start       = '1993-01'
#end         = '2013-01'
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

add_gmsl    = False # Add AVISO GMSL
if add_gmsl:
    rem_gmsl=0
    
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

# Window size for the experiment
winsize = 240

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

# Dictionary of Bounding Boxes to search thru
regiondict = {1:[150,180,5,50],
             2:[280-360,350-360,20,45],
             3:[300-360,360-360,50,75],
             4:[200-360,250-360,0,35],
             5:[50,105,-30,15],
             6:[280-330,360-360,-50,-20]
             }
#%%

# Load results
ldname      = "%s%s_Results_winsize%i.npz"%(datpath,expname,winsize)
ld          = np.load(ldname,allow_pickle=True)
clusts      =ld['clusters']
uncert      = ld['uncert']
count       = ld['count']
rempts      = ld['rempts']
Wks         = ld['Wks']
s_all       = ld['s_all']
s_by_clust  = ld['s_by_clust']
rngs        = ld['rngs']

#%% Load lowpased sla data
order = 5
order  = 5
tw     = 18 # filter size for time dim
outname = "%sSSHA_LP_%s_order%i_cutoff%i.npz" % (datpath,datname,order,tw)
ld2 =  np.load(outname,allow_pickle=True)

sla_lp = ld2['sla_lp']
lat5 = ld2['lat']
lon5 = ld2['lon']
times = ld2['times']


nlat5 = len(lat5)
nlon5 = len(lon5)
#%% Try Shfting, then calculating pattern correlation

test = clusts[0]
bbtest = [55,95,-25,-5]
bbtest2 = [55,95,-30,-10]


test1 = test.copy()

klon,klat = proc.sel_region(test,lon5,lat5,bbtest,returnidx=True)
selpoints = test[klat[:,None],klon[None,:]]

klon2,klat2 = proc.sel_region(test,lon5,lat5,bbtest2,returnidx=True)
test1[klat2[:,None],klon2[None,:]] = selpoints


# Calculate Pattern Correlation
R = slutil.patterncorr(test,test1)
print(R)

# Plot some results (Clusters Themselves)
proj = ccrs.PlateCarree(central_longitude=180)
fig,axs = plt.subplots(2,1,subplot_kw={'projection':proj},figsize=(4,5))

tests = [test,test1]
names = ['Before','After']
for a,t in enumerate(tests):
    
    ax = axs.flatten()[a]
    
    ax     = viz.add_coast_grid(ax)
    clusterPIC = tests[a]
    pcm=ax.pcolormesh(lon5,lat5,t,cmap=cmapn,transform=ccrs.PlateCarree())
    ax = viz.plot_box(bbtest,ax=ax)
    ax = viz.plot_box(bbtest2,ax=ax,color='r')
    
    fig.colorbar(pcm,ax=ax,fraction=0.025)
    ax.set_title("CESM-PiC Clusters (TEST)")

plt.suptitle("Shifting Pattern Downwards \n Pattern Correlation = %.3f" %R)
#plt.tight_layout()
plt.savefig("%sTEST_%s_Clusters_Shift.png"%(outfigpath,expname),dpi=200,bbox_inches='tight')



#plt.savefig("%sTEST2_%s_Clusters.png"%(outfigpath,expname),dpi=200,bbox_inches='tight')

#%% Try Multiplying all boxes by some value

test = clusts[0]
test1 = test.copy()
bbtest = [55,95,-25,-5]

klon,klat = proc.sel_region(test,lon5,lat5,bbtest,returnidx=True)
selpoints = test[klat[:,None],klon[None,:]]

test = test.flatten()
test1 = test1.flatten()
mults = [2,2,2,2,2,2]
for i in np.arange(1,7,1):
    test1[test==i] *= mults[i-1]
test1 = test1.reshape(nlat5,nlon5)
test = test.reshape(nlat5,nlon5)

#test1[klat[:,None],klon[None,:]] *= 200

#klon2,klat2 = proc.sel_region(test,lon5,lat5,bbtest2,returnidx=True)
#test1[klat2[:,None],klon2[None,:]] = selpoints

# Calculate Pattern Correlation
R = slutil.patterncorr(test,test1)
print(R)

# Plot some results (Clusters Themselves)
proj = ccrs.PlateCarree(central_longitude=180)
fig,axs = plt.subplots(2,1,subplot_kw={'projection':proj},figsize=(4,5))

tests = [test,test1]
for a,t in enumerate(tests):
    
    ax = axs.flatten()[a]
    
    ax     = viz.add_coast_grid(ax)
    clusterPIC = tests[a]
    pcm=ax.pcolormesh(lon5,lat5,t,cmap=cmapn,transform=ccrs.PlateCarree())
    #ax = viz.plot_box(bbtest,ax=ax)
    #ax = viz.plot_box(bbtest2,ax=ax,color='r')
    
    fig.colorbar(pcm,ax=ax,fraction=0.025)
    ax.set_title("CESM-PiC Clusters (TEST)")
    
plt.suptitle("Multiplying Pattern by 2 \n Pattern Correlation = %.3f" %R)
plt.tight_layout()
plt.savefig("%sTEST_%s_Clusters_Mult.png"%(outfigpath,expname),dpi=200,bbox_inches='tight')



# Calculate Pattern Correlation
R = slutil.patterncorr(test,test1)
print(R)



"""
Some notes on pattern correlation

- Shifting a portion southwards does result in a poorer match, and reduces the pattern
correlation

- Multiplying the whole pattern by a constant does not change the pattern correlation

- Multiplying each cluster by a value 

"""


#%% Organize silhouette values

nint = len(s_all)

silmap_all = np.zeros((nint,nlat5,nlon5)) * np.nan
clusters_all = np.zeros((nint,nlat5,nlon5)) * np.nan
for idx in tqdm(range(nint)): # Portions copied from script below
    
    # Read out the values
    clusterin    = clusts[idx] 
    uncertin     = uncert[idx]
    s_in         = s_all[idx]
    s_byclust_in = s_by_clust[idx]
    countin      = count[idx]
    rngin        = rngs[idx]
    rempts_in    = rempts[idx]
    
    # Recover clusterout for silhouette plotting
    remmask = rempts_in.copy()
    remmask[~np.isnan(remmask)] = np.nan # Convert all removed points to NaN
    remmask[np.isnan(rempts_in)] = 1
    clusterout,knan,okpts = proc.find_nan((clusterin*remmask).flatten(),0)
        
    # Ugly Fix, but not sure why sometimes s_in size doesnt match clusterin (non-nan)
    if len(clusterout) != len(s_in):
        print("Mismatch between clusterout (%i) and s_in (%i) for interval %i" % (len(clusterout),
                                                                                  len(s_in),idx))
        clusterout,knan,okpts = proc.find_nan((clusterin).flatten(),0)
    
    # Make Silhouette Map
    silmap = np.zeros(nlat5*nlon5)*np.nan
    silmap[okpts] = s_in
    silmap = silmap.reshape(nlat5,nlon5)
    silmap_all[idx,:,:] = silmap

    # Reassign clusters
    
#%% Plot map of (mean) silhouette values

vlims = [-.15,.15]

cmap="RdBu_r"
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})

silmap = np.nanmean(silmap_all,0)

ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,silmap,vmin=vlims[0],vmax=vlims[-1],cmap=cmap,transform=ccrs.PlateCarree())
ax.contour(lon5,lat5,silmap,levels=[0,np.nanmean(silmap)],colors='k',linewidths=0.75,linestyles=":",transform=ccrs.PlateCarree())

#ax.pcolormesh(lon5,lat5,silmap,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("Mean Silhouette Values for CESM-PiC Clusters (Year 400-2200) \n $s_{avg}$ = %.3f (Contoured)" % (np.nanmean(silmap)))
plt.savefig("%sCESM1PIC_MeanSilhouetteMap_%s.png"%(outfigpath,expname),dpi=200,bbox_inches='tight')

#%% Plot count of negative silhouette values

negcount = np.sum(silmap_all<0,axis=0)
silmap = negcount

cmap="Blues"
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})

ax     = viz.add_coast_grid(ax)
pcm=ax.pcolormesh(lon5,lat5,silmap,cmap=cmap,transform=ccrs.PlateCarree())
ax.contour(lon5,lat5,silmap,levels=[0,np.nanmean(silmap)],colors='k',linewidths=0.75,linestyles=":",transform=ccrs.PlateCarree())

#ax.pcolormesh(lon5,lat5,silmap,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax,fraction=0.025)
ax.set_title("Negative Silhouette Value Count for CESM-PiC Clusters (Year 400-2200) \n Mean Count = %.3f (Contoured)" % (np.nanmean(silmap)))
plt.savefig("%sCESM1PIC_CountNegSilhouetteMap_%s.png"%(outfigpath,expname),dpi=200,bbox_inches='tight')




#%% Try Visualizing something

def findk(arr,k,max=True):
    # From here: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    if max:
        idx = (-arr).argsort()[:k]
    else:
        idx = (arr).argsort()[:k]
    return idx
    
    

# Calculate average score
sscore = np.zeros(len(s_all))*np.nan
for i in range(len(s_all)):
    sscore[i] = s_all[i].mean()

k = 5
kmax = findk(sscore,k)
kmin = findk(sscore,k,max=False)

xtk = np.arange(0,22000,2400)
xtkl = (xtk/12).astype(int)+400


# Plot Timeseries of Silhouette Score
fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(sscore,color='gray',zorder=-1)
ax.scatter(kmax,sscore[kmax],marker="o",color="r",label="Max 5")
ax.scatter(kmin,sscore[kmin],marker="d",color="b",label="Min 5")
ax.set_xticks(xtk)
ax.set_xticklabels(xtkl)
ax.set_xlabel("Starting Year")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score for Clustering Results (Moving 20-yr windows)")
ax.grid(True,ls='dotted')
ax.legend()
plt.savefig("%sMinMax_sscore_k%i_%s_Clusters_Mult.png"%(outfigpath,k,expname),dpi=200,bbox_inches='tight')


# Now plot the clustering map and uncertaint of that point




for kmode in ['max','min']:
    for ii in range(k):
        print(ii)
        if kmode == "max":
            idx = kmax[ii]
        elif kmode == "min":
            idx = kmin[ii]
        else:
            print("kmode is invalid")
            break
        
        # Retrieve cluster information
        clusterin    = clusts[idx] 
        uncertin     = uncert[idx]
        s_in         = s_all[idx]
        s_byclust_in = s_by_clust[idx]
        countin      = count[idx]
        rngin        = rngs[idx]
        rempts_in    = rempts[idx]
        
        # Set some texts
        yearstr = "%i-%i" % (rngin[0].year,rngin[-1].year)
        
        # Adjust Cluster numbering
        clusternew,remapdict = slutil.remapcluster(clusterin,lat5,lon5,regiondict,returnremap=True)
        new_sbyclust = np.zeros(nclusters)
        for ks in remapdict.keys():
            newclass = remapdict[ks] # Class that k was remapped to
            new_sbyclust[newclass-1] = s_byclust_in[ks-1] # Reassign
            print("Reassigned new class %i"%newclass)
        
        # Recover clusterout for silhouette plotting
        remmask = rempts_in.copy()
        remmask[~np.isnan(remmask)] = np.nan # Convert all removed points to NaN
        remmask[np.isnan(rempts_in)] = 1
        clusterout,knan,okpts = proc.find_nan((clusterin*remmask).flatten(),0)
        
        # Ugly Fix, but not sure why sometimes s_in size doesnt match clusterin (non-nan)
        if len(clusterout) != len(s_in):
            clusterout,knan,okpts = proc.find_nan((clusterin).flatten(),0)
        
        # Plot 1: the silhouette
        fig,ax = plt.subplots(1,1)
        ax = slutil.plot_silhouette(clusterout,nclusters,s_in,ax1=ax,cmap=regioncolors)
        ax.grid(True,ls='dotted')
        ax.set_title("Silhouette Plot for CESM-PiC Clusters (Year %s) \n Mean Silhouette Coefficient = %.3f" % (yearstr,s_in.mean()))
        # Add dummy legend
        for i in range(nclusters):
            cid = i+1
            ax.axvline([-100],lw=5,color=regioncolors[i],label="Cluster %i, s = %.3f"%(cid,new_sbyclust[i]))
        ax.legend(fontsize=10)
        ax.set_xticks(np.arange(-.2,.6,.1))
        ax.set_xlim([-.25,.6])
        plt.savefig("%sCESM1PIC_%s_SilhouettePlot_k%s%i_%s.png"%(outfigpath,expname,kmode,ii,yearstr),dpi=200,bbox_inches='tight')
        
        # Plot 2: the silhoutte value map
        cmap="RdBu_r"
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
        plt.savefig("%sCESM1PIC_%s_SilhouetteMap_k%s%i_%s.png"%(outfigpath,expname,kmode,ii,yearstr),dpi=200,bbox_inches='tight')
        
        # Plot 3: the clustering result
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
        ax     = viz.add_coast_grid(ax)
        pcm=ax.pcolormesh(lon5,lat5,clusternew,cmap=cmapn,transform=ccrs.PlateCarree())
        fig.colorbar(pcm,ax=ax,fraction=0.025)
        ax.set_title("CESM-PiC Clusters (Year %s)" % (yearstr))
        plt.savefig("%sCESM1PIC_%s_ClustersMap_k%s%i_%s.png"%(outfigpath,expname,kmode,ii,yearstr),dpi=200,bbox_inches='tight')
        
        # Plot 4:  the Uncertainties
        vlm = [-10,10]
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
        ax     = viz.add_coast_grid(ax)
        pcm=ax.pcolormesh(lon5,lat5,uncertin,vmin=vlm[0],vmax=vlm[-1],cmap=cmap,transform=ccrs.PlateCarree())
        fig.colorbar(pcm,ax=ax,fraction=0.025)
        ax.set_title(r"CESM-PIC Cluster Uncertainty $(<\sigma^{2}_{in,x}>/<\sigma^{2}_{out,x}>)$"+" \n (Year %s) " % (yearstr))
        plt.savefig("%sCESM1PIC_%s_ClustersUncert_k%s%i_%s.png"%(outfigpath,expname,kmode,ii,yearstr),dpi=200,bbox_inches='tight')
        
        # End i loop

#%% Examine sea surface height values during that period

#%% Load in NAISST to see what is happening

# Left off at trying to plot AMV Index, but am having trouble matching the actual index...


ld3    = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM-PIC_NASSTI.npz")

nassti = ld3['nassti_full']

# Calculate AMV Index
nassti_ann = proc.ann_avg(nassti,0)


# Apply Low Pass Filter
cutofftime = 10
order      = 5
aa_sst = nassti_ann

# Design Butterworth Lowpass Filter
filtfreq = len(aa_sst)/cutofftime
nyquist  = len(aa_sst)/2
cutoff = filtfreq/nyquist
b,a    = butter(order,cutoff,btype="lowpass")
amv = filtfilt(b,a,aa_sst)


# Plot AMVV Index

amvidraw = nassti_ann
maskneg  = amvidraw<0
maskpos  = amvidraw>=0

fig,axs = plt.subplots(2,1)

times = np.arange(0,len(amvidraw+3))
timeplot = times
#timeplot = times[2:-1]

ax = axs[0]
ax.bar(timeplot[maskneg],amvidraw[maskneg],label="AMV-",color='b',width=1,alpha=0.5)
ax.bar(timeplot[maskpos],amvidraw[maskpos],label="AMV+",color='r',width=1,alpha=0.5)
ax.plot(timeplot,amv,label="10-yr Low-Pass Filter",color='k',lw=1.2)


ax = axs[1]
ax.plot()






