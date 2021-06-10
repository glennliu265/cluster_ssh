#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze CESM-PIC-window clustering results, specifically for the indian ocean

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
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210610/"
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
regionids = ("5: Indian-South Pacific Ocean",
             "1: Northwest Pacific",
             "4: East Pacific",
             "2: ST N. Atl",
             "3: SP N. Atl",
             "6: South Atlantic"
             )

# Make Region Colors
regioncolors = np.array(
                [[81,135,195],
                [233,51,35],
                [251,237,79],
                [73,161,68],
                [154,219,232],
                [138,39,113],
                ])/255
cmapn = (mpl.colors.ListedColormap(regioncolors))

# Dictionary of Bounding Boxes to search thru
regiondict = {1:[50,105,-15,15],
             2:[150,180,5,50],
             3:[200-360,250-360,0,35],
             4:[280-360,350-360,20,45],
             5:[300-360,360-360,50,75],
             6:[280-330,360-360,-50,-20]
             }
#%% Functions



def plot_IOD(idx,tint,startyr,sigma=1,ylm=None):
    
    #Dummy Variable
    varr = np.zeros((1,1,idx.shape[0]))
    kp,kn,kz = proc.get_posneg_sigma(varr,idx,sigma=sigma,return_id=True)    
    
    if ylm is None:
        ylm = np.nanmax(np.abs(idx))
    
    xtk    = np.arange(0,len(idx),12*tint)
    xtkl   = np.arange(0,int(len(idx)/12),tint) + startyr
    fig,ax = plt.subplots(1,1,figsize=(8,3))
    
    t = np.arange(0,idx.shape[0])
    
    ax.bar(t[kp],idx[kp],color='red',label="(+) IOD, Count = %i Months" % (idx[kp].shape[0]))
    ax.bar(t[kn],idx[kn],color='blue',label="(-) IOD, Count = %i Months" % (idx[kn].shape[0]))
    ax.bar(t[kz],idx[kz],color='gray',label="Neutral IOD, Count =  %i Months"% (idx[kz].shape[0]))
    ax.axhline([idx.std()],color="k",ls='dashed',lw=0.75)
    ax.axhline([-idx.std()],color="k",ls='dashed',lw=0.75)
    ax.grid(True,ls='dotted')
    #ax.set_title(ptitle)
    ax.set_ylim([-ylm,ylm])
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Dipole Mode Index $(\degree C)$")
    ax.set_xticks(xtk)
    ax.set_xticklabels(xtkl)
    ax.legend(fontsize=8,ncol=3)
    
    
    return fig,ax




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

#%% Find some extreme values

# Calculate averaged silhouette value over a particular region
bboxsil        = regiondict[1]
sreg,slon,slat = proc.sel_region(silmap_all.transpose(2,1,0),lon5,lat5,bboxsil)
      
s_allin        = np.nanmean(sreg,(0,1))

# Flatte the region, and count the number of unique values within that region
creg,_,_  = proc.sel_region(clusts.transpose(2,1,0),lon5,lat5,bboxsil)
nlatr,nlonr,nrngs = creg.shape
cregflat = creg.reshape(nlatr*nlonr,nrngs)
okdata,knan,okpts = proc.find_nan(cregflat,1) # Pick only non-nan points
npts = okdata.shape[0]
rcount = np.zeros(nrngs)
for t in range(nrngs):
    indata = okdata[:,t]
    rcount[t] = len(np.unique(indata))

def findk(arr,k,max=True):
    # From here: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    if max:
        idx = (-arr).argsort()[:k]
    else:
        idx = (arr).argsort()[:k]
    return idx

# Calculate average score
sscore = np.zeros(len(s_allin))*np.nan
for i in range(len(s_allin)):
    sscore[i] = s_allin[i].mean()

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


#%% Load the Dipole Mode Index

# Load DMI (Note that year 1-2 and the last year are dropped)
dmi = np.load(datpath+"CESM-PIC_DMI.npy")

# Replace into full array to match up timing
dmi_full = np.zeros(sla_lp.shape[0])*np.nan
dmi_full[24:-12] = dmi


#%% Compute an integrated Dipole Mode Index

ntime   = dmi_full.shape[0]
winsize = 240

# Preallocate
idmi_p = np.zeros(ntime-winsize)
idmi_n = np.zeros(ntime-winsize)

# Copy DMI, grab positive and negative values
dmi_n = dmi_full.copy()
dmi_p = dmi_full.copy()
dmi_n[dmi_full>0] = 0
dmi_p[dmi_full<0] = 0

# Now integrat over each period
for i in tqdm(range(ntime-winsize)):
    krng = np.arange(i,i+winsize+1)
    idmi_p[i] = np.sum(dmi_p[krng])
    idmi_n[i] = np.sum(dmi_n[krng])


idmi_total = idmi_p + idmi_n

#%% Plot the integrated INdex


fig,axs=plt.subplots(3,1,figsize=(12,10))

ax = axs[0]
ax.plot(idmi_n,color='b')
ax.set_title("Integrated (-) Dipole Mode Index (rho= %.3f)"%(np.corrcoef(sscore[~np.isnan(idmi_n)],idmi_n[~np.isnan(idmi_n)])[0,1]))


ax = axs[1]
#ax.plot(sscore,color='gray')
ax.plot(idmi_p,color='r')
ax.set_title("Integrated (-) Dipole Mode Index (rho= %.3f)"%(np.corrcoef(sscore[~np.isnan(idmi_p)],idmi_p[~np.isnan(idmi_p)])[0,1]))

ax = axs[2]
ax.plot(sscore,color='gray')
ax.set_title("Silhouette Score")

#ax.plot(idmi_total,color='k')
#ax.set_title("Integrated Total Dipole Mode Index (rho= %.3f)"%(np.corrcoef(sscore[~np.isnan(idmi_total)],idmi_total[~np.isnan(idmi_total)])[0,1]))

plt.savefig("%sIDMI_timeseries.png"%(outfigpath),dpi=200,bbox_tight='inches')


#%% Plot some histograms
tails = 2
conf  = 0.95
nbins = 50
invar1 = idmi_n
invar2 = idmi_p
title1 = "Integrated (-) Dipole Mode Index"
title2 = "Integrated (+) Dipole Mode Index"
fig,axs = plt.subplots(3,1,sharey=True,sharex=False,figsize=(8,10))

ax = axs[0]
ax = slutil.plothist(invar1,tails,conf,nbins,'b',ax=ax)
ax.grid(True,ls='dotted',zorder=-1)
ax.legend(fontsize=10)
ax.set_title("%s ($1\sigma=$%.2e)" % (title1,np.nanstd(invar1)))

ax = axs[1]
ax = slutil.plothist(invar2,tails,conf,nbins,'r',ax=ax)
ax.grid(True,ls='dotted',zorder=-1)
ax.legend(fontsize=10)
ax.set_title("%s ($1\sigma=$%.2e)" % (title2,np.nanstd(invar2)))

invar2 = sscore
title2 = "Silhouette Score"

ax = axs[2]
ax = slutil.plothist(invar2,tails,conf,nbins,'gray',ax=ax)
ax.grid(True,ls='dotted',zorder=-1)
ax.legend(fontsize=10)
ax.set_title("%s ($1\sigma=$%.2e)" % (title2,np.nanstd(invar2)))


plt.suptitle("Silhouette Score for Clustering Results")
plt.savefig("%sSilhouetteScore_Histogram_Synthetic.png"%(outfigpath),
            dpi=150,bbox_tight='inches')

plt.savefig("%sIDMI_distr.png"%(outfigpath),dpi=200,bbox_tight='inches')


#%% For each period, investigate the occurence of IOD...

kmax = findk(idmi_p,k,max=False)
kmin = findk(idmi_n,k,max=True)

    
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
        
        # Retrieve DMI Information
        dmi_in = dmi_full[idx:idx+240]
        
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
        clusterout,knan,okpts = proc.find_nan((clusternew*remmask).flatten(),0)
        
        # Ugly Fix, but not sure why sometimes s_in size doesnt match clusterin (non-nan)
        if len(clusterout) != len(s_in):
            clusterout,knan,okpts = proc.find_nan((clusternew).flatten(),0)
        
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
        
        # Plot IOD Index During This Period
        tint = 2 # Time interval in years
        ylm  = 3 # Maximum y values
        ptitle = r"CESM-PIC Dipole Mode Index \n (Year %s) \n $1\sigma=%.3f \degree C$ " % (yearstr,dmi_in.std())
        fn     ="%sCESM1PIC_%s_DMI_k%s%i_%s.png"%(outfigpath,expname,kmode,ii,yearstr)
        fig,ax = plot_IOD(dmi_in,tint,rngin[0].year,ylm=3)
        ax.set_title(ptitle)
        plt.savefig(fn,dpi=200,bbox_tight="inches")
        
        
        

    #plt.savefig(fn,dpi=200,bbox_tight="inches")







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






# Now plot the clustering map and uncertaint of that point





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



#%% Given the counts for each region, get some statistics


invars2 = [sscore,idmi_p,idmi_n,idmi_total]
titlesl = ("Silhouette Score","(+) Integrated Dipole Mode Index","(-) Integrated Dipole Mode Index","Total Integrated Dipole Mode Index")
fnl = ["sscore","idmip","idmin","idmit"]

colss = ['r','b','y','g']
nc,ncounts=np.unique(rcount,return_counts=True)

for v in range(4):
    invar = invars2[v]
    # Make histograms for each case
    fig,axs=plt.subplots(len(nc),1,sharex=True,figsize=(6,8))
    for i in range(len(nc)):
        
        idc = int(nc[i])  # Number of clusters in region
        kin = rcount==idc # Indices of when this happens
        
        idvar = invar[kin]
        ax = axs[i]
        ax = slutil.plothist(idvar,tails,conf,nbins,colss[i],ax=ax)
        ax.grid(True,ls='dotted',zorder=-1)
        ax.legend(fontsize=8)
        ax.set_title("%i Clusters count=%i ($1\sigma=$%.2e)" % (idc,ncounts[i],np.nanstd(idvar)))
    plt.suptitle("%s Distributions" % (titlesl[v]),y=.95)
    plt.savefig("%s%s_Histograms_by_Clust.png"%(outfigpath,fnl[v]),dpi=200,bbox_tight='inches')

    
    
#%%

fig,ax = plt.subplots(1,1)

ax.scatter()
    
    



