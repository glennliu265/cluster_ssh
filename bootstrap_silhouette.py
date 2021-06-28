#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Bootstrap silhouette values using timeseries
generated from cluster_synthetic

Created on Tue Jun  8 14:30:24 2021

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


#%% User Edits

# Set Paths
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210603/"

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
recalc      = False

# Synthetic Cluster Data ****
simlen = 5000
niter  = 1000

datname = "AVISO_%s_to_%s_remGMSL%i" % (start,end,rem_gmsl)
expname = "Synthetic_%iclusters_minpts%i_maxiters%i" % (nclusters,minpts,maxiter)
print(datname)
print(expname)

#Make Folders
#expdir = outfigpath+expname +"/"
expdir = outfigpath+"../20210610/"
checkdir = os.path.isdir(expdir)
if not checkdir:
    print(expdir + " Not Found!")
    os.makedirs(expdir)
else:
    print(expdir+" was found!")


# Load Regiondict and colormap
cmapn,regiondict = slutil.get_regions()

    
#%%
if recalc: # Recalculate the data if option is set
    #% Load Synthetic Cluster Data
    
    fn = "%sNoiseMaps_AVISO_niter%i.npz"%(datpath,niter)
    
    
    ld = np.load(fn,allow_pickle=True)
    wnout = ld['wnout']
    rnout = ld['rnout']
    lat   = ld['lat']
    lon   = ld['lon']
    arm1  = ld['ar1m']
    neffm = ld['neffm']
    
    #% Cluster and 
    #Calculate Silhouette values for each result (adapted from cluster_CESM-PIC_chunk.py)

    
    maxiter = 6
    
    # Get ranges for sliding window
    winsize = 240
    #rngs = []
    
    for mode in range(2): # Loop for red noise, white noise
        
        # Preallocate
        clusters   = []
        uncerts    = []
        counts     = []
        rempts     = []
        Wks        = []
        s_all      = []
        s_by_clust = []
    
        
        for i in tqdm(range(niter)):
            
            if mode == 0:
                sla_in = wnout[i,...]
                moden  = 'WhiteNoise'
            elif mode == 1:
                sla_in = rnout[i,...]
                moden  = 'RedNoise'
            
            # Do clustering
            allclusters,alluncert,allcount,rempt,allWk,alls,alls_byclust = slutil.elim_points(sla_in,lat,lon,nclusters,minpts,maxiter,expdir,
                                                                     viz=False,printmsg=False,calcsil=True)
            
            
            clusters.append(np.array(allclusters[-1])) # Save, and just take the final result # [niter x nlat x nlon]
            uncerts.append(np.array(alluncert[-1]))    # [niter x nlat x nlon]
            counts.append(np.array(allcount[-1]))      # [niter x nclusters]
            rempts.append(np.array(rempt))             # [lat x lon]
            Wks.append(np.array(allWk[-1]))            # [niter x nclusters]
            s_all.append(np.array(alls[-1]))           # [niter x nanpts]
            s_by_clust.append(alls_byclust[-1])        # [niter x ncluster]
            # End loop
        
        
        # Make everything to arrays
        clusters = np.array(clusters)  # [time,]
        uncerts  = np.array(uncerts)   
        counts   = np.array(counts)
        rempts  = np.array(rempts)
        Wks = np.array(Wks)
        s_all = np.array(s_all)
        s_by_clust = np.array(s_by_clust)
    
        
        np.savez("%s%s_Results_%s_niter%i.npz"%(datpath,expname,moden,niter),**{
            'lon':lon,
            'lat':lat,
            'clusters':clusters,
            'uncert':uncerts,
            'count':counts,
            'rempts':rempts,
            'Wks':Wks,
            's_all':s_all,
            's_by_clust':s_by_clust},allow_pickle=True)
else:
    print("Loading Existing Data")

#%% Reload the data
def load_data(fn):
    ld = np.load(fn,allow_pickle=True)
    
    lat = ld['lat']
    lon = ld['lon']
    
    c = ld['clusters']
    u = ld['uncert']
    cnt = ld['count']
    rp  = ld['rempts']
    w  = ld['Wks']
    
    sa = ld['s_all']
    sbc = ld['s_by_clust']
    
    return lat,lon,c,u,cnt,rp,w,sa,sbc
    
    
mn = ('WhiteNoise','RedNoise')
mnl = ('White Noise','Red Noise')
clusters   = []
uncerts    = []
counts     = []
rempts     = []
Wks        = []
s_all      = []
s_by_clust = []


for i in range(2):
    
    moden = mn[i]
    fn = "%s%s_Results_%s_niter%i.npz"%(datpath,expname,moden,niter)

    # Load Data
    lat,lon,c,u,cnt,rp,w,sa,sbc = load_data(fn)
    clusters.append(c)
    uncerts.append(u)
    counts.append(cnt)
    rempts.append(rp)
    Wks.append(w)
    s_all.append(sa)
    s_by_clust.append(sbc)
    
    # Calculate average silhouette valules
    
    
#%% Calculate Average Silhouette value for each map
# Do same for uncertainty


# Calculate Average Silhouette Values for each Map
sscore = np.zeros((2,niter))*np.nan
uscore = sscore.copy()

for i in range(2):
    sin = s_all[i]
    uin = uncerts[i]
    
    sv = np.zeros(niter)
    uv = sv.copy()
    for j in range(niter): # For each iteration...
        sv[j] = np.nanmean(sin[j])
        uv[j] = np.nanmean(uin[j,...])
    sscore[i,:] = sv
    uscore[i,:] = uv
    
#%% Plot Results

def interp_2pt(sortvar,findk,viz=False):
    """
    Quick 2-point linear interpolation of values in sortvar.
    Find index [np.floor(sortvar),np.floor(sortvar)+1]
    and interpolate

    Visualize result if viz=True    
    """
    # Get Bounds [round k down, k+1]
    llo = int(np.floor(findk))
    lhi = llo+1
    interpval = np.interp(findk,[llo,lhi],[sortvar[llo],sortvar[lhi]])
    
    # Visualize if set
    if viz:
        fig,ax =plt.subplots(1,1)
        ax.plot([llo,lhi],[sortvar[llo],sortvar[lhi]],marker="x")
        ax.scatter(findk,interpval,marker="o")
    return interpval
    


def calc_conf(invar,tails,conf,median=False):
    
    N = len(invar)
    
    sortvar = invar.copy()
    sortvar.sort()
    
    perc = (1 - conf)/tails
    
    
    # Lower Bounds
    lowid = perc*100
    lowbnd = interp_2pt(sortvar,lowid,viz=False)
    
    # Upper bounds
    hiid  = N - perc*100
    hibnd = interp_2pt(sortvar,hiid,viz=False)
    
    # Mean
    if median:
        mu  = np.median(sortvar)
    mu = np.mean(sortvar)
    
    return lowbnd,hibnd,mu

def plothist(invar,tails,conf,nbins,col,ax=None,alpha=0.5,fill=True,lw=1,median=False):
    if ax is None:
        ax = plt.gca()
    
    # Calculate Bounds
    lb,hb,mu = calc_conf(invar,tails,conf,median=median)
    ax.hist(invar,nbins,alpha=alpha,color=col,edgecolor=col,fill=fill,linewidth=lw)
    ax.axvline(lb,ls='dashed',color='k',label="Lower Bound = %.3e" % (lb))
    ax.axvline(mu,ls='solid',color='k',label="Mean = %.3e" % (mu),lw=2)
    ax.axvline(hb,ls='dashed',color='k',label="Upper Bound = %.3e" % (hb))
    
    return ax
    
    

conf  = 0.95
tails = 2
perc  = (1-conf)/2*100
plotsep = True

#invar = sscore[1]
#lb,hb,mu = calc_conf(invar,tails,conf)
#col = 'r'
nbins = 50


if plotsep:
    fig,ax = plt.subplots(1,1,sharey=True,sharex=False,figsize=(6,3))
else:
    fig,axs = plt.subplots(2,1,sharey=True,sharex=True,figsize=(6,7))
    ax = axs[0]
    
ax = plothist(sscore[0],tails,conf,nbins,'b',ax=ax)
ax.grid(True,ls='dotted',zorder=-1)
ax.legend(fontsize=10)
ax.set_title("Silhouette Score Histogram (White Noise)")
#ax.set_title("White Noise Maps ($1\sigma=$%.2e)" % np.std(sscore[0]))



if plotsep:
    xtk = ax.get_xticks()
    xlm = ax.get_xlim()
    ax.set_xlim(xlm)
    ax.set_xticks(xtk)
    plt.savefig("%sSilhouetteScore_Histogram_Synthetic_WhiteNoise.png"%(outfigpath),
            dpi=150,bbox_tight='inches')
    fig,ax = plt.subplots(1,1,sharey=True,sharex=False,figsize=(6,3))
    
else:
    ax = axs[1]
    
ax = plothist(sscore[1],tails,conf,nbins,'r',ax=ax)
ax.grid(True,ls='dotted',zorder=-1)
ax.legend(fontsize=10)
ax.set_title("Silhouette Score Histogram (Red Noise)")
#ax.set_title("Red Noise Maps ($1\sigma=$%.2e)" % np.std(sscore[1]))

if plotsep:
     ax.set_xlim(xlm)
     ax.set_xticks(xtk)
     
     plt.savefig("%sSilhouetteScore_Histogram_Synthetic_RedNoise.png"%(outfigpath),
            dpi=150,bbox_tight='inches')
else:
    plt.suptitle("Silhouette Score for Clustering Results")
    plt.savefig("%sSilhouetteScore_Histogram_Synthetic.png"%(outfigpath),
                dpi=150,bbox_tight='inches')



#%% plot same for uncertainties

fig,axs = plt.subplots(2,1,sharey=True,sharex=True)

ax = axs[0]
ax = plothist(uscore[0],tails,conf,nbins,'b',ax=ax,median=True)
ax.grid(True,ls='dotted',zorder=-1)
ax.legend(fontsize=10)
ax.set_title("White Noise Maps")

ax = axs[1]
ax = plothist(uscore[1],tails,conf,nbins,'r',ax=ax,median=True)
ax.grid(True,ls='dotted',zorder=-1)
ax.legend(fontsize=10)
ax.set_title("Red Noise Maps")

plt.suptitle("Uncertainty Metric for Clustering Results")

#%% Plot red and white noise with the maximum and minimum silhouette score



proj = ccrs.PlateCarree(central_longitude=180)
stat = sscore
labs = True


fig,axs = plt.subplots(2,2,subplot_kw={'projection':proj},figsize=(10,7))

for m in range(2):
    
    name = mnl[m]
    
    instat  = sscore[m]
    inclust = clusters[m]
    kmin   = np.argmin(instat)
    kmax   = np.argmax(instat)
    
    ax  = axs[m,0]
    ax  = slutil.add_coast_grid(ax,proj=proj,leftlab=labs,botlab=labs)
    pcm = ax.pcolormesh(lon,lat,inclust[kmin,:,:],cmap=cmapn,transform=ccrs.PlateCarree())
    ax.set_title("%s \n Min Silhouette Score (%.2e)"%(name,instat[kmin]))
    
    ax  = axs[m,1]
    ax  = slutil.add_coast_grid(ax,proj=proj,leftlab=labs,botlab=labs)
    pcm = ax.pcolormesh(lon,lat,inclust[kmax,:,:],cmap=cmapn,transform=ccrs.PlateCarree())
    ax.set_title("%s \n Max Silhouette Score (%.2e)"%(name,instat[kmax]))
fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='horizontal',shrink=0.55,pad=0.06)
plt.suptitle("Null Hypothesis Clustering Results",fontsize=14,y=.95)
plt.savefig("%sNullHypothesisClusteringResults_MaxMin.png"%(outfigpath),dpi=200,bbox_tight='inches')



#%% Examine a clustering case where it's only the distance Metric

# def load_msk_5deg():
#     start = '1993-01'
#     end   = '2013-01'
#     datpath = '/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/'
#     ld = np.load("%sSSHA_AVISO_%sto%s.npz" % (datpath,start,end),allow_pickle=True)
#     sla_5deg = ld['sla_5deg']
#     msk = sla_5deg.sum(0)
#     msk[~np.isnan(msk)] = 1
#     plt.pcolormesh(msk)
#     return msk
msk = slutil.load_msk_5deg()


_,nlat,nlon = clusters[0].shape


varin      = np.zeros((240,nlat,nlon)) * msk[None,:,:]
chardist   = 0
snamelong  = "Distance Only"
expname    = "AVISO_DistanceOnly_nclusters%i_Filtered_" % (nclusters)
distmode   = 1
absmode    = 0
uncertmode = 0

# ------------------
# Calculate Matrices
# ------------------
ntime,nlat,nlon = varin.shape
srho,scov,sdist,okdata,okpts,coords2=slutil.calc_matrices(varin,lon,lat,return_all=True)
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
for i in range(len(clusterout)):
    covpt     = scov[i,:]     #
    cid       = clusterout[i] #
    covin     = covpt[np.where(clusterout==cid)]
    covout    = covpt[np.where(clusterout!=cid)]
    
    if uncertmode == 0:
        uncertout[i] = np.mean(covin)/np.mean(covout)
    elif uncertmode == 1:
        uncertout[i] = np.median(covin)/np.median(covout)
uncert = np.zeros(nlat*nlon)*np.nan
uncert[okpts] = uncertout
uncert = uncert.reshape(nlat,nlon)

title = snamelong
expname += "_distmode%i_uncertmode_%i_absmode%i" % (distmode,uncertmode,absmode)
outfigpath = expdir


#
# Remap the clusters
#
# Adjust Cluster numbering
#cmapn,regiondict=slutil.get_regions()
clusternew,remapdict = slutil.remapcluster(clustered,lat,lon,regiondict,returnremap=True)
new_sbyclust = np.zeros(nclusters)

for ks in remapdict.keys():
    newclass = remapdict[ks] # Class that k was remapped to
    new_sbyclust[newclass-1] = s_byclust[ks-1] # Reassign
    print("Reassigned new class %i"%newclass)

# ---------------------
# Make Distance Plot
# ---------------------
proj = ccrs.PlateCarree(central_longitude=180)
labs = True

fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj},figsize=(6,5))
ax  = slutil.add_coast_grid(ax,proj=proj,leftlab=labs,botlab=labs)
pcm = ax.pcolormesh(lon,lat,clustered[:,:],cmap=cmapn,transform=ccrs.PlateCarree())
ax.set_title("Distance Only Null Hypothesis \n Silhouette Score (%.2e)"%(s_score))
fig.colorbar(pcm,orientation='horizontal',fraction=0.05,pad=0.07)
plt.savefig("%sDistanceOnly_Clusters.png"%outfigpath,dpi=200,bbox_tight='inches')


#plot_results(clustered,uncert,expname,lat5,lon5,outfigpath,title=title)

# ----------------------------------------
# Make some quick silhouette related plots
# ----------------------------------------
#cmap = cm.get_cmap("jet",nclusters)
fig,ax = plt.subplots(1,1)
ax,ccol = slutil.plot_silhouette(clusterout,nclusters,s,ax1=ax,returncolor=True)
ax.grid(True,ls='dotted')
ax.set_title("Silhouette Plot %s \n Mean Silhouette Coefficient = %.3f" % (snamelong,s.mean()))
# Add dummy legend
for i in range(nclusters):
    cid = i+1
    ax.axvline([-100],lw=5,color=ccol[i],label="Cluster %i, s = %.3f"%(cid,s_byclust[i]))
ax.legend(fontsize=10)
ax.set_xticks(np.arange(-1,1.1,.1))
ax.set_xlim([-1,1])
plt.savefig("%sSynthetic_SilhouettePlot_%s.png"%(outfigpath,expname),dpi=200,bbox_inches='tight')




