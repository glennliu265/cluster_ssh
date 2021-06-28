#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare output of cluster_AVISO and cluster_CESM

Created on Mon Mar 15 01:20:01 2021
@author: gliu
"""
from sklearn.metrics.pairwise import haversine_distances

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import pygmt
from tqdm import tqdm

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


#%%

# Set Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
# Set File Names
fn1 = "AVISO_1993-01_to_2013-01_remGMSL1_6clusters_minpts30_maxiters5_Results.npz"
fn2 = "AVISO_1993-01_to_2013-01_remGMSL0_6clusters_minpts30_maxiters5_Results.npz"
fn3 = "CESM_ens1_1993-01_to_2013-01_remGMSL1_6clusters_minpts30_maxiters5_Results.npz"
fn4 = "CESM_ens1_1993-01_to_2013-01_remGMSL0_6clusters_minpts30_maxiters5_Results.npz"
fns = [fn1,fn2,fn3,fn4]
expnames = ['AVISO','AVISO+GMSL','CESM-Ens1','CESM-Ens1+GMSL']


takemedian=True
plotdir = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210315/"

#%% Functions

def load_exp(filepath):
    ld = np.load(filepath,allow_pickle=True)
    lon5     = ld['lon']
    lat5     = ld['lat']
    sla      = ld['sla']
    clusters = ld['clusters']
    uncert   = ld['uncert']
    count    = ld['count']
    rempts   = ld['rempts']
    return [lon5,lat5,sla,clusters,uncert,count,rempts]


def retrieve_point(invar,kpt,nlat5,nlon5,okpts):
    """
    Retrieve values for a point  [kpt] given a distance or correlation matrix [var],
    and the indices of nonNaN values [okpts], and the lat/lon sizes [nlat5,nlon5]
    
    """
    # Get Value
    vrow = invar[kpt,:]
    
    # Place into variable
    mappt = np.zeros(nlat5*nlon5)*np.nan
    mappt[okpts] = vrow
    mappt = mappt.reshape(nlat5,nlon5)
    return mappt


#%% Load the data in

sshs     = []
clusters = []
uncerts  = []
#uncertsmedian=[]
sshok    = []
covs     = []
corrs    = []
okids     = [] # Indices of non-nan points

outids  =[]
inids   =[]
pcovins =[]
pcovouts=[]
for f in fns:
    
    
    lon5,lat5,sla,cluster,uncert,count,rempts = load_exp(datpath+f)
    sshs.append(sla)
    clusters.append(cluster)
    #uncerts.append(uncert)
    
    # Save SSH
    ntime,nlat,nlon=sla.shape
    sla = sla.reshape(ntime,nlat*nlon)
    okdata,knan,okpts=proc.find_nan(sla,0)
    sshok.append(okdata) # Save SSH
    okids.append(okpts)
    
    # Save Correlation and covariance
    npts = okdata.shape[1]
    scov = np.cov(okdata.T,okdata.T)
    srho = np.corrcoef(okdata.T,okdata.T)
    scov = scov[:npts,:npts]
    srho = srho[:npts,:npts]
    covs.append(scov)
    corrs.append(srho)
    
    # (Re)Calculate the Covariance for each point
    clusterout = cluster[-1].reshape(nlat*nlon)
    clusterout = clusterout[okpts]
    #uncertpt   = np.zeros((npts,clusterout.shape[0])) # [point,]
    pcovin      = [] # [pt][points inside region]
    outid       = []
    pcovout     = [] # [pt][points outside region]
    inid        = []
    uncertout = np.zeros(clusterout.shape)
    #uncertoutmed = np.zeros(clusterout.shape)
    for i in range(len(clusterout)):
        covpt     = scov[i,:]     # 
        cid       = clusterout[i] #
        covin     = covpt[np.where(clusterout==cid)]
        covout    = covpt[np.where(clusterout!=cid)]
        
        #uncertout[i] = np.median(covin)/np.median(covout)
        if takemedian:
            uncertout[i] = np.median(covin)/np.median(covout)
        else:
            uncertout[i] = np.mean(covin)/np.mean(covout)
            
        
        
        pcovin.append(covin)
        inid.append(np.where(clusterout==cid))
        pcovout.append(covout)
        outid.append(np.where(clusterout!=cid))
    pcovins.append(pcovin)
    pcovouts.append(pcovout)
    inids.append(inid)
    outids.append(outid)
    
    uncert = np.zeros(nlat*nlon)*np.nan
    uncert[okpts] = uncertout
    uncert = uncert.reshape(nlat,nlon)
    uncerts.append(uncert)
    
    print("Loaded "+f)

#%% Visualize covariance

gmsl   = uncerts[1]
nogmsl = uncerts[0]

if takemedian:
    uncertform = r"$Uncertainty $(median(\sigma^{2}_{in,x})/median(\sigma^{2}_{out,x}))$"
else:
    uncertform = r"$Uncertainty $(<\sigma^{2}_{in,x}>/<\sigma^{2}_{out,x}>)$"

# # Plot Historgram, GMSL retained
# plt.figure()
# plt.hist(gmsl)
# plt.xlabel(uncertform)
# plt.ylabel("Count")
# plt.title("Uncertainties for AVISO (GMSL retained)")
# plt.savefig(plotdir+"AVISO_remGMSL0.png",dpi=150)

# # Plot Historgram, GMSL removed
# plt.figure()
# plt.hist(nogmsl)
# plt.xlabel(r"Uncertainty $%s"%uncertform)
# plt.ylabel("Count")
# plt.title("Uncertainties for AVISO (no GMSL)")
# plt.savefig(plotdir+"AVISO_remGMSL1.png",dpi=150)


vm=[-2,2]
# Visualize uncertainty maps
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
ax     = slutil.add_coast_grid(ax)
pcm    = plt.pcolormesh(lon5,lat5,gmsl,vmin=vm[0],vmax=vm[-1],cmap='copper',transform=ccrs.PlateCarree())
if takemedian:
    ax.set_title(r"Uncertainty (AVISO with GMSL) $(median(\sigma^{2}_{in,x})/median(\sigma^{2}_{out,x}))$")
else:
    ax.set_title(r"Uncertainty (AVISO with GMSL) $(<\sigma^{2}_{in,x}>/<\sigma^{2}_{out,x}>)$")
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig(plotdir+"Uncertainty_AVISO_remGMSL0.png",dpi=200)

# Visualize uncertainty maps
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
ax     = slutil.add_coast_grid(ax)
pcm    = plt.pcolormesh(lon5,lat5,nogmsl,vmin=vm[0],vmax=vm[-1],cmap='copper',transform=ccrs.PlateCarree())
ax.set_title(r"Uncertainty (AVISO without GMSL) $(<\sigma^{2}_{in,x}>/<\sigma^{2}_{out,x}>)$")
if takemedian:
    ax.set_title(r"Uncertainty (AVISO without GMSL) $(median(\sigma^{2}_{in,x})/median(\sigma^{2}_{out,x}))$")
else:
    ax.set_title(r"Uncertainty (AVISO without GMSL) $(<\sigma^{2}_{in,x}>/<\sigma^{2}_{out,x}>)$")
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig(plotdir+"Uncertainty_AVISO_remGMSL1.png",dpi=200)

#%% Visualize the covariance for each point


# Indicate Point to find
lonf = 40#-70+360
latf = -35
kpt = np.where((coords[:,0] == lonf) * (coords[:,1] == latf))[0][0]
loctitle = "Lon %.1f Lat %.1f" % (lonf,latf)
locfn = "Lon%i_Lat%i" % (lonf,latf)
print("Found %.2f Lon %.2f Lat" % (coords[kpt,0],coords[kpt,1]))

# Plot Distance
distpt = retrieve_point(sdist,kpt,nlat5,nlon5,okpts)
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = add_coast_grid(ax)
ax.set_title("Distance (km) from %s" % (loctitle))
pcm = ax.pcolormesh(lon5,lat5,distpt,cmap='Blues')
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sDistance_%s.png"%(outfigpath,locfn),dpi=200,transparent=True)

# Plot Exponential Term
distpt = retrieve_point(expterm,kpt,nlat5,nlon5,okpts)
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = add_coast_grid(ax)
ax.set_title("${exp(-(Distance) / (2a^{2})})$ (km) from %s" % (loctitle))
pcm = ax.pcolormesh(lon5,lat5,distpt,cmap='Greens')
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sExpTerm_%s.png"%(outfigpath,locfn),dpi=200,transparent=True)

# Plot Correlation
distpt = retrieve_point(srho,kpt,nlat5,nlon5,okpts)
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = add_coast_grid(ax)
ax.set_title("Correlation with %s" % (loctitle))
pcm = ax.pcolormesh(lon5,lat5,distpt,cmap=cmocean.cm.balance,vmin=-1,vmax=1)
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sCorrelation_%s.png"%(outfigpath,locfn),dpi=200,transparent=True)

# Plot Final Distance Matrix
distpt = retrieve_point(distance_matrix,kpt,nlat5,nlon5,okpts)
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = add_coast_grid(ax)
ax.set_title("Distance Matrix with %s" % (loctitle))
pcm = ax.pcolormesh(lon5,lat5,distpt,cmap=cmocean.cm.dense)
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sDistanceMatrixFinal_%s.png"%(outfigpath,locfn),dpi=200,transparent=True)


#%%
i=1
print("Visualizing for experiment %s"%expnames[i])
scov = covs[i]


okpts = okids[i]
uncert = uncerts[i]
nlon5,nlat5 = [len(lon5),len(lat5)]

# Find a Point
#lonf = 220
lonf = 240
latf = -10
klon,klat = proc.find_latlon(lonf,latf,lon5,lat5)

# Make Coords
lonmesh,latmesh = np.meshgrid(lon5,lat5)
coords  = np.vstack([lonmesh.flatten(),latmesh.flatten()]).T
coords = coords[okpts,:]
kpt = np.where((coords[:,0] == lonf) * (coords[:,1] == latf))[0][0]
loctitle = "Lon: %.1f Lat: %.1f" % (lonf,latf)
locfn = "Lon%i_Lat%i" % (lonf,latf)
print("Found %.2f Lon %.2f Lat" % (coords[kpt,0],coords[kpt,1]))


# Get covariances in and out
scovin = pcovins[i][kpt]
scovout = pcovouts[i][kpt]

# First Check the point among the clusters
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
ax.set_title("Uncertainty Values for %s (%s)" % (loctitle,expnames[i]))
#uncert[uncert<0]=0
#uncert[uncert>2]=2
#pcm = ax.pcolormesh(lon5,lat5,uncert,vmin=-5,vmax=5,cmap='copper')
pcm = ax.pcolormesh(lon5,lat5,uncert)
ax.scatter([lonf],[latf],s=200,c='r',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sUncertFinal_%s_%s.png"%(plotdir,expnames[i],locfn),dpi=200,transparent=True)



# Plot Covariance matrix
covpt = retrieve_point(scov,kpt,nlat5,nlon5,okpts)
fig,ax = plt.subplots(1,1,figsize=(7,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
ax.set_title("Covariance Matrix for %s (%s)" % (loctitle,expnames[i]))
pcm = ax.pcolormesh(lon5,lat5,covpt,cmap=cmocean.cm.balance)
ax.scatter([lonf],[latf],s=200,c='k',marker='x')
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig("%sCovarianceeMatrixFinal_%s_%s.png"%(plotdir,expnames[i],locfn),dpi=200,transparent=True)

# Plot histogram of covariance values
bins = np.linspace(-0.003,0.003,20)
fig,ax = plt.subplots(1,1)
smeanin  = np.nanmean(scovin.flatten())
smeanout = np.nanmean(scovout.flatten())

#stackh = np.concatenate([scovout[:,None],scovin[:,None]],axis=0)
ax.hist(covpt.flatten(),bins,color='gray',edgecolor='w',lw=.75,alpha=1)
ax.hist(scovout.flatten(),bins,color='blue',edgecolor='w',lw=.75,alpha=0.8)
ax.hist(scovin.flatten(),bins,color='red',edgecolor='r',lw=.75,alpha=0.7)
ax.axvline([smeanout],color='cornflowerblue',ls='dashed',label=r"$<\sigma^{2}_{out,x}>$"+"= %.2e"%(smeanout))
ax.axvline([smeanin],color='r',ls='dashed',label=r"$<\sigma^{2}_{in,x}>$"+"= %.2e"%(smeanin))
ax.legend(fontsize=10)
ax.grid(True,ls='dotted')
ax.set_xlabel("SSH Covariance (cm^{2})")
ax.set_ylabel("Count")
ax.set_title("Histogram of Covariance Matrix for %s (%s)"% (loctitle,expnames[i]))
plt.tight_layout()
plt.savefig("%sCovarianceeMatrixHistogram_%s_%s.png"%(plotdir,expnames[i],locfn),dpi=200,transparent=True)

