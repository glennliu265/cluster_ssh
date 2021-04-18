#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster Synthetic

Testing clustering of synthetic timeseries

Created on Wed Mar 31 01:12:19 2021

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
from scipy import stats
# Custom Toolboxes
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/03_Scripts/cluster_ssh/")
from amv import proc,viz
import slutil
import yo_box as ybx
import tbx

#%% Load Aviso Data

# Set Paths
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210309/"

# Experiment Names
start       = '1993-01'
end         = '2013-01'
#start      = '1850-01'
#end        = '2100-12'
nclusters   = 6
rem_gmsl    = True
rem_seas    = True
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
expdir = outfigpath+"../20210414/"
checkdir = os.path.isdir(expdir)
if not checkdir:
    print(expdir + " Not Found!")
    os.makedirs(expdir)
else:
    print(expdir+" was found!")


#%% Functions

def cluster_ssh(sla,lat,lon,nclusters,distthres=3000,
                returnall=False,absmode=0,distmode=0,uncertmode=0):
    
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
        print("Found %i points in cluster %i" % (cnt,cid))
    uncert = np.zeros(nlat*nlon)*np.nan
    uncert[okpts] = uncertout
    uncert = uncert.reshape(nlat,nlon)
    
    if returnall:
        return clustered,uncert,cluster_count,Wk,srho,scov,sdist,distance_matrix
    return clustered,uncert,cluster_count,Wk



def plot_results(clustered,uncert,expname,lat5,lon5,outfigpath,title=None):
    
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
    if title is None:
        ax.set_title("Clustering Results \n nclusters=%i %s" % (nclusters,expname))
    else:
        ax.set_title(title)
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
    if title is None:
        ax1.set_title("Clustering Output (nclusters=%i) %s "% (nclusters,expname))
    else:
        ax1.set_title(title)
    plt.savefig(outfigpath+"Cluster_with_Shaded_uncertainties_%s.png" % expname,dpi=200)
    
    

    return fig,ax,fig1,ax1
    

def elim_points(sla,lat,lon,nclusters,minpts,maxiter,outfigpath,distthres=3000,
                absmode=0,distmode=0,uncertmode=0):
    
    ntime,nlat,nlon = sla.shape
    slain = sla.copy()
    
    # Preallocate
    allclusters = []
    alluncert   = []
    allcount    = []
    allWk = []
    rempts      = np.zeros((nlat*nlon))*np.nan
    
    # Loop
    flag = True
    it   = 0
    while flag or it < maxiter:
        
        expname = "iteration%02i" % (it+1)
        print("Iteration %i ========================="%it)
        
        # Perform Clustering
        clustered,uncert,cluster_count,Wk = cluster_ssh(slain,lat,lon,nclusters,distthres=distthres,
                                                     absmode=absmode,distmode=distmode,uncertmode=uncertmode)
        
        # Save results
        allclusters.append(clustered)
        alluncert.append(uncert)
        allcount.append(cluster_count)
        allWk.append(Wk)
        
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
    return allclusters,alluncert,allcount,rempts,allWk


#%%


# Load data (preproc, then anomalized)
st=time.time()
ld = np.load("%sSSHA_AVISO_%sto%s.npz" % (datpath,start,end),allow_pickle=True)
sla_5deg = ld['sla_5deg']
lon5 = ld['lon']
lat5 = ld['lat']
times = ld['times']
print("Loaded data in %.2fs"%(time.time()-st))

# Plotting utilities
cmbal = cmocean.cm.balance

#%% Aviso, Additional Preprocessing Steps

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
    out1 = slutil.remove_GMSL(ssha,lat5,lon5,timesyr,viz=True,testpoint=[lonf,latf])
    
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
    

# ---------------------
#%% Remove Seasonal Cycle
# ---------------------
if rem_seas:
    
    # Copy and reshape data
    sshc = ssha.copy()
    ntime = sshc.shape[0]
    sshc = sshc.reshape(ntime,nlat5*nlon5)
    
    # Get non nan points
    okdata,knan,okpts = proc.find_nan(sshc,0)
    
    # Remove seasonal cycle
    x,E = proc.remove_ss_sinusoid(okdata,semiannual=True)
    ssh_ss  = E@x
    okdata_ds = okdata - ssh_ss
    
    # Replace into data
    sshnew = np.zeros(sshc.shape)*np.nan
    sshnew[:,okpts] = okdata_ds
    sshnew = sshnew.reshape(ntime,nlat5,nlon5)
    
    sshss = np.zeros(sshc.shape)*np.nan
    sshss[:,okpts] = ssh_ss
    sshss = sshss.reshape(ntime,nlat5,nlon5)
    
    # Try another removal method
    ssha2 = ssha.copy()
    clim,ssha2 = proc.calc_clim(ssha2,0,returnts=1)
    ssha2 = ssha2 - clim[None,:,:,:]
    ssha2 = ssha2.reshape(ssha2.shape[0]*12,nlat5,nlon5)
    #ssha2 = ssha2.reshape(int(ntime/12)
    
    # Plot sample removal
    plotmons=60
    #klonss,klatss = proc.find_latlon(325,5,lon5,lat5)
    klonss,klatss = proc.find_latlon(330,50,lon5,lat5)
    fig,axs = plt.subplots(2,1)
    
    ax=axs[0]
    ax.plot(ssha[:plotmons,klatss,klonss],color='gray',label="Original")
    ax.plot(sshnew[:plotmons,klatss,klonss],color='k',label="Deseasoned (Sinusoid)")
    ax.plot(sshss[:plotmons,klatss,klonss],color='red',ls='dotted',label="Sinusoid Fit")
    ax.legend(ncol=3,fontsize=8)
    ax.grid(True,ls='dotted')
    ax.set_ylim([-.2,.2])
    ax.set_ylabel("SSH (cm)")
    
    ax = axs[1]
    ax.plot(ssha[:plotmons,klatss,klonss],color='gray',label="Original")
    ax.plot(ssha2[:plotmons,klatss,klonss],color='k',label="Deseasoned (Mean Cycle)")
    ax.plot(np.tile(clim[:,klatss,klonss],int(plotmons/12)),color='blue',ls='dotted',label="Mean Cycle")
    ax.legend(ncol=3,fontsize=8)
    ax.grid(True,ls='dotted')
    ax.set_ylim([-.2,.2])
    ax.set_xlabel("Months")
    ax.set_ylabel("SSH (cm)")
    
    plt.suptitle("Seasonal Cycle Removal at Lon %i Lat %i"%(lon5[klonss],lat5[klatss]))
    plt.savefig("%sSSRemoval_lon%i_lat%i.png"%(expdir,lon5[klonss],lat5[klatss]),dpi=150)

    ssha = ssha2.copy()


#
#%% Calculate some characteristic timescales
#
nlags = 60

# Again reshape
ssh3 = ssha.copy()
ssh3 = ssh3.reshape(ntime,nlat5*nlon5)
okdata,knan,okpts = proc.find_nan(ssh3,0)

# Calculate Lag Autocorrelation
lagac = np.zeros((nlags,okdata.shape[1]))
for l in range(nlags):
    lagac[l,:] = proc.pearsonr_2d(okdata[:ntime-l,:],okdata[l:,:],0)
sshac = np.zeros((nlags,nlat5*nlon5))
sshac[:,okpts] = lagac
sshac = sshac.reshape(nlags,nlat5,nlon5)



# Test Plot
klonss,klatss = proc.find_latlon(330,50,lon5,lat5)
r = sshac[1,klatss,klonss]
n_eff = int((ntime-nlags)*(1-r)/(1+r))
p = 0.05
tails = 2
ptilde    = p/tails
critval   = stats.t.ppf(1-ptilde,n_eff)
corrthres = np.sqrt(1/ ((n_eff/np.power(critval,2))+1))

fig,ax = plt.subplots(1,1)
ax.scatter(np.arange(0,nlags,1),sshac[:,klatss,klonss],20,color='r',marker="o")
ax.plot(sshac[:,klatss,klonss],color='r',label="ACF")
ax.set_title("Estimating Decay Timescale at Lon %i Lat %i"%(lon5[klonss],lat5[klatss]))
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag (Months)")
ax.axhline(1/np.exp(1),label="1/e",ls='dashed',color='k')
ax.axhline(corrthres,label="2-Tailed T-Test (DOF=%i), r=%.2f"%(n_eff,corrthres),ls='dashed',color='b')
ax.legend()
ax.grid(True,ls='dotted')
ax.set_xticks(np.arange(0,nlags+6,6))
plt.savefig("%sTimescale_lon%i_lat%i.png"%(expdir,lon5[klonss],lat5[klatss]),dpi=150)



# Find Nearest Month
msk = ssha.copy()
msk = msk.sum(0)
msk[~np.isnan(msk)] = 1
tau = sshac.copy()
tau = np.argmin(np.abs(tau-1/np.exp(1)),0)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
pcm = ax.pcolormesh(lon5,lat5,tau*msk,cmap='Blues')
fig.colorbar(pcm,ax=ax)
plt.savefig("%sTimescale_efold.png"%(expdir),dpi=150)







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

#%% Make Synthetic timeseries

msk = sla_lp.copy()
msk = msk.sum(0)
msk[~np.isnan(msk)] = 1


# Create white noise timeseries
wn = np.random.normal(0,1,sla_lp.shape)
wn *= msk[None,:,:]

# Create scaled form of timeseries
aviso_std = sla_lp.std(0)
wnstd = wn.copy()
wnstd *= aviso_std[None,:,:]

# Low Pass Filter The Timeseries
wnlp = slutil.lp_butter(wnstd,tw,order)


# Add a trend
#%% Do some clustering, with some experiments


# distmode
# 0: Default (Distance and Corr)
# 1: Distance Only
# 2: Corr Only

# uncertmode
# 0: Default (E(Cov_in) / E(Cov_out))
# 1: Median  (Med(Cov_in) / M(Cov_out))

# absmode
# 0: Default: Correlation and Covariances, no modification
# 1: Absolute Values: Take abs(corr) and abs(cov)
# 2: Anti: Take -1*corr, -1*cov

varin      = wnstd
snamelong  = "White Noise (Unfiltered)"
expname    = "AVISO_WhiteNoise_nclusters%i_" % (nclusters)
distmode   = 0
absmode    = 0
uncertmode = 0

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

# --------------------------
# Do Clustering (scipy)
# --------------------------
cdist      = squareform(distance_matrix,checks=False)
linked     = linkage(cdist,'weighted')
clusterout = fcluster(linked, nclusters,criterion='maxclust')

# --------------------------
# Replace into pull matrix
# --------------------------
clustered = np.zeros(nlat*nlon)*np.nan
clustered[okpts] = clusterout
clustered = clustered.reshape(nlat,nlon)

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
plot_results(clustered,uncert,expname,lat5,lon5,outfigpath,title=title)

# -------------------------------------------------------------
#%%   Calculate Clusters for particular experiments, for AVISO
# -------------------------------------------------------------

varin=sla_lp.copy()
absmode    =0
distmode   =0
uncertmode =0

expname = "Synthetic_%iclusters_minpts%i_maxiters%i" % (nclusters,minpts,maxiter)
expname += "_distmode%i_uncertmode_%i_absmode%i" % (distmode,uncertmode,absmode)
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

allclusters,alluncert,allcount,rempts,allWk = elim_points(varin,lat5,lon5,nclusters,minpts,maxiter,expdir,
                                                    absmode=absmode,distmode=distmode,uncertmode=uncertmode)


#
#%% Test Number of clusters
#
varin=sla_lp.copy()
absmode    =0
distmode   =0
uncertmode =0
nclustall = np.arange(1,31)

Wkall = []
Wknulls = []
clusters = []
cnull = []
for i in nclustall:
    clustered,uncert,cluster_count,Wk = cluster_ssh(varin,lat5,lon5,i,distthres=3000,
                                                         absmode=absmode,distmode=distmode,uncertmode=uncertmode)
    
    clusterednull,uncert,cluster_count,Wknull = cluster_ssh(wn,lat5,lon5,i,distthres=3000,
                                                         absmode=absmode,distmode=distmode,uncertmode=uncertmode)
    
    
    clusters.append(clusters)
    Wkall.append(Wk.mean())
    cnull.append(clusterednull)
    Wknulls.append(Wknull.mean())


Wkall = np.array(Wkall)
Wknulls = np.array(Wknulls)


fig,ax = plt.subplots(1,1)
# ax.plot(nclustall,Wkall/Wkall.max(),label="AVISO",color='r')
# ax.scatter(nclustall,Wkall/Wkall.max(),label="",color='r')

# ax.plot(nclustall,Wknulls/Wknulls.max(),label="White Noise, Distance Only(Null)",color='k')
# ax.scatter(nclustall,Wknulls/Wknulls.max(),label="",color='k')

ax.plot(nclustall,Wkall,label="AVISO",color='r')
ax.scatter(nclustall,Wkall,label="",color='r')

ax.plot(nclustall,Wknulls,label="White Noise, Distance Only(Null)",color='k')
ax.scatter(nclustall,Wknulls,label="",color='k')


ax.set_ylabel("Within Cluster Distance (km)")
ax.set_xlabel("Number of Clusters")
ax.set_xticks(nclustall[::2])
ax.legend()
    
#%% Examine typical distance and time decay scales

varin      = sla_lp
ntime,nlat,nlon = varin.shape
srho,scov,sdist,okdata,okpts,coords2=slutil.calc_matrices(varin,lon5,lat5,return_all=True)

i = 222

# Get Sorted Matrices
sortsrho  = np.zeros(srho.shape)
sortsdist = np.zeros(srho.shape)
sortidall = np.zeros(srho.shape)
npts = srho.shape[0]
for i in tqdm(range(npts)):
    
    # Get data for point
    distpt = sdist[i,:]
    rhopt  = srho[i,:]
    
    # Sort
    sortid = np.argsort(distpt)#Indices sorting by distance
    sortrho = rhopt[sortid] 
    sortdist = distpt[sortid]
    
    # Save
    sortsrho[i,:] = sortrho.copy()
    sortsdist[i,:] = sortdist.copy()
    sortidall[i,:] = sortid.copy()




i = 222

distpt = sortsdist[i,:]
rhopt = sortsrho[i,:]




E = np.ones((npts,2))
E[:,1] = -distpt
A = np.log(rhopt)
A = A[:,None]

import scipy
hey =scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(-b*t),  distpt,  rhopt)


def model_func(t, A, K, C):
    return A * np.exp(K * t) + C

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K


testa = rhopt.copy()
testa[testa<=0] = 1e-15
test = fit_exp_linear(distpt,testa)

fig,ax = plt.subplots(1,1)
ax.scatter(distpt,rhopt,20,color='k',alpha=0.8)
ax.set_title("Distance vs Correlation for Pt %i"%i)
ax.set_xlabel("Distance")
ax.set_ylabel("Correlation")
ax.axhline([0],ls='dashed',color='gray',lw=0.75)
labexp = "$%.3fe^{-%.3fd}$" % (test[0],test[1])
ax.plot(distpt,model_func(distpt,test[0],test[1],0),color='r',label=labexp)
ax.legend()
plt.savefig("%s../ExpFit.png"%outfigpath,dpi=200)


# --------------------------------
# Combine nearby values and average
dists1  = []#[[]]
counts1 = []#np.zeros((npts))
rhos1   = []#np.zeros((npts))
cnt = -1
dist0 = 999
tol=1
for i in tqdm(range(npts)):
    print(i)
    # Get Distance
    dist  = distpt[i]
    # Update i, depending on if distance is same
    if ~((dist < dist0+tol) and (dist > dist0-tol)):
        dists1.append(dist)
        rhos1.append(rhopt[i])
        counts1.append(1)
        cnt += 1
    else:
        # Add to the point
        dists1[cnt]  += dist
        rhos1[cnt]   += rhopt[i]
        counts1[cnt] += 1
    dist0 = dist
distfin = np.array(dists1)/np.array(counts1)
rhofin = np.array(rhos1)/np.array(counts1)
# ----------------------------------------

# Sort linear indices
fsdist = sortsdist.flatten()
fsid   = np.argsort(fsdist)

fsdistsort  =fsdist[fsid]
fsrhosort = sortsrho.flatten()[fsid]

distnew = []
rhonew  = []

# Calculate characterstic timescales by fitting an exponential function
plt.plot(fsrhosort), plt.xlim([0, 3000])

#%% Calculate characteristic timescale of decay

# Remove NaNs
invar = ssha.copy()
invar = invar.reshape(ntime,nlat5*nlon5)
okdata,knan,okpts = proc.find_nan(invar,dim=0)

npts = invar.shape[1]
nok = okdata.shape[1]

# Compute Lag 1 AR for each and effective DOF
ar1  = np.zeros(nok)
neff = np.zeros(nok) 
for i in tqdm(range(nok)):
    
    ts = okdata[:,i]
    r = np.corrcoef(ts[1:],ts[:-1])[0,1]
    ar1[i] = r
    neff[i] = ntime*(1-r)/(1+r)

# Replace into domain
ar1_map = np.zeros(npts)*np.nan
neff_map = np.zeros(npts)*np.nan
ar1_map[okpts] = ar1
neff_map[okpts] = neff
ar1_map = ar1_map.reshape(nlat5,nlon5)
neff_map = neff_map.reshape(nlat5,nlon5)

#%% Visualize some plots



fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
pcm = ax.pcolormesh(lon5,lat5,ar1_map,cmap="magma")
ax.set_title("Lag 1 Autocorrelation, AVISO (1993-2013)")
fig.colorbar(pcm,ax=ax,fraction=0.046)
plt.savefig("%s../AR1_noLP_Map.png"%outfigpath,dpi=200)


fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax = slutil.add_coast_grid(ax)
pcm = ax.pcolormesh(lon5,lat5,neff_map,vmin=0,vmax=240,cmap="Purples")
ax.set_title("Effective Degrees of Freedom, AVISO (1993-2013)")
fig.colorbar(pcm,ax=ax,fraction=0.046)
plt.savefig("%s../Neff_Map_noLP.png"%outfigpath,dpi=200)


