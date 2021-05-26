#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Sea Level Utilities

Created on Sun Mar 14 21:51:31 2021

@author: gliu
"""

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from tqdm import tqdm
from pylab import cm

from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import silhouette_score,silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.signal import butter, lfilter, freqz, filtfilt, detrend

import pygmt
import itertools
import glob
import time
import cmocean
import time
import sys

# Custom Scripts
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz
import yo_box as ybx
import tbx

# ----------------
# %% Preprocessing
# ----------------

def remove_GMSL(ssh,lat,lon,times,tol=1e-10,viz=False,testpoint=[330,50]):
    """
    Parameters
    ----------
    ssh : ARRAY [time x lat x lon]
        Sea Surface Height Anomalies to process
    lat : ARRAY [lat]
        Latitude values
    lon : ARRAY [lon]
        Longitude values 
    times : ARRAY [time]
        Time values to plot (years)
    tol : FLOAT, optional
        Tolerance to check if GMSL is zero. The default is 1e-10.
    viz : BOOL, optional
        Visualize GMSL removal. The default is False.
    testpoint : LIST [lon,lat]
        Longitude and latitude points to visualize removal at. The default is [330,50].

    Returns
    -------
    ssh : ARRAY [time x lat x lon]
        Sea surface height anomalies with GMSL removed
    gmslrem: ARRAY [time]
        Time series that was removed
    
    Additional outputs for viz == True:
        
        fig,ax that was visualized
        
    Note: Add latitude weights for future update...
    """
    # Calculate GMSL (Not Area Weighted)
    gmslrem = np.nanmean(ssh,(1,2))
    
    if np.any(gmslrem>tol):
        # Remove GMSL
        ssh_ori = ssh.copy()
        ssh     = ssh - gmslrem[:,None,None]
        
        # Plot Results
        if viz:
            lonf,latf = testpoint
            klon,klat = proc.find_latlon(lonf,latf,lon,lat)
            
            fig,ax = plt.subplots(1,1)
            #ax.set_xticks(np.arange(0,len(times)+1,12))
            ax.set_xticks(np.arange(0,len(times),12))
            ax.set_xticklabels(times[::12],rotation = 45)
            ax.grid(True,ls='dotted')
            
            ax.plot(ssh_ori[:,klat,klon],label="Original",color='k')
            ax.plot(ssh[:,klat,klon],label="Post-Removal")
            ax.plot(gmslrem,label="GMSL")
            
            ax.legend()
            ax.set_title("GMSL Removal at Lon %.2f Lat %.2f (%s to %s)" % (lon[klon],lat[klat],times[0],times[-1]))
            ax.set_ylabel("SSH (m)")
            return ssh,gmslrem,fig,ax
    else:
        print("GMSL has already been removed, largest value is %e" % (gmslrem.max()))
    return ssh,gmslrem


def lp_butter(varmon,cutofftime,order):
    """
    Design and apply a low-pass filter (butterworth)

    Parameters
    ----------
    varmon : 
        Input variable to filter (monthly resolution)
    cutofftime : INT
        Cutoff value in months
    order : INT
        Order of the butterworth filter

    Returns
    -------
    varfilt : ARRAY [time,lat,lon]
        Filtered variable

    """
    # Input variable is assumed to be monthy with the following dimensions:
    flag1d=False
    if len(varmon.shape) > 1:
        nmon,nlat,nlon = varmon.shape
    else:
        flag1d = True
        nmon = varmon.shape[0]
    
    # Design Butterworth Lowpass Filter
    filtfreq = nmon/cutofftime
    nyquist  = nmon/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
    # Reshape input
    if ~flag1d: # For 3d inputs, loop thru each point
        varmon = varmon.reshape(nmon,nlat*nlon)
        # Loop
        varfilt = np.zeros((nmon,nlat*nlon)) * np.nan
        for i in tqdm(range(nlon*nlat)):
            varfilt[:,i] = filtfilt(b,a,varmon[:,i])
        varfilt=varfilt.reshape(nmon,nlat,nlon)
    else: # 1d input
        varfilt = filtfilt(b,a,varmon)
    return varfilt


# -------------
# %% Clustering
# -------------

def calc_matrices(invar,lon,lat,return_all=False):
    """
    Calculate correlation, covariance, and distance matrices in preparation
    for clustering.

    Parameters
    ----------
    invar : ARRAY (Time x Lat x Lon)
        Input variable
    lon : ARRAY (Lon)
        Longitudes
    lat : ARRAY (Lat)
        Latitudes
    return_all : BOOL, optional
        Set to true to return non-nan points, indices, and coordinates. The default is False.

    Returns
    -------
    srho: ARRAY [npts x npts]
        Correlation Matrix
    scov: ARRAY [npts x npts]
        Covariance Matrix
    sdist: ARRAY [npts x npts]
        Distance Matrix

    """
    
    # ---------------------
    # Remove All NaN Points
    # ---------------------
    ntime,nlat,nlon = invar.shape
    varrs = invar.reshape(ntime,nlat*nlon)
    okdata,knan,okpts = proc.find_nan(varrs,0)
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
    
    if return_all:
        return srho,scov,sdist,okdata,okpts,coords2
    return srho,scov,sdist
    
def make_distmat(srho,sdist,distthres=3000,rhowgt=1,distwgt=1):
    """
    Make distance matrix, using output from calc_matrices
    
    dist = 1 - exp(-dist/(2a^2)) * corr

    Parameters
    ----------
    srho : ARRAY [npts x npts]
        DESCRIPTION.
    sdist : ARRAY [npts x npts]
        DESCRIPTION.
    distthres : INT, optional
        Point at which exponential term is 0.5. The default is 3000.
    rhowgt : FLOAT, optional
        Amount to weight the correlation matrix. The default is 1.
    distwgt : TYPE, optional
        Amount to weight the exponential (distance) term. The default is 1.

    Returns
    -------
    distance_matrix : ARRAY [npts x npts]
        Distance matrix for input to the clustering algorithm

    """
    
    # Calculate exponential term
    a_fac = np.sqrt(-distthres/(2*np.log(0.5))) # Calcuate so exp=0.5 when distance is distthres
    expterm = np.exp(-sdist/(2*a_fac**2))
    
    # Apply weighting to distance and correlation matrix
    # So that as wgt --> 0, value --> 1
    # and as wgt --> 1, value --> itself
    #expterm *= (expterm-1)*distwgt + 1
    #srho    *= ((np.abs(srho)-1)*rhowgt + 1)  * np.sign(srho)
    
    # Calculate 
    distance_matrix = 1-expterm*srho
    
    return distance_matrix

def calc_silhouette(distance_matrix,clusterout,nclusters):
        """
        Quick wrapper to calculate silhouette metrics.

        Parameters
        ----------
        distance_matrix : ARRAY [nsamples x nsamples]
            Distance metric used for clustering
        clusterout : ARRAY [nsamples]
            Clustering Labels

        Returns
        -------
        s_score : NUMERIC
            Averaged silhouette coefficient for all values
        s : ARRAY [nsamples,]
            Silhouette coefficient for each point
        s_bycluster : 
            DESCRIPTION.

        """
        
        # Calculate silhouette score (1 value)
        s_score = silhouette_score(distance_matrix,clusterout,metric="precomputed")
    
        # Calculate the silhouette for each point
        s       = silhouette_samples(distance_matrix,clusterout,metric='precomputed')
        #print("Calculated from SKlearn is %.3f" % s_score)
        
        # Calculate s for each cluster
        s_cluster = np.zeros(nclusters)
        counts    = s_cluster.copy()
        for i in range(len(clusterout)):
            cluster_id = clusterout[i]-1
            
            # Record Silhouette Score and Count
            s_cluster[cluster_id] += s[i]
            counts[cluster_id] += 1
        s_bycluster = s_cluster / counts
        
        return s_score,s,s_bycluster

# -----------------
# %% Visualization
# -----------------


def add_coast_grid(ax,bbox=[-180,180,-90,90],proj=None):
    """
    Add Coastlines, grid, and set extent for geoaxes
    
    Parameters
    ----------
    ax : matplotlib geoaxes
        Axes to plot on 
    bbox : [LonW,LonE,LatS,LatN], optional
        Bounding box for plotting. The default is [-180,180,-90,90].
    proj : cartopy.crs, optional
        Projection. The default is None.

    Returns
    -------
    ax : matplotlib geoaxes
        Axes with setup
    """
    if proj is None:
        proj = ccrs.PlateCarree()
    ax.add_feature(cfeature.COASTLINE,color='black',lw=0.75)
    ax.set_extent(bbox)
    gl = ax.gridlines(crs=proj, draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle="dotted",lw=0.75)
    gl.right_labels = False
    gl.top_labels = False
    return ax

def check_lpfilter(rawdata,lpdata,chkval,M,tw,dt=24*3600*30):
    """
    

    Parameters
    ----------
    rawdata : ARRAY [time x space]
        Raw Dataset (no NaNs)
    lpdata : ARRAY [time x space]
        Low-pass (LP) filtered dataset (no NaNs)
    chkval : INT
        Period (1/f) to check filter transfer function at (in units of dt)
    M : INT
        Bands to average over
    tw : INT
        Cutoff time low-pass filter was performed at (for plotting)
    dt : INT, optional
        Timestep size in seconds. The default is 24*3600*30.

    Returns
    -------
    lpspec : ARRAY [freq x space]
        Power spectra for LP filtered timeseries
    rawspec : ARRAY [freq x space]
        Power spectra for raw filtered timeseries
    p24 : ARRAY [space]
        Power spectra value for each point at chkval
    filtxfer : ARRAY [freq x space]
        Filter transfer function
    fig : mpl figure
    ax : mpl axes

    """
    # Set x-tick parameters
    xtk = [1/(10*12*dt),1/(24*dt),1/(12*dt),1/(3*dt),1/dt]
    xtkl = ['decade','2-yr','year','season','month']
    
    # Get number of points
    npts5 = lpdata.shape[1]
    
    # Compute power spectra for each point
    lpspec  = []
    rawspec = []
    for i in tqdm(range(npts5)):
        X_spec,freq,_=tbx.bandavg_autospec(rawdata[:,i],dt,M,.05)
        X_lpspec,_,_ =tbx.bandavg_autospec(lpdata[:,i],dt,M,.05)
        lpspec.append(X_lpspec)
        rawspec.append(X_spec)
    lpspec   = np.real(np.array(lpspec))
    rawspec  = np.real(np.array(rawspec))
    
    # Calculate filter transfer function
    filtxfer = lpspec/rawspec
    
    # Get index for the frequency of interest [k24mon]
    k24mon = np.argmin(np.abs(freq-chkval)) # Get index for 24 mon (chkval)
    if freq[k24mon] < chkval: # less than 24 months
        ids = [k24mon,k24mon+1]
    else:
        ids = [k24mon-1,k24mon]
    
    # Linearly interpolate to obtain value of spectrum at k24mon
    p24 = np.zeros(npts5)
    for it in tqdm(range(npts5)):
        p24[it] = np.interp(chkval,freq[ids],filtxfer[it,ids])
    
    # Plot results
    plotnum=npts5
    fig,axs= plt.subplots(2,1)
    ax = axs[0]
    ax.plot(freq,rawspec[:plotnum,:].T,label="",color='gray',alpha=0.25)
    ax.plot(freq,lpspec[:plotnum,:].T,label="",color='red',alpha=0.15)
    ax.set_xscale('log')
    ax.set_xticks(xtk)
    ax.set_xticklabels(xtkl)
    ax.set_title("Raw (gray) and Filtered (red) spectra")
    ax.grid(True,ls='dotted')
    
    ax = axs[1]
    plotp24 = p24.mean()#np.interp(chkval,freq[ids],filtxfer[:plotnum,:].mean(0)[ids]) #p24[:plotnum].mean()
    ax.plot(freq,filtxfer[:plotnum,:].T,label="",color='b',alpha=0.05,zorder=-1)
    ax.plot(freq,filtxfer[:plotnum,:].mean(0),label="",color='k',alpha=1)
    ax.scatter(chkval,[plotp24],s=100,marker="x",color='k',zorder=1)
    ax.set_ylim([0,1])
    ax.set_xscale('log')
    ax.set_xticks(xtk)
    ax.set_xticklabels(xtkl)
    ax.set_title("Filter Transfer Function (Filtered/Raw), %.3f" % (plotp24*100) +"%  Cutoff at " + "%i months" %(tw))
    ax.grid(True,ls='dotted')
    plt.suptitle("AVISO 5deg SSH Timeseries, %i-Band Average"% (M))
    plt.tight_layout()
    
    return lpspec,rawspec,p24,filtxfer,fig,ax


def plot_silhouette(clusterout,nclusters,s,cmap=None,ax1=None,xlm=[-.25, 1],returncolor=False):
        """
        Make a silhouette plot
        
        Parameters
        ----------
        clusterout : ARRAY [nsamples]
            Cluster Labels
        nclusters : INT
            Number of clusters
        s : ARRAY [nsamples]
            Silhouette coefficient for each cluster
        cmap : List of colors, optional
            Colors for each cluster. The default is None.
        ax1 : matplotlib axes, optional
            Axis to plot on. The default is None.
        xlm : [xlower, xupper], optional
            xlimits for plotting. The default is [-.25, 1].

        Returns
        -------
        ax1 : matplotlib axes
            matplotlib axes containing result.
        """
        # Adapted from: 
        #https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
        

    
        
        # Make Silhouette plot
        y_lower=10
        
        if ax1 is None:
            fig, ax1 = plt.subplots(1, 1)
        
        # Set x and y limits
        ax1.set_xlim(xlm)
        ax1.set_ylim([0, len(clusterout) + (nclusters + 1) * 10])
        
        s_score = s.mean()
        ccols = []
        # Make a plot, aggregating by cluster
        for i in range(nclusters):
            
            # Get silhouette scores for cluster and sort
            cid = i + 1
            ith_cluster_silhouette_values = s[clusterout == cid]
            ith_cluster_silhouette_values.sort()
            
            # Get y bounds for plotting
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            # Geet Colormap
            if cmap is None:
                color = mcm.nipy_spectral(float(cid) / nclusters)
                
            else:
                color = cmap[i]
            ccols.append(color)
                
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cid))
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
            print(cid)
            print(y_lower)
        
        # Labels
        ax1.set_title("Silhouette Plot")
        ax1.set_xlabel("Silhouette Coefficient")
        ax1.set_ylabel("Cluster Label")
        
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=s_score, color="red", linestyle="--")
        
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        
        if returncolor:
            return ax1,ccols
        return ax1
    
    
#
# %% Clustering Analysis
#

def patterncorr(map1,map2):
    # From Taylor 2001,Eqn. 1, Ignore Area Weights
    # Calculate pattern correation between two 2d variables (lat x lon)
    
    
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

# --------------------------
#%% Clustering, main scripts
# --------------------------
def cluster_ssh(sla,lat,lon,nclusters,distthres=3000,
                returnall=False,absmode=0,distmode=0,uncertmode=0,printmsg=True,
                calcsil=False):
    
    # ---------------------------------------------
    # Calculate Correlation, Covariance, and Distance Matrices
    # ---------------------------------------------
    ntime,nlat,nlon = sla.shape
    srho,scov,sdist,okdata,okpts,coords2=calc_matrices(sla,lon,lat,return_all=True)
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
        s_score,s,s_bycluster = calc_silhouette(distance_matrix,clusterout,nclusters)
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

def plot_results(clustered,uncert,expname,lat5,lon5,outfigpath,nclusters):
    
    # Set some defaults
    
    ucolors = ('Blues','Purples','Greys','Blues','Reds','Oranges','Greens')
    proj = ccrs.PlateCarree(central_longitude=180)
    cmap = cm.get_cmap("jet",nclusters)
    
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
    ax = add_coast_grid(ax)
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
    ax     = add_coast_grid(ax)
    pcm    = plt.pcolormesh(lon5,lat5,uncert,cmap='copper',transform=ccrs.PlateCarree())
    ax.set_title(r"Uncertainty $(<\sigma^{2}_{out,x}>/<\sigma^{2}_{in,x}>)$")
    fig.colorbar(pcm,ax=ax,fraction=0.02)
    plt.savefig(outfigpath+"Uncertainty.png",dpi=200)
    
    
    # Apply Thompson and Merrifield thresholds
    uncert[uncert>2] = 2
    uncert[uncert<0.5]=0
    
    # Plot Cluster Uncertainty
    fig1,ax1 = plt.subplots(1,1,subplot_kw={'projection':proj})
    ax1 = add_coast_grid(ax1)
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
            fig,ax,fig1,ax1 = plot_results(clustered,uncert,expname,lat,lon,outfigpath,nclusters)
            

        
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

