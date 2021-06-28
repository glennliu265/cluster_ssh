#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess AVISO
Read in monthly climatologial AVISO Data and combine the files
Should be 321 files (from 1993-01 to 2019-09)

Files were downloaded from AVISO FTP server (see notes on iCloud)
Checking for completeness using "avisoscrap.py"

https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/global/gridded-sea-level-anomalies-mean-and-climatology.html#c7273

Created on Tue Nov 17 00:58:21 2020

@author: gliu
"""

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import pygmt
from tqdm import tqdm

import glob
import time
import cmocean
import time
#import tqdm
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz
import yo_box as ybx
import tbx
from scipy.signal import butter, lfilter, freqz, filtfilt, detrend

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pylab import cm


from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

#%% User Edits

outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210309/"
resmooth   = False
debug      = True
rem_gmsl   = False

nclusters  = 6
tol        =2.5 # Search tolerance for spatial smoothing

mons3      = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
#%% Functions
def coarsen_byavg(invar,lat,lon,deg,tol,latweight=True,verbose=True,ignorenan=False):
    """
    Coarsen an input variable to specified resolution [deg]
    by averaging values within a search tolerance for each new grid box.
    
    Dependencies: numpy as np

    Parameters
    ----------
    invar : ARRAY [TIME x LAT x LON]
        Input variable to regrid
    lat : ARRAY [LAT]
        Latitude values of input
    lon : ARRAY [LON]
        Longitude values of input
    deg : INT
        Resolution of the new grid (in degrees)
    tol : TYPE
        Search tolerance (pulls all lat/lon +/- tol)
    
    OPTIONAL ---
    latweight : BOOL
        Set to true to apply latitude weighted-average
    verbose : BOOL
        Set to true to print status
    

    Returns
    -------
    outvar : ARRAY [TIME x LAT x LON]
        Regridded variable       
    lat5 : ARRAY [LAT]
        New Latitude values of input
    lon5 : ARRAY [LON]
        New Longitude values of input

    """

    # Make new Arrays
    lon5 = np.arange(0,360+deg,deg)
    lat5 = np.arange(-90,90+deg,deg)
    
    
    # Set up latitude weights
    if latweight:
        _,Y = np.meshgrid(lon,lat)
        wgt = np.cos(np.radians(Y)) # [lat x lon]
        invar *= wgt[None,:,:] # Multiply by latitude weight
    
    # Get time dimension and preallocate
    nt = invar.shape[0]
    outvar = np.zeros((nt,len(lat5),len(lon5)))
    
    # Loop and regrid
    i=0
    for o in range(len(lon5)):
        for a in range(len(lat5)):
            lonf = lon5[o]
            latf = lat5[a]
            
            lons = np.where((lon >= lonf-tol) & (lon <= lonf+tol))[0]
            lats = np.where((lat >= latf-tol) & (lat <= latf+tol))[0]
            
            varf = invar[:,lats[:,None],lons[None,:]]
            
            if latweight:
                wgtbox = wgt[lats[:,None],lons[None,:]]
                if ignorenan:
                    varf = np.nansum(varf/np.nansum(wgtbox,(0,1)),(1,2)) # Divide by the total weight for the box
                else:
                    varf = np.sum(varf/np.sum(wgtbox,(0,1)),(1,2)) # Divide by the total weight for the box
                
                
            else:
                if ignorenan:   
                    varf = np.nanmean(varf,axis=(1,2))
                else:
                    varf = varf.mean((1,2))
                
            outvar[:,a,o] = varf.copy()
            i+= 1
            msg="\rCompleted %i of %i"% (i,len(lon5)*len(lat5))
            print(msg,end="\r",flush=True)
    return outvar,lat5,lon5

def calc_sig(truncate,w):
    """
    calculate sigma for gaussian filter give desired filter size (in units of x and y)
    and the truncate parameter to be used (stdev cutoff for filter)
    https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
    w = 2*int(truncate*sigma + 0.5) + 1
    """
    return (w-2)/(2*truncate)

def load_aviso(datpath=None,verbose=True,numpy=False):
    """
    Inputs
    ------
    datpath : STR
        Path to AVISO nc files (if default, use preset datapath)
    verbose : BOOL
        Set to True to print messages
    numpy : BOOL
        Set to True to return as numpy arrays. Default is to load into
        dask dataset
        
    Outputs
    -------
    --- numpy == False ----
    dsall : Dataset
        Xarray Dataset 
    --- numpy == True ----  
    sla : ARRAY
        Sea level anomaly field
    lon : ARRAY (Longitude)
    lat : ARRAY (Latitude)
    times : ARRAY (cftime) 
    """
    if datpath is None:
        datpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/00_Raw/aviso/ncfiles/unzipped/*"
    # Get list of ncfiles
    nclist = glob.glob(datpath)
    nclist.sort()
    if verbose:
        print("Found %i nc files, starting from \n%s to...\n %s"% (len(nclist),nclist[0],nclist[-1]))
    
    # Try to read in all the files # time 321 lat 720 lon 1440
    st = time.time()
    dsall = xr.open_mfdataset(nclist,concat_dim='time')
    print("Opened in %.2fs"%(time.time()-st))
    
    if numpy:
        st = time.time()
        sla = dsall.sla.values
        lon = dsall.lon.values
        lat = dsall.lat.values
        times = dsall.time.values
        print("Loaded into numpy arrays in %.2fs"%(time.time()-st))
        return sla,lon,lat,times
    return dsall
        
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
    
def add_coast_grid(ax,bbox=[-180,180,-90,90],proj=None):
    if proj is None:
        proj = ccrs.PlateCarree()
    ax.add_feature(cfeature.COASTLINE,color='black',lw=0.75)
    ax.set_extent(bbox)
    gl = ax.gridlines(crs=proj, draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle="dotted",lw=0.75)
    gl.xlabels_top = False
    gl.ylabels_right = False
    return ax

def lp_butter(varmon,cutofftime,order):
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

#%% User inputs/load in dat

# Path to unzipped NC files
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/00_Raw/aviso/ncfiles/unzipped/*"

# Output Path
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/01_Data/01_Proc/"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/Scrap/"

#%% 1) Load Data
cmap = cmocean.cm.balance

# Load Aviso Data
sla,lon,lat,times = load_aviso(numpy=True)

# Get Dimensions [time x lat x lon]
ntime,nlat,nlon = sla.shape

# Retrieve the landmask
mask = sla.sum(0)
mask[~np.isnan(mask)] = 1

if debug: # Visualize raw data
    t=0
    fig,ax=plt.subplots(1,1)
    pcm=ax.pcolormesh(lon,lat,sla[t,:,:],vmin=-.5,vmax=.5,cmap=cmap)
    fig.colorbar(pcm,ax=ax)
    ax.set_title("Raw SSHA, time = %s"%(times[t]))

#%% 2) Apply Gaussian Spatial Filter, save landmask

# Replace in data array for filtering
sla = sla.transpose(1,2,0) # [lat x lon x time]

if resmooth:
    slasmooth = np.zeros((ntime,nlat,nlon))
    for i in tqdm.tqdm(range(ntime)):
        da = xr.DataArray(sla[:,:,i].astype('float32'),
                        coords={'lat':lat,'lon':lon},
                        dims={'lat':lat,'lon':lon},
                        name='sla')
        timestamp = times[i]
        smooth_field = pygmt.grdfilter(grid=da, filter="g500", distance="4",nans="i")
        slasmooth[i,:,:] = smooth_field.values
    
    np.save(outpath+"sla_filtered_g500_d4_nani.npy",slasmooth)
    #np.save("sla_filtered_g600_d4_nani.npy",slasmooth)
else:
    slasmooth = np.load(outpath+"sla_filtered_g500_d4_nani.npy")

if debug:
    fig,ax = plt.subplots(1,1)
    pcm = ax.pcolormesh(lon,lat,slasmooth[0,:,:],vmin=-0.4,vmax=0.4,cmap=cmap)
    fig.colorbar(pcm,ax=ax)

#%% 3) Apply Land Mask and coarsen the data

# Reapply Mask to correct for smoothed edges
sla_filt = slasmooth * mask[None,:,:]

# Apply Regridding (coarse averaging for now)
deg  = 5
tol=0.75
sla_5deg,lat5,lon5 = coarsen_byavg(sla_filt,lat,lon,deg,tol)

# Note ignoring nan leads to more coastal points
#sla_5deg,lat5,lon5 = coarsen_byavg(sla_filt,lat,lon,deg,tol,latweight=False,ignorenan=True)

# Plot Sample
if debug:
    cmap = cmocean.cm.balance
    fig,ax = plt.subplots(1,1)
    pcm = ax.pcolormesh(lon5,lat5,sla_5deg[0,:,:],cmap=cmap)
    fig.colorbar(pcm,ax=ax)


# Save 5 degree mask
mask5 = sla_5deg.sum(0)
mask5[~np.isnan(mask5)] = 1
np.save(outpath+"AVISO_landice_mask_5deg.npy",mask5)

#%% Restrict to a time

# Limit to particular period
start = '1993-01'
end   = '2013-01'

# Convert Datestrings
timesmon = np.datetime_as_string(times,unit="M")

# Find indices
idstart  = np.where(timesmon==start)[0][0]
idend    = np.where(timesmon==end)[0][0]

# Restrict Data
sla_5deg = sla_5deg[idstart:idend,:,:]
timeslim = timesmon[idstart:idend]
timesyr  = np.datetime_as_string(times,unit="Y")[idstart:idend]

# Save Coarsened data restrictd to the time period
outname = "%sSSHA_AVISO_%sto%s.npz" % (outpath,start,end)
np.savez(outname,**{
    'sla_5deg':sla_5deg,
    'lon':lon5,
    'lat':lat5,
    'times':times
    })

#%% Remove GMSL


if rem_gmsl:
    print("Removing GMSL")
    gmslrem = np.nanmean(sla_5deg,(1,2))
    
    sla_5deg_ori = sla_5deg.copy()
    sla_5deg     = sla_5deg - gmslrem[:,None,None]
    
    if np.all(gmslrem>1e-10):

    
        print("Saving GMSL")
        np.save(outpath+"AVISO_GMSL_%s_%s.npy"%(start,end),gmslrem)
    else:
        print("GMSL is already removed")
        plt.plot(gmslrem)
    if debug:
        # Test plot point
        lonf = 330
        latf = 40
        klon,klat = proc.find_latlon(lonf,latf,lon5,lat5)
    
        fig,ax = plt.subplots(1,1)
        ax.set_xticks(np.arange(0,240,12))
        ax.set_xticklabels(timesyr[::12],rotation = 45)
        ax.grid(True,ls='dotted')
        
        ax.plot(sla_5deg_ori[:,klat,klon],label="Original",color='k')
        ax.plot(sla_5deg[:,klat,klon],label="Post-Removal")
        ax.plot(gmslrem,label="GMSL")
        ax
        ax.legend()
        ax.set_title("GMSL Removal at Lon %.2f Lat %.2f (%s to %s)" % (lon5[klon],lat5[klat],start,end))
        ax.set_ylabel("SSH (m)")
        plt.savefig(outfigpath+"GMSL_Removal_lon%i_lat%i.png"% (lonf,latf),dpi=200)
        
else:
    print("GMSL Not Removed")

#%% 4) Low Pass Filter

order = 4
tw    = 15 # filter size for time dim




# Examine Low-Pass Filter
dt = 24*3600*30
M  = 5
xtk = [1/(10*12*dt),1/(24*dt),1/(12*dt),1/(3*dt),1/dt]
xtkl = ['decade','2-yr','year','season','month']



# Perform LowPass Filter
sla_lp = lp_butter(sla_5deg,tw,order)

# Sample Plot
klon,klat = proc.find_latlon(200,0,lon5,lat5)
fig,ax=plt.subplots(1,1)
ax.plot(sla_5deg[:,klat,klon],label="Unfiltered")
ax.plot(sla_lp[:,klat,klon],label="Filtered (Butterworth)")
ax.legend()



ntime,nlat5,nlon5 = sla_5deg.shape
gmsl_smooth = np.nanmean(sla_5deg,(1,2))
gmsl = np.nanmean(sla,(0,1))
gmsl_lp = np.nanmean(sla_lp,(1,2))  


# Plot Global Mean Sea Level
fig,ax = plt.subplots(1,1)
ax.set_title("Global Mean Sea Level")
ax.set_xticks(np.arange(0,240,12))
ax.set_xticklabels(timesyr[::12],rotation = 45)
ax.set_ylabel("SSHA (m)")
ax.set_xlabel("Time (Years)")
ax.grid(True,ls='dotted')
ax.plot(gmsl[idstart:idend],label="Raw",color='k')
ax.plot(gmsl_smooth,label="After Spatial Smoothing",color='r',ls='dashdot')
ax.plot(gmsl_lp,label="Low-Pass Filtered",color='b')
ax.legend()
plt.savefig(outfigpath+"GMSL.png",dpi=200)


outname = "%sSSHA_AVISO_%sto%s_LowPassFilter_order%i_cutoff%i.npz" % (outpath,start,end,order,tw)
np.savez(outname,**{
    'sla_lp':sla_lp,
    'lon':lon5,
    'lat':lat5,
    'times':times
    })
#%% Explore seasonal cycle removal at a particular point

# Locate Point
lonfss = 325
latfss = 10
fig,ax = plt.subplots(1,1)
pcm = ax.pcolormesh(lon5,lat5,sla_5deg[0,:,:],cmap=cmap)
ax.scatter([lonfss],[latfss],s=75,marker="X",color='k')
fig.colorbar(pcm,ax=ax)
ax.set_title("Lon %i Lat %i" % (lonfss,latfss))
plt.savefig(outfigpath+"Locator_lon%ilat%i.png" % (lonfss,latfss),dpi=200)

# Find Seasonal Cycle
klonss,klatss = proc.find_latlon(325,5,lon5,lat5)
slapt         = sla_5deg[:,klatss,klonss].reshape(20,12)
slaptlp       = sla_lp[:,klatss,klonss].reshape(20,12)

# Try Plotting Sea Level
fig,axs=plt.subplots(2,1,sharey=True)
ax = axs[0]
ax.set_title("Before Lowpass Filter")
ax.plot(mons3,slapt.T,alpha=0.2,color='gray',label="")
ax.plot(mons3,slapt.mean(0),color='k',label="Mean")
ax.set_ylim([-.2,.2])
ax.legend()
ax.grid(True,ls='dotted')

ax = axs[1]
ax.set_title("After Lowpass Filter")
ax.plot(mons3,slaptlp.T,alpha=0.2,color='gray',label="")
ax.plot(mons3,slaptlp.mean(0),color='k',label="Mean")
ax.grid(True,ls='dotted')

plt.suptitle("Seasonal Cycle at Lon %i Lat %i (%s to %s)"% (lonfss,latfss,start,end))
plt.tight_layout()
plt.savefig(outfigpath+"SeasonalCycle_Removal_LPF_lon%ilat%i.png"%(lonfss,latfss),dpi=200)

# Try removing Seasonal Cycle using Sinusoids
x,E = proc.remove_ss_sinusoid(slapt.flatten()[:,None])
slapt_ss  = E@x
slaptrm   = slapt.flatten()-slapt_ss.squeeze() 


fig,ax = plt.subplots(1,1)
ax.plot(sla_5deg[:,klatss,klonss],label="After Spatial Smoothing",color='k')
ax.plot(slapt_ss,label="Estimated Cycle",color='red')
ax.plot(slaptrm.squeeze(),label="Deseasonalized Data", color='b')


#%% 4.5) Remove NaN points and Examine Low pass filter

slars = sla_lp.reshape(ntime,nlat5*nlon5)

# Locate only non-Nan points
okdata,knan,okpts = proc.find_nan(slars,0)
npts = okdata.shape[1]


# Quick check low pass filter transfer function
lpdata  = okdata.copy()
rawdata = sla_5deg.reshape(ntime,nlat5*nlon5)[:,okpts]
lpspec  = []
rawspec = []
npts5 = okdata.shape[1]
for i in tqdm(range(npts5)):
    X_spec,freq,_=tbx.bandavg_autospec(rawdata[:,i],dt,M,.05)
    X_lpspec,_,_ =tbx.bandavg_autospec(lpdata[:,i],dt,M,.05)
    lpspec.append(X_lpspec)
    rawspec.append(X_spec)
lpspec   = np.array(lpspec)
rawspec  = np.array(rawspec)

filtxfer = lpspec/rawspec

k24mon = np.argmin(np.abs(freq-xtk[1])) # Get index for 24 mon
if freq[k24mon] < xtk[1]: # less than 24 months
    ids = [k24mon,k24mon+1]
else:
    ids = [k24mon-1,k24mon]
p24 = np.zeros(npts5)
for it in tqdm(range(npts5)):
    p24[it] = np.interp(xtk[1],freq[ids],filtxfer[it,ids])
    

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
plotp24 = np.interp(xtk[1],freq[ids],filtxfer[:plotnum,:].mean(0)[ids]) #p24[:plotnum].mean()
ax.plot(freq,filtxfer[:plotnum,:].T,label="",color='b',alpha=0.05,zorder=-1)
ax.plot(freq,filtxfer[:plotnum,:].mean(0),label="",color='k',alpha=1)
ax.scatter(xtk[1],[plotp24],s=100,marker="x",color='k',zorder=1)
ax.set_ylim([0,1])
ax.set_xscale('log')
ax.set_xticks(xtk)
ax.set_xticklabels(xtkl)
ax.set_title("Filter Transfer Function (Filtered/Raw), %.3f" % (plotp24*100) +"%  at 24 months")
ax.grid(True,ls='dotted')
plt.suptitle("AVISO 5deg SSH Timeseries, %i-Band Average"% (M))

plt.tight_layout()
plt.savefig("%sFilter_Transfer_%imonLP_%ibandavg_AVISO.png"%(outfigpath,tw,M),dpi=200)



#%% Try Iteratively Removing Based on Number of Points

def cluster_ssh(sla,lat,lon,nclusters,distthres=3000,
                returnall=False):
    # Input: time x lat x lon
    # Dependencies: 
    #    numpy as np
    
    
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

    # Apply rules from Thompson and Merrifield
    # if uncert > 2, set to 2
    # if uncert <0.5, set to 0
    uncertout[uncertout>2]   = 2
    uncertout[uncertout<0.5] = 0 
    
    # -----------------------
    # Replace into full array
    # -----------------------
    clustered = np.zeros(nlat5*nlon5)*np.nan
    clustered[okpts] = clusterout
    clustered = clustered.reshape(nlat5,nlon5)
    cluster_count = []
    for i in range(nclusters):
        cid = i+1
        cnt = (clustered==cid).sum()
        cluster_count.append(cnt)
        print("Found %i points in cluster %i" % (cnt,cid))
    uncert = np.zeros(nlat5*nlon5)*np.nan
    uncert[okpts] = uncertout
    uncert = uncert.reshape(nlat5,nlon5)
    
    if returnall:
        return clustered,uncert,cluster_count,srho,scov,sdist,distance_matrix
    return clustered,uncert,cluster_count



def plot_results(clustered,uncert,expname):
    
    # Set some defaults
    ucolors = ('Blues','Purples','Greys','Blues','Reds','Oranges')
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
    
    # Plot Cluster Uncertainty
    fig1,ax1 = plt.subplots(1,1,subplot_kw={'projection':proj})
    ax1 = add_coast_grid(ax1)
    for i in range(nclusters):
        cid = i+1
        cuncert = uncert.copy()
        cuncert[clustered!=cid] *= np.nan
        ax1.pcolormesh(lon5,lat5,cuncert,vmin=0,vmax=2,cmap=ucolors[i],transform=ccrs.PlateCarree())
        #fig.colorbar(pcm,ax=ax)
    ax1.set_title("Clustering Output (nclusters=%i) %s "% (nclusters,expname))
    plt.savefig(outfigpath+"Cluster_with_Shaded_uncertainties_%s.png" % expname,dpi=200)
    return fig,ax,fig1,ax1
    



def elim_points(sla,lat,lon,nclusters,minpts,maxiter,distthres=3000):
    
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
        fig,ax,fig1,ax1 = plot_results(clustered,uncert,expname)
        
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


maxiter = 30
allclusters,alluncert,allcount,rempts = elim_points(sla_lp,lat5,lon5,6,30,5)


cmap2 = cm.get_cmap("jet",len(allcount)+1)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
ax = add_coast_grid(ax)
pcm = ax.pcolormesh(lon5,lat5,rempts,cmap=cmap2,transform=ccrs.PlateCarree())
fig.colorbar(pcm,ax=ax)
ax.set_title("Removed Points")
plt.savefig(outfigpath+"RemovedPoints_by_Iteration_remGMSL%i_tol%i.png"%(rem_gmsl,tol),dpi=200)
plt.pcolormesh(lon5,lat5,rempts)



# clustered1,uncert1,cluster_count1 = cluster_ssh(sla_lp,lat5,lon5,nclusters,distthres=3000)
# fig,ax,fig1,ax1 = plot_results(clustered,uncert,'test')



#%% Old Script: Step by step clustering

# Get map of correlation values
srho = np.corrcoef(okdata.T,okdata.T)
scov = np.cov(okdata.T,okdata.T)

#Debug
if debug: # Check to make sure the points are not missing
    print(np.max(srho[:npts,:npts] - srho[npts:,npts:]))
srho = srho[:npts,:npts]
scov = scov[:npts,:npts]

#%% 6) Calculate Distance matrix

# Make pairs of coordinates (need to check if this is done properly)
lonmesh,latmesh = np.meshgrid(lon5,lat5)
coords = np.vstack([lonmesh.flatten(),latmesh.flatten()]).T
coords = coords[okpts,:]

# # Check to see if pairwise setup is correct
# ktest=1222
# ktest2 = 1141
# testx,testy = coords[ktest]
# testx2,testy2 = coords[ktest2]

# klon,klat = proc.find_latlon(testx,testy,lon5,lat5)
# klon2,klat2 = proc.find_latlon(testx2,testy2,lon5,lat5)
# #print("Value retrieved is %f. Corresponding value is %f"%(sla_5deg[0,klat,klon],sla_5deg.reshape(ntime,nlat5*nlon5)[0,ktest]))
# print("Value retrieved is %f. Corresponding value is %f"%(sla_5deg[0,klat,klon],okdata[0,ktest]))


# # Check correlation values
# ptcorr  = np.corrcoef(okdata[:,ktest],okdata[:,ktest2])[0,1]
# matcorr = srho[ktest,ktest2]



# Another attempt (NOTE THIS ONE WORKED!)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
from sklearn.metrics.pairwise import haversine_distances
coords1 = coords.copy()
#coords1[np.where(coords1[:,0]>=180),0] = -1*(coords1[np.where(coords1[:,0]>=180)] - 180)[:,0]

coords2      = np.zeros(coords1.shape)
coords2[:,0] = np.radians(coords1[:,1]) # First point is latitude
coords2[:,1] = np.radians(coords1[:,0]) # Second Point is Longitude
sdist = haversine_distances(coords2,coords2) * 6371
#sdist = haversine_distances(coords2) * 6371000/1000
#coords[ktest],coords[ktest2]
#sdist[ktest,ktest2],sdist[ktest2,ktest]
plt.pcolormesh(sdist),plt.colorbar(),plt.title("Distance Matrix")


bsas = [-34.83333, -58.5166646]
paris = [49.0083899664, 2.53844117956]
pt2 = [-65,110]
pt2_in_radians = [np.radians(_) for _ in pt2]
bsas_in_radians = [np.radians(_) for _ in bsas]
paris_in_radians = [np.radians(_) for _ in paris]
nmery = np.array([bsas_in_radians, paris_in_radians,pt2_in_radians])
result = haversine_distances(nmery)
result * 6371000/1000  # multiply by Earth radius to get kilometers
#%% 7) Put together distance matrix

distthres= 3000
a_fac = np.sqrt(-distthres/(2*np.log(0.5))) # Calcuate so exp=0.5 when distance is 3000km
expterm = np.exp(-sdist/(2*a_fac**2))
distance_matrix = 1-expterm*srho

#%%

#% Visualize data from a point
#kpt = 1222



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

#%% Do Clustering!




nclusters = 6

cdist = squareform(distance_matrix,checks=False)

# Theoretical Attempt on scipy
linked = linkage(cdist,'weighted')

#plt.figure(figsize=(10, 7))
# dendrogram(linked,
#             orientation='top',
#             labels=np.ar ange(0,cdist.shape[0],1),
#             distance_sort='descending',
#             show_leaf_counts=True)

clusterout = fcluster(linked, nclusters,criterion='maxclust')

clustered = np.zeros(nlat5*nlon5)*np.nan
clustered[okpts] = clusterout
clustered = clustered.reshape(nlat5,nlon5)

for i in range(nclusters):
    cid = i+1
    print("Found %i points in cluster %i" % ((clustered==cid).sum(),cid))

#lon5_180,clustered = proc.lon360to180(lon5,clustered.T[:,:,None],autoreshape=True)
#clustered = clustered.squeeze().T

# Plot Result
cmap = cm.get_cmap("jet",nclusters)
lonf = 40
latf = -35
proj = ccrs.PlateCarree(central_longitude=180)
bbox = [9,60,-30,-45]
#bbox = [-180,180,-90,90]

fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
#ax  = viz.init_map([1,360,-90,90],ax=ax)
#ax = add_coast_grid(ax,bbox=bbox,proj=proj)


ax.add_feature(cfeature.COASTLINE,color='black',lw=0.75)
#ax.set_extent(bbox)
gl = ax.gridlines(ccrs.PlateCarree(central_longitude=0),draw_labels=True,
              linewidth=2, color='gray', alpha=0.5, linestyle="dotted",lw=0.75)
gl.xlabels_top = False
gl.ylabels_right = False
    

pcm = ax.pcolormesh(lon5,lat5,clustered,cmap=cmap,transform=ccrs.PlateCarree(central_longitude=0))#,cmap='Accent')#@,cmap='Accent')
plt.colorbar(pcm,ax=ax,orientation='horizontal')
#ax.scatter([lonf+180],[latf],s=200,c='k',marker='x',transform=proj)
ax.set_title("Clustering Results \n nclusters=%i"% (nclusters))
plt.savefig("%sCluster_results_n%i.png"%(outfigpath,nclusters),dpi=200,transparent=True)

#plt.pcolormesh(nlat5,nlon5,clustered)

plt.show()

#%% Quantify Uncertainty in each group


uncertout = np.zeros(clusterout.shape)
for i in tqdm(range(len(clusterout))):
    covpt     = scov[i,:]     # 
    cid       = clusterout[i] #
    covin     = covpt[np.where(clusterout==cid)]
    covout    = covpt[np.where(clusterout!=cid)]
    #break
    uncertout[i] = np.mean(covin)/np.mean(covout)
    

# Apply rules from Thompson and Merrifield
# if uncert > 2, set to 2
# if uncert <0.5, set to 0
uncertout[np.abs(uncertout)>2]   = 2
uncertout[np.abs(uncertout)<0.5] = 0 
#uncertout[uncertout>2]   = 2
#uncertout[uncertout<0.5] = 0 


# Remap
uncert = np.zeros(nlat5*nlon5)*np.nan
uncert[okpts] = uncertout
uncert = uncert.reshape(nlat5,nlon5)



fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax = add_coast_grid(ax)
pcm=plt.pcolormesh(lon5,lat5,uncert,vmin=0,vmax=2,cmap='copper',transform=ccrs.PlateCarree())
#pcm=plt.pcolormesh(lon5,lat5,uncert,cmap='copper',transform=ccrs.PlateCarree())
ax.set_title(r"Uncertainty $(<\sigma^{2}_{out,x}>/<\sigma^{2}_{in,x}>)$")
fig.colorbar(pcm,ax=ax,fraction=0.02)
plt.savefig(outfigpath+"Uncertainty.png",dpi=200)



# Separate clusters out by section
ucolors = ('Blues','Purples','Greys','Blues','Reds','Oranges')
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax = add_coast_grid(ax)
for i in range(nclusters):
    
    cid = i+1
    print(cid)
    cuncert = uncert.copy()
    cuncert[clustered!=cid] *= np.nan
    print("Found %i points in cluster %i" % ((clustered==cid).sum(),cid))
    #cuncert[~np.isnan(cuncert)]=2
    ax.pcolormesh(lon5,lat5,cuncert,vmin=0,vmax=2,cmap=ucolors[i],transform=ccrs.PlateCarree())
    #fig.colorbar(pcm,ax=ax)
    
#ax.contour(lon5,lat5,clustered,colors='w')
ax.set_title("Clustering Output (nclusters=%i)"%nclusters)
plt.savefig(outfigpath+"Cluster_with_Shaded_uncertainties.png",dpi=200)





