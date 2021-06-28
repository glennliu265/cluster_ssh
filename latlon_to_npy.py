#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:56:00 2021

@author: gliu
"""


from scipy.io import loadmat
import numpy as np
ld = loadmat('/home/glliu/01_Data/CESM1_LATLON.mat')

lon = ld['LON'].squeeze()
lat = ld['LAT'].squeeze()

np.savez("/home/glliu/01_Data/cesm_latlon360.npz",**{
    'lon':lon,
    'lat':lat})
