#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Reading the Rousseeuw Paper  and trying out some figures to get a sense of the
Silhouette metric
Created on Mon May 17 18:07:11 2021

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt

a = np.arange(.01,2.01,.01)
b = 1
b2 = np.ones(a.shape)*b

rat = a/b

hoho = a.copy() * 0
hoho[rat<1] = 1 - a[rat<1]/b
hoho[rat==1] = 0
hoho2  =hoho.copy()
hoho2[rat>1] = 1 - a[rat>1]/b
hoho[rat>1] = b/a[rat>1] - 1
s = hoho.copy()


sform = (b-a)/np.maximum(a,b)

# fig,ax = plt.subplots(1,1,figsize=(4,3))
# ax.plot(a,s,label="Silhouette Coefficient")
# ax.plot(a,rat,label=r"$a_i/b_i$")
# #ax.plot(a,sform,label="Formula")
# #ax.plot(a,hoho2,label="Silhouette (Weighting Misclassified Heavier)")
# ax.axvline(b,label="$b_i$",color='k',ls='dashed')
# ax.legend(fontsize=8)
# ax.set_xlabel("$a_i$")
# ax.set_ylabel("Silhouette Coefficient")
# plt.tight_layout()
# plt.savefig("/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210519/silhouette_example.png",dpi=150)



fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.plot(rat,s,label="Silhouette Coefficient")
#ax.plot(a,rat,label=r"$a_i/b_i$")
#ax.plot(a,sform,label="Formula")
#ax.plot(a,hoho2,label="Silhouette (Weighting Misclassified Heavier)")
ax.axvline(1,label="",color='k',ls='dashed')
ax.legend(fontsize=8)
ax.set_xlabel("$a_i/b_i$")
ax.set_ylabel("Silhouette Coefficient")
plt.tight_layout()
plt.savefig("/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/02_Figures/20210519/silhouette_example.png",dpi=150)

