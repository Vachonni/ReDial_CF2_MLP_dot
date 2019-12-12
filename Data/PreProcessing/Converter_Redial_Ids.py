#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:22:45 2019

@author: nicholas
"""

import numpy as np

dict_movies_ReDOrId = np.load('/Users/nicholas/ReDial_CF2_MLP_dot/Data/PreProcessing/dict_movie_full_by_ReDOrId.npy', allow_pickle=True).item()


#%%

ReDiD_2_ReDOrId = {}
ReDOrId_2_ReDiD = {}

for k,v in dict_movies_ReDOrId.items():
    ReDOrId_2_ReDiD[k] = v['ReDiD']
    ReDiD_2_ReDOrId[v['ReDiD']] = k
    
    
#%%
    
np.save('/Users/nicholas/ReDial_Utils/ReDiD_2_ReDOrId.npy', ReDiD_2_ReDOrId)    
np.save('/Users/nicholas/ReDial_Utils/ReDOrId_2_ReDiD.npy', ReDOrId_2_ReDiD)