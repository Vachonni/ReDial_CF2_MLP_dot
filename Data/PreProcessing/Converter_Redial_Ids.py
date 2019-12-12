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
    dict_movies_full_by_ReDiD[v['ReDiD']]
    
    
#%%
    
np.save('/Users/nicholas/ReDial_Utils/ReDiD_2_ReDOrId.npy', ReDiD_2_ReDOrId)    
np.save('/Users/nicholas/ReDial_Utils/ReDOrId_2_ReDiD.npy', ReDOrId_2_ReDiD)


#%%

dict_movies_full_by_ReDiD = {}

for k,v in dict_movies_ReDOrId.items():
    # Add values of ReDOrId
    v['ReDOdId'] = k
    # New key will be ReDiD
    new_k = v['ReDiD']
    # Remove ReDiD from values
    v.pop('ReDiD')
    # Associate new key with reworked values
    dict_movies_full_by_ReDiD[new_k] = v 


#%%
    
    
np.save('/Users/nicholas/ReDial_CF2_MLP_dot/Data/PreProcessing/dict_movies_full_by_ReDiD.npy', dict_movies_full_by_ReDiD)    