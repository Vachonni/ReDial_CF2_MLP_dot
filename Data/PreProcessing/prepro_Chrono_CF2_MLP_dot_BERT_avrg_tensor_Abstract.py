#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:44:38 2019


Creating the BERT averaged representation of Abstract in a tensor


@author: nicholas
"""


import numpy as np
import torch

from BERTifying import Text_to_BERT_avrg

print('now Bertify is loaded')

if torch.cuda.is_available():
    PATH = '/home/vachonni/scratch/ReDial_CF2_MLP_dot/Data/'
else:    
    PATH = '/Users/nicholas/ReDial_CF2_MLP_dot/Data/'
    
dict_movies = np.load(PATH + 'PreProcessing/dict_movie_full_by_UiD.npy', \
                      allow_pickle=True).item()




#%%

# Treat items
print('ITEMS')

BERT_avrg = torch.zeros(48272, 768)

with torch.no_grad():
    count = 0
    
    for k, v in dict_movies.items():
        if count % 1000 == 0: print(count, k)
        BERT_avrg[k] = Text_to_BERT_avrg(v['abstract'])
        count += 1
        if count > 2: break
    

#%%
        
torch.save(BERT_avrg, PATH + 'item_BERT_avrg_Abstract.pt')  
