#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:44:38 2019


Adding the BERT averaged representation to the RT


@author: nicholas
"""


import pandas as pd
import torch

from BERTifying import Text_to_BERT_avrg

print('now Bertify is loaded')

if torch.cuda.is_available():
    df_user = pd.read_csv('/home/vachonni/scratch/ReDial_CF2_MLP_dot/Data/user_chrono_RT.csv')
    df_item = pd.read_csv('/home/vachonni/scratch/ReDial_CF2_MLP_dot/Data/movie_genres_RT.csv')
else:    
    df_user = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/user_chrono_RT.csv')
    df_item = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/movie_genres_RT.csv')


#%%


# BERT_avrg = [Text_to_BERT_avrg(text) for text in df_user['text']]

BERT_avrg = []
with torch.no_grad():
    
    for i, text in enumerate(df_user['text']):
        print(i)
        BERT_avrg.append(Text_to_BERT_avrg(text))
        

#%%

df_user['BERT_user'] = BERT_avrg

df_user.to_csv('user_chrono_RT_BERT_avrg.csv', index=False)