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


## Treat users
#print('USERS')
#
#BERT_avrg = []
#with torch.no_grad():
#    
#    for i, text in enumerate(df_user['text']):
#        if i % 1000 == 0: print(i)
#        BERT_avrg.append(Text_to_BERT_avrg(text))
#
#
#df_user['BERT_avrg'] = BERT_avrg
#
#df_user.to_csv('user_chrono_RT_BERT_avrg.csv', index=False)


#%%

# Treat items
print('ITEMS')

BERT_avrg = []
with torch.no_grad():
    
    for i, (title, genres) in enumerate(zip(df_item['title'], df_item['genres'])):
        if i % 1000 == 0: print(i)
        # If no genres were found for the movie, just leave a blank space
        if isinstance(title, float): 
            print(title)
            title=str(title)
        if isinstance(genres, float): 
            genres=''
        else: 
            genres='Genres: ' + genres
        BERT_avrg.append(Text_to_BERT_avrg(title + '. ' + genres))

    
df_item['BERT_avrg'] = BERT_avrg

df_item.to_csv('movie_genres_RT_BERT_avrg.csv', index=False)
