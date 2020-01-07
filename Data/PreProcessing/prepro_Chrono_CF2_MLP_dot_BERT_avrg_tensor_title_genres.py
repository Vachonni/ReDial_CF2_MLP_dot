#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:44:38 2019


Creating the BERT averaged representation of title + 'Genres: ' + genres in a tensor


@author: nicholas
"""


import pandas as pd
import torch

from BERTifying import Text_to_BERT_avrg

print('now Bertify is loaded')

if torch.cuda.is_available():
    df_user = pd.read_csv('/home/vachonni/scratch/ReDial_CF2_MLP_dot/Data/PreProcessing/user_chrono_RT.csv')
    df_item = pd.read_csv('/home/vachonni/scratch/ReDial_CF2_MLP_dot/Data/PreProcessing/movie_genres_RT.csv')
else:    
    df_user = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/PreProcessing/user_chrono_RT.csv')
    df_item = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/PreProcessing/movie_genres_RT.csv')


#%%


# Treat users
print('USERS')

BERT_avrg = torch.zeros(len(df_user.index), 768)
with torch.no_grad():
    
    for i, text in enumerate(df_user['text']):
        if i % 1000 == 0: print(i)
        BERT_avrg[i] = Text_to_BERT_avrg(text)
        

torch.save(BERT_avrg, 'embed_UserChrono_with_BERT_avrg.pt')


#%%

# Treat items
print('ITEMS')

BERT_avrg = torch.zeros(len(df_item.index), 768)
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
        BERT_avrg[i] = Text_to_BERT_avrg(title + '. ' + genres)


torch.save(BERT_avrg, 'embed_MovieTitlesGenres_with_BERT_avrg.pt')  
