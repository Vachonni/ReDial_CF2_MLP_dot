#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:04:28 2019


In prepraration for CF2 evalutation of new movies.

Finding "new movies" in targets of valid dataset
i.e. never mentionned in train data movies.


@author: nicholas
"""


import pandas as pd
from ast import literal_eval


# Load train and valid data
df_train_data = pd.read_csv("/Users/nicholas/ReDial_CF2/Data/ChronoTextSRGenres/Train.csv") 
df_valid_data = pd.read_csv("/Users/nicholas/ReDial_CF2/Data/ChronoTextSRGenres/Val.csv")

#%%

# Get all the movies mentionned in train
movies_in_train = {-1}
#count = 1

for str_ratings in df_train_data['ratings']:
    l_ratings = literal_eval(str_ratings)
    for (UiD, rating) in l_ratings[1:]:  # not taking -2, which indicate nb_movies_mentionned
        if UiD == -1: break
        movies_in_train.add(UiD)
#    count += 1
#    if count > 3: break

movies_in_train.discard(-1)
print(len(movies_in_train))


#%%

df_new_movies_liked = pd.DataFrame(columns = ['ConvID', 'text', 'ratings'])
df_new_movies_NOT_liked = pd.DataFrame(columns = ['ConvID', 'text', 'ratings'])

# Get all movies in targets of valid not in train
for ConvID, text, str_ratings in df_valid_data[['ConvID', 'text', 'ratings']].values:
    l_ratings = literal_eval(str_ratings)
    UiD, rating = l_ratings[1] # We check the next rating. 0 is nb movies mentionned
    if UiD not in movies_in_train:
        new_row = [ConvID, text, str({'UiD': UiD, 'qt_movie_mentionned': l_ratings[0][1]})]
        if rating == 1:
            df_new_movies_liked.loc[len(df_new_movies_liked)] = new_row
        if rating == 0:
            df_new_movies_NOT_liked.loc[len(df_new_movies_liked)] = new_row
    
#%%
            

df_new_movies_liked.to_csv('/Users/nicholas/ReDial_CF2/Data/new_movies_liked.csv', index=False)
df_new_movies_NOT_liked.to_csv('/Users/nicholas/ReDial_CF2/Data/new_movies_NOT_liked.csv', index=False)










