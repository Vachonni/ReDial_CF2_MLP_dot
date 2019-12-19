#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:27:26 2019


FOR BRET TRAINING IN CF2 - Creating str arrays for movies and user_choro


@author: nicholas
"""

import pandas as pd
import numpy as np





###### USERS


# Get user_chono_RT and turn it into a Numpy Array
users_RT = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/PreProcessing/user_chrono_RT.csv').to_numpy()
# Save it without the first column 
np.save('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDialML/str_UserChrono_RT.npy', users_RT[:,1])


#%%


####### MOVIES - REDIAL ONLY CASE



# Get the movie_dict of REDIAL 
movie_dict = np.load('/Users/nicholas/ReDial_CF2_MLP_dot/Data/PreProcessing/dict_movie_REDIAL_by_ReDOrId.npy', \
                     allow_pickle=True).item()

# Create the Abstract Numpay Array
str_MovieAbstract_RT = np.zeros((len(movie_dict),1), dtype=object)

for k,v in movie_dict.items():
    str_MovieAbstract_RT[k] = v['abstract']

np.save('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/str_MovieAbstract_RT.npy', str_MovieAbstract_RT)    


#%%



####### MOVIES - REDIAL + ML



# Get the movie_genres_RT and turn it into a Numpy Array
movie_RT = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/PreProcessing/movie_genres_RT.csv').to_numpy()

# Get conversion dict from UiD to ReDOrId:
UiD_2_ReDOrId = np.load('/Users/nicholas/ReDial_Utils/UiD_2_ReDOrId.npy', allow_pickle=True).item()

# Create the movie "Title: ... Genres:..." Numpay array
str_MovieTitlesGenres_RT_ALL = np.zeros((len(movie_RT),1), dtype=object)
# Create the movie "Title: ... Genres:..." Numpay array
str_MovieTitlesGenres_RT = np.zeros((len(UiD_2_ReDOrId),1), dtype=object)

for i in range(len(movie_RT)):
    if isinstance(movie_RT[i,1], str):
        title = movie_RT[i,1]
    else:
        title = ''
    if isinstance(movie_RT[i,2], str):
        genres = movie_RT[i,2]
    else:
        genres = ''
    str_MovieTitlesGenres_RT_ALL[i] = 'Title: ' + title + '.  Genres: ' + genres
    if i in UiD_2_ReDOrId:
        str_MovieTitlesGenres_RT[UiD_2_ReDOrId[i]] = 'Title: ' + title + '.  Genres: ' + genres

        
np.save('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/str_MovieTitlesGenres_RT.npy', str_MovieTitlesGenres_RT)
np.save('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDialML/str_MovieTitlesGenres_RT.npy', str_MovieTitlesGenres_RT_ALL)