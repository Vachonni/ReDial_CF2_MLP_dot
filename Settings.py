#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:38:14 2018


All constants for data pre-processing


@author: nicholas
"""

import numpy as np


# Number of movies treated in total (MovieLens + ReDial)
# Corresponds to last entry in '/Users/nicholas/ReDial/DataRaw/MoviesMergedId_updated181030.csv'
# or equivalently, the lenght of it
nb_movies_in_total = 48271



# List of genres used initially (for presentations and stats...)
genres = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 
          'documentary', 'drama', 'family', 'fantasy', 'noir', 'history', 
          'horror', 'music', 'musical', 'mystery', 'romance', 'sci-fi', 'scifi',
          'short', 'sport', 'superhero', 'thriller', 'war', 'western', 'kid',]



# List of genres used in ML
ML_genres = ['animation', 'adventure', 'drama', 'thriller', 'action', 'fantasy', 
             'mystery', 'horror', 'sci-fi', 'comedy', 'crime', 'romance', 'war', 
             'documentary', 'children', 'musical', 'western', 'imax', 'film-noir']



# List of all UiD of movies mentioned in ReDial

# Get the dict of conversions
ReDiD2UiD = np.load('./Data/ReDID2uID.npy').item()

l_ReDUiD = []
for value in ReDiD2UiD.values():
    l_ReDUiD.append(value)