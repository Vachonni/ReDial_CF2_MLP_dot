#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:10:00 2019


From ReDialML data to ReDial only


@author: nicholas
"""



import numpy as np
import pandas as pd

UiD_2_ReDOrId = np.load('/Users/nicholas/ReDial_Utils/UiD_2_ReDOrId.npy', allow_pickle=True).item()


#%%

PATHS =[
        '/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDialML/Train_UNIQUE.csv'
        ]

#%%


for path in PATHS:
    
    # Load .csv in DataFrame 
    df = pd.read_csv(path)
    
    # Changer heathers name 
    df.rename(columns={'movie_UiD':'movie_ReDOrId'}, inplace=True)
    
    # Change column name of UiD to ReDOrId
    data_np = df.to_numpy()
    
    # Change UiDs to ReDOrId
    for i in range(len(data_np)):
        print(i)
        data_np[i,4] = UiD_2_ReDOrId[data_np[i,4]]
            
    # Save the new file
    new_df = pd.DataFrame(data=data_np,    # values
                          index=df.index,  # 1st column as index
                          columns=df.comlums)  # 1st row as the column names
    
    
#%%    