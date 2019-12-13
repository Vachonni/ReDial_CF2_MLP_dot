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


"""

            TREAT DATA
            
"""



#%%

PATH_from = '/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDialML/'
files = [
         'Train_UNIQUE.csv',
         'Train_LIST.csv',
         'Train100.csv',
         'Val_UNIQUE.csv',
         'Val_LIST.csv',
         'Val100.csv',
         'Test_UNIQUE.csv',
         'Test_LIST.csv',
         'Test100.csv'
        ]
PATH_to = '/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/'


#%%


for file in files:
    
    # Load .csv in DataFrame 
    df = pd.read_csv(PATH_from+file)
    
    # Changer heathers name 
    df.rename(columns={'movie_UiD':'movie_ReDOrId'}, inplace=True)
    
    # Change column name of UiD to ReDOrId
    data_np = df.to_numpy()
    
    # Change UiDs to ReDOrId
    for i in range(len(data_np)):
        # If it's a UNIQUE or 100:
        if isinstance(data_np[i,4], int):
            data_np[i,4] = UiD_2_ReDOrId[data_np[i,4]]
        # If it's the LIST case:
        if isinstance(data_np[i,4], list):
            data_np[i,4] = [UiD_2_ReDOrId[item_id] for item_id in data_np[i,4]]
            
    # Save the new file
    new_df = pd.DataFrame(data=data_np,    # values
                          index=df.index,  # 1st column as index
                          columns=df.columns)  # 1st row as the column names
    new_df.to_csv(PATH_to+file, index=False)
    
    
#%%    
    

"""

            TREAT RELATIONAL TABLES (RT)

            
"""

#%%

import torch

RT = torch.load('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDialML/embed_MovieTitlesGenres_with_BERT_avrg.pt')


#%%

new_RT = torch.zeros(len(UiD_2_ReDOrId), RT.shape[1])

for k, v in UiD_2_ReDOrId.items():
    new_RT[v] = RT[k]

#%%
    
torch.save(new_RT, '/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/embed_MovieTitlesGenres_with_BERT_avrg.pt')












































