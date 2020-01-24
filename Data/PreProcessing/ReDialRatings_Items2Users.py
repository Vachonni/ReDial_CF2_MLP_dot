#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:14:23 2020


FROM: list of ReDial Ratings Users (ConvID) to Items (UiD)  
      ex: [[6950, [4788, 1426, 605], [1, 1, 1]], ...          - This is user 6950

TO: List of ReDial Ratings Items (ReDOrId) to Users (ConvOrId)  
    ex: [[0, [3444, 89, 7736], [1, 0, 1]],...}               - This is for one movies


    * Data is not Chrono
    * A users is a equivalent to a Conversation

@author: nicholas
"""

import numpy as np
import pickle


#%%


# Load converter (dict) from UiD to ReDOrId

Uid2ReDOrId = np.load('/Users/nicholas/ReDial_Utils/UiD_2_ReDOrId.npy', allow_pickle=True).item()


#%%

# Load all data (train-valid-test)...


with open('/Users/nicholas/ReDial/DataProcessed/ReDialRatings2ListTRAIN.txt', 'rb') as f:
    u2i_train = pickle.load(f)

with open('/Users/nicholas/ReDial/DataProcessed/ReDialRatings2ListVALID.txt', 'rb') as f:
    u2i_val = pickle.load(f)
    
with open('/Users/nicholas/ReDial/DataProcessed/ReDialRatings2ListTEST.txt', 'rb') as f:
    u2i_test = pickle.load(f)
    
#%%
    
# ...and merge. 
#For each movie, we need all the user's ratings. We'll then split train_val_test by movies.
    
u2i = u2i_train + u2i_val + u2i_test
    

#%%

# Initialize the dictionary for all movies with ReDOrId indices 
# (easier to always append than to check if key exists first)

dict_ReDial_ratings_ReDOrId2ConvOrId = {}

for i in range(7013):
    dict_ReDial_ratings_ReDOrId2ConvOrId[i] = [[],[]]



#%%

# Init converters for Conversation (users ID)
ConvID2ConvOrId = {}
ConvOrId2ConvID = {}


# For all conversation (users)...    
for ConvOrId, (ConvID, l_UiD, l_ratings) in enumerate(u2i):
    
    # ...and all movies rated by that user (conversation)
    for i in range(len(l_UiD)):
        # Convert UiD to ReDOrId
        ReDOrId =  Uid2ReDOrId[l_UiD[i]]
        # Get the rating
        rating = l_ratings[i]
        # Associate the (user, rating) for the right movie 
        dict_ReDial_ratings_ReDOrId2ConvOrId[ReDOrId][0].append(ConvOrId)
        dict_ReDial_ratings_ReDOrId2ConvOrId[ReDOrId][1].append(rating)
    
    # Update converters
    ConvID2ConvOrId.update({ConvID: ConvOrId})
    ConvOrId2ConvID.update({ConvOrId: ConvID})
   
    
    
#%%

# Some movies were mentioned but not rated (rated with a 2, meaning user didn't know the rating)

dict_ReDial_ratings_ReDOrId2ConvOrId_RATED = {}
dict_ReDial_ratings_ReDOrId2ConvOrId_NOT_RATED = {}
# To get the max quantity of ratings (usefull with fast.ai)
max_ratings = 0

for k,v in dict_ReDial_ratings_ReDOrId2ConvOrId.items():
    if v[0]==[]: 
        dict_ReDial_ratings_ReDOrId2ConvOrId_NOT_RATED.update({k:v})
    else:
        dict_ReDial_ratings_ReDOrId2ConvOrId_RATED.update({k:v})
        # Update the max_rating if bigger than acual
        qt_ratings = len(v[1])
        if qt_ratings > max_ratings: 
            max_ratings = qt_ratings
            most_rated = k
    
#%%
        
# Among the rated, split 80-10-10 (train-valid-test)

qt_data = len(dict_ReDial_ratings_ReDOrId2ConvOrId_RATED)

train_id_max = int(qt_data*0.8)
val_id_max = int(qt_data*0.9)

arr = np.arange(qt_data)
np.random.shuffle(arr) 

train_ids = arr[:train_id_max]
val_ids = arr[train_id_max:val_id_max]
test_ids = arr[val_id_max:]


#%%

# Turn dict into lists (to get wanted output)

list_ReDial_ratings_ReDOrId2ConvOrId = [ [k,v[0], v[1]] for k,v in \
                                        dict_ReDial_ratings_ReDOrId2ConvOrId_RATED.items()]

#%%
    
# Turn into numpy array for multiple index selection
arr_ReDial_ratings_ReDOrId2ConvOrId = np.array(list_ReDial_ratings_ReDOrId2ConvOrId)
    
    
#%%

list_ReDial_ratings_ReDOrId2ConvOrId_TRAIN = arr_ReDial_ratings_ReDOrId2ConvOrId[train_ids].tolist()
np.save('/Users/nicholas/ReDial_Utils/list_ReDial_ratings_ReDOrId2ConvOrId_TRAIN.npy', \
        list_ReDial_ratings_ReDOrId2ConvOrId_TRAIN)    
    
#%%

list_ReDial_ratings_ReDOrId2ConvOrId_VAL = arr_ReDial_ratings_ReDOrId2ConvOrId[val_ids].tolist()
np.save('/Users/nicholas/ReDial_Utils/list_ReDial_ratings_ReDOrId2ConvOrId_VAL.npy', \
        list_ReDial_ratings_ReDOrId2ConvOrId_VAL)  
    
#%%

list_ReDial_ratings_ReDOrId2ConvOrId_TEST = arr_ReDial_ratings_ReDOrId2ConvOrId[test_ids].tolist()
np.save('/Users/nicholas/ReDial_Utils/list_ReDial_ratings_ReDOrId2ConvOrId_TEST.npy', \
        list_ReDial_ratings_ReDOrId2ConvOrId_TEST)      
    
    
    
#%%


# Save converters

np.save('/Users/nicholas/ReDial_Utils/ConvID2ConvOrId.npy', ConvID2ConvOrId)
np.save('/Users/nicholas/ReDial_Utils/ConvOrId2ConvID.npy', ConvOrId2ConvID)



#%%
    














    
    # TESTS

#%%
    
ReDOrId2ReDiD = np.load('/Users/nicholas/ReDial_Utils/ReDOrId_2_ReDiD.npy', allow_pickle=True).item()
    

#%%

print(ReDOrId2ReDiD[no_mention[4]])    
    
    
#%%

for k,v in dict_ReDial_ratings_ReDOrId2ConvOrId.items():
    if 0 in v[1]: print(k)
    
    
    
#%%

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    