#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 07:45:39 2019


Turning text into averaged BERT representations


@author: nicholas
"""


# Get BERT averaged representation of texts in RT

# import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE = ', DEVICE )


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print('tokenizer loaded')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(DEVICE)
print('model loaded')

# df_user = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/user_chrono_RT.csv')
# df_item = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/movie_genres_RT.csv')


#%%


def Text_in_BERT(text, tokenizer=tokenizer, max_length=512):
    """
    From a string to a BERT input.
    The outpur is in a batch 1 format. (one line matrix format)
    
    NOTE: WHEN ONLY ONE SPEAKER:
        'position_ids' and 'token_type_ids' can be removed 
    """
    
    input_dict = {}
    encoded_dict = tokenizer.encode_plus(text, max_length=max_length)
    
    # Get the actual lenght of the input
    length = len(encoded_dict['input_ids'])
    qt_to_add = max_length - length 
    
    # Prepare inputs we'll add ('PAD' = token 100, with a special mask token)
    PAD_to_add = [100] * qt_to_add
    masks_to_add = [1] * qt_to_add
    
    # Add them
    encoded_dict['input_ids'] = encoded_dict['input_ids'] + PAD_to_add
    encoded_dict['special_tokens_mask'] = encoded_dict['special_tokens_mask'] + \
                                          masks_to_add                                    
                                      
    # Turn into torch tensors and add 'inputs_ids' to input_dict
    input_dict['input_ids'] = torch.tensor([encoded_dict['input_ids']]).long().to(DEVICE)
    encoded_dict['special_tokens_mask'] = torch.tensor([encoded_dict['special_tokens_mask']])
    
    # Add 'attention mask' tokens (it's the reverse of 'special_tokens_mask')
    input_dict['attention_mask'] = ((encoded_dict['special_tokens_mask'] -1) * -1).float().to(DEVICE)
    
    
    # Add token_type and position_ids 
    input_dict['token_type_ids'] = torch.zeros(1, max_length).long().to(DEVICE)
    input_dict['position_ids'] = torch.arange(max_length).unsqueeze(0).to(DEVICE)
    
    return input_dict
    



def BERT_avrg(last_hidden_layer):
    """
    From BERT last hidden layer to average of it.
    """
    
    return last_hidden_layer.mean(dim=1)
    



def Text_to_BERT_avrg(text, model=model, tokenizer=tokenizer, max_length=512):
    """
    From a string to BERT last hidden layer averaged 
    """
    
    input_to_bert = Text_in_BERT(text, tokenizer, max_length)
    last_hidden_layer = model(**input_to_bert)[0]
    avrg_last_hidden_layer = BERT_avrg(last_hidden_layer)
    
    return avrg_last_hidden_layer
    


#%%
    

# Example of usage: 

# [Text_to_BERT_avrg(text) for text in df_user['text'].head()]



#%%
    

import numpy as np

# We now convert all the str of ML - titles & genres

movie_str_ML = np.load('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDialML/str_MovieTitlesGenres_RT.npy',\
                       allow_pickle=True)

BERT_input_MovieTitlesGenres_dict = {}
    
for k, text in enumerate(movie_str_ML):
    if k == 0: print(text)
    # text is a list, so take first argument
    input_dict = Text_in_BERT(text[0])
    # Flatten the tensors in input dict
    input_dict = {k:v[0] for k,v in input_dict.items()}
    input_dict.pop('token_type_ids')
    input_dict.pop('position_ids')
    BERT_input_MovieTitlesGenres_dict[k] = input_dict
 #   if k == 0:break


np.save('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDialML/BERT_input_MovieTitlesGenres_dict.npy', \
        BERT_input_MovieTitlesGenres_dict)
    
    
    
#%%
    
    
# We now convert all the str of ReDial - titles & genres

movie_str = np.load('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/str_MovieTitlesGenres_RT.npy',\
                       allow_pickle=True)

BERT_input_MovieTitlesGenres_dict = {}
    
for k, text in enumerate(movie_str):
    if k == 0: print(text)
    # text is a list, so take first argument
    input_dict = Text_in_BERT(text[0])
    # Flatten the tensors in input dict
    input_dict = {k:v[0] for k,v in input_dict.items()}
    input_dict.pop('token_type_ids')
    input_dict.pop('position_ids')
    BERT_input_MovieTitlesGenres_dict[k] = input_dict
 #   if k == 0:break


np.save('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/BERT_input_MovieTitlesGenres_dict.npy', \
        BERT_input_MovieTitlesGenres_dict)
    
#%%

    
# We now convert all the str of ReDial - Abstract

movie_str = np.load('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/str_MovieAbstract_RT.npy',\
                       allow_pickle=True)

BERT_input_MovieAbstract_dict = {}
    
for k, text in enumerate(movie_str):
    if k == 0: print(text)
    # text is a list, so take first argument
    input_dict = Text_in_BERT(text[0])
    # Flatten the tensors in input dict
    input_dict = {k:v[0] for k,v in input_dict.items()}
    input_dict.pop('token_type_ids')
    input_dict.pop('position_ids')
    BERT_input_MovieAbstract_dict[k] = input_dict
  #  if k == 0:break


np.save('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/BERT_input_MovieAbstract_dict.npy', \
        BERT_input_MovieAbstract_dict)

    
    
#%%
    
    
# We now convert all the str of users_chrono

user_str = np.load('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDialML/str_UserChrono_RT.npy',\
                       allow_pickle=True)

BERT_input_UserChrono_dict = {}
    
for k, text in enumerate(user_str):
    if k == 0: print(text)
    # text in not a list
    input_dict = Text_in_BERT(text)
    # Flatten the tensors in input dict
    input_dict = {k:v[0] for k,v in input_dict.items()}
    input_dict.pop('token_type_ids')
    input_dict.pop('position_ids')
    BERT_input_UserChrono_dict[k] = input_dict
  #  if k == 0:break


#%%
np.save('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/BERT_input_UserChrono_dict.npy', \
        BERT_input_UserChrono_dict)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    