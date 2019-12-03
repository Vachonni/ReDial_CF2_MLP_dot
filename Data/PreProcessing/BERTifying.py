#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 07:45:39 2019


Turning text into averaged BERT representations


@author: nicholas
"""


# Get BERT averaged representation of texts in RT

import pandas as pd
import torch
from transformers import BertConfig, BertModel, BertTokenizer
 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE = ', DEVICE )


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased')
model.to(DEVICE)


#df_user = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/user_chrono_RT.csv')
#df_item = pd.read_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/movie_genres_RT.csv')


#%%


def Text_in_BERT(text, tokenizer=tokenizer, max_length=512):
    """
    From a string to a BERT input.
    """
    
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
                                      
    # Turn into torch tensors
    encoded_dict['input_ids'] = torch.tensor([encoded_dict['input_ids']]).long().to(DEVICE)
    encoded_dict['special_tokens_mask'] = torch.tensor([encoded_dict['special_tokens_mask']])
    
    # Add 'attention mask' tokens (it's the reverse of 'special_tokens_mask')
    encoded_dict['attention_mask'] = ((encoded_dict['special_tokens_mask'] -1) * -1).float().to(DEVICE)
    
    # Remove 'special_tokens_mask' from dict so it's ready for BERT
    encoded_dict.pop('special_tokens_mask')
    
    # Add token_type and position_ids 
    encoded_dict['token_type_ids'] = torch.zeros(1, max_length).long().to(DEVICE)
    encoded_dict['position_ids'] = torch.arange(max_length).unsqueeze(0).to(DEVICE)
    
    return encoded_dict
    



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


# Trying stuff



#
#
#
#
#
#for text in df_user['text'].head():
#    print(tokenizer.tokenize(text))
#
#
#
##%%
#
#
#
#tokenizer.encode_plus(['Have a nice day'], add_special_tokens=True, max_length=5)
#
##%%
#
#tokenizer.convert_ids_to_tokens(102)
#
##%%
#
#tokenizer.convert_tokens_to_ids(['PAD'])
#
#
#
##%%
#
#inp = text_to_BERT('Have a nivce day')
#
#model(**inp)
##%%
#model(torch.tensor([ 101,  2031,  1037,  9152, 25465,  2063,  2154,   100,   100,   100]).unsqueeze(0))[0]
#
##%%
#
#tokenizer.encode_plus('Have a nivce day', max_length=512)
#
##%%
#
#trr = text_to_BERT('Have a nivce day')
#
#for k,v in trr.items():
#    print(v.shape)
#
#print(trr)
#























