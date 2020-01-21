#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 07:45:39 2019


Turning BERT inputs into BERT pooler representations... 


        ...FINE TUNED with BertForSequenceClassification used for Recommendation
                                                                (Softmax on 48272)


@author: nicholas
"""


# Get BERT pooler value from pre-trained BERTReco

import numpy as np
import torch
from transformers import BertForSequenceClassification
 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE = ', DEVICE )

# Load on local or Compute Canada
if DEVICE == 'cpu':
    model = BertForSequenceClassification.from_pretrained('/Users/nicholas/ReDial_A19/Results/ChronoTextSRGenres_W20/model_out')
else:
    model = BertForSequenceClassification.from_pretrained('/home/vachonni/scratch/ReDial_A19/Results/ChronoTextSRGenres_W20/model_out')

model.to(DEVICE)
model.eval()
print('model loaded')


#%%

# Split the different children of the model

generator_of_models_modules = model.named_children()

dict_of_models_modules = {}

for name, module in generator_of_models_modules:
    dict_of_models_modules[name] = module

    
#%%

# Load data. BERT inputs format of user Chrono

if DEVICE == 'cpu':    
    data_user = np.load('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/BERT_input_UserChrono_dict.npy', \
                    allow_pickle=True).item()
else:
    data_user = np.load('/home/vachonni/scratch/ReDial_CF2_MLP_dot/Data/DataReDial/BERT_input_UserChrono_dict.npy', \
                    allow_pickle=True).item()     


#%%
    
def batch(d):
    """
    But dict of tensor in 1D to (1,-1) dimension 
    for BERT input of batch size 1 equivalent
    Also put on right DEVICE
    """
    for k,v in d.items():
        d[k] = d[k].view(1,-1).to(DEVICE)
        
    return d    
    
#%%    

# Treat all data and put result in a RT (torch tensor)

embed_UserChrono_BERTReco_FineTuned = torch.zeros(len(data_user), 768)

for k, v in data_user.items():
    v = batch(v)
    embed_UserChrono_BERTReco_FineTuned[k] = dict_of_models_modules['bert'](**v)[1].to('cpu')
    if k % 100 == 0: print(k) 
    
    
#%%    
    
# Save 

torch.save(embed_UserChrono_BERTReco_FineTuned, 'embed_UserChrono_BERTReco_FineTuned.pt')













    