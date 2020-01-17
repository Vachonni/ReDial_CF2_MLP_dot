#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:52:13 2019


Models used for CF2


@author: nicholas
"""


import torch
import torch.nn as nn
from transformers import BertModel
    


def DotProduct(tensor1, tensor2):
    
    logits = (tensor1 * tensor2).sum(dim=1)
    pred = torch.sigmoid(logits)
    
    return pred, logits




class MLP(nn.Module):
    """
    Input:
        user: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
        item: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
    Output:
        A tensor of shape (batch, 1) representing the predicted ratings of each user-item pair (+ the logits)
        
    """


    def __init__(self, input_size=2*768, hidden_size=512, output_size=1):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
          nn.Linear(input_size ,hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size ,output_size),
        )
        
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[2].weight)
       
        
    def forward(self, user, item):
        
        # Concatenate user and item
        user_item = torch.cat((user, item), dim = -1)
        
        # Make a prediction
        logits = self.model(user_item).squeeze()
        pred = torch.sigmoid(logits)

        return pred, logits






class TrainBERT(nn.Module):
    """ 
    A Model that takes in 2 BERT_input: user and item.
    
    Passed each through the SAME BERT_Model. 
    
    Averages the last_hidden_layer.
    
    Passed it through MLP model or DotProduct (depending on model)
    to get a prediction (and logits)
    """
    
    
    def __init__(self, model, input_size=2*768, hidden_size=512, output_size=1):
        super(TrainBERT, self).__init__()
        
        if model == 'TrainBERTDotProduct':
            self.merge = DotProduct
        elif model == 'TrainBERTMLP':
            self.merge = MLP(input_size, hidden_size, output_size)
        
        self.BERT = BertModel.from_pretrained('bert-base-uncased')
        
        
    def forward(self, user, item):
        
        # # Get user's BERT_avrg value
        # user_last_hidden_layer = self.BERT(**user)[0]
        # user_avrg_last_hidden_layer = user_last_hidden_layer.mean(dim=1)

        # # Get item's BERT_avrg value
        # item_last_hidden_layer = self.BERT(**item)[0]
        # item_avrg_last_hidden_layer = item_last_hidden_layer.mean(dim=1)    
        
        
        
        """ Trying with Pooler """
        
        # Get user's BERT_avrg value
        user_avrg_last_hidden_layer = self.BERT(**user)[1]

        # Get item's BERT_avrg value
        item_avrg_last_hidden_layer = self.BERT(**item)[1]
      
        
        """  """
        
        
        # Return pred and logits, according to matching factor
        return self.merge(user_avrg_last_hidden_layer, item_avrg_last_hidden_layer)









