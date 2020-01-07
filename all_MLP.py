#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:52:13 2019



                              ===   MLP_dot   ===


Input:
    user: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
    item: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
Output:
    A tensor of shape (batch, 1) representing the predicted ratings of each user-item pair (+ the logits)
    
    

@author: nicholas
"""




import torch
import torch.nn as nn
from transformers import BertModel
    


class all_MLP(nn.Module):
    """
    See top of file for description.
    """


    def __init__(self, input_size=2*768, hidden_size=512, output_size=1):
        super(all_MLP, self).__init__()
        
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
    
##    Passed it through all_MLP model to get a prediction (and logits)
    Do dot product
    """
    
    
    def __init__(self, input_size=2*768, hidden_size=512, output_size=1):
        super(TrainBERT, self).__init__()
        
        self.MLP = all_MLP(input_size, hidden_size, output_size)
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
        
        
        # # Pass the both through the MLP
        # return self.MLP(user_avrg_last_hidden_layer, item_avrg_last_hidden_layer)

        # Do dot product of both
        logits = (user_avrg_last_hidden_layer * item_avrg_last_hidden_layer).sum(dim=1)
        pred = torch.sigmoid(logits)
        
        return pred, logits









