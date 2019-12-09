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
    
    
Model is composed of 2 sub models: user_encoder and item_encoder. 

Dot producted is performed on these 2 encoders to obtain the prediction.




@author: nicholas
"""




import torch
import torch.nn as nn

    

class MLP_dot(nn.Module):
    """
    See top of file for description.
    """


    def __init__(self, input_size=2*768, hidden_size=512, output_size=1):
        super(MLP_dot, self).__init__()
        
        
        self.model = nn.Sequential(
          nn.Linear(input_size ,hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size ,output_size),
        )
        
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[2].weight)

        

        
    def forward(self, user, item):
        
        # Concatenate user and item
        user_item = torch.cat((user, item), dim =1)
        
        # Make a prediction
        logits = self.model(user_item).squeeze()
        pred = torch.sigmoid(logits)

        return pred, logits
















