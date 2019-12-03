#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:02:56 2019


Training Basic Transformer Recommender 


@author: nicholas
"""


########  IMPORTS  ########

import os
import sys
import json
import torch
from torch import optim
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

# Personnal imports
import TransformersNV
import Utils
import Settings 
import Arguments 




def main(args):
    
    
    ########################
    #                      # 
    #         INIT         #
    #                      # 
    ########################
    
    # Print agrs that will be used
    print(sys.argv)
    
    # Add 1 to nb_movies_in_total because index of movies starts at 1
    nb_movies = Settings.nb_movies_in_total + 1
    
    # Cuda availability check
    if args.DEVICE == "cuda" and not torch.cuda.is_available():
        raise ValueError("DEVICE specify a GPU computation but CUDA is not available")
      
    # Seed 
    if args.seed:
        manualSeed = 1
        # Python
     #   random.seed(manualSeed)
        # Numpy
        np.random.seed(manualSeed)
        # Torch
        torch.manual_seed(manualSeed)
        # Torch with GPU
        if args.DEVICE == "cuda":
            torch.cuda.manual_seed(manualSeed)
            torch.cuda.manual_seed_all(manualSeed)
            torch.backends.cudnn.enabled = False 
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    
    
    
#%%    
    
    ########################
    #                      # 
    #        MODEL         #
    #                      # 
    ########################
    
    from transformers import BertTokenizer, BertModel
    
    # OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
    import logging
    logging.basicConfig(level=logging.INFO)
      
    
    ######## GET PRETRAINED TOKENIZER
 
    # Load pre-trained model tokenizer (vocabulary)
# ============= >>>>>>>>>>  USE ENCODE WITH add_special_tokens=True
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
        
    ######## GET PRETRAINED MODEL
    
   # Load pre-trained model (weights) for USERS
    BERT_U = BertModel.from_pretrained('bert-base-uncased')

   # Load pre-trained model (weights) for ITEMS
    BERT_I = BertModel.from_pretrained('bert-base-uncased')
    
    
    
    

    
    
    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()
    
    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cpu')
    segments_tensors = segments_tensors.to('cpu')
    model.to('cpu')
    
    # Predict hidden states features for each layer
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model
        encoded_layers = outputs[0]
    # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
    assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)
    
#%%
    
    
    model = TransformersNV.BasicRecoTransformer(args.d_model, args.nhead, args.num_layers).to(args.DEVICE)
    criterion = torch.nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    
    ######## LOAD MODEL
    
    if args.preModel != 'none': 
        checkpoint = torch.load(args.preModel, map_location=args.DEVICE)    
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    

    ########################
    #                      # 
    #         DATA         #
    #                      # 
    ########################    
    
    
    ######## LOAD DATA 
    
    # R (ratings) - Format [ [UserID, [movies uID], [ratings 0-1]] ]   
    print('******* Loading SAMPLES from *******', args.dataPATH + args.dataTrain)
    train_data = json.load(open(args.dataPATH + args.dataTrain))
    valid_data = json.load(open(args.dataPATH + args.dataValid))
    # Use only samples where there is a genres mention
    valid_g_data = [[c,m,g,tbm] for c,m,g,tbm in valid_data if g != []]
    if args.DEBUG: 
        train_data = train_data[:128]
        valid_data = valid_data[:128]
    
    # G (genres) - Format [ [UserID, [movies uID of genres mentionned]] ]    
    print('******* Loading GENRES from *******', args.genresDict)
    dict_genresInter_idx_UiD = json.load(open(args.dataPATH + args.genresDict))
    
    # Getting the popularity vector 
    if not args.no_popularity:
        print('** Including popularity')
        popularity = np.load(args.dataPATH + 'popularity_vector.npy')
        popularity = torch.from_numpy(popularity).float()
    else: popularity = torch.ones(1)
        
    
    ######## CREATING DATASET 
    
    print('******* Creating torch datasets *******')
    train_dataset = Utils.RnGChronoDataset(train_data, dict_genresInter_idx_UiD, \
                                           nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                           args.no_data_merge, args.noiseTrain, args.top_cut)
    valid_dataset = Utils.RnGChronoDataset(valid_data, dict_genresInter_idx_UiD, \
                                           nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                           args.no_data_merge, args.noiseEval, args.top_cut)        
    
    
    ######## CREATE DATALOADER
    
    print('******* Creating dataloaders *******\n\n')    
    kwargs = {}
    if(args.DEVICE == "cuda"):
        kwargs = {'num_workers': 0, 'pin_memory': False}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,\
                                               shuffle=True, drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch,\
                                               shuffle=True, drop_last=True, **kwargs)    
    # For PredRaw - Loader of only 1 sample (user) 
    valid_bs1_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, **kwargs)
    ## For PredChrono
    #valid_chrono_loader = torch.utils.data.DataLoader(valid_chrono_dataset, batch_size=args.batch, shuffle=True, **kwargs)    
    
    
    
    
    ########################
    #                      # 
    #        TRAIN         #
    #                      # 
    ########################  
    
    
    train_losses = []
    valid_losses = []
        
    if args.DEBUG: args.epoch = 1
    
    
    for epoch in range(args.epoch):
    
        print('\n\n\n\n     ==> Epoch:', epoch, '\n')
        
        train_loss = Utils.TrainReconstruction(train_loader, model, criterion, optimizer, \
                                               args.zero1, args.weights, args.completionTrain)
        eval_loss = Utils.EvalReconstruction(valid_loader, model, criterion, \
                                             args.zero1, 100)
                
        train_losses.append(train_loss)
        valid_losses.append(eval_loss)
        losses = [train_losses, valid_losses]  
        
        print('\nEND EPOCH {:3d} \nTrain Reconstruction Loss on targets: {:.4f}\
              \nValid Reconstruction Loss on tragets: {:.4f}' \
              .format(epoch, train_loss, eval_loss))
    
        ######## PATIENCE - Stop if the Model didn't improve in the last 'patience' epochs
        patience = args.patience
        if len(valid_losses) - valid_losses.index(min(valid_losses)) > patience:
            print('--------------------------------------------------------------------------------')
            print('-                               STOPPED TRAINING                               -')
            print('-  Recent valid losses:', valid_losses[-patience:])
            print('--------------------------------------------------------------------------------')
            break   # End training
    
        
        ######## SAVE - First model and model that improves valid loss
        precedent_losses = valid_losses[:-1]
        # Cover 1st epoch for min([])'s error
        if precedent_losses == []: precedent_losses = [0]   
        if epoch == 0 or eval_loss < min(precedent_losses):
            print('\n\n   Saving...')
            state = {
                    'epoch': epoch,
                    'eval_loss': eval_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'losses': losses,
                    }
            if not os.path.isdir(args.id): os.mkdir(args.id)
            # Save at directory + (ML or Re) + _model.pth
            torch.save(state, args.id+args.dataTrain[0:2]+'_model.pth')
            print('......saved.')
            
        
        
        
        
        
        
        
        

if __name__ == '__main__':
    main(Arguments.args)














































    
    







































