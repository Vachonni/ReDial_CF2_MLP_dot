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
import pandas as pd
from statistics import mean
#import matplotlib.pyplot as plt

# Personnal imports
import MLP_dot
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
    
    
    # OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
    import logging
    logging.basicConfig(level=logging.INFO)
    
    
    # Create basic model
    model = MLP_dot.MLP_dot()
    model = model.to(args.DEVICE)
          
    
#    # LogitsLoss for stability. Weights for data imbalanced in rating = 0
#    w = (labels.float() - 1) * -90
    
    
    criterion = torch.nn.BCEWithLogitsLoss().to(args.DEVICE)        
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
          
    
    
#    ######## LOAD MODEL
#    
#    if args.preModel != 'none': 
#        checkpoint = torch.load(args.preModel, map_location=args.DEVICE)    
#        model.load_state_dict(checkpoint['state_dict'])
#        optimizer.load_state_dict(checkpoint['optimizer'])
#    
    
    
    

    ########################
    #                      # 
    #         DATA         #
    #                      # 
    ########################    
    
    
    ######## LOAD DATA 
    
    # R (ratings) - Format [ [UserID, [movies uID], [ratings 0-1]] ]   
    print('******* Loading SAMPLES from *******', args.dataPATH + args.dataTrain)
    df_train = pd.read_csv(args.dataPATH + args.dataTrain)
    df_valid = pd.read_csv(args.dataPATH + args.dataValid)
    # Turn DataFrame into an numpy array (easier iteration)
    train_data = df_train.values
    valid_data = df_valid.values
    # Load Relational Tables (RT) of BERT_avrg for users and items. Type: torch.tensor.
    # map_location is CPU because Dataset with num_workers > 0 should not return CUDA.
    user_RT = torch.load(args.dataPATH+'user_BERT_avrg.pt', map_location='cpu')
    item_RT = torch.load(args.dataPATH+'item_BERT_avrg.pt', map_location='cpu')    
#    # Use only samples where there is a genres mention
#    valid_g_data = [[c,m,g,tbm] for c,m,g,tbm in valid_data if g != []]
    if args.DEBUG: 
        train_data = train_data[:128]
        valid_data = valid_data[:128]
    
#    # G (genres) - Format [ [UserID, [movies uID of genres mentionned]] ]    
#    print('******* Loading GENRES from *******', args.genresDict)
#    dict_genresInter_idx_UiD = json.load(open(args.dataPATH + args.genresDict))
#    
#    # Getting the popularity vector 
#    if not args.no_popularity:
#        print('** Including popularity')
#        popularity = np.load(args.dataPATH + 'popularity_vector.npy')
#        popularity = torch.from_numpy(popularity).float()
#    else: popularity = torch.ones(1)
        
    
    ######## CREATING DATASET 
    
    print('******* Creating torch datasets *******')
    train_dataset = Utils.Dataset_MLP_dot(train_data, user_RT, item_RT, args.DEVICE)
    valid_dataset = Utils.Dataset_MLP_dot(valid_data, user_RT, item_RT, args.DEVICE)       
    
    
    ######## CREATE DATALOADER
    
    print('******* Creating dataloaders *******\n\n')    
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False}
    if (args.DEVICE == "cuda"):
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,\
                                               shuffle=True, drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch,\
                                               shuffle=True, drop_last=True, **kwargs)    
#    # For PredRaw - Loader of only 1 sample (user) 
#    valid_bs1_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, **kwargs)
#    ## For PredChrono
#    #valid_chrono_loader = torch.utils.data.DataLoader(valid_chrono_dataset, batch_size=args.batch, shuffle=True, **kwargs)    
#    
    
#%%    

    
    
    import time
    
    
    
    
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
        
        
        
        
        start_time = time.time()
        

        
        
        train_loss = Utils.TrainReconstruction(train_loader, model, criterion, optimizer, \
                                               args.weights, args.completionTrain, args.DEVICE)
        eval_loss = Utils.EvalReconstruction(valid_loader, model, criterion, 100, args.DEVICE)
        
        

        
        
        print('With {}, it took {} seconds'.format(args.num_workers, time.time() - start_time))   
        
        
        
                
        
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
            if not os.path.isdir(args.logPATH): os.mkdir(args.logPATH)
            # Save at directory + (ML or Re) + _model.pth
            torch.save(state, args.logPATH+args.dataTrain[0:2]+'_model.pth')
            print('......saved.')
            
        
        
        
        
        
        
        
        

if __name__ == '__main__':
    main(Arguments.args)














































    
    







































