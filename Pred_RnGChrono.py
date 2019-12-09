#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:15:28 2019


Basic Transformer Recommender - 
Predictions with models of type Ratings and Genres with Chronological Data


@author: nicholas
"""


########  IMPORTS  ########


import sys
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
      #  random.seed(manualSeed)
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
    
    # Global variable for runOrion.py (NDCGs for one model)
    NDCGs_1model = -1
    
    
    
    
    ########################
    #                      # 
    #        MODEL         #
    #                      # 
    ########################
    
    
    # Create basic model
    model = MLP_dot.MLP_dot()
    model = model.to(args.DEVICE)
    
    criterion = torch.nn.BCEWithLogitsLoss()

    
    print('******* Loading model *******')         
    
    checkpoint = torch.load(args.M1_path, map_location=args.DEVICE)
    
    model.load_state_dict(checkpoint['state_dict'])

    
    # For print: Liked or not predictions 
    print_not_liked = ''
    if args.pred_not_liked: print_not_liked = 'NOT '



    
    
    ########################
    #                      # 
    #         DATA         #
    #                      # 
    ########################    
    
    
    ######## LOAD DATA 
    
    print('******* Loading SAMPLES from *******', args.dataPATH + args.dataValid)
#    df_train = pd.read_csv(args.dataPATH + args.dataTrain)
    df_valid = pd.read_csv(args.dataPATH + args.dataValid)
    # Turn DataFrame into an numpy array (easier iteration)
#    train_data = df_train.values
    valid_data = df_valid.values
    # Load Relational Tables (RT) of BERT_avrg for users and items. Type: torch.tensor.
    # map_location is CPU because Dataset with num_workers > 0 should not return CUDA.
    user_BERT_RT = torch.load(args.dataPATH+'user_BERT_avrg_RT.pt', map_location='cpu')
    item_BERT_RT = torch.load(args.dataPATH+'item_BERT_avrg_RT.pt', map_location='cpu')    
#    # Use only samples where there is a genres mention
#    valid_g_data = [[c,m,g,tbm] for c,m,g,tbm in valid_data if g != []]
    if args.DEBUG: 
#        train_data = train_data[:128]
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
        
    
#    ######## CREATING DATASET 
#    
    print('******* Creating torch datasets *******')
#    train_dataset = Utils.Dataset_MLP_dot(train_data, user_BERT_RT, item_BERT_RT, for_pred=True)
    valid_dataset = Utils.Dataset_MLP_dot(valid_data, user_BERT_RT, item_BERT_RT)       
#    
#    
#    ######## CREATE DATALOADER
#    
    print('******* Creating dataloaders *******\n\n')    
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False}
    if (args.DEVICE == "cuda"):
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
#    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,\
#                                               shuffle=True, drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch,\
                                               shuffle=True, drop_last=True, **kwargs)    
##    # For PredRaw - Loader of only 1 sample (user) 
##    valid_bs1_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, **kwargs)
##    ## For PredChrono
##    #valid_chrono_loader = torch.utils.data.DataLoader(valid_chrono_dataset, batch_size=args.batch, shuffle=True, **kwargs)    
##    
    
    
    
    
    
    
    ##############################
    #                            # 
    #         PREDICTION         #
    #                            # 
    ##############################    
    
          
    # Make predictions (returns dictionaries)
    print("\n\nPrediction Chronological...")
  #  item_MLP_RT = model.item_encoder(item_BERT_RT.to(args.DEVICE))
    avrg_rank, MRR, RR, RE_1, RE_10, RE_50, NDCG = \
            Utils.Prediction(valid_data, model, user_BERT_RT, item_BERT_RT, \
                             args.completionPredChrono, args.ranking_method, \
                             args.DEVICE, args.topx)   
    # Print results
    print("\n  ====> RESULTS <==== \n")
    print("\n  ==> BY Nb of mentions, on to be mentionned <== \n")
        
    # List of metrics to evaluate and graph
    graphs_titles = ['AVRG_RANK','RE_1', 'RE_10','RE_50','MRR', 'NDCG']  # 'Avrg Pred Error', 'MMRR', 'Avrg Rank', 'MRR'
    graphs_data = [[avrg_rank,avrg_rank],[RE_1, RE_1],[RE_10, RE_10],[RE_50, RE_50],[MRR, MRR],[NDCG, NDCG]]  # Put twice because legacy of with / without genres
    # Evaluate + graph
    for i in range(len(graphs_titles)):
        avrgs = Utils.ChronoPlot(graphs_data[i], graphs_titles[i], args.logPATH)
        print(graphs_titles[i]+" on ReDial movies: {}={:.4f} and {}={:.4f} \n"\
              .format('withOUT genres', avrgs[0], \
                      'with genres', avrgs[1]))
        if graphs_titles[i] == 'NDCG':
            NDCGs_1model = avrgs
    
    return NDCGs_1model 


#%%
    
if __name__ == '__main__':
    main(Arguments.args)



























































