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
import torch
import pandas as pd
import numpy as np

# Personnal imports
import all_MLP
import Utils
import Arguments 


         
def main(args):                        
                    
    
    
    ########################
    #                      # 
    #         INIT         #
    #                      # 
    ########################
    
    # Print agrs that will be used
    print(sys.argv)
    
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
    model = all_MLP.all_MLP()
    model = model.to(args.DEVICE)
    
    print('\n******* Loading model *******')         
    
    checkpoint = torch.load(args.M1_path, map_location=args.DEVICE)
    
    model.load_state_dict(checkpoint['state_dict'])



    
    
    ########################
    #                      # 
    #         DATA         #
    #                      # 
    ########################    
    
    
    ######## LOAD DATA 
    
    
    print('\n******* Loading PRED samples from *******', args.dataPATH + args.dataPred)
    df_pred = pd.read_csv(args.dataPATH + args.dataPred)
    # Turn DataFrame into an numpy array (easier iteration)
    pred_data = df_pred.values
    
    # Load Relational Tables (RT) of BERT_avrg for users and items. Type: torch.tensor.
    # map_location is CPU because Dataset with num_workers > 0 should not return CUDA.
    user_BERT_RT = torch.load(args.dataPATH+args.user_BERT_RT, map_location='cpu')
    item_BERT_RT = torch.load(args.dataPATH+args.item_BERT_RT, map_location='cpu')    

    if args.DEBUG: 
        pred_data = pred_data[:128]
    
        
    
    
    
    
    ##############################
    #                            # 
    #         PREDICTION         #
    #                            # 
    ##############################    
    
          
    # Make predictions (returns dictionaries)
    print("\n\nPrediction Chronological...")
  #  item_MLP_RT = model.item_encoder(item_BERT_RT.to(args.DEVICE))
    avrg_rank, MRR, RR, RE_1, RE_10, RE_50, NDCG = \
            Utils.Prediction(pred_data, model, user_BERT_RT, item_BERT_RT, \
                             args.completionPredChrono, args.ranking_method, \
                             args.DEVICE, args.topx)   
    # Print results
    print("\n  ====> RESULTS <==== \n")
    print("\n  ==> BY Nb of mentions, on to be mentionned <== \n")
        
    # List of metrics to evaluate and graph
    graphs_data = [[avrg_rank,avrg_rank],[RE_1, RE_1],[RE_10, RE_10],[RE_50, RE_50],[MRR, MRR],[NDCG, NDCG]]  # Put twice because legacy of with / without genres
    graphs_titles = ['AVRG_RANK','RE_1', 'RE_10','RE_50','MRR', 'NDCG']  # 'Avrg Pred Error', 'MMRR', 'Avrg Rank', 'MRR'

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



























































