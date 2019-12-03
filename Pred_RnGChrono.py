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
import numpy as np
import matplotlib.pyplot as plt

# Personnal imports
import TransformersNV
import Utils
import Settings 
import Arguments 


         
def main(args):                        
                    
    
    ########  INIT  ########
    
    
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
    
    
    
    ######## DATA ########
    
    ######## LOAD DATA 
    # R (ratings) - Format [ [UserID, [movies uID], [ratings 0-1]] ]   
    print('******* Loading SAMPLES from *******', args.dataPATH + args.dataValid)
    #train_data = json.load(open(args.dataPATH + args.dataTrain))
    valid_data = json.load(open(args.dataPATH + args.dataValid))
    # Use only samples where there is a genres mention
    valid_g_data = [[c,m,g,tbm] for c,m,g,tbm in valid_data if g != []]
    if args.DEBUG: 
      #  train_data = train_data[:128]
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
    
    
    ######## CREATING DATASET ListRatingDataset 
    print('******* Creating torch datasets *******')
    #train_dataset = Utils.RnGChronoDataset(train_data, dict_genresInter_idx_UiD, \
    #                                       nb_movies, popularity, args.DEVICE, args.exclude_genres, \
    #                                       args.no_data_merge, args.noiseTrain, args.top_cut)
    #valid_dataset = Utils.RnGChronoDataset(valid_data, dict_genresInter_idx_UiD, \
    #                                       nb_movies, popularity, args.DEVICE, args.exclude_genres, \
    #                                       args.no_data_merge, args.noiseEval, args.top_cut)
    # FOR CHRONO (hence no_data_merge is True). With genres mentions or not
    valid_chrono_dataset = Utils.RnGChronoDataset(valid_data, dict_genresInter_idx_UiD, \
                                             nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                             True, args.noiseEval, args.top_cut)   
    # FOR CHRONO (hence no_data_merge is True) + use only samples where there is a genres mention
    valid_g_chrono_dataset = Utils.RnGChronoDataset(valid_g_data, dict_genresInter_idx_UiD, \
                                             nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                             True, args.noiseEval, args.top_cut)           
    
    
    ######## CREATE DATALOADER
    print('******* Creating dataloaders *******\n\n')    
    kwargs = {}
    if(args.DEVICE == "cuda"):
        kwargs = {'num_workers': 0, 'pin_memory': False}
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,\
    #                                           shuffle=True, drop_last=True, **kwargs)
    #valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch,\
    #                                           shuffle=True, drop_last=True, **kwargs)    
    # For PredChrono
    valid_chrono_loader = torch.utils.data.DataLoader(valid_chrono_dataset, batch_size=args.batch, shuffle=True, **kwargs)    
    valid_g_chrono_loader = torch.utils.data.DataLoader(valid_g_chrono_dataset, batch_size=args.batch, shuffle=True, **kwargs)    
    
    
    
    
    
    ########  MODEL  ######## 
        
    model1 = TransformersNV.BasicRecoTransformer(args.d_model, args.nhead, args.num_layers).to(args.DEVICE)
    criterion1 = torch.nn.BCELoss(reduction='none')

    
#    # Organize args layers 
#    if args.layer2 == 0:
#        layers = [nb_movies, args.layer1]
#    else: 
#        layers = [nb_movies, args.layer1, args.layer2]
#    
#    
#    # Load Model 1
    print('******* Loading BasicRecoTrans model *******')      
#    #    model_base = AutoEncoders.AsymmetricAutoEncoder(layers, nl_type=args.activations, \
#    #                                                    is_constrained=False, dp_drop_prob=0.0, \
#    #                                                    last_layer_activations=False, \
#    #                                                    lla = args.last_layer_activation).to(args.DEVICE)
#    #    model1 = AutoEncoders.GenresWrapperChrono(model_base, args.g_type).to(args.DEVICE)
#    #    checkpoint = torch.load(args.M1_path, map_location=args.DEVICE)
#    #    model1.load_state_dict(checkpoint['state_dict'])
#    
    checkpoint1 = torch.load(args.M1_path, map_location=args.DEVICE)
#    model_base1 = AutoEncoders.AsymmetricAutoEncoder(checkpoint1['layers'], \
#                                                     nl_type=checkpoint1['activations'], \
#                                                     is_constrained=False, dp_drop_prob=0.0, \
#                                                     last_layer_activations=False, \
#                                                     lla = checkpoint1['last_layer_activation']).to(args.DEVICE)
#    model1 = AutoEncoders.GenresWrapperChrono(model_base1, checkpoint1['g_type']).to(args.DEVICE)
    model1.load_state_dict(checkpoint1['state_dict'])
#    
#    if checkpoint1['loss_fct'] == 'BCEWLL':
#        criterion1 = torch.nn.BCEWithLogitsLoss(reduction='none')
#    elif checkpoint1['loss_fct'] == 'BCE':
#        criterion1 = torch.nn.BCELoss(reduction='none')
#    
    
    # For print: Liked or not predictions 
    print_not_liked = ''
    if args.pred_not_liked: print_not_liked = 'NOT '
    
    
    
    
    
    ########  CHRONO EVALUATION  ########
    
    
    # If one model (do with and without genres)
    if args.M2_path == 'none':
        
        # Make predictions (returns dictionaries)
        print("\n\nPrediction Chronological...")
        l1, l0, e1, e0, a1, a0, mr1, mr0, r1, r0, d1, d0 = \
             Utils.EvalPredictionRnGChrono(valid_chrono_loader, model1, criterion1, \
                                           args.zero1, False, args.pred_not_liked, \
                                           args.completionPredChrono, args.topx)
                        # without_genres is True because only one model, so do "without genres" pred
        """ ****** ======>>>>>>>  CHANGED TO without_genres to FALSE 
                                    because BasicRecoTransformer
                                  and to loader that includes genre and not genre
                                    (i.e. from valid_g_chrono_loader to valid_chrono_loader       
        # FOR BasicRecoTransformer - Adding this point so that there are results for WITHOUT genre """
        d0 = {2:[0,0,0,0,0,0]}
        
        
        # Print results
        print("\n  ====> RESULTS <==== \n")
        print("Global avrg pred error with {:.4f} and without {:.4f}".format(l1, l0)) 
        print("\n  ==> BY Nb of mentions, on to be mentionned <== \n")
                
        
#        
#        histo1 = []
#        histo0 = []
#        for k, v in sorted(e1.items()):
#            histo1 += [k for i in v]
#              
#        for k, v in sorted(e0.items()):
#            histo0 += [k for i in v]
#        
#        plt.hist(histo1, len(e1), alpha=0.3)
#        plt.hist(histo0, len(e0), alpha=0.3)    
#        plt.xlabel('Nb of mentionned movies before prediction')
#        plt.legend()
#        plt.show()
        
        
        # List of metrics to evaluate and graph
        graphs_titles = ['NDCG']  # 'Avrg Pred Error', 'MMRR', 'Avrg Rank', 'MRR'
        graphs_data = [[d0, d1]]  # [e0, e1], [mr0, mr1], [a0, a1], [r0, r1]
        # Evaluate + graph
        for i in range(len(graphs_titles)):
            avrgs = Utils.ChronoPlot(graphs_data[i], graphs_titles[i], args.id)
            print(graphs_titles[i]+" on {}liked ReDial movies: {}={:.4f} and {}={:.4f}"\
                  .format(print_not_liked, \
                          'withOUT genres', avrgs[0], \
                          'with genres', avrgs[1]))
            if graphs_titles[i] == 'NDCG':
                NDCGs_1model = avrgs
            
    
    
    
    
    # If two models
    else:   
        
        # Load Model 2
        print('\n******* Loading Model 2 *******')      
        checkpoint2 = torch.load(args.M2_path, map_location=args.DEVICE)
        model_base2 = AutoEncoders.AsymmetricAutoEncoder(checkpoint2['layers'], \
                                                         nl_type=checkpoint2['activations'], \
                                                         is_constrained=False, dp_drop_prob=0.0, \
                                                         last_layer_activations=False, \
                                                         lla = checkpoint2['last_layer_activation']).to(args.DEVICE)
        model2 = AutoEncoders.GenresWrapperChrono(model_base2, checkpoint2['g_type']).to(args.DEVICE)
        model2.load_state_dict(checkpoint2['state_dict'])
        
        if checkpoint2['loss_fct'] == 'BCEWLL':
            criterion2 = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif checkpoint2['loss_fct'] == 'BCE':
            criterion2 = torch.nn.BCELoss(reduction='none')
           
         
        # Make predictions (returns dictionaries)    
        print("\n\nPrediction Chronological Model1...")
        # Make prediction with and without genres in input
        l1, l0, e1, e0, a1, a0, mr1, mr0, r1, r0, d1, d0 = \
             Utils.EvalPredictionRnGChrono(valid_chrono_loader, model1, criterion1, \
                                           args.zero1, True, args.pred_not_liked, \
                                           args.completionPredChrono, args.topx)
                                 # without_genres True because do the "without genres" pred
        print("Prediction Chronological Model2...")                             
        l2, _, e2, _, a2, _, mr2, _, r2, _, d2, _ = \
             Utils.EvalPredictionRnGChrono(valid_chrono_loader, model2, criterion2, \
                                           args.zero1, False, args.pred_not_liked, \
                                           args.completionPredChrono, args.topx)
                                 # without_genres False because don't do the "without genres" pred
    
    
        # Print results
        print("\n  ====> RESULTS <==== \n")
      #  print("Global avrg pred error with {:.4f} and without {:.4f}".format(l1, l2))
        print("\n  ==> BY Nb of mentions, on to be mentionned <== \n")
        
        
        
        
        
        histo1 = []
        histo0 = []
        for k, v in sorted(e1.items()):
            histo1 += [k for i in v]
              
        for k, v in sorted(e2.items()):
            histo0 += [k for i in v]
        
        plt.hist(histo1, len(e1), alpha=0.3)
        plt.hist(histo0, len(e2), alpha=0.3)    
        plt.xlabel('Nb of mentionned movies before prediction')
        plt.legend()
        plt.show()
        
        
        # List of metrics to evaluate and graph
        graphs_titles = ['Avrg Pred Error', 'MMRR', 'NDCG']  # 'Avrg Rank', 'MRR'
        graphs_data = [[e0, e1, e2], [mr0, mr1, mr2], [d0, d1, d2]]  # [a0, a1, a2], [r0, r1, r2]
        # Evaluate + graph
        for i in range(len(graphs_titles)):
            avrgs = Utils.ChronoPlot(graphs_data[i], graphs_titles[i] , args.id, \
                                     [args.M1_label+'(out)', args.M1_label, args.M2_label])
            print(graphs_titles[i]+" on {}liked ReDial movies: {}={:.4f}, {}={:.4f} and {}={:.4f}"\
                  .format(print_not_liked, \
                          args.M1_label+'(out)', avrgs[0], \
                          args.M1_label, avrgs[1], \
                          args.M2_label, avrgs[2]))
            
    
    return NDCGs_1model 


#%%
    
if __name__ == '__main__':
    main(Arguments.args)



























































