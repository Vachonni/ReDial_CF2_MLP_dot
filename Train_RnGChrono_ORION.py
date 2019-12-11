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
import torch
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print('/n/n', sys.argv)
    
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
     
    
    # Create basic model
    model = all_MLP.all_MLP()
    model = model.to(args.DEVICE)       
    
    if args.model_output == 'Softmax':
        criterion = torch.nn.BCELoss() 
    elif args.model_output == 'sigmoid':
        criterion = torch.nn.BCEWithLogitsLoss()     
        
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    
    
    

    ########################
    #                      # 
    #         DATA         #
    #                      # 
    ########################    
    
    
    ######## LOAD DATA 
    
    if args.model_output == 'Softmax':
        args.dataTrain = 'Train_LIST.csv'
        args.dataValid = 'Val_LIST.csv'
    elif args.model_output == 'sigmoid':
        args.dataTrain = 'Train100.csv'
        args.dataValid = 'Val100.csv'
    
    print('\n******* Loading TRAIN samples from *******', args.dataPATH + args.dataTrain)
    df_train = pd.read_csv(args.dataPATH + args.dataTrain)
    print('******* Loading VALID samples from *******', args.dataPATH + args.dataValid)
    df_valid = pd.read_csv(args.dataPATH + args.dataValid)
    print('******* Loading PRED samples from *******', args.dataPATH + args.dataPred)
    df_pred = pd.read_csv(args.dataPATH + args.dataPred)
    # Turn DataFrame into an numpy array (easier iteration)
    train_data = df_train.values
    valid_data = df_valid.values
    pred_data = df_pred.values
    # Load Relational Tables (RT) of BERT_avrg for users and items. Type: torch.tensor.
    # map_location is CPU because Dataset with num_workers > 0 should not return CUDA.
    user_BERT_RT = torch.load(args.dataPATH + args.user_BERT_RT, map_location='cpu')
    item_BERT_RT = torch.load(args.dataPATH + args.item_BERT_RT, map_location='cpu')    

    if args.DEBUG: 
        train_data = train_data[:128]
        valid_data = valid_data[:128]
        pred_data = pred_data[:128]
            
    
    ######## CREATING DATASET 
    
    print('\n******* Creating torch datasets *******')
    train_dataset = Utils.Dataset_all_MLP(train_data, user_BERT_RT, item_BERT_RT, args.model_output)
    valid_dataset = Utils.Dataset_all_MLP(valid_data, user_BERT_RT, item_BERT_RT, args.model_output)       
    
    
    ######## CREATE DATALOADER
    
    print('\n******* Creating dataloaders *******\n\n')    
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False}
    if (args.DEVICE == "cuda"):
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,\
                                               shuffle=True, drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch,\
                                               shuffle=True, drop_last=True, **kwargs)    
       
    
#%%    

    
    
    import time
    
    
    
    
    ########################
    #                      # 
    #        TRAIN         #
    #                      # 
    ########################  
    
    
    train_losses = []
    valid_losses = []
    NDCG_to_plot = []
    RE10_to_plot = []
        
    if args.DEBUG: args.epoch = 1
    
    
    for epoch in range(args.epoch):
    
        print('\n\n\n\n     ==> Epoch:', epoch, '\n')
        
        
        
        
        start_time = time.time()
        

        
        
        train_loss = Utils.TrainReconstruction(train_loader, item_BERT_RT, model, \
                                               args.model_output, criterion, optimizer, \
                                               args.weights, args.completionTrain, args.DEVICE)
        eval_loss = Utils.EvalReconstruction(valid_loader, item_BERT_RT, model, args.model_output,
                                             criterion, args.weights, 100, args.DEVICE)
        
        

        
        
        print('With {}, it took {} seconds'.format(args.num_workers, time.time() - start_time))   
        
        
        
        
        
        
        
        
        # Make predictions (returns dictionaries)
        print("\n\nPrediction Chronological...")
  #      item_MLP_RT = model.item_encoder(item_BERT_RT.to(args.DEVICE))
        avrg_rank, MRR, RR, RE_1, RE_10, RE_50, NDCG = \
                    Utils.Prediction(pred_data, model, user_BERT_RT, item_BERT_RT, \
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
            if graphs_titles[i] == 'RE_10': RE10_to_plot.append(avrgs[0])
            if graphs_titles[i] == 'NDCG': NDCG_to_plot.append(avrgs[0])
            

        

    
    
    
    
    
                
        
        train_losses.append(train_loss.item())
        valid_losses.append(eval_loss)
        losses = [train_losses, valid_losses]  
        
        print('\nEND EPOCH {:3d} \nTrain Reconstruction Loss on targets: {:.4E}\
              \nValid Reconstruction Loss on tragets: {:.4E}' \
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
            # Save at directory + _model.pth
            torch.save(state, args.logPATH+'model.pth')
            
        # Training Curves plot - Save at each epoch
        plt.plot(losses[0], label='Train')  
        plt.plot(losses[1], label='Valid')  
        plt.plot(RE10_to_plot, label='Re@10')
        plt.plot(NDCG_to_plot, label='NDCG')
        plt.title('Training Curves', fontweight="bold")
        plt.xlabel('Epoch')
        plt.ylabel(str(criterion)[:3] + ' loss')
        plt.legend()
     #   plt.show()
        plt.savefig(args.logPATH+'Training_Curves.pdf')
        plt.close()
        print('......saved.')
            
        # Saving model corresponding to the last epoch (invariant of partience)
        state = {
                'epoch': epoch,
                'eval_loss': eval_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses,
                }
        # Save at directory + _model.pth
        torch.save(state, args.logPATH+'model_last_epoch.pth')
        
        
        
        
        
        
        

if __name__ == '__main__':
    main(Arguments.args)














































    
    







































