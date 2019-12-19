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
    print('\n', sys.argv)
    
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
     
    
    # Create model
            
    if args.model == 'TrainBERT':
        model = all_MLP.TrainBERT()
        model = model.to(args.DEVICE) 
        criterion = torch.nn.BCEWithLogitsLoss() 
        
    else:
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
        print('\n\n===---> WARNING: Moving to LIST DATA. Mandatory for Softmax evaluation <---===')
        args.dataTrain = 'Train_LIST.csv'
        args.dataValid = 'Val_LIST.csv'
    
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
    
    print('\n******* Loading RT *******', args.dataPATH + args.item_RT)
    # LOAD RT - According to the model
    if args.model == 'TrainBERT':
        user_RT = np.load(args.dataPATH + args.user_RT, allow_pickle=True).item()
        item_RT = np.load(args.dataPATH + args.item_RT, allow_pickle=True).item()
        # for ku, vu in user_RT.items():
        #     for k, v in vu.items():
        #         vu[k] = v[0].to(args.DEVICE)
        # for ki, vi in item_RT.items():
        #     for k, v in vi.items():
        #         vi[k] = v[0].to(args.DEVICE)
    else:
        # Load Relational Tables (RT) of BERT_avrg for users and items. Type: torch.tensor.
        # map_location is CPU because Dataset with num_workers > 0 should not return CUDA.
        user_RT = torch.load(args.dataPATH + args.user_RT, map_location='cpu')
        item_RT = torch.load(args.dataPATH + args.item_RT, map_location='cpu')    

    if args.DEBUG: 
        train_data = train_data[:128]
        valid_data = valid_data[:128]
        pred_data = pred_data[:128]
            
    
    ######## CREATING DATASET 
    
    print('\n******* Creating torch datasets *******')
    train_dataset = Utils.Dataset_TrainBERT(train_data, user_RT, item_RT, args.model_output)
    valid_dataset = Utils.Dataset_TrainBERT(valid_data, user_RT, item_RT, args.model_output)       
    
    
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
    RE10_training_plot = []
    NDCG_training_plot = []

        
    if args.DEBUG: args.epoch = 1
    
    
    for epoch in range(args.epoch):
    
        print('\n\n\n\n     ==> Epoch:', epoch, '\n')
        
        
        
        
        start_time = time.time()
        

        
        
        train_loss = Utils.TrainReconstruction(train_loader, item_RT, model, \
                                               args.model_output, criterion, optimizer, \
                                               args.weights, args.completionTrain, args.DEVICE)
        eval_loss = Utils.EvalReconstruction(valid_loader, item_RT, model, args.model_output,
                                             criterion, args.weights, 100, args.DEVICE)
        
        

        
        
        print('With {}, it took {} seconds'.format(args.num_workers, time.time() - start_time))   
        
        
        
        
        
        
        
        
        # Make predictions (returns dictionaries) 
        if args.completionPredEpoch != 0:
            print("\n\nPrediction Chronological...")
            avrg_rank, MRR, RR, RE_1, RE_10, RE_50, NDCG = \
                        Utils.Prediction(pred_data, model, user_RT, item_RT, \
                                         args.completionPredEpoch, args.ranking_method, \
                                         args.DEVICE, args.topx)   
            # Print results
            print("\n\n\n\n  ====> RESULTS <==== ")
            print("\n  ==> By qt_of_movies_mentioned, on to be mentionned movies <==\n")
                    
            # List of metrics to evaluate and graph
            #   Possible values: avrg_rank, MRR, RR, RE_1, RE_10, RE_50, NDCG 
            graphs_data = [avrg_rank, RE_1, RE_10, RE_50, MRR, NDCG]  
            graphs_titles = ['AVRG_RANK', 'RE_1', 'RE_10', 'RE_50', 'MRR', 'NDCG'] 
    
            # Evaluate + graph
            for i in range(len(graphs_titles)):
                subtitle = '_'+args.trial_id+'__last_epoch_'+str(args.completionPredEpoch)+'%_data' 
                avrgs = Utils.ChronoPlot(graphs_data[i], graphs_titles[i], \
                                         args.logInfosPATH, subtitle)
                if graphs_titles[i] == 'RE_10': RE10_training_plot.append(avrgs)
                if graphs_titles[i] == 'NDCG': NDCG_training_plot.append(avrgs)
            

        

    
    
    
    
    
                
        
        train_losses.append(train_loss.item())
        valid_losses.append(eval_loss)
        losses = [train_losses, valid_losses]  
        
        print('\n\nEND EPOCH {:3d} \nTrain Reconstruction Loss on targets: {:.4E}\
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
            if not os.path.isdir(args.logModelsPATH): os.makedirs(args.logModelsPATH, exist_ok=True)
            # Save at Models directory + model.pth
            torch.save(state, args.logModelsPATH+'model.pth')
            print('......saved.')
            
        # Training Curves plot - Save at each epoch
        plt.plot(losses[0], label='Train')  
        plt.plot(losses[1], label='Valid')  
        if args.completionPredEpoch != 0:
            plt.plot(RE10_training_plot, label='Re@10')
            plt.plot(NDCG_training_plot, label='NDCG')
        plt.title('Training Curves - ' + args.trial_id, fontweight="bold")
        plt.xlabel('Epoch')
        plt.ylabel(str(criterion)[:3] + ' loss')
        plt.legend()
     #   plt.show()
        plt.savefig(args.logInfosPATH+'Training_Curves.pdf')
        plt.close()
            
        # Saving model corresponding to the last epoch (invariant of patience)
        state = {
                'epoch': epoch,
                'eval_loss': eval_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses,
                }
        # Save at Models directory + model_last_epoch.pth
        torch.save(state, args.logModelsPATH + 'model_last_epoch.pth')
        
        
        
        
        
        
        

if __name__ == '__main__':
    main(Arguments.args)














































    
    







































