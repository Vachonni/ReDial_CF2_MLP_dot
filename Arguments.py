#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:46:45 2019


List of argumnents usable with parser 


@author: nicholas
"""

import argparse
import torch


parser = argparse.ArgumentParser(description='Train an MLP for CF2')



# Path where files will be logged (modifired in run.py and runORION.py)

parser.add_argument('--logPATH', type=str, metavar='',\
                    help='Path to directory used when saving files.')

    
# Data
    
parser.add_argument('--dataPATH', type=str, metavar='', default='/Data/DataReDial/', \
                    help='Path to datasets to train on')
parser.add_argument('--dataTrain', type=str, metavar='', default='Train_LIST.csv', \
                    help='File name of Dataset to train on')
parser.add_argument('--dataValid', type=str, metavar='', default='Val_UNIQUE.csv', \
                    help='File name of Dataset to for validation')
parser.add_argument('--dataPred', type=str, metavar='', default='Val_UNIQUE.csv', \
                    help='File name of Dataset to for prediction')    
parser.add_argument('--num_workers', type=int, metavar='', default=0, \
                    help='Qt CPU when loading data')
parser.add_argument('--user_RT', type=str, metavar='', default='BERT_input_UserChrono_dict.npy', \
                    help='File name where one index is one user input')
parser.add_argument('--item_RT', type=str, metavar='', default='BERT_input_MovieTitlesGenres_dict.npy', \
                    help='File name where one index is one item input')
parser.add_argument('--qt_random_ratings', type=int, metavar='', default=20, \
                    help='Quantity of random ratings added in training data')    



# Model

parser.add_argument('--model', type=str, metavar='', default='TrainBERTMLP', 
                    choices=['TrainBERTDotProduct', 'TrainBERTMLP', 'MLP'], \
                    help='Which model to use')    
parser.add_argument('--model_output', type=str, metavar='', default='sigmoid', 
                    choices=['Softmax', 'sigmoid'], \
                    help='How loss will be evaluated. \
                    With Softmax, masked BCE on all movies. \
                    With sigmoid, error on each. Should be used with 100 ramdom samples.')
parser.add_argument('--preModel', type=str, metavar='', default='none', \
                    help='Path to a pre-trained model to start with.')
    
    

# Training
    
parser.add_argument('--lr', type=float, metavar='', default=0.001, help='Learning rate')
parser.add_argument('--batch', type=int, metavar='', default=16, help='Batch size')
parser.add_argument('--epoch', type=int, metavar='', default=1000, help='Number of epoch')   
parser.add_argument('--weights', type=float, metavar='', default=1, \
                    help='Weights multiplying the errors on ratings of 0 (underrepresented) \
                    during training.  1 -> no weights, 5 -> 5 times the weight')
parser.add_argument('--patience', type=int, metavar='', default=5, \
                    help='number of epoch to wait without improvement in valid_loss before ending training')
parser.add_argument('--completionTrain', type=float, metavar='', default=100, \
                    help='% of data used during 1 training epoch ~ "early stopping"')
parser.add_argument('--completionEval', type=float, metavar='', default=100, \
                    help='% of data used during 1 eval epoch ~ "early stopping"')    
parser.add_argument('--completionPredEpoch', type=float, metavar='', default=100, \
                    help='% of data used for prediction during training (each epoch)')
parser.add_argument('--EARLY', default=False, action='store_true', \
                    help="If arg added, Train at 10%, Pred at 1% and PredChrono at 1%")


    

# ...for Pred file
    
parser.add_argument('--ranking_method', type=str, metavar='', default='average', \
                    help='How even ranks are assigned. Use "ordinal" if get an assert error \
                    of too many predictions equal. Can also be "average"')
parser.add_argument('--PredOnly', default=False, action='store_true', \
                    help='If arg added,run.py will only do prediction, no training.')
parser.add_argument('--model_path', type=str, metavar='', default='none', \
                    help='Path to model used for prediction. If none, determined in run.py')
parser.add_argument('--completionPredFinal', type=int, metavar='', default=100, \
                    help='% of data used for final prediction')


    
    
# Metrics
    
parser.add_argument('--topx', type=int, metavar='', default=100, \
                    help='for NDCG mesure, size of top ranks considered')



    
    

# Args to cover for ORION's (lack of 'choice' option or $SLURM_TMPDIR)
    
parser.add_argument('--trial_id', type=str, metavar='', default='Test',\
                    help='ORION - Unique trial experience. Used to reconstruc args.id.')
parser.add_argument('--working_dir', type=str, metavar='', default='.', \
                    help='ORION - Path to directory where experience is run.')

    

# Others
    
parser.add_argument('--seed', default=False, action='store_true', \
                    help="If arg added, random always give the same")

parser.add_argument('--DEVICE', type=str, metavar='', default='cuda', choices=['cuda', 'cpu'], \
                    help="Type of machine to run on")

parser.add_argument('--DEBUG', default=False, action='store_true', \
                    help="If arg added, reduced dataset and epoch to 1 for rapid debug purposes")



args = parser.parse_args()






# ASSERTIONS



# MODEL - DATA MATCH

# If model trains BERT, data needs to be in BERT input ready format
if args.model == 'TrainBERTDotProduct' or args.model == 'TrainBERTMLP':
    if args.user_RT[-3:] != 'npy' or args.item_RT[-3:] != 'npy':
        print("\n\n\n     ****************************")
        print("          ***   WARNING  ***")
        print("\n     Changing RT to make them BERT ready inputs ")
        print("     Using    BERT_input_MovieTitlesGenres_dict.npy    for items")
        print("\n     ****************************\n\n\n")
        args.user_RT = 'BERT_input_UserChrono_dict.npy'
        args.item_RT = 'BERT_input_MovieTitlesGenres_dict.npy'
# If not, it should be torch tensors containing BERT embeddings
else:
    if args.user_RT[-3:] != '.pt' or args.item_RT[-3:] != '.pt':
       print("\n\n\n     ****************************")
       print("          ***   WARNING  ***")
       print("\n     Changing RT to make them torch.Tensor with embeddings ")
       print("     Using    embed_MovieTitlesGenres_with_BERT_avrg.pt    for items")
       print("     Setting qt_random_ratings at 100")
       print("\n     ****************************\n\n\n")
       args.user_RT = 'embed_UserChrono_with_BERT_avrg.pt'
       args.item_RT = 'embed_MovieTitlesGenres_with_BERT_avrg.pt'  
       args.qt_random_ratings = 100





# Pourcentage
assert 0 <= args.completionTrain <=100,'completionTrain should be in [0,100]'
assert 0 <= args.completionPredEpoch <=100,'completionPredEpoch should be in [0,100]'
assert 0 <= args.completionPredFinal <=100,'completionPredFinal should be in [0,100]'





# CONVERSION
# (bunch of hyper-parameters group under a name for efficiency when running)

if args.EARLY:
    args.completionTrain = 10 
    args.completionPredEpoch = 10
    args.completionPredFinal = 10
    
    