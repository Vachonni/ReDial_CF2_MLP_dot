#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:46:45 2019


List of argumnents usable with parser 


@author: nicholas
"""

import argparse


parser = argparse.ArgumentParser(description='Train an MLP for CF2')



# Path where files will be logged (modifired in run.py and runORION.py)

parser.add_argument('--logPATH', type=str, metavar='',\
                    help='Path to directory used when saving files.')

    
# Data
    
parser.add_argument('--dataPATH', type=str, metavar='', default='/Data/DataReDial/', \
                    help='Path to datasets to train on')
parser.add_argument('--dataTrain', type=str, metavar='', default='Train100.csv', \
                    help='File name of Dataset to train on')
parser.add_argument('--dataValid', type=str, metavar='', default='Val100.csv', \
                    help='File name of Dataset to for validation')
parser.add_argument('--dataPred', type=str, metavar='', default='Val_UNIQUE.csv', \
                    help='File name of Dataset to for prediction')    
parser.add_argument('--num_workers', type=int, metavar='', default=0, \
                    help='Qt CPU when loading data')
parser.add_argument('--user_BERT_RT', type=str, metavar='', default='user_BERT_avrg_Title_Genres.pt', \
                    help='File name of tensor where one line is one user avrg BERT representation')
parser.add_argument('--item_BERT_RT', type=str, metavar='', default='item_BERT_avrg_Title_Genres.pt', \
                    help='File name of tensor where one line is one item avrg BERT representation')


# Model

parser.add_argument('--model_output', type=str, metavar='', default='sigmoid', 
                    choices=['Softmax', 'sigmoid'], \
                    help='How loss will be evaluated. \
                    With Softmax, masked BCE on all movies. \
                    With sigmoid, error on each. Should be used with 100 ramdom samples.')
parser.add_argument('--preModel', type=str, metavar='', default='none', \
                    help='Path to a pre-trained model to start with.')
    
    

# Training
    
parser.add_argument('--lr', type=float, metavar='', default=0.001, help='Learning rate')
parser.add_argument('--batch', type=int, metavar='', default=64, help='Batch size')
parser.add_argument('--epoch', type=int, metavar='', default=1000, help='Number of epoch')   
parser.add_argument('--weights', type=float, metavar='', default=1, \
                    help='Weights multiplying the errors on ratings of 0 (underrepresented) \
                    during training.  0 -> no weights, 5 -> 5 times the weight')
parser.add_argument('--patience', type=int, metavar='', default=2, \
                    help='number of epoch to wait without improvement in valid_loss before ending training')
parser.add_argument('--completionTrain', type=float, metavar='', default=100, \
                    help='% of data used during 1 training epoch ~ "early stopping"')
parser.add_argument('--completionPred', type=float, metavar='', default=100, \
                    help='% of data used for final prediction')
parser.add_argument('--completionPredEpoch', type=float, metavar='', default=100, \
                    help='% of data used for prediction during training (each epoch)')
parser.add_argument('--EARLY', default=False, action='store_true', \
                    help="If arg added, Train at 10%, Pred at 1% and PredChrono at 1%")


    

# ...for Pred file
    
parser.add_argument('--ranking_method', type=str, metavar='', default='min', \
                    help='How even ranks are assigned. Use "ordinal" if get an assert error \
                    of too many predictions equal. Can also be "average"')
parser.add_argument('--PredOnly', default=False, action='store_true', \
                    help='If arg added,run.py will only do prediction, no training.')
parser.add_argument('--M1_path', type=str, metavar='', default='none', \
                    help='Path to a Model 1. Will pred with + without genres')
parser.add_argument('--M1_label', type=str, metavar='', default='none', \
                    help='Label for Model 1')
parser.add_argument('--completionPredChrono', type=float, metavar='', default=100, \
                    help='% of data used for prediction')

    

    
# Genres 
    
parser.add_argument('--genresDict', type=str, metavar='', default='dict_genresInter_idx_UiD.json', \
                    help='File name of Dict of genres')
parser.add_argument('--top_cut', type=int, metavar='', default=100, \
                    help='number of movies in genres vector (for torch Dataset)')

    
    
# Metrics
    
parser.add_argument('--topx', type=int, metavar='', default=100, \
                    help='for NDCG mesure, size of top ranks considered')



    
# Global (training(s) and prediction) with file runOrion.py
    
parser.add_argument('--NOpreTrain', default=False, action='store_true', \
                    help="If arg added, skips the pre-training on ML")

    
    

# Args to cover for ORION's (lack of 'choice' option or $SLURM_TMPDIR)
    
parser.add_argument('--ORION_NOpreTrain', type=int, metavar='', default=-1, choices=[-1,0,1], \
                    help='Pretraining on ML or not, will be transformed in runOrion.py. ')
parser.add_argument('--ORION_g_type', type=int, metavar='', default=-1, choices=[-1,0,1,2], \
                    help='g_type, will be transformed in runOrion.py (only 3 last types)')
parser.add_argument('--trial_id', type=str, metavar='', default='./test',\
                    help='ORION - Unique trial experience. USed to reconstruc args.id.')
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


# if args.loss_fct == 'BCE':
#     assert args.last_layer_activation != 'none','Need last layer activation with BCE'
# if args.loss_fct == 'BCEWLL':
#     assert args.last_layer_activation == 'none',"Last layer activation must be 'none' with BCEWLL"
    
    

# Pourcentage
assert 0 <= args.completionTrain <=100,'completionTrain should be in [0,100]'
assert 0 <= args.completionPred <=100,'completionPred should be in [0,100]'
assert 0 <= args.completionPredEpoch <=100,'completionPredEpoch should be in [0,100]'





# CONVERSION
# (bunch of hyper-parameters group under a name for efficiency when running)

if args.EARLY:
    args.completionTrain = 10 
    args.completionPred = 1
    args.completionPredEpoch = 1
    
    