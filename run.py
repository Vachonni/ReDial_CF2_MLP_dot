#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:42:14 2019


CF2 - Run Training and Prediction 


@author: nicholas
"""



########  IMPORTS  ########


import os
import json
import sys

# Get all arguemnts
from Arguments import args 
import Train_RnGChrono_ORION
import Pred_RnGChrono





########  ARGUMENTS  ########


# (need for Orion, from $SLURM_TMPDIR, adapted elsewhere)
args.logModelsPATH = args.working_dir + '/Results/Models/' + args.trial_id + '/'
args.logInfosPATH = args.working_dir + '/Results/Infos/' + args.trial_id + '/'

# Making the --dataPATH 
args.dataPATH = args.working_dir + args.dataPATH 

print(vars(args))




# Print only when working with $SLURM_TMPDIR (i.e. with 'local' in working dir)
if 'local' in args.working_dir:
    print('\n\n')
    print('============================================')
    print('==== Mind only when using $SLURM_TMPDIR ====')
    
    print('\nIN $SLURM_TMPDIR:')
    files = os.listdir(args.working_dir)
    for name in files:
        print(name)
        
    print('\nIN $SLURM_TMPDIR/Data/:')
    files = os.listdir(args.working_dir+'/Data/')
    for name in files:
        print(name)
    
    print('\nIN {}:'.format(args.dataPATH))
    files = os.listdir(args.dataPATH)
    for name in files:
        print(name)

    print('============================================\n\n')




# Save all arguments values
if not os.path.isdir(args.logInfosPATH): os.makedirs(args.logInfosPATH, exist_ok=True)
with open(args.logInfosPATH + 'arguments.json', 'w') as fp:
    json.dump(vars(args), fp, sort_keys=True, indent=4)
    fp.write('\n\n'+str(sys.argv))





########  TRAINING  ########

print('\n\n\n\n')
print('**********************')
print('*      TRAINING      *')
print('**********************\n')
#args.dataTrain = 'Train_EQUAL.csv' 
#args.dataValid  = 'Val_EQUAL.csv' 
#args.completionTrain = 100 
# args.completionPredEpoch = 0 
args.seed = True 

# Execute training on ReDial
if not args.PredOnly:
    Train_RnGChrono_ORION.main(args) 
    
    
    
    
    
########  PREDICTION  ########

print('\n\n\n\n')
print('**********************')
print('*  FINAL PREDICTION  *')
print('**********************\n')
# Set args for prediction of one model, 
args.seed = True
if args.model_path == 'none':
    args.model_path = args.logModelsPATH + 'model.pth'   
if args.DEBUG and args.completionPredFinal!= 0:
    args.completionPredFinal = 5

# Execute prediction on the ReDial model 
NDCGs_1model = Pred_RnGChrono.main(args)




