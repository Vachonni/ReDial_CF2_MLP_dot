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
args.logPATH = args.working_dir + '/Results/' + args.trial_id + '/' 

# Making the --dataPATH 
args.dataPATH = args.working_dir + '/Data/'     

print(vars(args))


# Save all arguments values
if not os.path.isdir(args.logPATH): os.makedirs(args.logPATH, exist_ok=True)
with open(args.logPATH+'arguments.json', 'w') as fp:
    json.dump(vars(args), fp, sort_keys=True, indent=4)
    fp.write('\n\n'+str(sys.argv))





########  TRAINING  ########



args.dataTrain = 'Train_UNIQUE.csv' 
args.dataValid  = 'Val_UNIQUE.csv' 
args.completionTrain = 100 
args.completionPred = 0
args.completionPredEpoch = 0 
args.seed = True 

# Execute training on ReDial
if not args.PredOnly:
    Train_RnGChrono_ORION.main(args) 
    
    
    
    
    
########  PREDICTION  ########

# Set args for prediction of one model, 
args.seed = True
args.M1_path = args.logPATH + 'model.pth'   
if args.DEBUG and args.completionPredChrono != 0:
    args.completionPredChrono = 1

# Execute prediction on the ReDial model 
NDCGs_1model = Pred_RnGChrono.main(args)




