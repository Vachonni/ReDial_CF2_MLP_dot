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


# Making the --id the log folder (this is a strage name, should be changed to args.logPATH)
# (need for Orion, from $SLURM_TMPDIR, adapted elsewhere)
args.id = args.working_dir + '/Results/' + args.trial_id + '/' 

# Making the --dataPATH 
args.dataPATH = args.working_dir + '/Data/'     

print(vars(args))


# Save all arguments values
if not os.path.isdir(args.id): os.makedirs(args.id, exist_ok=True)
with open(args.id+'arguments.json', 'w') as fp:
    json.dump(vars(args), fp, sort_keys=True, indent=4)
    fp.write('\n\n'+str(sys.argv))





########  TRAINING  ########


# NO Pretraing on ML    
if args.NOpreTrain: 
    args.dataTrain = 'ReDialRnGChronoTRAIN.json' 
    args.dataValid  = 'ReDialRnGChronoVALID.json' 
    args.noiseTrain = False
    args.noiseEval = False
    args.completionTrain = 100 
    args.completionPred = 0
    args.completionPredEpoch = 0 
    args.activations = 'relu'
    args.last_layer_activation = 'softmax'
    args.loss_fct = 'BCE'
    args.seed = True 
    
    # Execute training on ReDial
    if not args.PredOnly:
        Train_RnGChrono_ORION.main(args) 
    
    
    
    
    
########  PREDICTION  ########

# Set args for prediction of one model, 
args.seed = True
args.M1_path = args.id + 'Re_model.pth'   
if args.DEBUG and args.completionPredChrono != 0:
    args.completionPredChrono = 1


# Execute prediction on the ReDial model 
# (No need for args.no_data_merge, it's treated in Pred_RnGChrono.main)
NDCGs_1model = Pred_RnGChrono.main(args)
