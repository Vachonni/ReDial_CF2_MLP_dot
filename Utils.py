#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:53:48 2018


Classes and functions for ReDial project.


@author: nicholas
"""

import ast
import numpy as np
from torch.utils import data
import torch
#import nltk
import matplotlib.pyplot as plt
from statistics import mean, stdev
import scipy.stats as ss
#import Settings






########################
#                      # 
#       DATASET        #
#                      # 
########################  




class Dataset_all_MLP(data.Dataset):
    """
    Dataset to use when *_RT are tensors of embeddings as the returns are.
         
    
    INPUT: 
        data: A Numpy Array shape (qt_data, 6), where each column is:
             [data_idx, ConvID, qt_movies_mentioned, user_id, item_id, rating]
             Chrono data's from ReDial
        user_RT: torch tensor (in cuda) of shape (qt_user, embedding_size). 
                 Kind of a Retational Table.
                 Each line is the embedding representation of corresponding user_id     
        item_RT: torch tensor (in cuda) of shape (qt_item, embedding_size). 
                 Kind of a Retational Table.
                 Each line is the embedding representation of corresponding item_id
        model_output: Softmax or sigmoid. How the loss will be evaluated.

    RETURNS (for one data point): 
        Always a 5-tuple (with some -1 when variable not necessary, depends of model_output)
            user's embedding
            item_id
            item's embedding
            rating (or list of ratings) corresponding user and item
            masks on ratings when list of ratings (for Softmax)
    """
    
    
    def __init__(self, data, user_RT, item_RT, model_output):
  
        self.data = data
        self.user_RT = user_RT
        self.item_RT = item_RT
        self.model_output = model_output
        # Quantity of users 
        # (Not always unique users. In chrono, users repeat for varying qt of movies mentionned)
        self.qt_user = len(data)
        self.qt_item = len(item_RT)
        

        
    def __len__(self):
        "Total number of samples."
        
        return len(self.data)



    def __getitem__(self, index):
        "Generate one sample of data."    
        
        # Get items in 'index' position 
        data_idx, ConvID, qt_movies_mentionned, user_id, item_id, rating = self.data[index]        
        
        if self.model_output == 'Softmax':
            
            # Convert str to list
            l_item_id = ast.literal_eval(item_id)
            l_ratings = ast.literal_eval(rating)
            
            # Turn the list of item id's and list of ratings to full tensors (size len of items)
            full_ratings = torch.zeros(self.qt_item)
            # masks are 1 if a raing is available, 0 if not
            full_masks = torch.zeros(self.qt_item)
            for i, item in enumerate(l_item_id):
                full_ratings[item] = l_ratings[i]
                full_masks[item] = 1
            
            return  self.user_RT[user_id], -1, -1, full_ratings, full_masks
        
        
        else:   
            item_id = int(item_id)
            return  self.user_RT[user_id], item_id, self.item_RT[item_id], rating.astype(float), -1
        




class Dataset_TrainBERT(data.Dataset):
    """    
    Dataset to use when *_RT are dict of BERT_input.
        
    
    INPUT: 
        data: A Numpy Array shape (qt_data, 6), where each column is:
             [data_idx, ConvID, qt_movies_mentioned, user_id, item_id, rating]
             Chrono data's from ReDial
        user_RT: A dict of BERT_input (len = qt_user). 
                 Kind of a Retational Table.
                 Each item is a dict containing 4 tensors (inputs, masks, positional, token_types) 
                 corresponding to user's key    
        item_RT: A Numpy Array of shape (qt_item, 1). 
                 Kind of a Retational Table.
                 Each item is a dict containing 4 tensors (inputs, masks, positional, token_types) 
                 corresponding to item's key
        model_output: Softmax or sigmoid. How the loss will be evaluated.

    
    RETURNS (for one data point): 
        Always a 5-tuple (with some -1 when variable not necessary, depends of model_output)
            user's BERT_input format
            item_id
            item's BERT_input format
            rating (or list of ratings) corresponding user and item
            masks on ratings when list of ratings (for Softmax)
    """
    
    
    def __init__(self, data, user_RT, item_RT, model_output):
  
        self.data = data
        self.user_RT = user_RT
        self.item_RT = item_RT
        self.model_output = model_output
        # Quantity of users 
        # (Not always unique users. In chrono, users repeat for varying qt of movies mentionned)
        self.qt_user = len(data)
        self.qt_item = len(item_RT)
        

        
    def __len__(self):
        "Total number of samples."
        
        return len(self.data)



    def __getitem__(self, index):
        "Generate one sample of data."    
        
        # Get items in 'index' position 
        data_idx, ConvID, qt_movies_mentionned, user_id, item_id, rating = self.data[index]        
        
        if self.model_output == 'Softmax':
            
            # Convert str to list
            l_item_id = ast.literal_eval(item_id)
            l_ratings = ast.literal_eval(rating)
            
            # Turn the list of item id's and list of ratings to full tensors (size len of items)
            full_ratings = torch.zeros(self.qt_item)
            # masks are 1 if a raing is available, 0 if not
            full_masks = torch.zeros(self.qt_item)
            for i, item in enumerate(l_item_id):
                full_ratings[item] = l_ratings[i]
                full_masks[item] = 1
            
            return  self.user_RT[user_id], -1, -1, full_ratings, full_masks
        
        
        else:   
            item_id = int(item_id)
            return  self.user_RT[user_id], item_id, self.item_RT[item_id], rating.astype(float), -1
        





########################
#                      # 
#       TRAINING       #
#                      # 
########################  



def TrainReconstruction(train_loader, item_RT, model, model_output, criterion, optimizer, \
                        weights_factor, completion, DEVICE):
    
    model.train()
    train_loss = 0
    train_loss_no_weight = 0
    
    # For print pusposes 
    nb_batch = len(train_loader) * completion / 100
    qt_of_print = 5
    print_count = 0  
    
    # Put on right DEVICE
    if hasattr(model, 'BERT'):
        for k_item, v_item in item_RT.items():
            for k, v in v_item.items():
                v_item[k] = v.to(DEVICE)
    else:
        item_RT = item_RT.to(DEVICE)
    
    
    print('\nTRAINING')
     
    for batch_idx, (user, _, item, targets, masks) in enumerate(train_loader):
        
        # Put on right DEVICE
        if hasattr(model, 'BERT'):
            user = {k:v.to(DEVICE) for k, v in user.items()}
            item = {k:v.to(DEVICE) for k, v in item.items()}
        else:
            user = user.to(DEVICE)
            item = item.to(DEVICE)
        targets = targets.to(DEVICE)
        masks = masks.to(DEVICE)
        
        # Early stopping
        if batch_idx > nb_batch: 
            print(' *EARLY stopping')
            break
        
        # Print update
        if True:  #batch_idx > nb_batch / qt_of_print * print_count:
            print('Batch {:4d} out of {:4.1f}.    Reconstruction Loss on targets: {:.4E}, no weights: {:.4E}' \
                  .format(batch_idx, nb_batch, train_loss/(batch_idx+1), train_loss_no_weight/(batch_idx+1)))  
            print_count += 1    
    
        optimizer.zero_grad()   
        
        # Make prediction
        if model_output == 'Softmax':
            # user is batch x embedding_size. item is qt_items x embedding_size.
            # Put in batch dimension for model (who will concat along embed (dim = -1))
            user = user.unsqueeze(1).expand(-1, len(item_RT), -1)
            item = item_RT.expand(user.shape[0], -1, -1)
            _, logits = model(user, item)

        elif model_output == 'sigmoid':
            # Proceed one at a time
            _, logits = model(user, item)
        
        # Consider wieghts
        if model_output == 'sigmoid':
            # Add weights on targets rated 0 (w_0) because outnumbered by targets 1
            w_0 = (targets - 1) * -1 * (weights_factor - 1)
            w = torch.ones(len(targets)).to(DEVICE) + w_0
            criterion.weight = w
    
        # Evaluate loss
        if model_output == 'Softmax':
            pred = torch.nn.Softmax(dim=0)(logits)
            # Use only the predictions where there a rating was really available
            pred = pred * masks
            loss = criterion(pred, targets)
        elif model_output == 'sigmoid':
            loss = criterion(logits, targets)
        
        # Consider weights (evaluate without)
        if model_output == 'sigmoid':
            criterion.weight = None
            loss_no_weight = criterion(logits, targets).detach()
            train_loss_no_weight += loss_no_weight.detach()

        loss.backward()
        optimizer.step()
        
        train_loss += loss.detach()
        
    train_loss /= nb_batch
        
    return train_loss






########################
#                      # 
#      EVALUATION      #
#                      # 
########################  
    


def EvalReconstruction(valid_loader, item_RT, model, model_output, criterion, \
                       weights_factor, completion, DEVICE):
    model.eval()
    eval_loss = 0
    eval_loss_no_weight = 0
    
    # For print pusposes 
    nb_batch = len(valid_loader) * completion / 100
    qt_of_print = 5
    print_count = 0
    
    # Put on right DEVICE
    if hasattr(model, 'BERT'):
        for k_item, v_item in item_RT.items():
            for k, v in v_item.items():
                v_item[k] = v.to(DEVICE)
    else:
        item_RT = item_RT.to(DEVICE)
    
    
    print('\nEVALUATION')
    
    with torch.no_grad():
        for batch_idx, (user, _, item, targets, masks) in enumerate(valid_loader):
            
            # Put on right DEVICE
            if hasattr(model, 'BERT'):
                user = {k:v.to(DEVICE) for k, v in user.items()}
                item = {k:v.to(DEVICE) for k, v in item.items()}
            else:
                user = user.to(DEVICE)
                item = item.to(DEVICE)
            targets = targets.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # Early stopping 
            if batch_idx > nb_batch: 
                print(' *EARLY stopping')
                break
            
            # Print update
            if batch_idx > nb_batch / qt_of_print * print_count:
                print('Batch {:4d} out of {:4.1f}.    Reconstruction Loss on targets: {:.4E}, no weights: {:.4E}'\
                      .format(batch_idx, nb_batch, eval_loss/(batch_idx+1), eval_loss_no_weight/(batch_idx+1)))  
                print_count += 1
                
            # Make prediction
            if model_output == 'Softmax':
                # user is batch x embedding_size. item is qt_items x embedding_size.
                # Put in batch dimension for model (who will concat along dim =1)
                user = user.unsqueeze(1).expand(-1, len(item_RT), -1)
                item = item_RT.expand(user.shape[0], -1, -1)
                _, logits = model(user, item)
                
            elif model_output == 'sigmoid':
                # Proceed one at a time
                _, logits = model(user, item)
            
            # Consider wieghts
            if model_output == 'sigmoid':
                # Add weights on targets rated 0 (w_0) because outnumbered by targets 1
                w_0 = (targets - 1) * -1 * (weights_factor - 1)
                w = torch.ones(len(targets)).to(DEVICE) + w_0
                criterion.weight = w
        
            # Evaluate loss
            if model_output == 'Softmax':
                pred = torch.nn.Softmax(dim=0)(logits)
                # Use only the predictions where there a rating was really available
                pred = pred * masks
                loss = criterion(pred, targets)
            elif model_output == 'sigmoid':
                loss = criterion(logits, targets)
            
            # Consider weights (evaluate without)
            if model_output == 'sigmoid':
                criterion.weight = None
                loss_no_weight = criterion(logits, targets).detach()
                eval_loss_no_weight += loss_no_weight.detach()
                
            eval_loss += loss.detach()
            
        eval_loss /= nb_batch    
                
    return eval_loss
                
                






########################
#                      # 
#      PREDICTION      #
#                      # 
########################  


def Prediction(pred_data, model, user_RT, item_RT, completion, \
               ranking_method, DEVICE, topx=100):
    """
    Prediction on targets = to be mentionned movies...
    
    """
    
    model.eval()
    
    # For print pusposes 
    nb_batch = len(pred_data) * completion / 100
    qt_of_print = 5
    print_count = 0
    
    Avrg_Ranks = {}
    MRR = {}
    RR = {}
    NDCG = {}
    RE_1 = {}
    RE_10 = {}
    RE_50 = {}
                
    pred_on_user = None
    l_items_id = []
        
    
    with torch.no_grad():
        for batch_idx, (_, _, qt_movies_mentionned, user_id, item_id, rating) in enumerate(pred_data):
            
            # Early stopping 
            if batch_idx > nb_batch or nb_batch == 0: 
                print('EARLY stopping')
                break
            
            # Print Update
            if batch_idx > nb_batch / qt_of_print * print_count:
                print('Batch {} out of {}'.format(batch_idx, nb_batch))
                print_count += 1
                               
            # Put on right DEVICE (what will be used for prediction)
            user_RT = user_RT.to(DEVICE)
            item_RT = item_RT.to(DEVICE)
            
            
            ### Need to accumualte all movies for the same user (= same qt_movies_mentions)
            # If first time, set pred_on_user to the first one
            if pred_on_user == None: 
                pred_on_user = user_id
                pred_on_qt_m_m = qt_movies_mentionned
            # If we changed user 
            if pred_on_user != user_id:
                
                """ Make the prediction on the pred_on user """
                # Get user's embedding
                user_embed = user_RT[pred_on_user]
                # Adapt shape for model: embedding_size -> qt_items x embedding_size 
                user_embed_broad = user_embed.expand(len(item_RT), -1)
                # Make predictions on all movies 
                pred = model(user_embed_broad, item_RT)[0]   # model returns (pred, logits)
                
                # Insure their is at least one target movie (case where new user starts with rating 0)
                # (if not, go to next item and this sample not considered (continue))
                if l_items_id == []: 
                    if rating == 1:
                        l_items_id = [item_id]    
                    else:
                        l_items_id = [] 
                    continue

                # ... get Ranks for targets 
                ranks, avrg_rk, mrr, rr, re_1, re_10, re_50, ndcg = \
                                            Ranks(pred, l_items_id, ranking_method, topx)
                
                # Add Ranks results to appropriate dict
                if pred_on_qt_m_m in RR.keys():
                    Avrg_Ranks[pred_on_qt_m_m].append(avrg_rk)
                    MRR[pred_on_qt_m_m].append(mrr)
                    RR[pred_on_qt_m_m].append(rr)
                    NDCG[pred_on_qt_m_m].append(ndcg)
                    RE_1[pred_on_qt_m_m].append(re_1)
                    RE_10[pred_on_qt_m_m].append(re_10)
                    RE_50[pred_on_qt_m_m].append(re_50)
                else:
                    Avrg_Ranks[pred_on_qt_m_m] = [avrg_rk]
                    MRR[pred_on_qt_m_m] = [mrr]
                    RR[pred_on_qt_m_m] = [rr]
                    NDCG[pred_on_qt_m_m] = [ndcg]
                    RE_1[pred_on_qt_m_m] = [re_1]
                    RE_10[pred_on_qt_m_m] = [re_10]
                    RE_50[pred_on_qt_m_m] = [re_50]
                """ Done"""   
               
                # Update pred_on with the one in the for loop, were done with pred_on
                pred_on_user = user_id
                pred_on_qt_m_m = qt_movies_mentionned
                if rating == 1:
                    l_items_id = [item_id]    
                else:
                    l_items_id = [] 
                
            
            # If same user, add information
            else:
                # Prediction only on positive mentions
                if rating == 1:
                    l_items_id.append(item_id)
            
            

    return Avrg_Ranks, MRR, RR, RE_1, RE_10, RE_50, NDCG







########################
#                      # 
#       METRICS        #
#                      # 
########################  


def DCG(v, top):
    """
    (Discounted Cumulative Gain)   
    Needed to compare rankings when the number of item compared are not the same
    and/or when relevance is not binary
    
    V is vector of ranks, lowest is better
    top is the max rank considered 
    Relevance is 1 if items in rank vector, 0 else
    """
    
    discounted_gain = 0
    
    for i in np.round(v):
        if i <= top:
            discounted_gain += 1/np.log2(i+1)

    return round(discounted_gain, 2)




def nDCG(v, top, nb_values=0):
    """
    DCG normalized with what would be the best evaluation.
    
    nb_values is the max number of good values there is. If not specified or bigger 
    than top, assumed to be same as top.
    """
    if nb_values == 0 or nb_values > top: nb_values = top
    dcg = DCG(v, top)
    idcg = DCG(np.arange(nb_values)+1, top)
    
    return round(dcg/idcg, 2)
    
  

    
def RR(v):
    """
    Gives a value in [0,1] for the first relevant item in list.
    1st = 1 and than lower until cloe to 0.
    Only consern with FIRST relevant item in the list.
    """
    return 1/np.min(v)




    
def Recall_at_k_one_item(v, k=1):
    """
    Recall@K for one item. 1 if the smallest rank in v is smaller than k, 0 if not.
    """
    
    return 1 if np.min(v) <= k else 0




def Ranks(all_values, indices_to_rank, ranking_method, topx = 0):
    """
    Takes 2 numpy array and return, for all values in indices_to_rank,
    the ranks, average ranks, MRR and nDCG for ranks smaller than topx
    """    
    # If topx not mentionned (no top), it's for all the values
    if topx == 0: topx = len(all_values)
    
    if ranking_method == 'min':
        qt_uniq = len(all_values.unique())
       # plt.hist(all_values, 100, [0.0,1.0])
       # plt.show
        assert qt_uniq > len(all_values) * 0.98, \
               "{} of predictions are equal, which is more than 2%. \
               USE --ranking_method 'ordinal'".format(1 - (qt_uniq/len(all_values)))
    
    ranks = ss.rankdata((-1*all_values).cpu(), method=ranking_method)[indices_to_rank]
        
    ndcg = nDCG(ranks, topx, len(indices_to_rank))
    
    if ranks.sum() == 0: print('warning, should always be at least one rank')
    
    return ranks, ranks.mean(), round(float((1/ranks).mean()),4), RR(ranks), \
           Recall_at_k_one_item(ranks, 1), Recall_at_k_one_item(ranks, 10), \
           Recall_at_k_one_item(ranks, 50), ndcg






########################
#                      # 
#        PLOTS         #
#                      # 
########################  

    
def ChronoPlot(metrics, title, PATH, subtitle = ''):
    """
    Plot graph of metrics 
    
    metrics is a dict where each key is the qt_of_movies_mentioned and values are 
    list of metric values for difference data predictions.
    
    We do the mean of values for each key and then plot accrodingly.   
        ** Needs minimum of 5 vallues to be considered
    A print of mean metric values by qt_of_movies_mentioned is done, 
    and one if done for the global mean (indep on qt_of_moveis_mentioned)
    
    
    RETURNS global mean for this metrics (all values, indepedant of qt_of_movies_mentioned)
    """
        
    # Strat with empty values
    dx = []
    dy = []
    derr = []
    dall = []     # global mean to return
        
    # For each key
    for k, v in sorted(metrics.items()):
        if len(v) < 5:
            continue
        else:
            dx.append(k)
            dy.append(mean(v))
            derr.append(stdev(v))
            dall += v

    # MAKING GRAPH
    plt.errorbar(dx, dy, derr, elinewidth=0.5, label=title)
    # Adding BERT for recommendation on Redial and ML
    plt.errorbar([0,1,2,3,4,5,6,7,8], [0.18954871794871794, 0.20591032608695653, 0.18370689655172415, 0.13998529411764707, 0.13518518518518519, 0.12472826086956522, 0.11848101265822784, 0.13777777777777778, 0.11130434782608696], label='BERT ReDialML')    
    plt.title(title + '  ' + subtitle, fontweight="bold")
    plt.xlabel('Nb of mentionned movies before prediction')
    plt.legend()
  # plt.show()
    plt.savefig(PATH+title+subtitle+'.pdf')
    plt.close()
    
    # Printing by qt_of_movies_mentioned
    print(title, ' (all):  ', dy)
    
    # ...for all users, independently of qt_of_movies_mentioned
    mean_value = mean(dall)
    print(title, '(mean):  ', mean_value)
    
    return mean_value
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






