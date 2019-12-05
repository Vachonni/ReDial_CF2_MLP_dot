#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:53:48 2018


Classes and functions for ReDial project.


@author: nicholas
"""

import numpy as np
from torch.utils import data
import torch
#import nltk
import matplotlib.pyplot as plt
from statistics import mean, stdev
import scipy.stats as ss

#import Settings




"""
DATASET - Sub classes of Pytorch DATASET to prepare for Dataloader

"""




class Dataset_MLP_dot(data.Dataset):
    """    
    
    ****** Now inputs and targets are seperated in data ******
    
    
    INPUT: 
        data: A Numpy Array shape (39706, 6), where each column is:
             [data_idx, ConvID, qt_movies_mentioned, user_chrono_id, movie_UiD, rating
             Chrono data's from ReDial
        user_RT: torch tensor (in cuda) of shape (39706, 768). Kind of a Retational Table.
                 Each line is the BERT avrg representation of corresponding user_chrono_id     
        item_RT: torch tensor (in cuda) of shape (48272, 768). Kind of a Retational Table.
                 Each line is the BERT avrg representation of corresponding movie_UiD

    
    RETURNS (for one data point):
        user's BERT avrg representation
        item's BERT avrg representation
        rating corresponding
    """
    
    
    def __init__(self, data, user_RT, item_RT):
        self.data = data
        self.user_RT = user_RT
        self.item_RT = item_RT


        
    def __len__(self):
        "Total number of samples."
        
        return len(self.data)



    def __getitem__(self, index):
        "Generate one sample of data."
        
        # Get items in 'index' position 
        data_idx, ConvID, qt_movies_mentionned, user_id, item_id, rating = self.data[index]    
        
        return  self.user_RT[user_id], self.item_RT[item_id], rating.astype(float)
        






"""

TRAINING AND EVALUATION 

"""



def TrainReconstruction(train_loader, model, criterion, optimizer, weights_factor, completion, DEVICE):
    
    model.train()
    train_loss = 0
    train_loss_no_weight = 0
    nb_batch = len(train_loader) * completion / 100
    
    
#    """ """
#    pred_mean_values = []
#    
#    """ """
   
    
    print('\nTRAINING')
     
    for batch_idx, (user, item, targets) in enumerate(train_loader):
        
        # Put on right DEVICE
        user = user.to(DEVICE)
        item = item.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # Early stopping
        if batch_idx > nb_batch: 
            print(' *EARLY stopping')
            break
        
        # Print update
        if batch_idx % 100 == 0: 
            print('Batch {:4d} out of {:4.1f}.    Reconstruction Loss on targets: {:.4f}, no weights: {:.4f}' \
                  .format(batch_idx, nb_batch, train_loss/(batch_idx+1), train_loss_no_weight/(batch_idx+1)))  
                
        
        optimizer.zero_grad()   

        pred, logits = model(user, item)

           
#        
#        """ To look into pred values evolution during training"""
#        pred_mean_values.append((pred.detach()).mean())
#        """ """

        # Add weights on targets rated 0 (w_0) because outnumbered by targets 1
        w_0 = (targets - 1) * -1 * (weights_factor - 1)
        w = torch.ones(len(targets)).to(DEVICE) + w_0
        criterion.weight = w
    
        loss = criterion(logits, targets)
        
        criterion.weight = None
        loss_no_weight = criterion(logits, targets).detach()

        loss.backward()
        optimizer.step()
        
#        # Remove weights for other evaluations
#        criterion.weight = None
#        loss_no_weights = (criterion(pred, targets) * masks[1]).sum() 
#        loss_no_weights /= nb_ratings
#        train_loss += loss_no_weights.detach()
        
        train_loss += loss
        train_loss_no_weight += loss_no_weight
        
    train_loss /= nb_batch
        
    return train_loss





def EvalReconstruction(valid_loader, model, criterion, weights_factor, completion, DEVICE):
    model.eval()
    eval_loss = 0
    eval_loss_no_weight = 0
    nb_batch = len(valid_loader) * completion / 100
    
    print('\nEVALUATION')
    
    with torch.no_grad():
        for batch_idx, (user, item, targets) in enumerate(valid_loader):
            
            # Put on the right DEVICE
            user = user.to(DEVICE)
            item = item.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Early stopping 
            if batch_idx > nb_batch: 
                print(' *EARLY stopping')
                break
            
            # Print update
            if batch_idx % 100 == 0: 
                print('Batch {:4d} out of {:4.1f}.    Reconstruction Loss on targets: {:.4f}, no weights: {:.4f}'\
                      .format(batch_idx, nb_batch, eval_loss/(batch_idx+1), eval_loss_no_weight/(batch_idx+1)))  
                               
            pred, logits = model(user, item)  
      
        
            # Add weights on targets rated 0 (w_0) because outnumbered by targets 1
            w_0 = (targets - 1) * -1 * (weights_factor - 1)
            w = torch.ones(len(targets)).to(DEVICE) + w_0
            criterion.weight = w
        
            loss = criterion(logits, targets)
            
            criterion.weight = None
            loss_no_weight = criterion(logits, targets).detach()
        
            eval_loss += loss
            eval_loss_no_weight += loss_no_weight
            
    
    eval_loss /= nb_batch 
    
    return eval_loss







"""

PREDICTION

"""




def Prediction(valid_data, model, user_BERT_RT, item_MLP_RT, completion, \
               ranking_method, DEVICE, topx=100):
    """
    Prediction on targets = to be mentionned movies...
    
    ** Only works with RnGChronoDataset **
    
    """
    
    model.eval()
    nb_batch = len(valid_data) * completion / 100
    
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
        for batch_idx, (_, _, qt_movies_mentionned, user_id, item_id, rating) in enumerate(valid_data):
            
            # Early stopping 
            if batch_idx > nb_batch or nb_batch == 0: 
                print('EARLY stopping')
                break
            
            # Print Update
            if batch_idx % 1000 == 0:
                print('Batch {} out of {}'.format(batch_idx, nb_batch))
                               
            # Put on the right DEVICE (what will be used for prediction)
            user_BERT_RT = user_BERT_RT.to(DEVICE)
            item_MLP_RT = item_MLP_RT.to(DEVICE)
     #       user_id = user_id.to(DEVICE)
     #       item_id = item_id.to(DEVICE)
            
            
            ### Need to accumualte all movies for the same user (= same qt_movies_mentions)
            # If first time, set pred_on_user to the first one
            if pred_on_user == None: 
                pred_on_user = user_id
                pred_on_qt_m_m = qt_movies_mentionned
            # If we changed user 
            if pred_on_user != user_id:
                
                """ Make the prediction on the pred_on user """
                # Get user's MLP_dot representation
                user_MLP_dot = model.user_encoder(user_BERT_RT[pred_on_user])
                # Make predictions on all movies (1x128, 128x48272)
                logits = torch.mm(user_MLP_dot.unsqueeze(0), torch.transpose(item_MLP_RT, 0, 1))[0]
                pred = torch.sigmoid(logits)
                
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
               
                # Update pred_on with actual data
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








"""

OTHERS

"""



# DCG (Discounted Cumulative Gain)   
 
# Needed to compare rankings when the numbre of item compared are not the same
# and/or when relevance is not binary

def DCG(v, top):
    """
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
    
  
# RR (Reciprocal Rank)
    
# Gives a value in [0,1] for the first relevant item in list.
# 1st = 1 and than lower until cloe to 0.
# Only consern with FIRST relevant item in the list.
    
def RR(v):
    return 1/np.min(v)



# Recall@K for one item. 1 if the smallest rank in v is smaller than k, 0 if not.
    
def Recall_at_k_one_item(v, k=1):
    return 1 if np.min(v) <= k else 0



#def Ranks(all_values, values_to_rank, topx = 0):
#    """
#    Takes 2 numpy array and return, for all values in values_to_rank,
#    the ranks, average ranks, MRR and nDCG for ranks smaller than topx
#    """    
#    # If topx not mentionned (no top), it's for all the values
#    if topx == 0: topx = len(all_values)
#    
#    # Initiate ranks
#    ranks = np.zeros(len(values_to_rank))
#    
#    for i,v in enumerate(values_to_rank):
#        ranks[i] = len(all_values[all_values > v]) + 1
#        
#    ndcg = nDCG(ranks, topx, len(values_to_rank))
#    
#    if ranks.sum() == 0: print('warning, should always be at least one rank')
#    
#    return ranks, ranks.mean(), round(float((1/ranks).mean()),4), RR(ranks), \
#           Recall_at_k_one_item(ranks, 1), Recall_at_k_one_item(ranks, 10), \
#           Recall_at_k_one_item(ranks, 50), ndcg
    










def Ranks(all_values, indices_to_rank, ranking_method, topx = 0):
    """
    Takes 2 numpy array and return, for all values in indices_to_rank,
    the ranks, average ranks, MRR and nDCG for ranks smaller than topx
    """    
    # If topx not mentionned (no top), it's for all the values
    if topx == 0: topx = len(all_values)
    
    if ranking_method == 'min':
        qt_uniq = len(all_values.unique())
        assert qt_uniq > 48272 * 0.99, \
               "{} of predictions are equal, which is more than 1%. \
               USE --ranking_method 'ordinal'".format((1 - (qt_uniq/48272)))
    
    ranks = ss.rankdata(all_values.cpu(), method=ranking_method)[indices_to_rank]
        
    ndcg = nDCG(ranks, topx, len(indices_to_rank))
    
    if ranks.sum() == 0: print('warning, should always be at least one rank')
    
    return ranks, ranks.mean(), round(float((1/ranks).mean()),4), RR(ranks), \
           Recall_at_k_one_item(ranks, 1), Recall_at_k_one_item(ranks, 10), \
           Recall_at_k_one_item(ranks, 50), ndcg








    
def ChronoPlot(l_d, title, PATH, l_label= ['withOUT genres', 'with genres']):
    """
    Plot graph of list ofdicts, doing mean of values for each key
    """
    dmean = []    # global mean to return
    
#    d0x = []
#    d0y = []
#    d0err = []
#    d0mean = []    # global mean to return
    
    # For each dictionary
    for i in range(len(l_d)):
        dx = []
        dy = []
        derr = []
        dall = []
        
        # For each key
        for k, v in sorted(l_d[i].items()):
            if len(v) < 5:
                continue
            else:
                dx.append(k)
                dy.append(mean(v))
                derr.append(stdev(v))
                dall += v

#    for k, v in sorted(d0.items()):
#        if len(v) < 5:
#            continue
#        else:
#            d0x.append(k)
#            d0y.append(mean(v))
#            d0err.append(stdev(v))
#            d0mean += v
    
#    plt.plot(d1x, d1y, label=label1)
#    plt.plot(d0x, d0y, label=label2)  
        plt.errorbar(dx, dy, derr, elinewidth=0.5, label=l_label[i])
        print(title, ' ',l_label[i],' CHRONO VALUES:', dy)
        dmean.append(mean(dall))
        
    # ADDING BERT
    plt.errorbar([0,1,2,3,4,5,6,7,8], [0.18954871794871794, 0.20591032608695653, 0.18370689655172415, 0.13998529411764707, 0.13518518518518519, 0.12472826086956522, 0.11848101265822784, 0.13777777777777778, 0.11130434782608696], label='BERT')    
    
    plt.title(title, fontweight="bold")
    plt.xlabel('Nb of mentionned movies before prediction')
    plt.legend()
  # plt.show()
    plt.savefig(PATH+title+'.pdf')
    plt.close()
    
    return dmean
    
    
    
    
    
def EpochPlot(tup, title=''):
    """
    Plot graph of 4-tuples, doing mean of values
    """
        
    ygl = [wgl for (wgl, wnl, wgn, wnn) in tup]
    ynl = [wnl for (wgl, wnl, wgn, wnn) in tup]
    ygn = [wgn for (wgl, wnl, wgn, wnn) in tup]
    ynn = [wnn for (wgl, wnl, wgn, wnn) in tup]
    
    plt.plot(ygl, 'C0', label='Genres + Liked')
    plt.plot(ynl, 'C0--', label='No Genres + Liked')
    plt.plot(ygn, 'C1', label='Genres + Not Liked')
    plt.plot(ynn, 'C1--', label='No Genres & Not Liked')    
    plt.title(title, fontweight="bold")
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






