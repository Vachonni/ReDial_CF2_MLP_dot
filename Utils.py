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


#import Settings




"""
DATASET - Sub classes of Pytorch DATASET to prepare for Dataloader

"""




class RnGChronoDataset(data.Dataset):
    """    
    
    ****** Now inputs and targets are seperated in data ******
    
    
    INPUT: 
        RnGlist format is:
            ["ConvID", [(UiD, Rating) mentionned], ["genres"], [(UiD, Rating) to be mentionned]]
        top_cut is the number of movies in genres vector
        If data from a non-chrono dataset (eg: ML), all data in mentionned (to be mentionned empty).
    
    RETUNRS:
        masks, (inputs, genres) and targets. 
        Genres is vector with value for top_cut movies of intersection of genres mentionned 
        by user, normalized (or deduced for ML).

    """
    
    def __init__(self, RnGlist, dict_genresInter_idx_UiD, nb_movies, popularity, DEVICE, \
                 exclude_genres=False, no_data_merge=False, noise=False, top_cut=100):
        self.RnGlist = RnGlist
        self.dict_genresInter_idx_UiD = dict_genresInter_idx_UiD
        self.nb_movies = nb_movies
        self.popularity = popularity
        self.DEVICE = DEVICE
        self.exclude_genres = exclude_genres
        self.no_data_merge = no_data_merge
        self.noise = noise
        self.top_cut = top_cut
        
        
    def __len__(self):
        "Total number of samples. Here one sample corresponds to a new mention in Conversation"
        return len(self.RnGlist)


    def __getitem__(self, index):
        "Generate one sample of data."
        
        # Get list of movies and ratings for user number (=index) 
        ConvID, l_inputs, l_genres, l_targets = self.RnGlist[index]
        
        # Init
        # For inputs, we consider maximum 8 movies (after, too little data in ReDial)
        # Add 48273 for the empty embedding at position 48273 (when not 8 inputs)
        inputs = torch.zeros(9, dtype=torch.int64) + 48273
        # Size 9 because fist token is the [RECO] token, at idx 48272
        inputs[0] = 48272 
        targets = torch.zeros(self.nb_movies)   
        masks_inputs = torch.zeros(self.nb_movies)
        masks_targets = torch.zeros(self.nb_movies)
        genres = torch.zeros(self.nb_movies)

        # If we merge data (mentionned and to be mentionned)
        if not self.no_data_merge:
            sum_i_t = l_inputs + l_targets
            l_inputs = sum_i_t
            l_targets = sum_i_t
        
        # Manage noise
        if self.noise:
            min_input = max(2, len(l_inputs))    # Insure p choice is at least 1
            max_input = min(7, min_input)        # Insure not more than qt in l_inputs
            p = torch.randint(1, max_input, (1,)).type(torch.uint8)  
            ind_to_take = torch.randperm(len(l_inputs))[:p]
            l_inputs = torch.IntTensor(l_inputs)
            l_inputs = l_inputs[ind_to_take]
            # Same for targets 
            p = torch.randint(1, max_input, (1,)).type(torch.uint8)  
            ind_to_take = torch.randperm(len(l_inputs))[:p]
            l_targets = torch.IntTensor(l_targets)
            l_targets = l_targets[ind_to_take]        
        
        # Inputs
        input_idx = 1   # 0 is the [RECO] token
        for uid, rating in l_inputs:
            if rating == 1:
                inputs[input_idx] = uid
                input_idx += 1
            masks_inputs[uid] = 1
            if input_idx >= 8: break
                
        # Targets 
        for uid, rating in l_targets:
            targets[uid] = rating
            masks_targets[uid] = 1
        
        # Genres
        if not self.exclude_genres:
            # Turn list of genres into string
            str_genres = str(l_genres)
            # Try - if no movies of that genres (key error)
            try:
                genres_idx, l_genres_uid = self.dict_genresInter_idx_UiD[str_genres] 
            except:
       #         print('No movie with genres:', str_genres)
                genres_idx = 1
            # If there is a genres...   (no else needed, since already at 0)
            if genres_idx != 1:
                for uid in l_genres_uid:
                    genres[uid] = 1.0
                    
                """normalization and popularity"""
                # If we want popularity (i.e. args.no_popularity == True)
                if not torch.equal(self.popularity, torch.ones(1)):     
                    # Include popularity in genres
                    genres = genres * self.popularity
                    # Take top 100 movies according to popularity
                    genres_cut = torch.zeros(self.nb_movies)
                    genres_cut[genres.topk(self.top_cut)[1]] = genres.topk(self.top_cut)[0]
                    genres = genres_cut  
                # Normalize vector
                genres = torch.nn.functional.normalize(genres, dim=0)
        
        
        return (masks_inputs.to(self.DEVICE), masks_targets.to(self.DEVICE)), \
               (inputs.to(self.DEVICE), (genres_idx, genres.to(self.DEVICE))), \
               targets.to(self.DEVICE)








"""

TRAINING AND EVALUATION 

"""



def TrainReconstruction(train_loader, model, criterion, optimizer, zero1, weights_factor, completion):
    model.train()
    train_loss = 0
    nb_batch = len(train_loader) * completion / 100
    
    
    
    """ """
    pred_mean_values = []
    
    """ """
   
    print('TRAINING')
     
    for batch_idx, (masks, inputs, targets) in enumerate(train_loader):
        
        # Early stopping
        if batch_idx > nb_batch: 
            print(' *EARLY stopping')
            break
        
        # Print update
        if batch_idx % 100 == 0: 
            print('Batch {:4d} out of {:4.1f}.    Reconstruction Loss on targets: {:.4f}'\
                  .format(batch_idx, nb_batch, train_loss/(batch_idx+1)))  
                
        # Add weights on targets rated 0 because outnumbered by targets 1
        weights = (masks[1] == 1).float() * (targets == 0).float() \
                                * weights_factor + torch.ones_like(targets)
        criterion.weight = weights.float()
        

        """ UNcomment TO TRAIN WITHOUT GENRES"""
        # Genres removed from inputs 
#        inputs[1][0] = torch.ones(inputs[0].size(0), dtype=torch.uint8)
#        inputs[1][1] = torch.zeros(inputs[0].size(0), 48272)

        # re-initialize the gradient computation
        optimizer.zero_grad()   


        # Making inputs 0 = not seen, -1 = not liked and 1 = liked
        if zero1 == 1:
            inputs[0] = 2*inputs[0] - masks[0]        
        # Making inputs 0 = not seen, 1 = not liked and 2 = liked
        if zero1 == 2:
            inputs[0] = inputs[0] + masks[0]
            
            
        pred = model(inputs)

            
        
        
        """ To look into pred values evolution during training"""
#        if model.model_pre.lla == 'none':
#            pred_mean_values.append((torch.nn.Sigmoid()(pred.detach())).mean())
#        else:
#            pred_mean_values.append((pred.detach()).mean())
        pred_mean_values.append((pred.detach()).mean())
        """ """
        
        
        
     
        """ Try reconstruction including genres """
#        targets = targets + model.g * inputs[1][1]
#        masks = masks * (inputs[1][1] != 0).float()
     
        

        loss = (criterion(pred, targets) * masks[1]).sum()
        assert loss >= 0, 'Getting a negative loss in training - IMPOSSIBLE'
        # Using only predictions of movies that were rated in targets
     #   pred = pred * masks
        nb_ratings = masks[1].sum()
        loss /= nb_ratings
        loss.backward()
        optimizer.step()
        
        # Remove weights for other evaluations
        criterion.weight = None
        loss_no_weights = (criterion(pred, targets) * masks[1]).sum() 
        loss_no_weights /= nb_ratings
        train_loss += loss_no_weights.detach()
        
    train_loss /= nb_batch
        
    return (pred_mean_values, train_loss) 





def EvalReconstruction(valid_loader, model, criterion, zero1, completion):
    model.eval()
    eval_loss = 0
    nb_batch = len(valid_loader) * completion / 100
    
    print('\nEVALUATION')
    
    with torch.no_grad():
        for batch_idx, (masks, inputs, targets) in enumerate(valid_loader):
            
            # Early stopping 
            if batch_idx > nb_batch: 
                print(' *EARLY stopping')
                break
            
            # Print update
            if batch_idx % 100 == 0: 
                print('Batch {:4d} out of {:4.1f}.    Reconstruction Loss on targets: {:.4f}'\
                      .format(batch_idx, nb_batch, eval_loss/(batch_idx+1)))  
                   
            # Making inputs 0 = not seen, -1 = not liked and 1 = liked
            if zero1 == 1:
                inputs[0] = 2*inputs[0] - masks[0]        
            # Making inputs 0 = not seen, 1 = not liked and 2 = liked
            if zero1 == 2:
                inputs[0] = inputs[0] + masks[0]
            
            pred = model(inputs)  
            
            # Using only movies that were rated in targets
        #   pred = pred * masks
            nb_ratings = masks[1].sum()
       #     del(masks)
            loss = (criterion(pred, targets) * masks[1]).sum()
            assert loss >= 0, 'Getting a negative loss in eval - IMPOSSIBLE'
            loss = loss / nb_ratings
            eval_loss += loss
    
    eval_loss /= nb_batch 
    
    return eval_loss




def EvalPredictionGenresRaw(loader, model, criterion, zero1, completion):
    """
    Same as EvalPredictionGenres, but values returned are complete (not their mean)
    """
    model.eval()
    
    # In order: g (genres) and l (liked). If not, then used 'n'
    l_loss_gl = []
    l_loss_nl = []
    l_loss_gn = []
    l_loss_nn = []
    l_avrg_rank_gl= []
    l_avrg_rank_nl= []
    l_avrg_rank_gn= []
    l_avrg_rank_nn= []
    l_rr_gl = []
    l_rr_nl = []    
    l_rr_gn = []    
    l_rr_nn = []
    l_ndcg_gl = []
    l_ndcg_nl = []
    l_ndcg_gn = []
    l_ndcg_nn = []
    
    nb_batch = len(loader) * completion / 100
    
    with torch.no_grad():
        # For each user
        for batch_idx, (masks, inputs, targets) in enumerate(loader):
            
            
            # Early stopping 
            if batch_idx > nb_batch or nb_batch == 0: 
                print('EARLY stopping')
                break
                
              
            # Print Update
            if batch_idx % 1000 == 0:
                print('Batch {} out of {}.'.format(batch_idx, nb_batch))   
                
            # Making inputs 0 = not seen, -1 = not liked and 1 = liked
            if zero1 == 1:
                inputs[0] = 2*inputs[0] - masks[0]        
            # Making inputs 0 = not seen, 1 = not liked and 2 = liked
            if zero1 == 2:
                inputs[0] = inputs[0] + masks[0]

           
            count_pred = 0
            # For each movie in inputs
            for i, mask in enumerate(masks[0][0]):        # second [0] because loader returns 'list of list' (batches usually > 1)
                if mask == 1:                             # If a rated movie    
                    
                    # Get the rating for movie i
                    r = inputs[0][0][i].clone().detach()         # To insure deepcopy and not reference
                    # "Hide" the rating of movie m
                    inputs[0][0][i] = 0
                    # Get the predictions
                    pred = model(inputs)
                    # Put the ratings in original condition  
                    inputs[0][0][i] = r
                    # Evaluate error
                    error = criterion(pred[0][i], r)
                    # Ranking of this prediction among all predictions. 
                    # ranks is a 2D array of size (nb of pred with same value as m, position of value)
                    
                    
                    """ Adding Sigmoid to pred if BCELogits used """
                    if model.model_pre.lla == 'none':
                        pred = torch.nn.Sigmoid()(pred)
                    """ """
                    
                 #   ranks = (torch.sort(pred[0][Settings.l_ReDUiD], descending=True)[0] == pred[0][i]).nonzero() + 1
#                    ranks_old = (torch.sort(pred[0], descending=True)[0] == pred[0][i]).nonzero() + 1
                    """ Trying to use function Ranks for consistency"""
                    _, avrg_rank, _, rr, ndcg = Ranks(pred[0], pred[0][i].view(-1))
           #         ranks = torch.from_numpy(ranks).view(1,-1)
#                    if (ranks_old[0,0] == ranks[0,0].long()).all().sum() != 1:
#                        print('\n\n** DIFFERENT RANKS **',ranks_old[0,0], ranks[0,0].long(),'\n\n')
                    """ """
                 #   print("Value of rating (r) is:", r)
                 
                    # Manage value of r (rating) according to zero1
                    r_liked = False
                    if zero1 == 0 and r == 1: r_liked = True
                    if zero1 == 1 and r == 1: r_liked = True
                    if zero1 == 2 and r == 2: r_liked = True
                    
                    # with genres and liked (gl case) 
                    if inputs[1][0][0] != 1 and r_liked:
                        l_loss_gl.append(error.item())
                        l_avrg_rank_gl.append(avrg_rank)
                        l_rr_gl.append(rr)
                        l_ndcg_gl.append(ndcg)
                    # with NO genres and liked (nl case) 
                    if inputs[1][0][0] == 1 and r_liked:
                        l_loss_nl.append(error.item())
                        l_avrg_rank_nl.append(avrg_rank)
                        l_rr_nl.append(rr)
                        l_ndcg_nl.append(ndcg)
                    # with genres and DISliked (gn case) 
                    if inputs[1][0][0] != 1 and not r_liked:
                        l_loss_gn.append(error.item())
                        l_avrg_rank_gn.append(avrg_rank)
                        l_rr_gn.append(rr)
                        l_ndcg_gn.append(ndcg)
                    # with NO genres and DISliked (nn case) 
                    if inputs[1][0][0] == 1 and not r_liked:
                        l_loss_nn.append(error.item())
                        l_avrg_rank_nn.append(avrg_rank)
                        l_rr_nn.append(rr)
                        l_ndcg_nn.append(ndcg)
                    
                 # Early stoopping
                    count_pred += 1 
                    if count_pred > len(masks[0][0].nonzero()) * completion / 100: 
                     #   print('stop at {} prediction for user'.format(count_pred), len(masks[0][0].nonzero()))
                        break


    return l_loss_gl, l_loss_nl, l_loss_gn, l_loss_nn, \
           l_avrg_rank_gl, l_avrg_rank_nl, l_avrg_rank_gn, l_avrg_rank_nn, \
           l_rr_gl, l_rr_nl, l_rr_gn, l_rr_nn, \
           l_ndcg_gl, l_ndcg_nl, l_ndcg_gn, l_ndcg_nn






def EvalPredictionRnGChrono(valid_loader, model, criterion, zero1, without_genres, \
                            pred_not_liked, completion, topx=100):
    """
    Prediction on targets = to be mentionned movies...
    
    ** Only works with RnGChronoDataset **
    
    """
    model.eval()
    nb_batch = len(valid_loader) * completion / 100
    
    eval_loss_with_genres = 0
    eval_loss_without_genres = 0
    
    results_error_with_genres = {}
    results_error_without_genres = {}
    results_Avrg_Ranks_with_genres = {}
    results_Avrg_Ranks_without_genres = {}
    results_MRR_with_genres = {}
    results_MRR_without_genres = {}
    results_RR_with_genres = {}
    results_RR_without_genres = {}
    results_DCG_with_genres = {}
    results_DCG_without_genres = {}
                
                
    with torch.no_grad():
        for batch_idx, (masks, inputs, targets) in enumerate(valid_loader):
            
            # Early stopping 
            if batch_idx > nb_batch or nb_batch == 0: 
                print('EARLY stopping')
                break
            
            # Print Update
            if batch_idx % 10 == 0:
                print('Batch {} out of {}.  Loss:{}'\
                      .format(batch_idx, nb_batch, eval_loss_with_genres/(batch_idx+1)))
             
            # Making inputs 0 = not seen, -1 = not liked and 1 = liked
            if zero1 == 1:
                inputs[0] = 2*inputs[0] - masks[0]        
            # Making inputs 0 = not seen, 1 = not liked and 2 = liked
            if zero1 == 2:
                inputs[0] = inputs[0] + masks[0]
            
    # WITH GENRES
            # Make a pred
            pred = model(inputs)  
            
            # LOSS - Using only movies to be mentionned that were rated
       #     pred_masked = pred * masks[1]
            nb_ratings = masks[1].sum()
            loss = (criterion(pred, targets) * masks[1]).sum()
            loss = loss / nb_ratings
            eval_loss_with_genres += loss
    
    
            """ Adding Sigmoid to pred if BCELogits used """
        #    if model.model_pre.lla == 'none':
        #        pred = torch.nn.Sigmoid()(pred)
            """ """
    
    
            # NRR & NDCG
            # Need to evaluate each samples seperately, since diff number of targets
            # For each sample in the batch
#            for i, sample in enumerate(pred[:,Settings.l_ReDUiD]):
            for i in range(len(pred)):
                
                
                # Prediction on DISLIKED targets
                if pred_not_liked:
                    l_or_n_targets = masks[1][i] - targets[i]
                # ...or on LIKED targets
                else:
                    l_or_n_targets = targets[i]
                    
                    
                # Insure their is at least one target movie 
                # (if not, sample not considered)
                if l_or_n_targets.sum() == 0: continue
            
                
                # Get error on pred ratings
                error = ((criterion(pred[i], l_or_n_targets) * masks[1][i]).sum() / masks[1][i].sum()).item()
                # ... get Ranks for targets (not masks[1] because only care about liked movies)
                rk, avrg_rk, mrr, rr, ndcg = Ranks(pred[i], \
                                                  pred[i][l_or_n_targets.nonzero().flatten().tolist()],\
                                                  topx)  
                # Get the number of inputs mentionned before prediction
                qt_mentionned_before = masks[0][i].sum(dtype=torch.uint8).item()
                
                # Add Ranks results to appropriate dict
                if qt_mentionned_before in results_RR_with_genres.keys():
                    results_error_with_genres[qt_mentionned_before].append(error)
                    results_Avrg_Ranks_with_genres[qt_mentionned_before].append(avrg_rk)
                    results_MRR_with_genres[qt_mentionned_before].append(mrr)
                    results_RR_with_genres[qt_mentionned_before].append(rr)
                    results_DCG_with_genres[qt_mentionned_before].append(ndcg)
                else:
                    results_error_with_genres[qt_mentionned_before] = [error]
                    results_Avrg_Ranks_with_genres[qt_mentionned_before] = [avrg_rk]
                    results_MRR_with_genres[qt_mentionned_before] = [mrr]
                    results_RR_with_genres[qt_mentionned_before] = [rr]
                    results_DCG_with_genres[qt_mentionned_before] = [ndcg]

            
            
    # WITHOUT GENRES
            if without_genres:
                # Make a pred with genres removed from inputs. Genres indx at 1 and Genres UiD at 0 
                inputs[1][0] = inputs[1][0] + 0 + 1
                inputs[1][1] = inputs[1][1] * 0
                pred = model(inputs)  
                
                # LOSS - Using only movies to be montionned that were rated
           #     pred_masked = pred * masks[1]
                nb_ratings = masks[1].sum()
                loss = (criterion(pred, targets) * masks[1]).sum()
                loss = loss / nb_ratings
                eval_loss_without_genres += loss
        
        
                """ Adding Sigmoid to pred if BCELogits used """
                if model.model_pre.lla == 'none':
                    pred = torch.nn.Sigmoid()(pred)
                """ """
        
        
                # NRR & NDCG
                # Need to evaluate each samples seperately, since diff number of targets
                # For each sample in the batch
    #            for i, sample in enumerate(pred[:,Settings.l_ReDUiD]):
                for i in range(len(pred)):
                    
                    
                    # Prediction on DISLIKED targets
                    if pred_not_liked:
                        l_or_n_targets = masks[1][i] - targets[i]
                    # ...or on LIKED targets
                    else:
                        l_or_n_targets = targets[i]
        
        
                    # Insure their is at least one target movie 
                    # (if not, sample not considered)
                    if l_or_n_targets.sum() == 0: continue
                
            
                    # Get error on pred ratings
                    error = ((criterion(pred[i], l_or_n_targets) * masks[1][i]).sum() / masks[1][i].sum()).item()         
                    # ... get Ranks for targets (not masks[1] because only care about liked movies)
                    rk, avrg_rk, mrr, rr, ndcg = Ranks(pred[i], \
                                                      pred[i][l_or_n_targets.nonzero().flatten().tolist()],\
                                                      topx)
                    # Get the number of inputs mentionned before prediction
                    qt_mentionned_before = masks[0][i].sum(dtype=torch.uint8).item()
                    
                    # Add Ranks results to appropriate dict
                    if qt_mentionned_before in results_RR_without_genres.keys():
                        results_error_without_genres[qt_mentionned_before].append(error)
                        results_Avrg_Ranks_without_genres[qt_mentionned_before].append(avrg_rk)
                        results_MRR_without_genres[qt_mentionned_before].append(mrr)
                        results_RR_without_genres[qt_mentionned_before].append(rr)
                        results_DCG_without_genres[qt_mentionned_before].append(ndcg)
                    else:
                        results_error_without_genres[qt_mentionned_before] = [error]
                        results_Avrg_Ranks_without_genres[qt_mentionned_before] = [avrg_rk]
                        results_MRR_without_genres[qt_mentionned_before] = [mrr]
                        results_RR_without_genres[qt_mentionned_before] = [rr]
                        results_DCG_without_genres[qt_mentionned_before] = [ndcg]
    
         #  if batch_idx > 10: break
    
        eval_loss_with_genres /= nb_batch 
        eval_loss_without_genres /= nb_batch


    return eval_loss_with_genres, eval_loss_without_genres, \
           results_error_with_genres, results_error_without_genres, \
           results_Avrg_Ranks_with_genres, results_Avrg_Ranks_without_genres, \
           results_MRR_with_genres, results_MRR_without_genres,\
           results_RR_with_genres, results_RR_without_genres,\
           results_DCG_with_genres, results_DCG_without_genres







#"""
#
#CLASSES
#
#"""
#
#
#class Conversation:
#    """
#    Class to work with the original Conversation Data from ReDial
#    """
#    
#    def __init__(self, json):
#        
#        self.json = json
#        self.id = json["conversationId"]
#        self.movie_mentions = [k for k in json["movieMentions"]]
#        self.movie_seek_liked = [m for m in json["initiatorQuestions"] if \
#                                 json["initiatorQuestions"][m]["liked"] == 1 and \
#                                 json["initiatorQuestions"][m]["liked"] == 1]
#        self.movie_seek_notliked = [m for m in json["initiatorQuestions"] if \
#                                   json["initiatorQuestions"][m]["liked"] == 0 and \
#                                   json["initiatorQuestions"][m]["liked"] == 0]
#        self.seek_wId = self.json["initiatorWorkerId"]
#        self.recom_wId = self.json["respondentWorkerId"]
#    
#    
#            
#    def getSeekRecomText(self):
#        self.seek_text = []
#        self.recom_text = []
#        
#        for msg in self.json["messages"]:
#            if msg["senderWorkerId"] == self.seek_wId:
#                self.seek_text.append(msg["text"])
#            else:
#                self.recom_text.append(msg["text"])
#                               
#                
#    def getSeekerGenres(self, genres_to_find):
#        # Get unique genres mentionned in all ut of seekers 
#        self.genres_seek = getGenresListOfTextToOneList(self.seek_text, genres_to_find)
#
#
#def getGenresFromOneText(text, genres_to_find):
#    """
#    Take a string
#    Returns list of genres (strings) mentionned in text
#    
#    EXAMPLE: 
#        In: "Hey everybody, meet warren, he's a kid a bit drama"
#        Out: ['drama', 'kid']
#    """
#    genres_in_text = []
#    # Get list of unique words mentionned in text
#    words = nltk.word_tokenize(text.lower())
#    for g in genres_to_find:
#        for w in words:
#            if g == w: 
#                genres_in_text.append(g)
#    return genres_in_text
#
#
#
#def getGenresListOfTextToOneList(l_text, genres_to_find):
#    """
#    Take a list of strings
#    Returns list of unique genres (strings) mentionned in all texts
#    
#    EXAMPLE:
#        In: ["Hey everybody, meet warren, he's a kid a bit drama", 
#             "Sentence with no genre",
#             "Genres repeating, like drama",
#             "Horror movies are fun"]
#        Out: ['drama', 'kid', 'horror']
#    """
#        
#    l_genres = []
#    for text in l_text:
#        genres_in_text = getGenresFromOneText(text, genres_to_find)
#        # Concat only if genres retreived  
#        if genres_in_text != []:
#            l_genres += genres_in_text
#    # Return without duplicates
#    return list(set(l_genres))
#





"""

OTHERS

"""


#def Splitting(l_items, ratio_1, ratio_2, ratio_3):
#    """
#    Splitting a list of items randowly, into sublists, according to ratios.
#    Returns the 3 sublist (2 could be empty)
#    """
#    # Make sure ratios make sense
#    if ratio_1 + ratio_2 + ratio_3 != 1:
#        raise Exception("Total of ratios need to be 1, got {}".format(ratio_1 + ratio_2 + ratio_3))
#    size_1 = round(ratio_1 * len(l_items))
#    size_2 = round(ratio_2 * len(l_items))
#    np.random.shuffle(l_items)
#    sub_1 = l_items[:size_1]
#    sub_2 = l_items[size_1:size_1+size_2]
#    sub_3 = l_items[size_1+size_2:]
#
#    return sub_1, sub_2, sub_3 
#
#
#
#def SplittingDataset(full_dataset, ratio_train, ratio_valid, ratio_test):
#    """
#    Splitting a torch dataset into Train, Valid and Test sets randomly.
#    Returns the 3 torch datasets
#    """
#    train_size = round(ratio_train * len(full_dataset))
#    valid_size = round(ratio_valid * len(full_dataset))
#    test_size = len(full_dataset) - train_size - valid_size
#    # Split train & valid from test 
#    train_n_valid_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size + valid_size, test_size])
#    # Split train and valid
#    train_dataset, valid_dataset = torch.utils.data.random_split(train_n_valid_dataset, [train_size, valid_size])
#    return train_dataset, valid_dataset, test_dataset
    


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

    

def Ranks(all_values, values_to_rank, topx = 0):
    """
    Takes 2 numpy array and return, for all values in values_to_rank,
    the ranks, average ranks, MRR and nDCG for ranks smaller than topx
    """    
    # If topx not mentionned (no top), it's for all the values
    if topx == 0: topx = len(all_values)
    
    # Initiate ranks
    ranks = np.zeros(len(values_to_rank))
    
    for i,v in enumerate(values_to_rank):
        ranks[i] = len(all_values[all_values > v]) + 1
        
    ndcg = nDCG(ranks, topx, len(values_to_rank))
    
    if ranks.sum() == 0: print('warning, should always be at least one rank')
    
    return ranks, ranks.mean(), round(float((1/ranks).mean()),4), RR(ranks), ndcg
    


    
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






