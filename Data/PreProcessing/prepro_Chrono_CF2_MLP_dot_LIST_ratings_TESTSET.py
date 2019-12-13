#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:27:39 2019


FROM: ReDial Conversations 

TO: A CSV with columns: ConvID(int) - Qt_movies_mentioned(int) - user_text(str) - item_text(str) - ratings (0,1)
    Each line = 1 user text , 1 movie text , rating given by that user to that movie


@author: nicholas
"""


###############
### IMPORTS
###############


import numpy as np
import json
import re
import pandas as pd



###############
### SETTINGS
###############


# Path to the Conversation file.           
Conv_PATH = '/Users/nicholas/Desktop/data_10021/test_data.json'
# Path to dict of conversion from ReDialId to UniversalID
ReDID2uID_PATH = '/Users/nicholas/ReDial/DataProcessed/ReDID2uID.npy'
# Path to dict of UiD to genres of this movie
genres_PATH = '/Users/nicholas/ReDial/DataProcessed/dict_UiD_genres.txt'


count_conv = 0 
max_nb_movie_rated = 19  # Evaluated elsewhere 




###############
### LOADING
###############


# Loading MovieLens in a DataFrame, only relevant columns
#df = pd.read_csv(ML_PATH, usecols = ['userId', 'movieId', 'rating'])

# Loading dict from ReDialId to UniversalID
ReDID2uID = np.load(ReDID2uID_PATH, allow_pickle=True).item()
# dict from UniversalID to ReDialId 
uID2ReDID = {v: k for k, v in ReDID2uID.items()}

## Loading a numpy array with [text with @, text with titles, _, _]
#text_data = np.load('/Users/nicholas/GitRepo/DialogueMovieReco/Data/DataConvFilm.npy')

# Loading dict of genres by movies UiD
with open (genres_PATH, 'rb') as f:
    UiD_genres_dict = json.load(f)





#%%


#####################
### TEXT MANAGEMENT 
#####################


### TO EXTRACT Movies Mentions with @
re_filmId = re.compile('@[0-9]{5,6}')
### TO EXTRACT Date after movie title
re_date = re.compile(' *\([0-9]{4}\)| *$') 

### To get the movie titles with a ReDial ID (when replacing in text directly)
#df_filmID = pd.read_csv('/Users/nicholas/Desktop/data_10021/movies_db.csv')

### To get the movie titles with UiD and ReDial ID (when replacing in text directly)
df_movie_merge = pd.read_csv('/Users/nicholas/Desktop/data_10021/MoviesMergedId_updated181030.csv', encoding='latin')


## Function called by .sub
## A re method to substitue matching pattern in a string
## Here, substitute film ID with film NL title   (COMMENTED ADDITION OF GENRE IN TEXT)
#
#def filmIdtoString(match):
#    filmId = match.group()[1:]               # Remove @ 
#    if df_filmID[df_filmID['movieId'] == int(filmId)].empty:
#        print('Unknow movie', filmId)
#        film_str = filmId                    # Put film ID since can't find title
#    else:
#        film_str_with_date = df_filmID[df_filmID['movieId'] == int(filmId)][' movieName'].values[0]
#        film_str = re_date.sub("", film_str_with_date)  # Remove date for more NL
#    #    film_genres = UiD_genres_dict[str(ReDID2uID[int(filmId)])]
#    #    film_str += " (" + " ".join(film_genres) + ")"
#    
#    return film_str


# Function called by .sub
# A re method to substitue matching pattern in a string
# Here, substitute film ID with film NL title   (COMMENTED ADDITION OF GENRE IN TEXT)

def filmIdtoString(match):
    filmId = match.group()[1:]               # Remove @ 
    film_str_with_date = df_movie_merge[df_movie_merge['databaseId'] == \
                                        int(filmId)]['movieName'].values[0]
    film_str = re_date.sub("", film_str_with_date)  # Remove date for more NL
    
    return film_str

#
## Function to return the KB info about a movie. Now KB is Movie title fallowed by genre
## INPUT: Movie UiD (int)
## OUTPOUT: KB about the movie (str), tirle(str), genres(str)
#    
#def KB_of_movie(m_UiD):
#    # Convert uID to ReDID
#    m = uID2ReDID[m_UiD]
#    # get the title
#    if df_filmID[df_filmID['movieId'] == m].empty:
#        print('Unknow movie', m)
#        title = str(m)                    # Put film ID since can't find title
#    else:
#        film_str_with_date = df_filmID[df_filmID['movieId'] == m][' movieName'].values[0]
#        title = re_date.sub("", film_str_with_date)  # Remove date for more NL
#    # get the genre
#    genre = ', '.join(UiD_genres_dict[str(m_UiD)])
#        
#    return title + '. Genre: ' + genre, title, genre


# Function to return the KB info about a movie. Now KB is Movie title fallowed by genre
# INPUT: Movie UiD (int)
# OUTPOUT: KB about the movie (str), tirle(str), genres(str)
    
def KB_of_movie(m):
    # get the title
    film_str_with_date = df_movie_merge[df_movie_merge['index'] == m]['movieName'].values[0]
    title = re_date.sub("", film_str_with_date)  # Remove date for more NL
    # get the genre
    genre = ', '.join(UiD_genres_dict[str(m)])
        
    return title + '. Genre: ' + genre, title, genre


#%%

# Create train and user data
test_data = []


# Create user and item Relational Tables
user_RT = []
u_id = 0
data_id = 0


# For all conversations
for line in open(Conv_PATH, 'r'):
    
    # Data for this conversation
    data = []
    uRT = []

    # Load conversation in a dict
    conv_dict = json.loads(line)
    
    # Get the conversation ID
    ConvID = int(conv_dict['conversationId'])
    # Get Seeker ID
    seekerID = conv_dict['initiatorWorkerId']
    # Get Recommender ID
    recommenderID = conv_dict['respondentWorkerId']

    ##### Get a list of all the ratings and movies_rated 
    l_ratings = []
    # First, get an non-empty movie form 
    # (seeker first, recommender 2nd, drop if none)
    if conv_dict['initiatorQuestions'] != []:
        questions_dict = conv_dict['initiatorQuestions']
    elif conv_dict['respondentQuestions'] != []:
        questions_dict = conv_dict['respondentQuestions']
    else:
        continue
    # Second, treat all movies in movie form
    for movieReDID, values in questions_dict.items():
        # If we know the rating (==2 would be did not say)
        if values['liked'] == 0 or values['liked'] == 1:
            # Get the movie uID
            movieuID = ReDID2uID[int(movieReDID)]
            # Get the rating according to the liked value
            rating = values['liked']
            l_ratings.append((movieuID, rating))
            l_movies_rated = [m for m, r in l_ratings]

    ##### Manage all texts in a Conversation 1x1     
    text_buffer = ""
    count_original_mentions = 0
    for message in conv_dict['messages']:
        # scan for @ movies mentions
        all_movies = re_filmId.findall(message['text'])
        
        
        ### If movies are mentionned in this message
        if all_movies != []:
             # If first utterance has a movie mention, simply add this message in text buffer
            if text_buffer == "":  
                speaker = message['senderWorkerId']
                if speaker == seekerID: speakerID = 'S:: '
                else: speakerID = 'R:: '   
                message_in_NL = re_filmId.sub(filmIdtoString, message['text']) 
                text_buffer += speakerID + message_in_NL + ' '  # Add new text with NL title
                count_original_mentions += len(all_movies)   # Count these mentions
                # Check for extrem case where only one rated movie and 
                # there is a movie mention in first utterance. If so, go to next Conv
                if count_original_mentions >= len(l_ratings): break
                continue            # But don't try to predict on empty str
                
            # Get the movies mentionned as UiD
            movies_found = [ ReDID2uID[int(film[1:])] for film in all_movies ]                

            l_ratings_to_come = []
            # If next rating (of movies never mentionned yet) is mentionned in movies found, 
            # use this list of ratings (of movies never mentionned yet) as ratings to come
            if l_movies_rated[count_original_mentions] in movies_found:
                l_ratings_to_come = l_ratings[count_original_mentions:]
            if l_ratings_to_come != []:              
#                # Fill to have list of same lenght
#                fill_size = max_nb_movie_rated - len(l_ratings_to_come)
#                filling = [(-1,0)] * fill_size
#                l_ratings_to_come += filling
#                # Add the number of movies mentioned
#                l_ratings_to_come = [(-2,count_original_mentions)] + l_ratings_to_come
                
                
                
                
#                # For all ratings to come, add a data point 
##                qt_movies_mentioned = l_ratings_to_come[0][1]
#                for m,r in l_ratings_to_come:
##                    if m == -1: break       # Reached the filling
#                    data.append([data_id, ConvID, count_original_mentions, u_id, m, r])
#                    data_id += 1
                
                
                # From [(movies, ratings)] to [[movies], [ratings]]
                l_m = []
                l_r = []
                for m, r in l_ratings_to_come:
                    l_m.append(m)
                    l_r.append(r)
                    
                data.append([data_id, ConvID, count_original_mentions, u_id, l_m, l_r])
                data_id += 1
                
                
                count_original_mentions += sum([movies_found_i in l_movies_rated[count_original_mentions:] \
                                                for movies_found_i in movies_found])
                uRT.append([u_id, text_buffer])
                u_id += 1
#                # Put list of ratings in text type (for .csv purposes in BertReco)
#                l_ratings_to_come = str(l_ratings_to_come)               
#                data.append([ConvID, text_buffer, \
#                             l_ratings_to_come])

                # It's finish for this Conv id all mentions have been turned into data point
                if count_original_mentions >= len(l_ratings): break
          
            
        ### If no movies are mentioned in this message, to next one
        ## Identify the speaker
        # If it's first message (case without movie mention in message)
        if text_buffer == "":
            speaker = message['senderWorkerId']
            if speaker == seekerID: speakerID = 'S:: '
            else: speakerID = 'R:: '   
        # If not first message, check if same speaker as previous text
        else:
            # Same speaker, don't mention anything
            if speaker == message['senderWorkerId']:
                speakerID = ""
            # New speaker, get his ID    
            else:
                speaker = message['senderWorkerId']
                if speaker == seekerID: speakerID = 'S:: '
                else: speakerID = 'R:: '   
                
        message_in_NL = re_filmId.sub(filmIdtoString, message['text']) 
        text_buffer += speakerID + message_in_NL + ' '  # Add new text with NL title 

        
        
    # Update user Relational Table
    user_RT += uRT
    
    # Put data in the right set 
    test_data += data
         
    count_conv += 1
  #  if count_conv > 7: break
   

#%%

# Creating a DataFrame and saving it

df = pd.DataFrame(test_data)
df.columns = ['data_idx', 'ConvID', 'qt_movies_mentioned', 'user_chrono_id', 'movie_UiD', 'rating']
df.to_csv('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDialML/Test_LIST.csv', index=False)














