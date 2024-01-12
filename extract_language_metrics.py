# import libraries
import pandas as pd
import os
import json
import numpy as np
from nltk import word_tokenize
import warnings
import time
warnings.filterwarnings("ignore")

from utils import clean_text, clean_comments


#### Load collective action dictionary defined by Smith et al. in "After Aylan Kurdi: How Tweeting About Death, Threat, and Harm Predict Increased Expressions of Solidarity With Refugees Over Time"####
df_collective_action = pd.read_csv("./collective_action_dic.csv", sep=";", header=None, names=["word"])


#### Functions ####

def match_word(word, word_list):
    '''
    Function to match word with list of words.

    Args:
    word (str): input word to be matched
    word_list (list): list of words to match with

    Returns:
    True if word matches with any word in word_list, False otherwise
    '''
    for w in word_list:
        if w.endswith("*"):
            if word.startswith(w[:-1]):
                return True
        else:
            if word == w:
                return True
    return False


def get_coll_action(df, text_column):
    '''
    Function to get collective action relative frequency from text.

    Args:
    df (dataframe): dataframe with text column
    text_column (str): name of text column

    Returns:
    df (dataframe): dataframe with collective action relative frequency
    '''
    # small preprocessing of text for matching
    
    # remove punctuation
    df[text_column] = df[text_column].str.replace('[^\w\s]','')
    # tokenize
    df[text_column] = df[text_column].apply(lambda x: word_tokenize(x))
    # lowercase
    df[text_column] = df[text_column].apply(lambda x: [word.lower() for word in x])
    # length of text
    df["text_length"] = df[text_column].apply(lambda x: len(x))

    # # if length of text is less than 3, set to nan
    # df.loc[df["text_length"] < 3, text_column] = np.nan

    # freq of collective action words, use match_word function, take into account nan values
    df["collective_action freq"] = df[text_column].apply(lambda x: sum([match_word(word, df_collective_action["word"].values) for word in x]) if x is not np.nan else np.nan)
    df["collective_action rfreq"] = df["collective_action freq"] / df["text_length"]
    
    return df

#### Prepare comments data ####

# read comments
data = []
for file in os.listdir("./data/comments/"):
    if file.endswith(".json"):
        with open("./data/comments/"+file, "r") as f:
            data.append(json.load(f))

data = [item for sublist in data for item in sublist] 

# as dataframe
df_comments = pd.DataFrame(data)

# clean comments
yt_comments =[] 
for i in range(len(df_comments)):
    video_id = df_comments.iloc[i]["VideoID"]
    yt_comments.append([[clean_comments(item[1]), video_id] for item in df_comments.iloc[i]["Comments"]])
# flatten list
yt_comments_all = [item for sublist in yt_comments for item in sublist]
# remove empty comments
yt_comments_all = [item for item in yt_comments_all if item[0] != ""]
# use dict to remove duplicates
yt_comments = list(dict.fromkeys([tuple(item) for item in yt_comments_all]))


#### Extract collective action features ####

list_for_csv = []
tot_n_videos = 0    

### agency-oriented

## load data
for file in os.listdir("./data/examples"):
    if file.startswith("self_mformer_wisescale_all_noscaled0"):

        # time
        start_time = time.time()

        # cluster label
        cluster_label = file.split("_")[-1].split(".")[0]

        # read file videos as dataframe from csv
        data_videos = pd.read_csv("./data/examples/" + file)

        # remove duplicate videos
        data_videos = data_videos.drop_duplicates(subset="Video ID")

        # clean text
        data_videos["Video Transcript"] = data_videos["Video Transcript"].apply(clean_text)

        # remove videos with empty transcript
        data_videos = data_videos[data_videos["Video Transcript"] != ""]

        # reset index
        data_videos = data_videos.reset_index(drop=True)

        tot_n_videos += len(data_videos)

        # get all comments of videos in that cluster
        tot_indices = []
        for video_id in data_videos["Video ID"]:
            # retrieve comments
            indices = [j for j in range(len(yt_comments)) if yt_comments[j][1] == video_id]
            tot_indices.extend(indices)
        
        # get comments
        comments = [yt_comments[i] for i in tot_indices]
        # get liwc features
        df_comments = pd.DataFrame(comments, columns=["text", "VideoID"])
        df_comments = get_coll_action(df_comments, "text")

        # time
        print("--- %s seconds ---" % (time.time() - start_time))
                
        # add narrative column
        df_comments["narrative"] = "self_"+cluster_label
        # save comment id (take index from yt_comments)
        df_comments["comment_id"] = [i for i in tot_indices]
        list_for_csv.append(df_comments)

    
### communal-oriented
        
## load data
for file in os.listdir("./data/examples"):
    if file.startswith("group_mformer_wisescale_all_noscaled0"):

        # time
        start_time = time.time()

        # cluster label
        cluster_label = file.split("_")[-1].split(".")[0]

        # read file videos as dataframe from csv
        data_videos = pd.read_csv("./data/examples/" + file)

        # remove duplicate videos
        data_videos = data_videos.drop_duplicates(subset="Video ID")

        # clean text
        data_videos["Video Transcript"] = data_videos["Video Transcript"].apply(clean_text)

        # remove videos with empty transcript
        data_videos = data_videos[data_videos["Video Transcript"] != ""]

        # reset index
        data_videos = data_videos.reset_index(drop=True)

        tot_n_videos += len(data_videos)

        # get all comments of videos in that cluster
        tot_indices = []
        for video_id in data_videos["Video ID"]:
            # retrieve comments
            indices = [j for j in range(len(yt_comments)) if yt_comments[j][1] == video_id]
            tot_indices.extend(indices)
        
        # get comments
        comments = [yt_comments[i] for i in tot_indices]
        # get liwc features
        df_comments = pd.DataFrame(comments, columns=["text", "VideoID"])
        df_comments = get_coll_action(df_comments, "text")

        # time
        print("--- %s seconds ---" % (time.time() - start_time))

        # add narrative column
        df_comments["narrative"] = "group_"+cluster_label
        # save comment id (take index from yt_comments)
        df_comments["comment_id"] = [i for i in tot_indices]

        list_for_csv.append(df_comments)

#### Save ####
        
df = pd.concat(list_for_csv)

df = df.loc[:,~df.columns.str.endswith(' freq')]

# if results folder does not exist, create it
if not os.path.exists("./results"):
    os.makedirs("./results")

df.to_csv("./results/collective_action_features_comments.csv", index=False)