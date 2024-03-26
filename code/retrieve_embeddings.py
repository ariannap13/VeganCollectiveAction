# import packages
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import os
import json
from utils import clean_text, clean_comments

#### Suffix definition: type of embeddings to be extracted ####
analyze_comments = False
normalize_emb = False 

if analyze_comments:
    suffix = "withtrans_comments"
elif not analyze_comments:
    suffix = "withtrans"
else:
    suffix = "notrans"

if normalize_emb:
    suffix = suffix + "_norm"

if not os.path.exists("./embedding_data"):
    os.makedirs("./embedding_data")

#### Functions definition ####
    
def get_embeddings(data, type):
    '''
    Function to get embeddings for a list of texts.

    Args:
    data (list): list of texts
    type (str): type of data. Either "video" or "comment"

    Returns:
    embeddings (list): list of embeddings
    '''

    sentences = []
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda") # good performance, fast according to https://www.sbert.net/docs/pretrained_models.html

    for i in range(len(data)):
        if type == "video":
            video_id = data.iloc[i]["Video ID"]
            sentences.append([clean_text(data["Video Transcript"].values[i]), video_id])

        else:
            video_id = data.iloc[i]["VideoID"]
            sentences.append([[clean_comments(item[1]), video_id] for item in data.iloc[i]["Comments"]])
    
    if type == "video":
        # remove if text is empty
        sentences = [item for item in sentences if item[0] != ""]

        # get only sentences
        sentences = [item[0] for item in sentences]

        if normalize_emb:
            embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)        
        else:
            embeddings = model.encode(sentences, show_progress_bar=True)
    else:
        sentences_all = [item for sublist in sentences for item in sublist]
        # remove elements with empty text
        sentences_all = [item for item in sentences_all if item[0] != ""]
        sentences = list(dict.fromkeys([tuple(item) for item in sentences_all]))

        # get only sentences
        sentences = [item[0] for item in sentences]
        print("length of texts:", len(sentences), flush=True)

        if normalize_emb:
            embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)
        else:
            embeddings = model.encode(sentences, show_progress_bar=True)

    return embeddings
    
#### Retrieve embeddings ####

if not analyze_comments:
    ### video content

    # retrieve narratives
    for file in os.listdir("../data/examples"):

        # agency-oriented
        if (file.startswith("self_mformer_wisescale_all_noscaled0")) and (".csv" in file):
            identity = "self"
            # focus on 1 narrative at a time
            cluster_label = file.split("_")[-1].split(".")[0]
            # read file as dataframe
            data = pd.read_csv("../data/examples/" + file)

            data = data.drop_duplicates(subset="Video ID")

            embeddings = get_embeddings(data, "video")
            with open("./embedding_data/embeddings_"+identity+"_"+suffix+"_cl_"+cluster_label+".pkl", "wb") as f:
                pickle.dump(embeddings, f)
        
        # communal-oriented
        if (file.startswith("group_mformer_wisescale_all_noscaled0")) and (".csv" in file):
            identity = "group"
            # focus on 1 narrative at a time
            cluster_label = file.split("_")[-1].split(".")[0]
            # read file as dataframe
            data = pd.read_csv("../data/examples/" + file)

            data = data.drop_duplicates(subset="Video ID")
            
            embeddings = get_embeddings(data, "video")

            # save embeddings
            with open("./embedding_data/embeddings_"+identity+"_"+suffix+"_cl_"+cluster_label+".pkl", "wb") as f:
                pickle.dump(embeddings, f)
                
else:
    ### comments

    # open all comments files in data/comments
    data = []
    for file in os.listdir("../data/comments/"):
        if file.endswith(".json"):
            with open("../data/comments/"+file, "r") as f:
                data.append(json.load(f))

    data = [item for sublist in data for item in sublist] 

    # as dataframe
    data = pd.DataFrame(data)
    
    embeddings = get_embeddings(data, "comment")

    # save embeddings
    with open("./embedding_data/embeddings_"+suffix+".pkl", "wb") as f:
        pickle.dump(embeddings, f)


