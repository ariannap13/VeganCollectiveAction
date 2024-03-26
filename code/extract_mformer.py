# import libraries
import os
import pickle
import re
import spacy
nlp = spacy.load('en_core_web_sm')
import warnings
import pandas as pd
from transformers import AutoModelForSequenceClassification
from roberta_utils import predict
from utils import clean_text

#### Define arguments ####
tag = "nomeatmay"
topic_n = 1 # define according to the topic you want to analyze from topic_modeling.py exploration

data_path = "./data/"+tag+"/"
is_baseline = False
text_col = "Video Transcript Clean"
verbose = True

if not is_baseline:
    output_path = data_path+"retrieved_videos_mformer.csv"
else:
    output_path = data_path+"retrieved_videos_mformer_baseline.csv"
device = "cuda" # "cuda" or "cpu"
batch_size = 32

#### Functions definition ####

# score texts using mformer models
def predict_df(df, text_col, output_path, device="cuda", batch_size=32):
    """
    Use a RobertaForSequenceClassification model to predict a list of texts.

    Args:
        df: a dataframe with a column of texts
        text_col: name of the column with the texts
        output_path: path to save the dataframe with the scores
        device: torch device. Defaults to "cuda".
        batch_size: batch size. Defaults to 32.

    Returns:
        a dataframe with the scores for the texts in df
    """
    for f, path in labelers.items():
        print(f, flush=True)
        print(path, flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device)
        y_score = predict(X=df[text_col].tolist(), model=model, batch_size=batch_size, path=path, device=device)
        df[f"{f}_score"] = y_score
    df.to_csv(output_path)

    
#### Clean Video Transcript column ####

# load dataframe with retrieved videos
if not is_baseline:
    with open(data_path+"retrieved_videos_all_topic.pickle", "rb") as token:  
        retrieved_videos = pickle.load(token)

else:
    df_all = []
    for YEAR in range(2014,2024):
        with open(data_path+"retrieved_filtered_baseline_video_"+str(YEAR)+".pickle", "rb") as token:    
            retrieved_videos = pickle.load(token)
        df_all.append(retrieved_videos)

    retrieved_videos = pd.concat(df_all)

# filter out None values in Video Transcript column
retrieved_videos = retrieved_videos[retrieved_videos["Video Transcript"].notnull()]

# filter by topic - select only video related to the challenge or to the vegan lifestyle, delete recipes
if not is_baseline:
    retrieved_videos = retrieved_videos[retrieved_videos["topic"]==topic_n] 

# apply clean function to Video Transcript column
retrieved_videos['Video Transcript Clean'] = retrieved_videos['Video Transcript'].apply(lambda x: clean_text(x))
# remove all rows for which Video Transcript Clean is empty and reset index
retrieved_videos = retrieved_videos[retrieved_videos['Video Transcript Clean'] != ''].reset_index(drop=True)

#### Score texts ####

# Mformer models
labelers = {
    "authority": "joshnguyen/mformer-authority",
    "care": "joshnguyen/mformer-care",
    "fairness": "joshnguyen/mformer-fairness",
    "loyalty": "joshnguyen/mformer-loyalty",
    "sanctity": "joshnguyen/mformer-sanctity"
}

# Check if the data file exists
assert os.path.exists(data_path), f"Data file does not exist at {data_path}."

# Check if the output file exists
if os.path.exists(output_path):
    warnings.warn(f"Output file already exists at {output_path}. It will be overwritten.")

# Load the texts
if verbose:
    print(f"Loading data from {data_path}...", flush=True)

# Score the texts
if verbose:
    print(f"Scoring texts...", flush=True)

predict_df(df=retrieved_videos,
            text_col=text_col,
            output_path=output_path,
            device=device,
            batch_size=batch_size)

if verbose:
    print("Done!", flush=True)


