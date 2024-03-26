# import packages
import pandas as pd
import os
import json
import pickle
import numpy as np
from sklearn.metrics import silhouette_samples
from utils import clean_text, clean_comments

#### Prepare comments data ####

### Comments

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
# remove duplicates
yt_comments = list(dict.fromkeys([tuple(item) for item in yt_comments_all]))


# open embedding file for comments
with open("./embedding_data/embeddings_withtrans_comments_norm.pkl", "rb") as f:
    embeddings_yt_comments = pickle.load(f)


#### Narratives-comments alignment ####

## agency-oriented

# open files
tot_emb_self = []
tot_labels_self = [] 
tot_video_ids_self = []
for file in os.listdir("./data/examples"):
    if file.startswith("self_mformer_wisescale_all_noscaled0"):

        # retrieve cluster label
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
        
        # open embedding file for videos within cluster
        with open("./embedding_data/embeddings_self_withtrans_norm_cl_" + cluster_label + ".pkl", "rb") as f:
            embeddings_self = pickle.load(f)

        # save embeddings and labels
        tot_emb_self.append(embeddings_self)
        tot_labels_self.append([cluster_label for i in range(len(embeddings_self))])

        # save list of video ids
        tot_video_ids_self.append(list(data_videos["Video ID"].values))

        # create list of video-comment pairs and save comments embeddings
        list_video_comments_pairs_yt = []
        for i, video in data_videos.iterrows():

            # get embedding of video
            video_id = video["Video ID"]
            emb_video = embeddings_self[i]

            # get indices of comments 
            indices = [j for j in range(len(yt_comments)) if yt_comments[j][1] == video_id]
            if len(indices) == 0:
                list_video_comments_pairs_yt.append([emb_video, None, video_id])
                continue
            # get embeddings of comments of those indices
            emb_comments = embeddings_yt_comments[indices]
            # calculate centroid of comments
            centroid_comments = np.mean(emb_comments, axis=0)

            # save pair
            list_video_comments_pairs_yt.append([emb_video, centroid_comments, video_id])
        
        # compute cosome similarity between video and centroid of comments
        list_cosine_videocomm = []
        for i, pair in enumerate(list_video_comments_pairs_yt):

            emb_video = pair[0]
            emb_comments = pair[1]
            video_id = pair[2]
            if emb_comments is None:
                continue
            cos_sim_pair = np.dot(emb_video, emb_comments)/(np.linalg.norm(emb_video)*np.linalg.norm(emb_comments))
            list_cosine_videocomm.append([cos_sim_pair, video_id])

        # save cosine similarity between video and centroid of comments as video-comment alignment
        with open("./results/cosine_videocomm_self_"+cluster_label+"_all_noscaled.pkl", "wb") as f:
            pickle.dump(list_cosine_videocomm, f)
         
# compute silhouette score
tot_emb_self = np.concatenate(tot_emb_self, axis=0)
tot_labels_self = np.concatenate(tot_labels_self, axis=0)
silhouette_score_self = silhouette_samples(tot_emb_self, tot_labels_self, metric="cosine")

# flatten list of video ids
tot_video_ids_self = [item for sublist in tot_video_ids_self for item in sublist]

# dataframe with video ids and silhouette scores
df_silhouette = pd.DataFrame({"VideoID": tot_video_ids_self, "Silhouette": silhouette_score_self})

# save dataframe
df_silhouette.to_csv("./results/silhouette_scores_ids_self_all_noscaled.csv")



## communal-oriented

# open files
tot_emb_group = []
tot_labels_group = []
tot_video_ids_group = []
for file in os.listdir("./data/examples/"):
    if file.startswith("group_mformer_wisescale_all_noscaled0_"):

        # retrieve cluster label
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
        
        # open embedding file for videos within cluster
        with open("./embedding_data/embeddings_group_withtrans_norm_cl_" + cluster_label + ".pkl", "rb") as f:
            embeddings_self = pickle.load(f)

        # save embeddings and labels
        tot_emb_self.append(embeddings_self)
        tot_labels_self.append([cluster_label for i in range(len(embeddings_self))])

        # save list of video ids
        tot_video_ids_self.append(list(data_videos["Video ID"].values))

        # create list of video-comment pairs and save comments embeddings
        list_video_comments_pairs_yt = []
        for i, video in data_videos.iterrows():

            # get embedding of video
            video_id = video["Video ID"]
            emb_video = embeddings_self[i]

            # get indices of comments 
            indices = [j for j in range(len(yt_comments)) if yt_comments[j][1] == video_id]
            if len(indices) == 0:
                list_video_comments_pairs_yt.append([emb_video, None, video_id])
                continue
            # get embeddings of comments of those indices
            emb_comments = embeddings_yt_comments[indices]
            # calculate centroid of comments
            centroid_comments = np.mean(emb_comments, axis=0)

            # save pair
            list_video_comments_pairs_yt.append([emb_video, centroid_comments, video_id])
        
        # compute cosome similarity between video and centroid of comments
        list_cosine_videocomm = []
        for i, pair in enumerate(list_video_comments_pairs_yt):

            emb_video = pair[0]
            emb_comments = pair[1]
            video_id = pair[2]
            if emb_comments is None:
                continue
            cos_sim_pair = np.dot(emb_video, emb_comments)/(np.linalg.norm(emb_video)*np.linalg.norm(emb_comments))
            list_cosine_videocomm.append([cos_sim_pair, video_id])


        with open("./results/cosine_videocomm_group_"+cluster_label+"_all_noscaled.pkl", "wb") as f:
            pickle.dump(list_cosine_videocomm, f)
   

# compute silhouette score
tot_emb_group = np.concatenate(tot_emb_group, axis=0)
tot_labels_group = np.concatenate(tot_labels_group, axis=0)
silhouette_score_group = silhouette_samples(tot_emb_group, tot_labels_group, metric="cosine")

# flatten list of video ids
tot_video_ids_group = [item for sublist in tot_video_ids_group for item in sublist]

# dataframe with video ids and silhouette scores
df_silhouette = pd.DataFrame({"VideoID": tot_video_ids_group, "Silhouette": silhouette_score_group})

# save dataframe 
df_silhouette.to_csv("./results/silhouette_scores_ids_group_all_noscaled.csv")

