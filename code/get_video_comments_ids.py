# import libraries
import pandas as pd

#### Define tag ####
tag = "nomeatmay"
dir = "./data/"+tag+"/"

#### Get list of video ids for easier comments retrieval ####

# open retrieved videos mformer
df = pd.read_csv(dir+"retrieved_videos_mformer.csv")

# get video ids
video_ids = df["Video ID"].tolist()

# dataframe with video ids
df_video_ids = pd.DataFrame(video_ids, columns=["Video ID"])

# save
df_video_ids.to_csv(dir+"video_ids_mformer.csv", index=False)