# import modules
import pickle
import pandas as pd
import os

#### Define tag ####
tag = "veganuary"

#### Merge retrieved videos ####

# loop through years
for YEAR in range(2014,2024):

    # open files in data folder that start with retrieved_video_YEAR and end with .pickle
    files = []
    for i in os.listdir("./data/"+tag+"/"):
        if i.startswith("retrieved_target_video_"+str(YEAR)) and i.endswith(".pickle"):
            files.append(i)

    # concatenate the dataframes in the files
    retrieved_videos = pd.DataFrame()
    for i in files:
        with open("./data/"+tag+"/"+i, "rb") as token:    
            df = pickle.load(token)
        retrieved_videos = pd.concat([retrieved_videos, df])

    retrieved_videos = retrieved_videos.drop_duplicates(subset="Video ID")

    # drop video with video category id 10 (music)
    retrieved_videos = retrieved_videos[retrieved_videos["Video Category ID"] != 10]

    # reset index
    retrieved_videos = retrieved_videos.reset_index(drop=True)

    # save dataframe to pickle file
    with open("./data/"+tag+"/retrieved_allvideos_"+str(YEAR)+".pickle", "wb") as token:    
        pickle.dump(retrieved_videos, token)
