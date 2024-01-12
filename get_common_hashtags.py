# import modules
import pickle
import pandas as pd

#### Get list of common hashtags ####

# set parameters
years = list(range(2014,2024))
tag = "veganuary"
basic_keyword = "veganuary"

# count hashtags 
all_hashtags = []
for year in years:
    
    # load the retrieved videos for the year
    with open("./data/"+tag+"/retrieved_target_video_"+str(year)+"_key_"+basic_keyword+".pickle", "rb") as token:    
        retrieved_videos = pickle.load(token)

    # find hashtags in Video Description column 
    hashtags = {}
    for i in range(len(retrieved_videos)):
        if retrieved_videos["Video Description"][i] != None:
            for j in retrieved_videos["Video Description"][i].split():
                if j.startswith("#"):
                    if j[1:] in hashtags:
                        # check if hashtag as already been counted for the video
                        if retrieved_videos["Video ID"][i] not in hashtags[j[1:]]:
                            hashtags[j[1:]].append(retrieved_videos["Video ID"][i])
                    else:
                        hashtags[j[1:]] = [retrieved_videos["Video ID"][i]]
    
    # replace the list of video ids with the number of videos in which the hashtag appears
    for key in hashtags:
        hashtags[key] = len(hashtags[key])
    
    # append the dictionary to the list of dictionaries
    all_hashtags.append(hashtags)

# merge the dictionaries in the list of dictionaries
merged_hashtags = {}
for i in all_hashtags:
    for key in i:
        if key in merged_hashtags:
            merged_hashtags[key] += i[key]
        else:
            merged_hashtags[key] = i[key]

# create a dataframe with the hashtags and their frequency, sort by frequency
df = pd.DataFrame.from_dict(merged_hashtags, orient='index', columns=["frequency"])
df = df.sort_values(by="frequency", ascending=False)

# save the dataframe
df.to_csv("./data/"+tag+"/common_hashtags_"+basic_keyword+".csv")





