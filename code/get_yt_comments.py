# import required modules
from pathlib import Path
import pandas as pd
import json
from utils import youtube_authenticate, get_video_comments


# define tag
tag = "nomeatmay"

# if comments folder does not exist, create it
if not Path("../data/comments").exists():
    Path("../data/comments").mkdir()

# define scopes for API access
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]


#### Main program, data retrieval ####

youtube = youtube_authenticate(SCOPES)

# open transcript file
df_videos = pd.read_csv("../data/"+tag+"/video_ids_mformer.csv")

# remove possible duplicates
df_videos = df_videos.drop_duplicates(subset="Video ID")

# get list of video ids
video_ids = df_videos["Video ID"].tolist()

video_data = []
iter_videos = 0 
all_data = []

for video_id in video_ids:

    # if json file already exists, open
    if Path(f"../data/comments/"+tag+".json").exists():
        with open(f"../data/comments/"+tag+".json", "r") as f:
            all_data = json.load(f) # in this way, we will overwrite the json file if we run the script again
        
        # if video already in json, skip
        if any(d["VideoID"] == video_id for d in all_data):
            iter_videos += 1
            
            if iter_videos % 10 == 0:
                print("N. videos:", iter_videos ,"/", len(video_ids))

            continue

    comments = get_video_comments(youtube, video_id=video_id)
    
    video_data = {
        'VideoID': video_id,
        'Comments': comments
    }

    all_data.append(video_data)

    iter_videos += 1

    if iter_videos % 10 == 0:
        print("N. videos:", iter_videos ,"/", len(video_ids))

    # save as json
    with open("../data/comments/"+tag+".json", "w") as f:
        json.dump(all_data, f, indent=4)
