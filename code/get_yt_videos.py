# import required modules
from youtube_transcript_api import YouTubeTranscriptApi
import os
import pickle
import pandas as pd
from utils import youtube_authenticate, search_videos, get_video_details

#### Global variables ####
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

#### Main program, data retrieval ####

# set keyword and tag (i.e. challenge of interest)
KEYWORD = "hey"
tag = "veganury"
# define whether we are looking fot target or baseline videos
baseline = False

# loop over years
for YEAR in range(2014, 2024):
    # change start and end date according to the reference challenge
    START_DATE = str(YEAR-1)+"-12-01T00:00:00Z"
    END_DATE = str(YEAR)+"-02-01T00:00:00Z"

    youtube = youtube_authenticate(SCOPES)

    # Step 1: Search for videos based on the keyword
    if baseline:
        # Search for videos based on the keyword, video category id 22 (People & Blogs, most popular category)
        search_results = search_videos(youtube, threshold_api_units = 8000, n_nextpage=4, publishedAfter=START_DATE, publishedBefore=END_DATE, relevanceLanguage="en", videoCategoryId=22, type="video", part='id', maxResults=50, order="date")
    else:
        earch_results = search_videos(youtube, threshold_api_units = 8000, n_nextpage=10, q=KEYWORD, publishedAfter=START_DATE, publishedBefore=END_DATE, relevanceLanguage="en", type="video", part='id', maxResults=50, order="date")
    
    video_data = []
    
    for video in search_results:
        video_id = video['id']['videoId']
        
        # Step 2: Retrieve video details
        video_details = get_video_details(youtube, video_id)
        
        if video_details:
            video_channel = video_details['snippet']['channelId']   
            video_title = video_details['snippet']['title']
            video_description = video_details['snippet']['description']
            video_timestamp = video_details['snippet']['publishedAt']
            try:
                video_views = video_details['statistics']['viewCount']
            except:
                video_views = None
            video_category_id = video_details['snippet']['categoryId']
            try:
                video_like_count = video_details['statistics']['likeCount']
            except:
                video_like_count = None
            try:
                video_dislike_count = video_details['statistics']['dislikeCount']
            except:
                video_dislike_count = None
            try:
                video_comment_count = video_details['statistics']['commentCount']
            except:
                video_comment_count = None

            
            video_data.append({
                'Video ID': video_id,
                'Channel ID': video_channel,
                'Video Title': video_title,
                'Video Timestamp': video_timestamp,
                'Video Description': video_description,
                'Video Views': video_views,
                'Video Category ID': video_category_id,
                'Video Like Count': video_like_count,
                'Video Dislike Count': video_dislike_count,
                'Video Comment Count': video_comment_count
            })

    # Step 3: Create a DataFrame structure
    df = pd.DataFrame(video_data)

    # get transcripts if automatic caption is available
    try:
        video_ids = list(df["Video ID"].values)

        transcript_list, unretrievable_videos = YouTubeTranscriptApi.get_transcripts(video_ids, continue_after_error=True)

        list_transcripts = []

        for video_id in video_ids:

            if video_id in transcript_list.keys():

                srt = transcript_list.get(video_id)

                text_list = []
                for i in srt:
                    text_list.append(i['text'])

                text = '.'.join(text_list)
                list_transcripts.append(text)
                
            else:
                list_transcripts.append(None)

        df["Video Transcript"] = list_transcripts

        # if data folder does not exist, create it
        if not os.path.exists("./data/"+tag):
            os.makedirs("./data/"+tag)
        
        # save as pickle
        if baseline:
            with open("./data/"+tag+"/retrieved_baseline_video_"+str(YEAR)+"_key_"+KEYWORD+".pickle", "wb") as token:    
                pickle.dump(df, token)
        else:
            with open("./data/"+tag+"/retrieved_target_video_"+str(YEAR)+"_key_"+KEYWORD+".pickle", "wb") as token:    
                pickle.dump(df, token)  

    except:
        print("No videos for "+str(YEAR)+".")  
