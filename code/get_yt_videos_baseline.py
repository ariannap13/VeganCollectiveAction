# import required modules
from youtube_transcript_api import YouTubeTranscriptApi
import os
import pickle
import pandas as pd
from utils import youtube_authenticate, search_videos, get_video_details

#### Global variables ####
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

#### Functions definition ####

# Function to search for videos based on a keyword
def search_videos(service, threshold_api_units, n_nextpage=3, **kwargs):
    '''
    Returns a list of videos based on keyword search parameters.

    Args:
        service: the YouTube API service object already initialized
        threshold_api_units: the maximum number of API units we can use in this query
        **kwargs: keyword arguments to be passed to service.search().list()

    Returns:
        a list of dicts representing each video.
    '''
    all_results = []
    next_page_token = ''

    total_api_units_used = 0

    count_nextpage = 0

    while total_api_units_used < threshold_api_units and next_page_token is not None and count_nextpage<=n_nextpage:
        # Calculate the number of API units needed for the upcoming request (e.g., 100 units for search)
        api_units_needed = 100

        # Check if making the next request would exceed the threshold
        if total_api_units_used + api_units_needed <= threshold_api_units:
    
            search_results = service.search().list(**kwargs).execute()
            all_results.extend(search_results.get('items', []))

            total_api_units_used += api_units_needed

            if total_api_units_used + api_units_needed <= threshold_api_units:
            # Check for more pages of results
                next_page_token = search_results.get('nextPageToken')
                total_api_units_used += api_units_needed
                print("Total units used: %d" % total_api_units_used)
                if not next_page_token:
                    break
                kwargs['pageToken'] = next_page_token
            else:
                print("Reached API unit threshold. Stopping requests.")
                break
        else:
            print("Reached API unit threshold. Stopping requests.")
            break

        count_nextpage+=1
                
    return all_results

# Function to retrieve video details
def get_video_details(service, video_id):
    '''
    Returns a detailed description of a video.

    Args:
        service: the YouTube API service object already initialized
        video_id: the ID of the video for which we want to retrieve details

    Returns:
        a list of dicts representing video details.
    '''
    video_details = service.videos().list(part='snippet,statistics', id=video_id).execute()
    return video_details.get('items', [])[0] if video_details.get('items', []) else None


#### Main program, data retrieval ####

# Define tag
tag = "veganuary"

# Loop over years
for YEAR in range(2014, 2024):
    START_DATE = str(YEAR-1)+"-12-01T00:00:00Z"
    END_DATE = str(YEAR)+"-02-01T00:00:00Z"

    youtube = youtube_authenticate(SCOPES)

    # Step 1: Search for videos based on the keyword, video category id 22 (People & Blogs, most popular category)
    search_results = search_videos(youtube, threshold_api_units = 8000, n_nextpage=4, publishedAfter=START_DATE, publishedBefore=END_DATE, relevanceLanguage="en", videoCategoryId=22, type="video", part='id', maxResults=50, order="date")
    
    video_data = []
    
    for video in search_results:
        video_id = video['id']['videoId']
        
        # Step 2: Retrieve video details
        video_details = get_video_details(youtube, video_id)
        
        if video_details:
            video_title = video_details['snippet']['title']
            video_description = video_details['snippet']['description']
            video_timestamp = video_details['snippet']['publishedAt']
            try:
                video_views = video_details['statistics']['viewCount']
            except:
                video_views = None
            # also get likes, dislikes, comments, and category id
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
                'Video Title': video_title,
                'Video Timestamp': video_timestamp,
                'Video Description': video_description,
                'Video Views': video_views,
                'Video Category ID': video_category_id,
                'Video Like Count': video_like_count,
                'Video Dislike Count': video_dislike_count,
                'Video Comment Count': video_comment_count
            })

    # Create a DataFrame-like structure
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

        # if folder does not exist, create it
        if not os.path.exists("./data/"+tag):
            os.makedirs("./data/"+tag)
        
        # save the dataframe
        with open("./data/"+tag+"/retrieved_baseline_video_"+str(YEAR)+".pickle", "wb") as token:    
            pickle.dump(df, token)  

    except:
        print("No videos for "+str(YEAR)+".")  
