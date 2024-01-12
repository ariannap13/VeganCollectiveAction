import re
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import sys


### Data Cleaning ###

def clean_text(text, STOPWORDS=[], topic_model = False):
    '''
    Function to clean text. 

    Args: 
    text (string)
    STOPWORDS (list): list of stopwords to remove, default is empty list
    topic_model (bool): if True, also remove stopwords

    Returns:
    cleaned text (string).
    '''

    #replace . with space
    output = text.replace('.', ' ')

    #remove tags such as [Music]
    output = re.sub(r'\[.*?\]', '', output)

    #lowercase
    output = output.lower()

    #remove punctuation
    output = re.sub(r'[^\w\s]', ' ', output)

    #remove extra space
    output = re.sub(r'\s+', ' ', output).strip()

    if topic_model:

        #remove stop words
        output = ' '.join([i for i in output.split() if i not in STOPWORDS]) 

        # remove numbers from strings
        output = ' '.join([i for i in output.split() if not i.isdigit()])

        #remove extra space
        output = re.sub(r'\s+', ' ', output).strip()
    
    #remove string if string only contains punctuation
    if sum([i.isalpha() for i in output])== 0:
        output = ''
        
    #remove string if length<3
    if len(output.split()) < 3:
        output = ''

    if topic_model:

        # get list of unique words in output and remove strings with less than 10 unique words
        if len(set(output.split())) < 10:
            output = ''
    
    else:
            
        # get list of unique words in output and remove strings with less than 5 unique words
        if len(set(output.split())) < 5:
            output = ''

    return output

def clean_comments(text):
    '''
    Function to clean comments.

    Args:
    text (str): input text

    Returns:
    output (str): cleaned text
    '''

    # if text type is float or NoneType, return None
    if type(text) == float or type(text) == type(None):
        output = ""
        return output
    
    # remove urls
    output = re.sub(r'http\S+', '', text)

    #remove mentions
    output = re.sub(r'@\S+', '', output)

    # remove comments that have less than 5 unique words
    if len(set(output.split())) < 5:
        output = ""
    
    return output


### YouTube API ###

# Function to authorize API access using OAuth2
def youtube_authenticate(SCOPES):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "./credentials.json"
    creds = None
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists("./token.pickle"):
        with open("./token.pickle", "rb") as token:
            creds = pickle.load(token)
    # if there are no (valid) credentials availablle, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open("./token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build(api_service_name, api_version, credentials=creds)

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


# Function to retrieve comments from a video
def get_video_comments(service, video_id, **kwargs):
    '''
    Function to retrieve comments from a video.

    Args:
    service (googleapiclient.discovery.Resource): authenticated YouTube API service instance
    video_id (str): YouTube video ID
    kwargs: arguments to be passed to comments().list()

    Returns:
    comments (list): list of top-level comments
    '''

    comments = []

    try:
        results = service.commentThreads().list(videoId=video_id, part='snippet,replies', **kwargs).execute()
    except:
        print("Error with video:", video_id, sys.exc_info()[0])
        return comments

    n_comments = 0
    while results:
        for item in results['items']:
            # top level comments
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comment_id = item['snippet']['topLevelComment']['id']
            comment_parent = item['snippet']['topLevelComment']['snippet']['videoId']
            comment_likes = item['snippet']['topLevelComment']['snippet']['likeCount']
            comment_published_at = item['snippet']['topLevelComment']['snippet']['publishedAt']
            comment = [comment_id, comment_text, comment_parent, comment_likes, comment_published_at]
            comments.append(comment)
            n_comments += 1

            if n_comments > 1000:
                return comments

            # check for replies
            if item['snippet']['totalReplyCount'] > 0:
                if "replies" not in item:
                    continue
                for reply_item in item['replies']['comments']:
                    reply_text = reply_item['snippet']['textDisplay']
                    reply_id = reply_item['id']
                    reply_parent = reply_item['snippet']['parentId']
                    reply_likes = reply_item['snippet']['likeCount']
                    reply_published_at = reply_item['snippet']['publishedAt']
                    reply = [reply_id, reply_text, reply_parent, reply_likes, reply_published_at]
                    comments.append(reply)
                    n_comments += 1

                    if n_comments > 1000:
                        return comments

        # Check for more pages of comments
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            try:
                results = service.commentThreads().list(videoId=video_id, part='snippet,replies', **kwargs).execute()
            except:
                print("Error with video:", video_id, sys.exc_info()[0])
                return comments
        else:
            break

    return comments