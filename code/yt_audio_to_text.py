# import required modules
import whisper
from langdetect import detect
from pytube import YouTube
import pickle

#### Define tag ####
tag = "nomeatmay"

#### Whisper transcription ####

# loop over years
for YEAR in range(2014,2024):

    # load the retrieved videos for the year
    with open("../data/"+tag+"/retrieved_allvideos_"+str(YEAR)+".pickle", "rb") as token:    
        retrieved_videos = pickle.load(token)

    # language filter on the description of the videos in retrieved_videos with detect from langdetect
    list_to_drop = []
    for i in range(len(retrieved_videos)):
        try:
            language = detect(retrieved_videos["Video Description"].values[i])
            if language != "en":
                list_to_drop.append(i)
        except:
            try:
                # if Video Description is None, check Video Title
                language = detect(retrieved_videos["Video Title"].values[i])
                if language != "en":
                    list_to_drop.append(i)
            except:
                # if Video Title is None, print Video Description and Video ID
                print(retrieved_videos["Video Description"].values[i])
                print("Error in language detection")

    retrieved_videos = retrieved_videos.drop(list_to_drop)

    # check which values in the Video Transcript column are None and save the video ids in a list  
    video_ids = []
    for i in range(len(retrieved_videos)):
        if retrieved_videos["Video Transcript"].values[i] == None:
            video_ids.append(retrieved_videos["Video ID"].values[i])

    # retrieve Whisper captions for the videos in video_ids
    counter = 0
    for id in video_ids:

        # Create the URL for the YouTube video
        url = "https://www.youtube.com/watch?v=" + id

        # Create a YouTube object from the URL
        try:
            yt = YouTube(url)
        except:
            print("Error in video download")
            # delete row corresponding to video ID from retrieved_videos
            retrieved_videos = retrieved_videos[retrieved_videos["Video ID"] != id]
            continue

        # Check if the video is longer than 10 minutes
        if yt.length > 600:
            print("Video too long")
            # delete row corresponding to video ID from retrieved_videos
            retrieved_videos = retrieved_videos[retrieved_videos["Video ID"] != id]
            continue
        
        # Get the audio stream
        try:
            audio_stream = yt.streams.filter(only_audio=True).first()
            # Download the audio stream
            output_path = "YoutubeAudios"
            filename = "audio_"+id+".mp3"
            audio_stream.download(output_path=output_path, filename=filename)

            # Load the base model and transcribe the audio
            model = whisper.load_model("base")
            result = model.transcribe("./YoutubeAudios/audio_"+id+".mp3")

            transcribed_text = result["text"]
            retrieved_videos.loc[retrieved_videos["Video ID"] == id, "Video Transcript"] = transcribed_text
            
        except:
            print("Error in audio download or transcription")
            # delete row corresponding to video ID from retrieved_videos
            retrieved_videos = retrieved_videos[retrieved_videos["Video ID"] != id]

        counter += 1

        if counter % 10 == 0:
            print("Counter:", counter)

    # reset index of retrieved_videos
    retrieved_videos = retrieved_videos.reset_index(drop=True)

    # save dataframe to pickle file
    with open("../data/"+tag+"/retrieved_allvideos_"+str(YEAR)+".pickle", "wb") as token:    
        pickle.dump(retrieved_videos, token)