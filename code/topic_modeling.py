# import libraries
import pickle
import pandas as pd
import en_core_web_md
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from langdetect import detect
from wordcloud import STOPWORDS
from utils import clean_text

#### Define tag ####
tag = "nomeatmay"

#### Functions definition ####

def get_top_words(model, feature_names, n_top_words):
    '''
    Function to get top words in each topic.

    Args:
    model: topic model
    feature_names (list): output feature names from vectorizer
    n_top_words (int): number of top words to print

    Returns:
    topic_list (list): list of lists of top words in each topic
    '''

    topic_list = []
    for _, topic in enumerate(model.components_):
        topic_list.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topic_list


#### Clean and prepare data ####

# create a dataframe by concatenating all the dataframes for each year  
retrieved_videos = pd.DataFrame()
for year in range(2014, 2024):
    with open("../data/"+tag+"/retrieved_allvideos_"+str(year)+".pickle", "rb") as token:
        retrieved_videos_year = pickle.load(token)
    retrieved_videos = pd.concat([retrieved_videos, retrieved_videos_year], ignore_index=True)

# filter out None values in Video Transcript column
retrieved_videos = retrieved_videos[retrieved_videos["Video Transcript"].notnull()]
retrieved_videos = retrieved_videos.reset_index(drop=True)

# language filter on the transcript, after Whisper captions have been retrieved
list_to_drop = []
for i in range(len(retrieved_videos)):
    try:
        language = detect(retrieved_videos["Video Transcript"].values[i])
        if language != "en":
            list_to_drop.append(i)
    except:
        print(retrieved_videos["Video Transcript"].values[i])
        print("Error in language detection")
retrieved_videos = retrieved_videos.drop(list_to_drop)


# expand list of stop words with custom-made list
STOPWORDS = stopwords.words('english')
add_list = ["uh", "well", "oh", "ah", "hes", "gonna", "going", "got", "okay", "hi", "hello", "hey", "yall", "like", "really", "yeah", "um", "im", "ive", "id", "ill", "youre", "youve", "youll", "youd", "youd", "shes", "dont", "arent", "isnt", "wasnt", "werent", "wont", "wouldnt", "shouldnt", "couldnt", "cant", "didnt", "doesnt", "hadnt", "hasnt", "havent", "aint", "thats", "theres", "whats", "whos", "wheres", "whens", "whys", "hows", "couldve", "shouldve", "wouldve", "mightve", "mustve", "cantve", "didnt", "doesnt", "hadnt", "hasnt", "havent", "aint", "thats", "theres", "whats", "whos", "wheres", "whens", "whys", "hows", "couldve", "shouldve", "wouldve", "mightve", "mustve", "cantve", "theyre", "actually", "theyve", "weve"]
STOPWORDS.extend(add_list)

# clean Video Transcript column
retrieved_videos['Video Transcript Clean'] = retrieved_videos['Video Transcript'].apply(lambda x: clean_text(x, STOPWORDS, topic_model=True))
retrieved_videos = retrieved_videos[retrieved_videos['Video Transcript Clean'] != ''].reset_index(drop=True)

# remove words related to a certain POS tag
nlp = en_core_web_md.load()
# Tags to remove from the text
removal= ['ADV', 'AUX', 'PRON','CONJ','INTJ','CCONJ','SCONJ','PUNCT','PART','DET','ADP','SPACE','NUM','SYM']
tokens = []
for summary in nlp.pipe(retrieved_videos["Video Transcript Clean"]):
   proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal and not token.is_stop and token.is_alpha]
   string_transcript = " ".join(proj_tok)
   tokens.append(string_transcript)

retrieved_videos["clean_transcript"] = tokens


### Perform LDA topic modeling ####

# Create Document-Term Matrix
vectorizer = CountVectorizer(max_df=0.5,min_df=5)

# Fit and transform your text data
X_count = vectorizer.fit_transform(retrieved_videos['clean_transcript'])

#Step 2: Assign topic to each document after having chosen the number of topics
n_comp = 2
lda_model = LatentDirichletAllocation(n_components=2,               # Number of topics
                                        max_iter=10,               # Max learning iterations
                                        learning_method='online',
                                        random_state=100,          # Random state
                                        batch_size=16,            # n docs in each learning iter
                                        evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                        n_jobs = -1,               # Use all available CPUs
                                        )
lda_output = lda_model.fit_transform(X_count)

#get topic ID assigned to each document, not the probability of assignment
topic_id = np.argmax(lda_output, axis=1)

n_top_words = 10
topic_list = get_top_words(lda_model, vectorizer.get_feature_names_out(), n_top_words)

# assign topic_list value corresponding to topic_id to each row in retrieved_videos dataframe
retrieved_videos['topic'] = topic_id
retrieved_videos['topic_of_reference_words'] = retrieved_videos['topic'].apply(lambda x: topic_list[x])

# save dataframe as pickle file
retrieved_videos.to_pickle("../data/"+tag+"/retrieved_videos_all_topic.pickle")
