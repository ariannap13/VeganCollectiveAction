# ReadMe

This file serves as a documentation of the code related to the paper "The Role of Discourse Narratives in Shaping Collective Action:
Plant-Based Challenges on YouTube as a Case Study".

## Packages

All analyses were run on Python 3.8.16. Required libraries and versions are specified in `requirements.txt`.

## Overall pipeline

### Video data collection and cleaning

1. **get_yt_videos.py**: Get YouTube videos given keywords and timeframe of reference.
2. **get_common_hashtags.py**: Get the most frequent hashtags in video descriptions to expand the keyword search list. Then run **get_yt_videos.py** again with new keywords to continue data collection.
3. **merge_videos_year.py**: Get a single file per year and remove videos with music as a category.
4. **yt_audio_to_text.py**: Whisper get transcripts from audio when not auto-captioned.
5. **topic_modeling.py**: Topic modeling, excluding topic not relevant to the theoretical narrative framework.
6. **get_yt_videos_baseline.py**: Get YouTube videos for the baseline set.

### Narrative mapping

7. **extract_mformer.py**: Extract moral dimensions scores through `mformer` for the target video set or the baseline video set, given a dataset of reference (i.e. challenge).
8. **clustering.ipynb**: Scale moral scores with baseline, merge data and extract collective identity scores to map video into *communal-oriented* and *agency-oriented*. Then, use UMAP and HDBSCAN clusters to extract narrative groups.

### Comments data collection and cleaning

9. **get_video_comments_ids.py**: Get video IDs for which to extract comments based on those videos mapped to the theoretical framework.
10. **get_yt_comments.py**: Get YouTube comments.

### Reactions analysis

11. **extract_language_metrics.py**: extract relative frequency of collective action markers from comments.
12. **analyze_language_metrics.ipynb**: Language metrics analysis.
13. **retrieve_embeddings.py**: extract S-BERT embeddings of videos and comments.
14. **extract_video_comments_metrics.py**: Extract alignment of video and comments.
15. **analyze_video_comments_metrics.ipynb**: Semantic similarity analysis.
16. **regression.ipynb**: OLS regression to predict collective action levels given the number of videos, the level of moral foundations, the silhouette score given the clustering into narrative types and the alignment of video and comments. 

### General comments
The file `utils.py` contains useful functions that are imported throughout the pipeline.