import filecmp
import os
import shutil
import re
import json
import pandas as pd
import time
import logging
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
import zipfile
import matplotlib.pyplot as plt
import sys

user_name = sys.argv[1]

print(user_name)

print("Started the script")

# unzip file
def unzip_file(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

unzip_file(f'Data/{user_name}/my_spotify_data.zip', f'Data/{user_name}')

cid = '7fffb74083874b72a336d4e4b35ca2db'
secret = '66f0b32349a445a5a7173ebeb3dd741a'

path = f'Data/{user_name}/my_spotify_data/Spotify Account Data'
filenames = os.listdir(path)
os.makedirs('Data', exist_ok=True)


combined_data = []

for filename in filenames:
    if filename.startswith("StreamingHistory_music") and filename.endswith(".json"):
        path = os.path.join(f'Data/{user_name}/my_spotify_data/Spotify Account Data', filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            combined_data.extend(data)

df = pd.DataFrame(combined_data)

def correctInvalidTimes(df, time_col):
    def correct_time(row):
        try:
            pd.to_datetime(row, format="%Y-%m-%d %H:%M")
            return row
        except ValueError:
            parts = row.split(' ')
            date = parts[0]
            time = parts[1]
            hours, minutes = time.split(':')
            hours = int(hours)
            minutes = int(minutes)

            # Correct invalid minutes
            if minutes >= 60:
                extra_hours = minutes // 60
                minutes = minutes % 60
                hours += extra_hours
            
            # Correct invalid hours
            if hours >= 24:
                hours = hours % 24

            corrected_time = f"{date} {hours:02}:{minutes:02}"
            return corrected_time

    df[time_col] = df[time_col].apply(correct_time)
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', format="%Y-%m-%d %H:%M")
    return df

def filterShortSongs(df, duration_col='msPlayed', min_duration=30000):
    return df[df[duration_col] >= min_duration]

df['endTime'] = pd.to_datetime(df['endTime'])

latest_date = df['endTime'].max()

two_months_before_latest = latest_date - pd.DateOffset(months=2)

df = df[df['endTime'] >= two_months_before_latest]

df = correctInvalidTimes(df, "endTime")
df = filterShortSongs(df)

df.to_csv(f'Data/{user_name}/filtered_streaming_history.csv')

uniqueSongs = df.drop(columns=['endTime', 'msPlayed']).drop_duplicates()

def append_to_csv(df, filepath=f'Data/{user_name}/newSongsWithURIs.csv'):
    df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)

def get_uris_and_append(df):
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret), requests_timeout=10, retries=10)

     # Replace NaN with empty strings in the relevant columns
    df['trackName'] = df['trackName'].fillna('').astype(str)
    df['artistName'] = df['artistName'].fillna('').astype(str)

    for i in tqdm(range(df.shape[0])):
        item = df.iloc[i:i+1]  # Keep as dataframe
        track = item["trackName"].values[0].strip()
        artist = item["artistName"].values[0].strip()

        try:
            searchResults = spotify.search(q=f"track:{track} artist:{artist}", type="track")
            if searchResults['tracks']['items']:
                track_info = searchResults['tracks']['items'][0]
                track_link = track_info['external_urls']['spotify']
                track_URI = track_link.split("/")[-1].split("?")[0]
                item["uri_link"] = track_URI
            else:
                item["uri_link"] = "None"
        except Exception as e:
            print(f"Error for track {track} by {artist}: {e}")
            item["uri_link"] = "None"
        append_to_csv(item)
        time.sleep(0.1)
        
# get_uris_and_append(uniqueSongs)

logging.basicConfig(filename=f'Data/{user_name}/audio_features.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def append_to_csv(df, filepath='final_audio_features.csv'):
    """Appends DataFrame to CSV."""
    try:
        df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)
        logging.info(f"Successfully appended {len(df)} records to {filepath}")
    except Exception as e:
        logging.error(f"Error while writing to CSV: {e}")

def divide_chunks(l, n):
    """Divides list into chunks of size n."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_audio_features(df, filepath=f'Data/{user_name}/final_audio_features.csv'):
    """Fetches audio features for tracks and appends to CSV after each chunk."""
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret),
                              requests_timeout=10, retries=10)

    tracks_uri_list = df["uri_link"].tolist()
    artist = df["artistName"].tolist()
    track = df["trackName"].tolist()
    tracks_uri_chunks = list(divide_chunks(tracks_uri_list, 50)) 

    for chunk_index, chunk in enumerate(tqdm(tracks_uri_chunks)):
        audio_features = []
        try:
            features = spotify.audio_features(tracks=chunk)
            if features:
                for feature in features:
                    if feature is None:  
                        logging.warning(f"Audio feature unavailable for chunk {chunk_index} - skipping track.")
                        audio_features.append(None)  
                    else:
                        audio_features.append(feature)
            else:
                logging.warning(f"Empty response from Spotify API for chunk {chunk_index}.")
                audio_features = [None] * len(chunk)  
        except Exception as e:
            logging.error(f"Error fetching audio features for chunk {chunk_index}: {e}")
            audio_features = [None] * len(chunk) 
        time.sleep(0.2)  
        if len(audio_features) < len(chunk):
            audio_features.extend([None] * (len(chunk) - len(audio_features))) 

        chunk_result = pd.DataFrame()
        chunk_result["trackName"] = track[chunk_index * 50:(chunk_index + 1) * 50]
        chunk_result["artistName"] = artist[chunk_index * 50:(chunk_index + 1) * 50]
        chunk_result["spotify_uri"] = chunk
        chunk_result["audio_features"] = audio_features

        append_to_csv(chunk_result, filepath=filepath)
        
uniqueSongs = pd.read_csv(f'Data/{user_name}/newSongsWithURIs.csv')

# get_audio_features(uniqueSongs)

import ast

def expand_audio_features(df):
    songs_dict = df.to_dict('records')
    
    for song in songs_dict:
        song['audio_features'] = ast.literal_eval(song['audio_features'])
        
        for feature in song['audio_features']:
            song[feature] = song['audio_features'][feature]
            
    final_songs_df = pd.DataFrame(songs_dict)
    
    final_songs_df = final_songs_df.drop(columns=['audio_features', 'id', 'uri', 'track_href', 'analysis_url', 'type'])
    
    return final_songs_df

songs_df = pd.read_csv(f'Data/{user_name}/final_audio_features.csv')

songs_df = songs_df.dropna()

songs_df = expand_audio_features(songs_df)

def mapUserToSongs(df, songs_df):
    merged_df = pd.merge(df, songs_df, on=['trackName', 'artistName'], how='inner')
    merged_df.to_csv(f'Data/{user_name}/user_listening_history.csv', index=False)

mapUserToSongs(df, songs_df)
user_df = pd.read_csv(f'Data/{user_name}/user_listening_history.csv')
feature_columns = ['valence', 'energy', 'danceability', 'loudness']

def extract_hour(df, time_column):
    """Extract the hour from the datetime column."""
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    df['hour_of_day'] = df[time_column].dt.hour
    return df

def compute_diurnal_patterns(df, feature_columns, time_column):
    """Compute the mean diurnal pattern of audio features."""
    df = extract_hour(df, time_column)
    
    df = df.dropna(subset=['hour_of_day'] + feature_columns)

    diurnal_patterns = df.groupby('hour_of_day')[feature_columns].mean()
    
    diurnal_patterns.to_csv(f'Data/{user_name}/diurnal_patterns.csv', index=False)
    
    return diurnal_patterns

def plot_diurnal_patterns(df, feature_columns, time_column):
    """Plot diurnal patterns for features in the given DataFrame."""
    
    diurnal_patterns = compute_diurnal_patterns(df, feature_columns, time_column)
    
    # for feature in feature_columns:
    #     plt.figure(figsize=(10, 6))
        
    #     plt.plot(diurnal_patterns.index, diurnal_patterns[feature], label='Diurnal Pattern', color='blue')
        
    #     plt.title(f'Diurnal Pattern of {feature.capitalize()}')
    #     plt.xlabel('Hour of Day')
    #     plt.ylabel(f'{feature.capitalize()} Value')
    #     plt.xticks(range(0, 24))
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

plot_diurnal_patterns(user_df, feature_columns, 'endTime')

def extract_day_of_week(df, time_column):
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    df['day_of_week'] = df[time_column].dt.dayofweek
    return df

def compute_daywise_patterns(df, feature_columns, time_column):
    df = extract_day_of_week(df, time_column)
    df = df.dropna(subset=['day_of_week'] + feature_columns)
    daywise_patterns = df.groupby('day_of_week')[feature_columns].mean()
    
    daywise_patterns.to_csv(f'Data/{user_name}/daywise_week_patterns.csv', index=False)
    
    return daywise_patterns

def plot_daywise_patterns(df, feature_columns, time_column):
    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daywise_patterns = compute_daywise_patterns(df, feature_columns, time_column)
    
    # for feature in feature_columns:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(daywise_patterns.index, daywise_patterns[feature], label='Daywise Pattern', color='blue')
    #     plt.title(f'Daywise Pattern of {feature.capitalize()}')
    #     plt.xlabel('Day of Week')
    #     plt.ylabel(f'{feature.capitalize()} Value')
    #     plt.xticks(ticks=range(7), labels=day_labels)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

plot_daywise_patterns(user_df, feature_columns, 'endTime')

def extract_day_of_month(df, time_column):
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    df['day_of_month'] = df[time_column].dt.day
    return df

def compute_daywise_month_patterns(df, feature_columns, time_column):
    df = extract_day_of_month(df, time_column)
    df = df.dropna(subset=['day_of_month'] + feature_columns)
    daywise_month_patterns = df.groupby('day_of_month')[feature_columns].mean()
    
    daywise_month_patterns.to_csv(f'Data/{user_name}/daywise_month_patterns.csv', index=False)
        
    return daywise_month_patterns

def plot_daywise_month_patterns(df, feature_columns, time_column):
    daywise_month_patterns = compute_daywise_month_patterns(df, feature_columns, time_column)
    
    # for feature in feature_columns:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(daywise_month_patterns.index, daywise_month_patterns[feature], label=feature.capitalize(), color='blue')
        
    #     plt.title(f'Daywise Pattern of {feature.capitalize()} Over a Month')
    #     plt.xlabel('Day of the Month')
    #     plt.ylabel(f'{feature.capitalize()} Value')
    #     plt.xticks(range(1, 32))
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

plot_daywise_month_patterns(user_df, feature_columns, 'endTime')