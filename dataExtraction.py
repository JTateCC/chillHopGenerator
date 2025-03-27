import yt_dlp
import librosa
import librosa.display
import uuid
import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import random
from itertools import product
from tqdm import tqdm



def search_and_download(term, output_dir='downloads', count=10, log_file='downloads_log.csv'):
    os.makedirs(output_dir, exist_ok=True)
    downloaded_ids = load_download_ids(log_file)

    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'extract_flat': False,
        'noplaylist': True,
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    search_query = f'ytsearch{count}:{term}'

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_query, download=False)['entries']
        for entry in info:
            vid = entry['id']
            if vid in downloaded_ids:
                print(f"Skipping duplicate: {vid}")
                continue

            print(f"Downloading: {vid} ({entry['title']})")
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])

            # Rename the file
            old_path = os.path.join(output_dir, f"{vid}.mp3")
            guid_name = str(uuid.uuid4()) + ".mp3"
            new_path = os.path.join(output_dir, guid_name)
            os.rename(old_path, new_path)

            log_download(vid, guid_name, term, log_file)


def extract_data_from_mp3(mp3_path):
    y, sr = librosa.load(mp3_path, sr=None)

    # Tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_count = len(beat_frames)

    # MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)  # mean across time for each coeff

    # Chroma (harmonic content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral centroid (brightness)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid)

    # Zero-crossing rate (percussiveness)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Roll-off (how much energy is in high frequencies)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)

    # Assemble all features into a dictionary
    feature_dict = {
        'file': mp3_path,
        'tempo': tempo,
        'beat_count': beat_count,
        'zcr': zcr_mean,
        'rolloff': rolloff_mean,
        'spec_centroid': spec_centroid_mean
    }

    # Add MFCCs
    for i, val in enumerate(mfcc_mean):
        feature_dict[f'mfcc_{i + 1}'] = val

    # Add Chroma
    for i, val in enumerate(chroma_mean):
        feature_dict[f'chroma_{i + 1}'] = val

    return feature_dict



def load_download_ids(duplicate_csv_path):
    if not os.path.exists(duplicate_csv_path):
        return set()
    with open(duplicate_csv_path, 'r') as f:
        return set(row['youtube_id'] for row in csv.DictReader(f))

def log_download(youtube_id, guid_name, term, csv_path):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['youtube_id', 'file_name', 'search_term'])
        writer.writerow([youtube_id, guid_name, term])

def generate_lofi_search_terms():
    genres = ["lo-fi", "chillhop", "jazzhop", "instrumental beats", "boom bap", "downtempo"]
    moods = ["relaxing", "sleepy", "rainy day", "late night", "cozy", "dreamy", "ambient", "mellow", "moody"]
    contexts = ["study", "sleep", "focus", "background", "night drive", "reading", "coding", "work"]
    types = ["mix", "playlist", "full album", "compilation", "1 hour", "2 hour"]

    combinations = []
    for g, m, c, t in product(genres, moods, contexts, types):
        combinations.append(f"{m} {g} {c} {t}")

    # Optionally shuffle or limit
    random.shuffle(combinations)
    return combinations[:100]  # return top 100 search terms

# Example usage

def bulk_download_mp3():
    search_terms = generate_lofi_search_terms()
    for term in search_terms[:10]:
        search_and_download(term)


def build_data_frame(folder_path='downloads', output_csv='chillhop_dataset.csv'):
    feature_list = []

    mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]

    print(f"Found {len(mp3_files)} MP3s in {folder_path}. Extracting features...\n")

    for mp3_file in tqdm(mp3_files):
        mp3_path = os.path.join(folder_path, mp3_file)
        try:
            features = extract_data_from_mp3(mp3_path)
            feature_list.append(features)
        except Exception as e:
            print(f"❌ Error with {mp3_file}: {e}")
            continue

    df = pd.DataFrame(feature_list)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Dataset saved to {output_csv} with {len(df)} rows.")

    return df

def main_data_extraction():
    bulk_download_mp3()
    build_data_frame()

