import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split

from util import Audio 

def load_wav_audio(audio_dir, separated=False):
    """
    Load WAV audio files, convert to mel-spectrograms, and create samples.
    """
    audio = Audio()
    print(f'Processing audio data from "{audio_dir}" folder...')
    data = {}
    
    for cls in os.listdir(audio_dir): 
        cls_dir = os.path.join(audio_dir, cls)
        data[cls] = []

        if not os.path.isdir(cls_dir):
            continue
        
        if separated:
            for track in os.listdir(cls_dir): 
                track_dir = os.path.join(cls_dir, track)
                for fname in os.listdir(track_dir):  
                    if not fname.endswith('.wav'):
                        continue

                    file_path = os.path.join(track_dir, fname)
                    print(f'-> Processing {file_path}...')
                    
                    mel = audio.audio_to_mel(file_path)
                    print('\tConverted to mel-spectrogram!')

                    k = mel.shape[1] // 25
                    mels = audio.mel_sample(mel, width=384, k=k)
                    print('\tSampled mel-spectrogram slices!')

                    if mels is not None:
                        data[cls].append(mels)
        else:
            for fname in os.listdir(cls_dir):
                if not fname.endswith('.wav'):
                    continue

                file_path = os.path.join(cls_dir, fname)
                print(f'-> Processing {file_path}...')

                mel = audio.audio_to_mel(file_path)
                print('\tConverted to mel-spectrogram!')

                k = mel.shape[1] // 25
                mels = audio.mel_sample(mel, width=384, k=k)
                print('\tSampled mel-spectrogram slices!')

                if mels is not None:
                    data[cls].append(mels)
        
        if data[cls]:
            data[cls] = np.concatenate(data[cls], axis=0)
        else:
            print(f'Warning: No valid data found for class "{cls}"')
    
    return data

def create_dataset(data):

    os.makedirs('./data/train', exist_ok=True)
    os.makedirs('./data/test', exist_ok=True)

    for cls in data:
        if len(data[cls]) == 0:
            print(f'Warning: No data found for class "{cls}". Skipping...')
            continue

        train, test = train_test_split(data[cls], test_size=0.1, random_state=101)
        np.save(f'./data/train/{cls}.npy', train)
        np.save(f'./data/test/{cls}.npy', test)
        print(f'Dataset for class "{cls}" saved: {len(train)} train, {len(test)} test samples.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, help='Path to the audio directory.')
    parser.add_argument('--separated', action='store_true', help='Process separated components (e.g., bass, drums).')
    args = parser.parse_args()


    data = load_wav_audio(args.audio_dir, separated=args.separated)

    create_dataset(data)
