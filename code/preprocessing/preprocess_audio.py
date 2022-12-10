import pandas as pd
import numpy as np
from glob import glob
import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks

if __name__ == "__main__":
    features = []
    sample_rate = 16000
    chunk_length_ms = 1000*5
    input_path = "input/audio/"
    output_path = "output/audio_features/"
    files = glob(input_path + '*.mp3')
    
    for file in files:
        file_features = []
        sound = AudioSegment.from_file(file)
        # Change to one channel and 16kHz sampling rate
        # Then split audio files into 5-sec chunks
        sound = sound.set_channels(1)
        sound = sound.set_frame_rate(sample_rate)
        chunks = make_chunks(sound, chunk_length_ms)
        for i, chunk in enumerate(chunks):
            samples = chunk.get_array_of_samples()
            X = np.array(samples).astype(np.float32)/32768
            # Extract audio features, including mel-freq cepestral coefs, mel spectrogram, and chromagram
            stft = np.abs(librosa.stft(X))
            result = np.array([])
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, fmax=8000).T, axis=0)
            result = np.hstack((result, mfccs))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, fmax=8000).T, axis=0)
            result = np.hstack((result, mel))
            file_features.append(result)
        file_features = pd.DataFrame(np.array(file_features), columns=range(180))
        file_features['index'] = file_features.index
        file_features['date'] = file.split('/')[-1].split('.')[0]
        features.append(file_features)
    # Concatenate the results, format date, and save
    features = pd.concat(features, ignore_index=True)
    features['date'] = pd.to_datetime(features['date'])
    features.to_csv(output_path + "features.csv", index=False)