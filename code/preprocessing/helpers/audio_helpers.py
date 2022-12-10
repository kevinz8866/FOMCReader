import glob
import librosa # to extract speech features
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import pandas as pd


def to_one_channel(inpath, outpath, split_length=None):
    """ This function converts all conference audios into one-channel wav files
    """
    for file in glob.glob(inpath):
        out = outpath + file.split('/')[-1].split('.')[-2]
        #if file.endswith('.wav'):
        sound = AudioSegment.from_file(file)
        sound = sound.set_channels(1)
        if split_length != None:
            chunk_length_ms = 1000*split_length
            chunks = make_chunks(sound, chunk_length_ms)
            for i, chunk in enumerate(chunks):
                chunk.export(out + ' ' + str(i) + '.wav', format="wav")
        else:
            sound.export(out + '.wav', format="wav")

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
   # with soundfile.SoundFile(file_name) as sound_file:
    X, sample_rate = librosa.load(file_name, sr=16000)
    print(sample_rate)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, fmax=8000).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate, fmax=8000).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))
    return result
