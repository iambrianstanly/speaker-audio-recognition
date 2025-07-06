
import librosa

def extract_mfcc(path):
    audio, sr = librosa.load(path, sr=None, duration=1)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs
