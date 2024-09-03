
import os
import librosa
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def main():        
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='enter the file path')
    
    args = parser.parse_args()
    
    # file_path = './data/test/Benjamin/1000.wav'
    audio, sr = librosa.load(args.file_path, sr=None, duration=1)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Normalize MFCC features
    mfccs = StandardScaler().fit_transform(mfccs)
    
    xinput = np.expand_dims(mfccs.T, axis=0)
    
    
    model = keras.models.load_model('./model/model.h5')
    y_pred = model.predict(xinput)
    
    class_names = [
        "Benjamin",
        "Jens",
        "Julia",
        "Margaret",
        "Nelson"
    ]
    
    class_name = class_names[np.argmax(y_pred)]
    
    print('predicted class', class_name)
    
if __name__ == '__main__':
    import sys
    sys.exit(main())