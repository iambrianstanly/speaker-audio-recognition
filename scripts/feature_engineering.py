import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import yaml
import argparse
import numpy as np
from src.feature_engineering.feature_extraction import extract_mfcc
from src.feature_engineering.normalization import run_normalisation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(root_dir):

    features = []
    labels = []
    class_map = {}  # Maps folder name to integer label
    current_label = 0

    for speaker_name in sorted(os.listdir(root_dir)):  # Sort for consistency
        speaker_dir = os.path.join(root_dir, speaker_name)
        if not os.path.isdir(speaker_dir):
            continue

        # Assign a numeric label for each class (folder name)
        class_map[speaker_name] = current_label

        for file in os.listdir(speaker_dir):
            if file.endswith(".wav"):
                path = os.path.join(speaker_dir, file)
                mfccs = extract_mfcc(path)
                features.append(mfccs.T)  # Shape: (time_steps, n_mfcc)
                labels.append(current_label)

        current_label += 1

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return features, labels


def save_train_data(config):
    X_train_full, y_train_full = load_data(config["train_dir"])

    X_train_full_norm = run_normalisation(X_train_full, config, mode="train")

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full_norm, y_train_full,
                                                          test_size=0.1, random_state=42, shuffle=True)

    np.savez(config["train_save_path"], X=X_train, y=y_train)
    np.savez(config["valid_save_path"], X=X_valid, y=y_valid)


def save_test_data(config):
    X_test, y_test = load_data(config["test_dir"])

    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1]) # temporal features are sustained
    X_test_norm = run_normalisation(X_test_reshaped, config, mode="test")
    
    np.savez(config["test_save_path"], X=X_test, y=y_test)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Add config file for feature engineering")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_train_data(config)
    save_test_data(config)



if __name__ == "__main__":
    main()


