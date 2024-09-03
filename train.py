
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from create_data import extract_features


def main():
    # Set the parent directory for speaker folders
    parent_dir = "data/train"
    
    # List of speaker folders
    speaker_folders = [
        "Benjamin",
        "Jens",
        "Julia",
        "Margaret",
        "Nelson"
    ]
    
    # Extract features and labels
    X, y = extract_features(parent_dir, speaker_folders)
    
    
    # Encode labels with explicit classes
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    label_encoder.classes_ = np.array(speaker_folders)
    
    
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Print the shapes of training and validation data
    print("Training Data Shape:", X_train.shape)
    print("Validation Data Shape:", X_val.shape)
    
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(speaker_folders), activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    # Train the model with EarlyStopping
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[early_stopping])
    
    # model.save('./model/model.h5')
    
    pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
    plt.savefig('./output/accuracy.png')
    
    pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
    plt.savefig('./output/loss.png')
    
    
    # pd.DataFrame(history.history).to_csv('./history.csv', index=False)
    
if __name__ == '__main__':
    import sys
    sys.exit(main())
