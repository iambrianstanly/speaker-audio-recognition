import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Model 
tf.random.set_seed(42)


def get_model(config):
    inputs = Input(shape=config["input_shape"])
    x = LSTM(128, return_sequences=False)(inputs)
    x = Dense(64, activation="relu")(x)
    x = Dense(config["n_class"], activation="softmax")(x)
    model = Model(inputs=inputs, outputs=x, name="Speech_Recognizer")
    return model