import numpy as np
import tensorflow as tf # tf 2.14.1


def get_dataloaders(config):

    if config["mode"] == "train":

        train_data = np.load(config["train_dir"])
        valid_data = np.load(config["valid_dir"])

        print("train feature shape", train_data['X'].shape)
        print("valid feature shape", valid_data['X'].shape)

        print("train label shape", train_data['y'].shape)
        print("validation llabel shape", valid_data['y'].shape)

        train_ds = tf.data.Dataset.from_tensor_slices((train_data["X"], train_data["y"]))
        train_ds = train_ds.shuffle(buffer_size=config["buffer_size"], seed=42)
        train_ds = train_ds.batch(config["batch_size"])

        valid_ds = tf.data.Dataset.from_tensor_slices((valid_data["X"], valid_data["y"]))
        valid_ds = valid_ds.shuffle(buffer_size=config["buffer_size"], seed=42)
        valid_ds = valid_ds.batch(config["batch_size"])

        return train_ds, valid_ds
    
    elif config["mode"] == "eval":
        test_data = np.load(config["test_dir"])

        return test_data

    




