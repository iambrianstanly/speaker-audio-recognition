import tensorflow as tf


def train_model(model, train_ds, valid_ds, config):
    
    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=["accuracy"])
        
    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=2, 
        restore_best_weights=True
    )

    check_point = tf.keras.callbacks.ModelCheckpoint(
        filepath=config["checkpoint_path"],
        monitor="val_loss",
        save_best_only=True
    )

    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=config["log_dir"])
    
    # Train the model with EarlyStopping
    model.fit(train_ds, validation_data=valid_ds, epochs=config["epochs"], callbacks=[early_stopping, check_point, tensorboard_callbacks])
    