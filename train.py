import build_models
import argparse
import load_data
import vectorize_data
from config import *

import tensorflow as tf


FLAGS = None

def train_sequence_model(data,
                         learning_rate=LEARNING_RATE,
                         epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         blocks=2,
                         filters=64,
                         dropout_rate=DROPOUT_RATE,
                         embedding_dim=200,
                         kernel_size=3,
                         pool_size=3,
                         num_classes=NUM_CLASSES):
   
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Vectorize texts.
    x_train, x_val, word_index = vectorize_data.sequence_vectorize(
            train_texts, val_texts)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    # Create model instance.
    model = build_models.sepcnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=x_train.shape[1:],
                                     num_classes=num_classes,
                                     num_features=num_features)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_accuracy'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save(f"./models/sepcnn_epoch_{epochs}_loss_{history['val_loss'][-1]}.h5")
    return history['val_accuracy'][-1], history['val_loss'][-1]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATA_PATH)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=int, default=LEARNING_RATE)
    FLAGS, unparsed = parser.parse_known_args()

    data = load_data.load_tweet_dataset(FLAGS.data_dir)

    train_sequence_model(data, epochs=FLAGS.epochs, learning_rate=FLAGS.lr,num_classes=FLAGS.num_classes)