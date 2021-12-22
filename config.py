import tensorflow as tf

NUM_CLASSES = 3
LABELS = ["Neutral", "Positive", "Negative"]

TOP_K = 20000
MAX_SEQUENCE_LENGTH = 500
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
DROPOUT_RATE = 0.2
EPOCHS = 10

DATA_PATH = "./data/finale.csv"

METRICS = [
           tf.keras.metrics.Accuracy(name='accuracy'),
           tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall')
]
