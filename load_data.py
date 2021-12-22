import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from config import *

def load_tweet_dataset(data_path, val_split = 0.15, seed=123):

    data  = pd.read_csv(data_path, encoding="utf-8")
    X_train, X_test, y_train, y_test  = _split_train_test_data(data, val_split, seed)
    train_texts = list(X_train)
    test_texts = list(X_test)
    train_labels = np.array(y_train)
    test_labels = np.array(y_test)

    
    train_labels = to_categorical(train_labels, NUM_CLASSES)
    test_labels = to_categorical(test_labels, NUM_CLASSES)

    return ((train_texts, train_labels), (test_texts, test_labels))

def _split_train_test_data(data, val_split, seed):

    data = train_test_split(data["text"], data["Sentiment"], test_size=val_split, random_state=seed, stratify=data["Sentiment"])
    return data