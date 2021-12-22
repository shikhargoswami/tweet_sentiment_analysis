from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from config import *


def sequence_vectorize(train_texts, val_texts):
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index

def preprocess_text(sample_text):
    # print(sample_text)
    sample_text = list([sample_text])
    # print(sample_text)
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(sample_text)
    preprocessed_text = tokenizer.texts_to_sequences(sample_text)
    # print(preprocessed_text)
    max_length = 40 # same as x_train
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH
    preprocessed_text = sequence.pad_sequences(preprocessed_text, maxlen=max_length)
    # print(preprocessed_text)
    return preprocessed_text
