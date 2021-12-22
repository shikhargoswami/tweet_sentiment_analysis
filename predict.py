from json import load
from best_model import choose_best_model
from tensorflow.keras.models import load_model
import build_models
from vectorize_data import preprocess_text
from config import *
import argparse
from clean_data import clean

def predict(text):

    best_model_path = choose_best_model("./models/")
    model = load_model(best_model_path)

    # model.load_weights(best_model_path)
    cleaned_text = clean(text)
    preprocessed_text = preprocess_text(cleaned_text)

    scores = model.predict(preprocessed_text)
    print(scores)
    predicted_labels = scores.argmax(axis=1)
    # print(predicted_labels)
    return ([LABELS[label] for label in predicted_labels][0])


if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str)

    FLAGS, unparsed = parser.parse_known_args()
    predict(FLAGS.text)
