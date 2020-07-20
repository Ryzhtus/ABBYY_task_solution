import os
import string
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize

with open('models/logistic_regression/logistic_regression.pkl', 'rb') as file:
    logisitc_regression_model = pickle.load(file)

with open('models/logistic_regression/tf_idf_logistic_regression.pkl', 'rb') as file:
    tf_idf = pickle.load(file)

def predict(path: str):
    # функция предсказания, path - путь к папке с текстами
    directory = os.listdir(path)
    directory = sorted(directory, key=lambda x: int(x[:-4]))
    predictions = []

    for file in directory:
        open_file = open(path + '/' + file, 'r+')
        read_file = open_file.read()
        answer = predict_once(read_file)
        predictions.append(answer)

    data_for_csv = {'File Name': directory, 'Predictions': predictions}
    results = pd.DataFrame(data=data_for_csv)
    results.to_csv('data/predictions_abbyy.csv', index=False)


def predict_once(text: str):
    # функция предсказания для входной строки
    cleaned_text = text.translate(str.maketrans('', '', string.punctuation))
    cleaned_text = cleaned_text.lower()
    tokenized_text = word_tokenize(cleaned_text)
    vector = tf_idf.transform(tokenized_text)
    prediction = logisitc_regression_model.predict(vector)

    return max(prediction)

predict('data/data_abbyy')
