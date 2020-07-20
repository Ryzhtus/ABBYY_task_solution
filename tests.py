import string
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score

def clean_data():
    data = pd.read_csv('data/corpus_for_tests_py/dataset.csv')

    data = data[data['language'].isin(['English', 'Russian', 'Indonesian', 'Chinese', 'Korean', 'Turkish'])]

    data.loc[data['language'] == "Russian", "language"] = 1
    data.loc[data['language'] != 1, "language"] = 0

    data.reset_index(inplace=True)
    data = data.sample(frac=1).reset_index(drop=True)
    data['Text'] = data['Text'].replace(r'[{}]'.format(string.punctuation), '', regex=True)
    data['Text'] = [text.lower() for text in data['Text']]
    data['Text'] = [word_tokenize(text) for text in data['Text']]
    data = data.astype({"Text": str})

    return data


def test_log_reg():
    rnn_model = torch.load(PATH)

    data_test = clean_data()
    X = data_test['Text']
    y = data_test['language']
    y = y.astype('int')

    TF_IDF_X = tf_idf.transform(X)

    preds = logisitc_regression_model.predict(TF_IDF_X)
    preds = preds.astype('int')

    print(accuracy_score(y, preds))
    print(f1_score(y, preds))

def test_rnn():
    with open('models/logistic_regression/logistic_regression.pkl', 'rb') as file:
        rnn_model = pickle.load(file)

    preds = logisitc_regression_model.predict(TF_IDF_X)
    preds = preds.astype('int')

    print(accuracy_score(y, preds))
    print(f1_score(y, preds))

