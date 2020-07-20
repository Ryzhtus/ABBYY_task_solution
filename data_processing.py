import pandas as pd

languages = ['ru', 'en', 'de', 'uk', 'be']

data_train = pd.DataFrame()
data_test = pd.DataFrame()

for lang in languages:
    train = open('data/wiki_' + str(lang) + '/' + str(lang) + '_train.txt', 'r+')
    test = open('data/wiki_' + str(lang) + '/' + str(lang) + '_test.txt', 'r+')

    # 1 - руссий, 0 - все остальные
    if lang == 'ru':
        label = 1
    else:
        label = 0

    for line in train:
        if line[0] == '=':
            pass
        else:
            line = line.strip()
            row_text = pd.Series([line, label])
            data_train = data_train.append(row_text, ignore_index=True)

    for line in test:
        if line[0] == '=':
            pass
        else:
            line = line.strip()
            row_text = pd.Series([line, label])
            data_test = data_test.append(row_text, ignore_index=True)


index_train = data_train[data_train[0] == '\n'].index
data_train.drop(index_train, inplace=True)
data_train.columns = ['text', 'lang']
data_train.to_csv('data/wiki_train.csv', index=False)

index_test = data_test[data_test[0] == '\n'].index
data_test.drop(index_test, inplace=True)
data_test.columns = ['text', 'lang']
data_test.to_csv('data/wiki_test.csv', index=False)