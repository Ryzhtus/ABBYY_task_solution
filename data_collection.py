from nlp_data_py import WikiDataset
from nlp_data_py import Book, Splitter

languages = ['ru', 'en', 'de', 'uk', 'be']
pages = {'ru': ['ABBYY', 'Яндекс', 'Google_(компания)', 'Microsoft', 'Apple', 'Mail.ru',
                'Московский физико-технический институт', 'Эйлер,_Леонард', 'Россия', 'Германия', 'Великобритания',
                'Украина', 'Европа', 'Соединённые_Штаты_Америки'],
         'en': ['ABBYY', 'Yandex', 'Google', 'Microsoft', 'Apple', 'Mail.ru',
                'Moscow_Institute_of_Physics_and_Technology', 'Leonhard_Euler', 'United_Kingdom',
                'Germany', 'United_States'],
         'de': ['ABBYY', 'Yandex', 'Google', 'Microsoft', 'Apple', 'Mail.ru',
                'Moskauer_Institut_für_Physik_und_Technologie', 'Leonhard_Euler', 'Vereinigte_Staaten'],
         'uk': ['ABBYY', 'Яндекс', 'Google', 'Microsoft', 'Московський_фізико-технічний_інститут', 'Леонард_Ейлер',
                'Велика_Британія', 'Росія', 'Німеччина', 'Україна'],
         'be': ['ABBYY', 'Яндэкс', 'Google', 'Microsoft', 'Маскоўскі_фізіка-тэхнічны_інстытут', 'Леанард_Эйлер',
                'Расія', 'Германія', 'Украіна', 'Беларусь', 'Вялікабрытанія']}

for lang in languages:
    scanned_pickle = './data/wiki_' + str(lang) + '/scanned_' + str(lang) + '.pkl'
    save_dataset_path = './data/wiki_' + str(lang) + '/'

    book_def: Book = Book(chunk_splitter='(?<=[.!?]) +', chunks_per_page=2)
    splitter: Splitter = Splitter(split_ratios=[0.66, 0.33], dataset_names=[str(lang) + '_train', str(lang) + '_test'],
                                  shuffle=False)

    WikiDataset.create_dataset_from_wiki(lang=lang,
                                         seeds=pages.get(lang),
                                         match=".*neuro",
                                         recursive=True, limit=2,
                                         scanned_pickle=scanned_pickle,
                                         save_dataset_path=save_dataset_path,
                                         book_def=book_def,
                                         splitter=splitter)