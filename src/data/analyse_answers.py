import click
from tqdm import tqdm

import pandas as pd

from preprocess import unfold_abbreviation, read_abbreviations_dictionary
from preprocess import spell_check, drop_stopwords
from preprocess import tokenize_drop_punkt, normalize_tokens

import pymorphy2

# константы
MA = pymorphy2.MorphAnalyzer()
ABBREVIATIONS = read_abbreviations_dictionary()
EXTRA_STOPWORDS = ['участок', 'отдел', 'группа', 'другой', 
                   'работа', 'госуправление', 'менеджер',
                   'посредничество', 'школьник', 'безработный', 
                   'студент', 'студентка', 'ученик', 'ученица',
                   'поиск', 'учёба', 'школа',
                   'родитель', 'шея', 'курс', 'стипендия',
                   'мама', 'пенсионер', 'пенсия',
                   'год', 'класс', 'маленький']

# функции
def count_verbs(words: list) -> int:
    '''Возвращает количество глаголов в списке слов'''
    cnt = 0
    
    for word in words:
        # для слесарь у pymorh на первом месте глагол слесарить
        if word != 'слесарь':
            p = MA.parse(word)[0]
            if 'VERB' in p.tag or 'INFN' in p.tag:
                cnt += 1

    return cnt

def find_words(what: list, where: list) -> int:
    '''Ищет превое вхождение любого элемента what в where 
    '''
    found = 0
    for word in where:
        if word.lower() in what:
            found = 1
            break

    return found

def check_negative(words: list) -> int:
    '''Определяет, есть ли среди слов отрицания
    '''
    negaties = ['нет', 'не', 'ни', 'никем', 'без', 'никто', 'некем', 'некто']
    return find_words(what=negaties, where=words)

def check_past_or_future(words: list) -> int:
    '''Определяет, есть ли упоминания прошлого или будущего
    '''
    past_future = ['бывший', 'бывшая', 'будущий', 'будушая']
    return find_words(what=past_future, where=words)

@click.command()
@click.argument('source', type=click.Path(exists=True))
@click.argument('destination', type=click.Path())
@click.option('--col_name', '-c', help='Название столбца с тектом')
def analyse_answers(source: str, destination: str, col_name: str) -> None:
    '''Анализирует открытые ответы и дополняет датасет результатами анализа:
    
    \b
    - расшифровывает аббервиатуры,
    - исправляет опечатки,
    - выделяет токены,
    - сохраняет существительные и прилагательные в нормальной форме,
    - подсчитывает количество слов
    - количество слов, не найденных в словаре,
    - наличие отрицания (не, ни, без)
    - количество глаголов
    
    \b
    Аргументы:
        - source -- исходный файл,
        - destination -- файл с результатами
    '''

    # читаем данные и проверяем наличие столбца col_name
    df = pd.read_csv(source, index_col=0, low_memory=False)
    if col_name not in df.columns:
        print(f'в данных нет столбца {col_name}')
        return
    
    # добавим новые столбцы
    df['corrected'] = ''
    for col in ['n_out_of_vocab', 'n_words', 'n_verbs', 'has_negative', 'has_past_future']:
        df[col] = 0

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        answer = row[col_name]

        # в B2B части ответы содержат категорию и занчение, разделенные "|"
        # нам нужно только значение в sour
        if '|' in answer:
            parts = [w.strip() for w in answer.split('|')]
            # для IT категории в профессии не хватает категории, оставим IT
            if parts[0] == 'IT' and parts[1] in ['Администратор', 'Аналитик', 'Архитектор']:
                answer = ' '.join(parts)
            else:
                answer = parts[1].replace('IT-', 'IT ') if 'IT-' in parts[1] else parts[1]  

        words = answer.split()

        # заменим аббревиатуры расшифровкой
        words = unfold_abbreviation(words, ABBREVIATIONS)

        # исправим опечатки и посчитаем количество слов вне словаря 
        words, n_out_of_vocab = spell_check(words)

        # определим, есть ли отрицание в ответе
        has_negative = check_negative(words)

        # посчитаем количество глаголов
        n_verbs = count_verbs(words)

        # разобьем на токены и удалим стоп-слова
        tokens = tokenize_drop_punkt(' '.join(words))

        # приведем токены к номальной форме, оставим только существительные и прилагательные
        normalized_tokens = normalize_tokens(tokens)
        
        # удалим стоп-слова
        normalized_tokens = drop_stopwords(normalized_tokens, extra_stop_words=EXTRA_STOPWORDS) 

        # удалим повторы и отсортируем
        normalized_tokens = list(set(normalized_tokens))
        normalized_tokens.sort()

        # проверим, есть ли упоминаение будущего или прошлого
        has_past_future = check_past_or_future(words)

        # добавим к датасету результаты анализа
        df.loc[i, 'corrected'] = ' '.join(words)
        df.loc[i, 'n_out_of_vocab'] = n_out_of_vocab
        df.loc[i, 'n_words'] = len(words)
        df.loc[i, 'n_verbs'] = n_verbs
        df.loc[i, 'has_negative'] = has_negative
        df.loc[i, 'has_past_future'] = has_past_future
        df.loc[i, 'tokenized'] = ' '.join(tokens)
        df.loc[i, 'normalized'] = ' '.join(normalized_tokens)

    # сохраним результат
    df.to_csv(destination)

if __name__ == '__main__':
    analyse_answers()