import re
import pandas as pd
import os

import pymorphy2

from nltk.tokenize import word_tokenize
from nltk import download
from nltk.corpus import stopwords

from symspellpy import SymSpell, Verbosity

download('punkt')
download('stopwords')

# константы
IS_ALPHA_RE = re.compile('[\w-]{2,}')
STOPWORDS = stopwords.words('russian')
SYM_SPELL = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)

BASE_PATH = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
DICTIONARY_PATH = BASE_PATH + '/models/symspell/professions.txt'
ABBREVIATIONS_PATH = BASE_PATH + '/datasets/external/abbreviation.csv'
EXTRASTOPWORDS_PATH = BASE_PATH + '/datasets/external/extrastopwords.csv'

SYM_SPELL.load_dictionary(DICTIONARY_PATH, 0, 1)

MA = pymorphy2.MorphAnalyzer()

def tokenize_drop_punkt(sentence: str) -> list:
    '''Разбивает предложение на токены без знаков пунктуации
    '''
    return [w.lower() for w in word_tokenize(sentence, language='russian') if IS_ALPHA_RE.match(w)]

def drop_stopwords(words: list, extra_stop_words: list = []) -> list:
    '''Удаляет стоп-слова из списка слов
    '''
    sw = STOPWORDS + extra_stop_words
    return [w for w in words if w not in sw]

def normalize_tokens(tokens: list) -> list:
    '''Оставляет среди токенов только существительные и прилагательные в нормальной форме
    '''
    words = []
    for word in tokens:
        p = MA.parse(word)[0]
        # для слесарь у pymorh на первом месте глагол слесарить, добавим исключение
        if (word == 'слесарь'):
            words.append(word)
        elif ('NOUN' in p.tag) or ('ADJF' in p.tag) or ('ADJS' in p.tag):
            words.append(p.normal_form)
    return words  

def read_abbreviations_dictionary() -> dict:
    '''Считывает словарь из CSV файла и преобразует в dict'''
    data = pd.read_csv(ABBREVIATIONS_PATH, sep=';')
    return dict(zip(data.abbreviation.to_list(), 
                    data.meaning.to_list()))

def read_stopwords_dictionary() -> list:
    '''Считывает словарь из CSV файла и преобразует в dict'''
    data = pd.read_csv(EXTRASTOPWORDS_PATH, sep=';')
    return data.stopword.to_list()

def unfold_abbreviation(words: list, abbreviations: dict) -> list:
    '''Расшифровывает аббревиатуры и возвращает список слов'''
    unfolded_words = []

    for word in words:
        w = abbreviations[word].split() if  word in abbreviations else [word]
        unfolded_words += w
                    
    return unfolded_words

def spell_check(words: list, tolower=True) -> tuple:
    '''Заменяет опечатки в переданных словах.
    Возвращает:
        список исправленных слов
        количество слов, не найденных в словаре 
    '''
    n_out_of_vocab = 0
    corrected_words = []

    for word in words:
        if tolower:
            word = word.lower()
            
        corrected = word 

        if len(word) > 3:
            suggestions = SYM_SPELL.lookup(word, Verbosity.CLOSEST, include_unknown=True)
            if(len(suggestions) > 0):
                suggest = suggestions[0]
                if suggest.count > 0:
                    corrected = suggest.term 
                else:
                    n_out_of_vocab += 1
            else:
                n_out_of_vocab += 1
        else:
            n_out_of_vocab += 1

        corrected_words.append(corrected)

    return corrected_words, n_out_of_vocab  