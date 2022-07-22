import re
import pandas as pd
import os

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

DICTIONARY_PATH = ('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2]) 
                   + '/models/symspell/professions.txt')

SYM_SPELL.load_dictionary(DICTIONARY_PATH, 0, 1)


def tokenize_drop_punkt(sentence: str) -> list:
    '''Разбивает предложение на токены без знаков пунктуации
    '''
    return [w.lower() for w in word_tokenize(sentence, language='russian') if IS_ALPHA_RE.match(w)]

def drop_stopwords(words: list) -> list:
    '''Удаляет стоп-слова из списка слов
    '''
    return [w for w in words if w not in STOPWORDS]

def read_abbreviations_dictionary(filename: str='../datasets/external/abbreviation.csv') -> dict:
    '''Считывает словарь из CSV файла и преобразует в dict'''
    data = pd.read_csv('../datasets/external/abbreviation.csv', sep=';')
    return dict(zip(data.abbreviation.to_list(), 
                    data.meaning.to_list()))

def unfold_abbreviation(words: list, abbreviations: dict) -> list:
    '''Расшифровывает аббревиатуры и возвращает список слов'''
    unfolded_words = []

    for word in words:
        w = abbreviations[word].split() if  word in abbreviations else [word]
        unfolded_words += w
                    
    return unfolded_words

def spell_check(words: list) -> tuple:
    '''Заменяет опечатки в переданных словах.
    Возвращает:
        список исправленных слов
        количество слов, не найденных в словаре 
    '''
    n_out_of_vocab = 0
    corrected_words = []

    for word in words:
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