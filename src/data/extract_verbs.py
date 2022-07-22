import click
from regex import F
from tqdm import tqdm
import pandas as pd
import pymorphy2 

from preprocess import drop_stopwords, tokenize_drop_punkt, spell_check

MA = pymorphy2.MorphAnalyzer()

def check_for_verbs(tokens: list) -> list:
    '''Ищет в tokens глаголы
    Возвращает список глагол - время
    '''
    result = []

    for token in tokens:
        p = MA.parse(token)[0]
        if 'VERB' in p.tag:
            result.append([f'{token} ({p.normal_form})', p.tag.tense])
        elif 'INFN' in p.tag:
            result.append([f'{token} ({p.normal_form})', 'infn'])
    
    return result

@click.command()
@click.argument('source', type=click.Path(exists=True))
@click.argument('destination', type=click.Path())
@click.option('--col_name', '-c', help='Название столбца с тектом')
def extract_verbs(source: str, destination: str, col_name: str) -> None:
    '''
    Извлекает из текста глаголы и сохраняет информацию об их морфололической форме 

    \b
    Аргументы:
        - source - пусть к csv файлу с исходными данными
        - destination - путь к csv файлу с рузультатами выделения глалогов
    '''
    
    # читаем данные и проверяем наличие столбца col_name
    df = pd.read_csv(source, low_memory=False)
    if col_name not in df.columns:
        print(f'в данных нет столбца {col_name}')
        return
    
    # выделим уничкальные значения и посчитаем их количество
    frequency_table = df[col_name].value_counts().reset_index()
    frequency_table.columns = ['term', 'count']

    verbs = []
    tenses = []
    counts = []

    # пройдем по всем строкам и определим наличие глаголов
    for i, row in tqdm(frequency_table.iterrows(), total=frequency_table.shape[0]):
        answer = row['term'].lower()

        # исправим опечатки
        spell_checked, n_out = spell_check(answer.split(' '))

        # разобъем на токен и удалим стоп-слова
        tokens = drop_stopwords(tokenize_drop_punkt(' '.join(spell_checked)))

        # проверим наличие глаголов
        result = check_for_verbs(tokens)

        # если глаголы есть, сохраним результат
        if len(result): 
            for r in result:
                verbs.append(r[0])
                tenses.append(r[1])
                counts.append(row['count'])
    
    # подсчитаем количество уникальных глаголов
    verbs_df = pd.DataFrame({'verb': verbs, 'tense': tenses, 'count': counts})\
               .groupby(by=['verb', 'tense']).sum().reset_index()

    # сохраним результат
    verbs_df.to_csv(destination, index=False)

    print('Найдено ', verbs_df.shape[0], 
          MA.parse('глагол')[0].make_agree_with_number(verbs_df.shape[0]).word)  
    print('Сохранено в ', destination)

if __name__ == '__main__':
    extract_verbs()
