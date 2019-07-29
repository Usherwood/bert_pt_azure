#!/usr/bin/env python

"""Functions for processing the data at stages 1 and 2"""

import csv
import os

import pandas as pd
import numpy as np
from collections import Counter

from text_preprocessing import tokenizer_word, clean_text_for_language_model

__author__ = "Peter J Usherwood and Steve Smit"
__python_version__ = '3.6'


def read_in_chunks(file, chunk_size=1024):
    last = ""
    while True:
        data = file.read(chunk_size)
        data = data.replace('<', '_').replace('>','_')  # Hack fix to deal with <unk> that arnt read as whole, will cause natural <>s to appear as _
        words = tokenizer_word(last + data, tokenize_punc=True)

        if data and data[-1] == ' ':
            last = ''
        else:
            last = words.pop()
        yield words
    yield last


def read_raw_data_preprocess_and_save(dataset_file_map,
                                      models_dir,
                                      dataset_dir,
                                      input_type,
                                      split_clitics,
                                      remove_numbers,
                                      base_folder="preprocessed_ulm_data"):
    """
    Read raw data in chunks, preprocess it according to preprocessing for language models, save in chunks in a single
    folder if the input type is 'all' or in 3 corresponding folders if given as 'train', 'validate', and 'test' files.

    :param dataset_file_map: Dict organizing the inputs. Either has a key 'all' with a single filename containing all
    the data. Or will have 3 keys 'train', 'validate', 'test' each with a filename corresponding to the appropriate data
    set.
    :param models_dir: pathlib path to directory where all processed data dfor the model will be stored (note this is
    a root, the code will create the substructure)
    :param dataset_dir: pathlib path to the directory where the raw data is held
    :param input_type: Str, either tokens, csv, or txt.
    :param split_clitics: Bool, if true splits clitics into sperate tokens, NOTE IF DATA IS ALREADY SPLIT ON CLITICS
    SET TO FALSE

    :return: Saves preprocessed data into corresponding folders in chunks
    """

    dataset_file_map, file_split = _process_dataset_file_map(dataset_file_map, models_dir)

    for dataset_key in dataset_file_map.keys():

        zx = 0

        if file_split == 'all':
            preprocessed_data_dir = models_dir / base_folder
        else:
            preprocessed_data_dir = models_dir / base_folder / dataset_key

        if input_type == 'tokens' or input_type == 'txt':
            cleaned_tokens = []
            pieces = []
            with open(dataset_dir / dataset_file_map[dataset_key], 'r', encoding='utf-8') as inp:
                for ix, piece in enumerate(read_in_chunks(inp)):
                    text = " ".join(piece)
                    if ix == 0:
                        text = '.' + text
                    cleaned_tokens += tokenizer_word(clean_text_for_language_model(text=text,
                                                                                   remove_numbers=remove_numbers),
                                                     tokenize_punc=True,
                                                     split_clitics=split_clitics)
                    if ix % 500 == 0 and ix != 0:
                        if len(cleaned_tokens) == 0:
                            break
                        zxth_data_path = preprocessed_data_dir / 'data{}.csv'.format(zx)
                        with zxth_data_path.open("w", encoding="utf-8") as output:
                            print(len(cleaned_tokens))
                            writer = csv.writer(output, lineterminator='\n')
                            writer.writerow(cleaned_tokens)
                        cleaned_tokens = []
                        zx += 1
        elif input_type == 'csv':
            df = pd.read_csv(dataset_dir / dataset_file_map[dataset_key])
            text = " _doc_ ".join(df['text'])

            cleaned_tokens = tokenizer_word(clean_text_for_language_model(text=text,
                                                                          remove_numbers=remove_numbers),
                                            tokenize_punc=True,
                                            split_clitics=split_clitics)

            z_data_path = preprocessed_data_dir / 'data.csv'.format(zx)
            with z_data_path.open("w", encoding="utf-8") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerow(cleaned_tokens)

    return True


def create_vocab_df(preprocessed_training_data_dir):

    vocab_df = pd.DataFrame(columns=['Word'])

    for file in os.listdir(preprocessed_training_data_dir):
        with open(os.path.join(preprocessed_training_data_dir, file), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            words = list(reader)[0]
        words_as_set = set(words)
        print('Number of words %d' % len(words_as_set))

        count_dict = Counter(words)
        vocab_df_current = pd.DataFrame(np.array([list(dict(count_dict).keys()), list(dict(count_dict).values())]).T,
                                        columns=['Word', 'Freq'])

        vocab_df_current['Freq'] = vocab_df_current['Freq'].astype(np.float64)
        vocab_df = vocab_df.append(vocab_df_current)
        del vocab_df_current

    vocab_df = vocab_df.groupby(vocab_df['Word']).agg({'Freq': sum, 'Word': 'first'})
    vocab_df = vocab_df.sort_values(by=['Freq'], ascending=False)

    return vocab_df


def _process_dataset_file_map(dataset_file_map, models_dir):

    if dataset_file_map['all']:
        file_split = 'all'
        dataset_file_map.pop('train', None)
        dataset_file_map.pop('validate', None)
        dataset_file_map.pop('test', None)
    elif not dataset_file_map.get('train') or not dataset_file_map.get('validate') or not dataset_file_map.get('test'):
        raise ValueError('Input must have a file of all data OR 3 files of train test validate')
    else:
        (models_dir / "preprocessed_ulm_data" / "train").mkdir(exist_ok=True)
        (models_dir / "preprocessed_ulm_data" / "test").mkdir(exist_ok=True)
        (models_dir / "preprocessed_ulm_data" / "validate").mkdir(exist_ok=True)
        file_split = 'split'
        dataset_file_map.pop('all')

    return dataset_file_map, file_split
