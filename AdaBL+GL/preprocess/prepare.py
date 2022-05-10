import os

import re

from preprocess.lex.token import Tokenizer
from preprocess.lex.word_sim import WordSim
from preprocess.dataset import CodeSearchDataset
import pandas as pd


import zlib
# from itertools import zip

def source_code_decompress(compressed_sc):
    sc_code = zlib.decompress(bytes.fromhex(compressed_sc)).decode()
    return sc_code

def bl_prepare(conf, data_path, project, data_type, output_db_path, train_mode=True, train_db_path=None):

    if not train_mode:
        train_corpus_path = re.sub(r'\.db$', '.txt', train_db_path)
        print(f"reading the training corpus from { train_corpus_path}")
        # train_corpus_path =  re.sub(r'\.db$', '.txt', train_db_path)

    core_term_path = os.path.join(conf['data']['core_term_path'])
    fasttext_corpus_path = re.sub(r'\.db$', '.txt', output_db_path)

    tokenizer = Tokenizer()

    data_path = os.path.join(data_path, project)

    bl_df_file_name = os.path.join(data_path, f"{project}_{data_type}_br_sc_.csv")

    br_sc_df_tmp = os.path.join(os.path.dirname(output_db_path), f"{project}_{data_type}_processed_.pickle")

    if not os.path.isfile(br_sc_df_tmp):
        print(f"Reading BL data frame from {bl_df_file_name}")
        br_sc_df = pd.read_csv(bl_df_file_name)
        br_sc_df = br_sc_df.dropna()
        br_sc_df["reports_processed"]  = br_sc_df.report_processed.apply(lambda x: x.split("-|-"))
        # br_sc_df["file_content"] = br_sc_df.file_content.apply(source_code_decompress)

        br_sc_df["file_content_processed"] = br_sc_df.file_content_processed.apply(lambda x: x.split("-|-"))

        train_corpus_content = []
        if not train_mode:
            with open(train_corpus_path, 'r') as f:
                train_corpus_content = f.readlines()

        print(f"Writing FastText corpus to {fasttext_corpus_path}")

        with open(fasttext_corpus_path, 'w') as f:
            f.writelines(train_corpus_content)
            f.write('\n')
            for report_processed, sc_processed in zip(br_sc_df.reports_processed , br_sc_df.file_content_processed):
                f.write(' '.join(report_processed) + '\n')
                f.write(' '.join(sc_processed) + '\n')

        word_sim = WordSim(core_term_path, pretrain=(conf['model']['pretrained_wordvec'] == str(True)), update=True, fasttext_corpus_path=fasttext_corpus_path)
        query_max_len = int(conf['data']['query_max_len'])
        code_max_len = int(conf['data']['code_max_len'])
        br_sc_df["reports_processed"]  = br_sc_df["reports_processed"].apply(lambda x: x[:query_max_len])
        br_sc_df["file_content_processed"]  = br_sc_df["file_content_processed"].apply(lambda x: x[:code_max_len])

        br_sc_df["reports_processed_len"] = br_sc_df.reports_processed.apply(len)

        br_sc_df["file_content_processed_len"] = br_sc_df.file_content_processed.apply(len)

        tmp_br_sc_path = train_db_path
        br_sc_df.to_pickle(br_sc_df_tmp)
    
    else:
        word_sim = WordSim(core_term_path, pretrain=(conf['model']['pretrained_wordvec'] == str(True)), update=False, fasttext_corpus_path=fasttext_corpus_path)
        br_sc_df = pd.read_pickle(br_sc_df_tmp)

    
    if train_mode:
        CodeSearchDataset.create_bl_dataset(br_sc_df, word_sim, output_db_path,
                                         int(conf['data']['query_max_len']),
                                         int(conf['data']['code_max_len']),
                                         int(conf['train']['neg_top_k']),
                                         int(conf['train']['neg_sample_size']))
    else:
        CodeSearchDataset.create_bl_dataset(br_sc_df, word_sim, output_db_path,
                                         int(conf['data']['query_max_len']),
                                         int(conf['data']['code_max_len']),
                                         int(conf['train']['valid_neg_sample_size']),
                                         int(conf['train']['valid_neg_sample_size']))

def prepare(conf, code_path, nl_path, output_db_path, train_mode=True, train_db_path=None):

    if not train_mode:
        train_corpus_path = os.path.join(conf['data']['wkdir'], re.sub(r'\.db$', '.txt', train_db_path))

    core_term_path = os.path.join(conf['data']['wkdir'], 'conf/core_terms.txt')
    fasttext_corpus_path = os.path.join(conf['data']['wkdir'], re.sub(r'\.db$', '.txt', output_db_path))

    data = Tokenizer().parse(nl_path, code_path)

    train_corpus_content = []
    if not train_mode:
        with open(train_corpus_path, 'r') as f:
            train_corpus_content = f.readlines()
    with open(fasttext_corpus_path, 'w') as f:
        f.writelines(train_corpus_content)
        f.write('\n')
        for item in data:
            f.write(' '.join(item[0]) + '\n')
            f.write(' '.join(item[1]) + '\n')
    word_sim = WordSim(core_term_path, pretrain=(conf['model']['pretrained_wordvec'] == str(True)), update=True, fasttext_corpus_path=fasttext_corpus_path)

    if train_mode:
        CodeSearchDataset.create_dataset(data, word_sim, output_db_path,
                                         int(conf['data']['query_max_len']),
                                         int(conf['data']['code_max_len']),
                                         int(conf['train']['neg_top_k']),
                                         int(conf['train']['neg_sample_size']))
    else:
        CodeSearchDataset.create_dataset(data, word_sim, output_db_path,
                                         int(conf['data']['query_max_len']),
                                         int(conf['data']['code_max_len']),
                                         int(conf['train']['valid_neg_sample_size']),
                                         int(conf['train']['valid_neg_sample_size']))
