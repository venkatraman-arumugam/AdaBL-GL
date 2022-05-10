from __future__ import absolute_import
from concurrent.futures import process

import os
import argparse
import configparser
import logging

import re
import time

import torch

from learning.codesearcher import CodeSearcher
from preprocess.lex.token import Tokenizer
from preprocess.lex.word_sim import WordSim
from preprocess.prepare import prepare, bl_prepare
from preprocess.dataset import CodeSearchDataset, MatchingMatrix

from time import gmtime, strftime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser("train and test code search model")
    parser.add_argument("-p", "--prepare", action="store_true", default=False, help="Prepare dataset first.")
    parser.add_argument("--mode", choices=["train", "eval", "debug", "statistics","predict", "test"],
                        default="predict",
                        help="The mode to run. Use `train` mode to train a model;"
                        "Use `eval` mode to evaluate the model;"
                        "Use `predict` mode to make predictions")
    parser.add_argument("-v", "--verbose", default=True, help="Print verbose info.")
    # parser.add_argument("--project", type=str)
    parser.add_argument("--data_setting", type=str)
    parser.add_argument("--model_type", type=str)
    option = parser.parse_args()
    return option

# def get_config():
#     basedir = os.path.dirname(__file__)
#     config = configparser.ConfigParser()
#     config.read(os.path.join(basedir, './conf/config.ini'))
#     config.set('data', 'wkdir', basedir)
#     return config

def get_config(processed_data_basedir, data_settings, project, model_type):
    
    time_folder = folder_structure(model_type)

    bl_model_basedir = f"models/{model_type}/{data_settings}/{project}"

    # bl_model_basedir = f"{time_folder}/{data_settings}/{project}"
    
    if not os.path.isdir(bl_model_basedir):
        os.makedirs(bl_model_basedir)


    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), './conf/config.ini'))
    config.set('data', 'wkdir', processed_data_basedir)
    config.set('data', 'modedir', bl_model_basedir)

    # result_write_folder= f"Results/{model_type}/{data_settings}"

    result_write_folder = f"{time_folder}/{data_settings}"

    if not os.path.isdir(result_write_folder):
        os.makedirs(result_write_folder)

    result_write_file = os.path.join(result_write_folder, f"{project}_result.txt")
    
    
    
    result_write_folder= f"{time_folder}/{data_settings}/ranking_results"
        
    if not os.path.isdir(result_write_folder):
        os.makedirs(result_write_folder)
    
    
    rank_result_write_file = os.path.join(result_write_folder, f"{project}.csv")
    
    logger.info(f"writing results to {result_write_file}")

    config.set('data', 'result', result_write_file)
    
    config.set('data', 'ranking_results', rank_result_write_file)
    config.set('data', 'model_type', model_type)
    
    return config



def folder_structure(run_name):
    result_path=os.path.join(os.getcwd(),"Result",run_name,strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path


def main():
    
    option = parse_args()
    data_base_path = "/home/varumuga/scratch/Thesis/research/AdaCS/bl_dataset"
    data_setting = option.data_setting
    model_type = option.model_type
    # project = option.project
    # data_setting = option.data_setting
    # processed_data_basedir = f"processed_data_output/{data_setting}/{project}"
    # if not os.path.isdir(processed_data_basedir):
    #     os.makedirs(processed_data_basedir)

    # data_path = os.path.join(data_base_path, data_setting)
    # output_db_path  = "{data_setting}/{project}"

    # conf = get_config(processed_data_basedir, data_setting,  project)
    # # if not os.path.isdir(output_db_path):
    # #     os.makedirs(output_db_path)
    # train_db_path = os.path.join(processed_data_basedir, f"{project}_train_codevec.db")

    # test_db_path = os.path.join(processed_data_basedir, f"{project}_test_codevec.db")
    # projects =  ['AspectJ', 'Eclipse_Platform_UI', 'JDT', 'SWT', 'Birt', 'Tomcat']
    # 'Eclipse_Platform_UI', 'JDT',
    # 'SWT', 'Birt', 'Tomcat'
    
    if option.prepare:
        # , 'Eclipse_Platform_UI', 'JDT', 'SWT', 'Birt', 'Tomcat'
        projects =  ['AspectJ']
        # projects = ["JDT"]
        for project in projects:            
            processed_data_basedir = f"processed_data_output/{data_setting}/{project}"
            if not os.path.isdir(processed_data_basedir):
                os.makedirs(processed_data_basedir)

            data_path = os.path.join(data_base_path, data_setting)

            conf = get_config(processed_data_basedir, data_setting,  project, model_type)

            train_db_path = os.path.join(processed_data_basedir, f"{project}_train_codevec.db")

            test_db_path = os.path.join(processed_data_basedir, f"{project}_test_codevec.db")
            logger.info("preparing dataset...")
            #prepare(conf, conf['data']['train_code_path'], conf['data']['train_nl_path'], conf['data']['train_db_path'], train_mode=True)
            #prepare(conf, conf['data']['valid_code_path'], conf['data']['valid_nl_path'], conf['data']['valid_db_path'], train_mode=False, train_db_path=conf['data']['train_db_path'])
            
            start = time.time()
            print("Prepraring Train data")
            bl_prepare(conf, data_path, project, "train", train_db_path, train_mode=True, train_db_path=train_db_path)
            print(f"Train data done in {time.time() - start}")

            start = time.time()
            print("Prepraring Test data")
            bl_prepare(conf, data_path, project, "test", test_db_path, train_mode=False, train_db_path=train_db_path)
            print(f"Test data preparation done in {time.time() - start}")
        # prepare(conf, conf['data']['test_code_path'], conf['data']['test_nl_path'], conf['data']['test_db_path'], train_mode=False, train_db_path=conf['data']['train_db_path'])
    elif option.mode == 'train':
        # 'AspectJ', 'Eclipse_Platform_UI', 'SWT', 'Birt', 
        # 'AspectJ', 'Eclipse_Platform_UI', 'SWT', 'Birt', 
        projects =  ['AspectJ', 'Eclipse_Platform_UI', 'SWT', 'Birt','JDT', 'Tomcat']

        for project in projects:            
            start = time.time()
            processed_data_basedir = f"processed_data_output/{data_setting}/{project}"
            if not os.path.isdir(processed_data_basedir):
                os.makedirs(processed_data_basedir)

            data_path = os.path.join(data_base_path, data_setting)

            conf = get_config(processed_data_basedir, data_setting,  project, model_type)

            train_db_path = f"{project}_train_codevec.db"

            test_db_path = f"{project}_test_codevec.db"

            conf.set('data', 'train_db_path', train_db_path)
            conf.set('data', 'test_db_path', test_db_path)
            word_emb_model_path = os.path.join("/home/varumuga/scratch/Thesis/research/AdaCS/processed_data_output/", data_setting, project, f"{project}_train_codevec.model")
            conf.set('data', 'word_emb_model_path', word_emb_model_path)
            sc_data_path = os.path.join("bl_dataset", data_setting, project, f"{project}_train_br_sc_.csv")
            conf.set('data', 'sc_datapath', sc_data_path)
            logger.info(f"start training model in {data_setting} for {project}")
            searcher = CodeSearcher(conf)
            searcher.train()
            end = time.time() - start
            logger.info(f"Total Time for training model {end}")
            logger.info(f"done training model in {{data_setting}} for {project}")
    
    elif option.mode == "test":
        #  'AspectJ', 'Eclipse_Platform_UI', 'JDT',
        projects =  [ 'SWT', 'Birt','Tomcat']
        num = 1
        for project in projects:            
            start = time.time()
            processed_data_basedir = f"processed_data_output/{data_setting}/{project}"

            if not os.path.isdir(processed_data_basedir):
                os.makedirs(processed_data_basedir)

            data_path = os.path.join(data_base_path, data_setting)

            conf = get_config(processed_data_basedir, data_setting,  project, model_type)

            train_db_path = f"{project}_train_codevec.db"

            test_db_path = f"{project}_test_codevec.db"

            conf.set('data', 'train_db_path', train_db_path)
            conf.set('data', 'test_db_path', test_db_path)
            word_emb_model_path = os.path.join("/home/varumuga/scratch/Thesis/research/AdaCS/processed_data_output/", data_setting, project, f"{project}_test_codevec.model")
            conf.set('data', 'word_emb_model_path', word_emb_model_path)
            sc_data_path = os.path.join("bl_dataset", data_setting, project, f"{project}_test_br_sc_.csv")
            conf.set('data', 'sc_datapath', sc_data_path)
            result_write_folder= f"Results/{model_type}/{data_setting}"

            if not os.path.isdir(result_write_folder):
                os.makedirs(result_write_folder)

            result_write_file = os.path.join(result_write_folder, f"{project}_result.txt")

            logger.info(f"start testing model in {data_setting} for {project}")

            logger.info(f"writing results to {result_write_file}")

            conf.set('data', 'result', result_write_file)
            searcher = CodeSearcher(conf)
            searcher.load_model(int(num))
            searcher.test()
            end = time.time() - start
            logger.info(f"Total Time for testing model {end}")
            logger.info(f"done testing model in {data_setting} for {project}")

    elif option.mode == 'eval':
        num = input('Please input the epoch of the model to be loaded: ')
        searcher = CodeSearcher(conf)
        searcher.load_model(int(num))
        print('load model successfully.')
        searcher.eval2()
    elif option.mode == 'predict':
        num = input('Please input the epoch of the model to be loaded: ')
        path = input('Please input the save path of model outputs: ')
        searcher = CodeSearcher(conf)
        searcher.load_model(int(num))
        print('load model successfully.')
        searcher.predict(path)
    elif option.mode == 'statistics':
        s = input('Please input the relative data path (e.g. "domain/test"):')
        paths = s.strip().split(';')
        data = []
        for x in paths:
            base_path = os.path.join(conf['data']['wkdir'], './data/'+x)
            data += Tokenizer().parse(base_path + '.nl', base_path + '.code')
        data = [item for item in data if len(item[0]) and len(item[0])<=int(conf['data']['query_max_len']) and len(item[1])<=int(conf['data']['code_max_len'])]
        print('|utterances| = ' + str(len(data)))
        c = 0
        for item in data:
            c += len(item[0])
        print('|natural language tokens| = ' + str(c))
        c = 0
        for item in data:
            c += len(item[1])
        print('|code tokens| = ' + str(c))
        c = set()
        for item in data:
            for w in item[0]:
                c.add(w)
        print('|unique natural language tokens| = ' + str(len(c)))
        for item in data:
            for w in item[1]:
                c.add(w)
        print('|unique code tokens| = ' + str(len(c)))
        nlMap = [0 for _ in range(int(conf['data']['query_max_len'])+1)]
        codeMap = [0 for _ in range(int(int(conf['data']['code_max_len'])/10)+1)]
        for item in data:
            nlMap[len(item[0])] += 1
            codeMap[int(len(item[1])/10)] += 1
        print(nlMap)
        print(codeMap)
    elif option.mode == 'debug':
        line = input('Please input two item ids, seperated by space: ')
        eles = line.strip().split()
        data = Tokenizer().parse(os.path.join(conf['data']['wkdir'], conf['data']['test_nl_path']),
                                 os.path.join(conf['data']['wkdir'], conf['data']['test_code_path']))
        fasttext_corpus_path = os.path.join(conf['data']['wkdir'], re.sub(r'\.db$', '.txt', conf['data']['test_db_path']))
        core_term_path = os.path.join(conf['data']['wkdir'], 'conf/core_terms.txt')
        word_sim = WordSim(core_term_path, pretrain=(conf['model']['pretrained_wordvec'] == str(True)), fasttext_corpus_path=fasttext_corpus_path)
        for a in range(len(data)):
            if data[a][2] == eles[0]:
                for b in range(len(data)):
                    if data[b][2] == eles[1]:
                        matrix = MatchingMatrix(data[a][0], data[b][1], data[a][2], word_sim, conf['data']['query_max_len'])
                        for i in range(len(matrix)):
                            for j in range(len(matrix[0])):
                                print('%5.2f' % data.matrix[i][j], end=', ')
                            print()
                        break
                break

if __name__ == '__main__':
    main()
