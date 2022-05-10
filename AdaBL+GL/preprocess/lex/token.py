import re
import os

from gensim.parsing import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd

class Tokenizer:

    def __init__(self):
        self.p = PorterStemmer()

    # def parse_bl(self, data_path, project, data_type):

       
    #     return self.__combine(self.__parse_file(os.path.join(data_path, project, f"{project}_{}_br_sc_.csv")), True, True, 
    #                           self.__parse_file(os.path.join(data_path, project, f"{project}_train_sc"), False, True))

    def parse(self, nl_path, code_path):
        return self.__combine(self.__parse_file(nl_path, True, True), 
                              self.__parse_file(code_path, False, True))

    def parse_nl(self, nl_path):
        return self.__parse_file(nl_path, True, True)

    def parse_code(self, code_path):
        return self.__parse_file(code_path, False, True)

    @staticmethod
    def __combine(nl_dict, code_dict):
        ret = []
        for key in sorted([int(key) for key in nl_dict.keys()]):
            ret.append((nl_dict[str(key)], code_dict[str(key)], str(key)))
        return ret


    def _parse_df(dataset_file_path, br_rm_stop_words, sc_rm_stop_words, br_stem, sc_stem):
        df = pd.read_csv(dataset_file_path)
        df.report = df.report.appy(lambda x : __get_tokens(x, br_rm_stop_words, br_stem))

        df.code = df.code.appy(lambda x : __get_tokens(x, br_rm_stop_words, br_stem))

        return df

    def __parse_file(self, file_path, rm_stopwords=False, stem=False):
        ret = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 0:
                    p = line.index('\t')
                    idx = line[: p]
                    tokens = self.__get_tokens(line[p + 1:], rm_stopwords, stem)
                    ret[idx] = tokens
        return ret


    def get_tokens(self, content, rm_stopwords=False, stem=False):
        try:
            
            return self.__get_tokens(content, rm_stopwords, stem)
        except:
            print(content)
            raise Exception() 

    def __get_tokens(self, content, rm_stopwords=False, stem=False):
        words = [word for word in re.split('[^A-Za-z]+', content) if len(word) > 0]
        ret = []
        for word in words:
            ret += self.__camel_case_split(word)
        tmp = []
        for word in ret:
            if rm_stopwords:
                word = remove_stopwords(word)
            if len(word) > 0:
                if stem:
                    word = self.p.stem(word)
                tmp.append(word)
        ret = tmp
        return ret

    @staticmethod
    def __camel_case_split(word):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', word)
        return [m.group(0).lower() for m in matches]
