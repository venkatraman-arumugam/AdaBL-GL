from __future__ import division

import pyspark
import pandas as pd
# import ray.data as ray_pd
# import modin.pandas as ray_pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
import warnings
import javalang
import re
import glob
import math
import time
from scipy import spatial
import scipy.spatial.distance
import xml.etree.ElementTree as ET
import requests
import multiprocessing
from tqdm import tqdm_notebook
from time import gmtime, strftime
from random import randint
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import zlib
import pathlib
import time
from ast import literal_eval
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType,StructField,IntegerType,Row
from pyspark.sql.functions import udf, col


def split_natural_lang(doc):
    """
    @Receives: a document in natural language (bugreport)
    @Process: splits it as described in BugLocator
    @Return: a list of lower cased words
    """
    try:
        wordList=[]
        word=''
        for char in doc:
            if char.isalnum() or char=='\'':
                word+=char
            else:
                if len(word)>0:
                    wordList.append(word)
                    word=''
        if len(word)>0:
            wordList.append(word)
        return wordList
    except:
        return []


def code_splitter(sourceCode):
    """
    @Receives: a code
    @Process: splits it same as described in BugLocator
    @Return: a list of lower cased words
    """
    contentBuf = []
    wordBuf = []
    for char in sourceCode:
        if ((char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z')):
            wordBuf.append(char)
            continue
        length = len(wordBuf)
        if (length != 0):
            k = 0
            for i in range(length-1):
                j=i+1
                first_char = wordBuf[i]
                second_char = wordBuf[j]
                if ((first_char >= 'A' and first_char <= 'Z') and (second_char >= 'a' and second_char <= 'z')):
                    contentBuf.append(wordBuf[k:i])
                    contentBuf.append(' ')
                    k = i
                    continue
                if ((first_char >= 'a' and first_char <= 'z') and (second_char >= 'A' and second_char <= 'Z')):
                    contentBuf.append(wordBuf[k:j])
                    contentBuf.append(' ')
                    k = j
                    continue
            if (k < length):
                contentBuf.append(wordBuf[k:])
                contentBuf.append(" ")
            wordBuf=[]
    words=''
    for each in contentBuf:
        if isinstance(each,str):
            words+=each
        else: 
            for term in each: 
                words+=term
    words= words.split()
    contentBuf = []
    for i in range(len(words)):
        if (words[i].strip()!="" and len(words[i]) >= 2):
            contentBuf.append(words[i])
    return contentBuf



def general_preprocessor(doc,mode):
    """
    @Receives: a document (code or bug report denoted by mode)
    @Process: processes the docucument by stemming and removing stop-words and converting to lower cases
    @Return: a list of lower cased words
    """
    JavaKeywords=["abstract", "continue", "for", 
                "new", "switch", "assert", "default", "goto", "package", 
                "synchronized", "boolean", "do", "if", "private", "this", 
                "break", "double", "implements", "protected", "throw", "byte", 
                "else", "import", "public", "throws", "case", "enum", 
                "instanceof", "return", "transient", "catch", "extends", "int", 
                "short", "try", "char", "final", "interface", "static", "void", 
                "class", "finally", "long", "strictfp", "volatile", "const", 
                "float", "native", "super", "while", "org", "eclipse", "swt", 
                "string", "main", "args", "null", "this", "extends", "true", 
                "false"]
    stop_words=["a", "a's", "able", "about", "above",
                "according", "accordingly", "across", "actually", "after",
                "afterwards", "again", "against", "ain't", "all", "allow",
                "allows", "almost", "alone", "along", "already", "also",
                "although", "always", "am", "among", "amongst", "an", "and",
                "another", "any", "anybody", "anyhow", "anyone", "anything",
                "anyway", "anyways", "anywhere", "apart", "appear",
                "appreciate", "appropriate", "are", "aren't", "around", "as",
                "aside", "ask", "asking", "associated", "at", "available",
                "away", "awfully", "b", "be", "became", "because", "become",
                "becomes", "becoming", "been", "before", "beforehand",
                "behind", "being", "believe", "below", "beside", "besides",
                "best", "better", "between", "beyond", "both", "brief", "but",
                "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot",
                "cant", "cause", "causes", "certain", "certainly", "changes",
                "clearly", "co", "com", "come", "comes", "concerning",
                "consequently", "consider", "considering", "contain",
                "containing", "contains", "corresponding", "could", "couldn't",
                "course", "currently", "d", "definitely", "described",
                "despite", "did", "didn't", "different", "do", "does",
                "doesn't", "doing", "don't", "done", "down", "downwards",
                "during", "e", "each", "edu", "eg", "eight", "either", "else",
                "elsewhere", "enough", "entirely", "especially", "et", "etc",
                "even", "ever", "every", "everybody", "everyone", "everything",
                "everywhere", "ex", "exactly", "example", "except", "f", "far",
                "few", "fifth", "first", "five", "followed", "following",
                "follows", "for", "former", "formerly", "forth", "four",
                "from", "further", "furthermore", "g", "get", "gets",
                "getting", "given", "gives", "go", "goes", "going", "gone",
                "got", "gotten", "greetings", "h", "had", "hadn't", "happens",
                "hardly", "has", "hasn't", "have", "haven't", "having", "he",
                "he's", "hello", "help", "hence", "her", "here", "here's",
                "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
                "hi", "him", "himself", "his", "hither", "hopefully", "how",
                "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie",
                "if", "ignored", "immediate", "in", "inasmuch", "inc",
                "indeed", "indicate", "indicated", "indicates", "inner",
                "insofar", "instead", "into", "inward", "is", "isn't", "it",
                "it'd", "it'll", "it's", "its", "itself", "j", "just", "k",
                "keep", "keeps", "kept", "know", "knows", "known", "l", "last",
                "lately", "later", "latter", "latterly", "least", "less",
                "lest", "let", "let's", "like", "liked", "likely", "little",
                "look", "looking", "looks", "ltd", "m", "mainly", "many", 
                "may", "maybe", "me", "mean", "meanwhile", "merely", "might",
                "more", "moreover", "most", "mostly", "much", "must", "my",
                "myself", "n", "name", "namely", "nd", "near", "nearly",
                "necessary", "need", "needs", "neither", "never",
                "nevertheless", "new", "next", "nine", "no", "nobody", "non",
                "none", "noone", "nor", "normally", "not", "nothing", "novel",
                "now", "nowhere", "o", "obviously", "of", "off", "often", "oh",
                "ok", "okay", "old", "on", "once", "one", "ones", "only",
                "onto", "or", "other", "others", "otherwise", "ought", "our",
                "ours", "ourselves", "out", "outside", "over", "overall",
                "own", "p", "particular", "particularly", "per", "perhaps",
                "placed", "please", "plus", "possible", "presumably",
                "probably", "provides", "q", "que", "quite", "qv", "r",
                "rather", "rd", "re", "really", "reasonably", "regarding",
                "regardless", "regards", "relatively", "respectively", "right",
                "s", "said", "same", "saw", "say", "saying", "says", "second",
                "secondly", "see", "seeing", "seem", "seemed", "seeming",
                "seems", "seen", "self", "selves", "sensible", "sent",
                "serious", "seriously", "seven", "several", "shall", "she",
                "should", "shouldn't", "since", "six", "so", "some",
                "somebody", "somehow", "someone", "something", "sometime",
                "sometimes", "somewhat", "somewhere", "soon", "sorry",
                "specified", "specify", "specifying", "still", "sub", "such",
                "sup", "sure", "t", "t's", "take", "taken", "tell", "tends",
                "th", "than", "thank", "thanks", "thanx", "that", "that's",
                "thats", "the", "their", "theirs", "them", "themselves",
                "then", "thence", "there", "there's", "thereafter", "thereby",
                "therefore", "therein", "theres", "thereupon", "these", "they",
                "they'd", "they'll", "they're", "they've", "think", "third",
                "this", "thorough", "thoroughly", "those", "though", "three",
                "through", "throughout", "thru", "thus", "to", "together",
                "too", "took", "toward", "towards", "tried", "tries", "truly",
                "try", "trying", "twice", "two", "u", "un", "under",
                "unfortunately", "unless", "unlikely", "until", "unto", "up",
                "upon", "us", "use", "used", "useful", "uses", "using",
                "usually", "uucp", "v", "value", "various", "very", "via",
                "viz", "vs", "w", "want", "wants", "was", "wasn't", "way",
                "we", "we'd", "we'll", "we're", "we've", "welcome", "well",
                "went", "were", "weren't", "what", "what's", "whatever",
                "when", "whence", "whenever", "where", "where's", "whereafter",
                "whereas", "whereby", "wherein", "whereupon", "wherever",
                "whether", "which", "while", "whither", "who", "who's",
                "whoever", "whole", "whom", "whose", "why", "will", "willing",
                "wish", "with", "within", "without", "won't", "wonder",
                "would", "would", "wouldn't", "x", "y", "yes", "yet", "you",
                "you'd", "you'll", "you're", "you've", "your", "yours",
                "yourself", "yourselves", "z", "zero","quot"]
    
    porter = PorterStemmer()
    Java_keyWords=[porter.stem(each.strip().lower()) for each in JavaKeywords]
    natural_stop_words=[porter.stem(each.strip().lower()) for each in stop_words]
    
    processed_doc=[]
    if mode=="code":
        splitted_doc=[porter.stem(term.lower()) for term in code_splitter(doc)]
        processed_doc=[term for term in splitted_doc if not(term in Java_keyWords or
                                                            term in natural_stop_words or len(term)<2)]
    elif mode=="text":
        splitted_doc=[porter.stem(term.lower()) for term in split_natural_lang(doc)]
        processed_doc=[term for term in splitted_doc if not(term in natural_stop_words or len(term)<2)]
    return processed_doc


def classNames_methodNames(node):
    result=''
    if isinstance(node,javalang.tree.MethodDeclaration) or isinstance(node,javalang.tree.ClassDeclaration):
        return node.name.lower()+' '
    if not (isinstance(node,javalang.tree.PackageDeclaration) or
        isinstance(node,javalang.tree.FormalParameter) or
       isinstance(node,javalang.tree.Import)):
        if node:
            if isinstance(node, javalang.ast.Node):
                for childNode in node.children:
                    result+=classNames_methodNames(childNode)
    return result
    
def traverse_node(node,i=0):
    i+=1
    result=''
    if not(isinstance(node,javalang.tree.PackageDeclaration)
            or isinstance(node,javalang.tree.FormalParameter)            
            or isinstance(node,javalang.tree.Import)
            or isinstance(node,javalang.tree.CompilationUnit)):
        if node:
            if (isinstance(node,int) or isinstance(node,str) or isinstance(node,float)) and i==2:
                result+=node+' '
            if isinstance(node, javalang.ast.Node):
                for childNode in node.children:
                    result+=traverse_node(childNode,i)
    return result

def code_parser(code):
    try:
        tree = javalang.parse.parse(code)
        return ''.join([traverse_node(node) for path, node in tree]) + ' ' + ''.join([classNames_methodNames(node)
                                                                                      for path, node in tree])
    except Exception as e: 
        # print(e)
        return ''

    
def async_sourcecode_procesing(source_code):
#     print(source_code)
#     try:
    sc_code = zlib.decompress(bytes.fromhex(source_code)).decode()
#     except:
#         print(source_code)
# #     print(sc_code)
    processed_code=general_preprocessor(code_parser(sc_code),'code')
#     print(processed_code)
    return Row('file_content_processed', 'size')(zlib.compress("-|-".join(processed_code).encode("utf-8")).hex(), len(processed_code))

def loadBugs2df(report):
#     print(report)
    return general_preprocessor(report,"text")


spark = SparkSession \
    .builder \
    .appName("BL") \
    .config("spark.driver.memory", "25g") \
    .config("spark.some.config.option", "") \
    .getOrCreate()


all_sc_df_spark_df = spark.read.option('header', True).option('inferSchema', True).option('delimiter', ',').csv("/home/varumuga/scratch/Thesis/Replications/Globug/globug_bl_dataset/combined_dataset/allSourceCodes.csv")

schema = StructType([
    StructField("file_content_processed", StringType(), False),
    StructField("size", IntegerType(), False)])

udf_br_processor = udf(async_sourcecode_procesing, schema) 
# udf_br_processor = udf(lambda x: async_sourcecode_procesing(x), StringType()) # if the function returns an int
sc_processed_df = all_sc_df_spark_df.withColumn("output", udf_br_processor(col("file_content"))) #"_3" being the column name of the column you want to consider
sc_processed_df = sc_processed_df.select("cid", "file_content", "output.file_content_processed", "output.size")
# print(len(sc_processed_df))
start = time.time()
sc_processed_df.write.mode("overwrite").parquet("/home/varumuga/scratch/Thesis/Replications/Globug/globug_bl_dataset/combined_dataset/allSourceCodesProcessed")
end = time.time()
print(end - start)