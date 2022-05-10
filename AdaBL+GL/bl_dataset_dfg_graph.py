from tree_sitter import Language, Parser
from parser.parser_utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
import pandas as pd
from parser.DFG import DFG_java
import zlib
import hashlib

from typing import List
from dataclasses import dataclass
import time
import os
import json

import modin.pandas as pd
import ray

ray.init(_plasma_directory="/tmp") 


node_and_edges_path = "graphs"


if not os.path.isdir(node_and_edges_path):
    os.makedirs(node_and_edges_path)

def source_code_decompress(compressed_sc):
    sc_code = zlib.decompress(bytes.fromhex(compressed_sc)).decode()
    return sc_code

def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
        code_tokens=[]
    return code_tokens,dfg



def construct_dfg(uncompressed_code):
    dfg_function={
        'java':DFG_java
    }
    code = source_code_decompress(uncompressed_code)
    
    code_md_hash = hashlib.md5(code.encode('utf-8')).hexdigest()
    
    # print(code_md_hash)

    code_dfg_path = os.path.join(node_and_edges_path, f"{code_md_hash}.json")
    
    # print("Writing to Path ", code_dfg_path)
    
    if not os.path.isfile(code_dfg_path):
        parsers={}        
        for lang in dfg_function:
            LANGUAGE = Language('parser/my-languages.so', lang)
            parser = Parser()
            parser.set_language(LANGUAGE) 
            parser = [parser,dfg_function[lang]]    
            parsers[lang]= parser

        code_tokens,dfg = extract_dataflow(code, parsers["java"], "java")
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)  
        # dfg = dfg[:dfg_length]

        nodes = []

        edges = [[], []]
        for idx, items in enumerate(dfg):
            nodes.append(items[0])

            comes_from_nodes = items[4]

            for comes_from_node in comes_from_nodes:
                edges[0].append(comes_from_node)
                edges[1].append(idx)


        nodes_and_edges = {}
        nodes_and_edges["nodes"] = nodes
        nodes_and_edges["edges"] = edges

        # node_and_edges_folder_path = os.path.join(node_and_edges_path, project, str(bug_id))

        with open(code_dfg_path, "w") as f:
            json.dump(nodes_and_edges, f)
    
    return ""



allSC_df = pd.read_csv("/home/varumuga/scratch/Thesis/research/AdaCS/bl_dataset/combined/allSourceCodes.csv")


start = time.time()
allSC_df_ = allSC_df.file_content.apply(construct_dfg)
end = time.time() - start

print(allSC_df_.unique().shape)
print(end)