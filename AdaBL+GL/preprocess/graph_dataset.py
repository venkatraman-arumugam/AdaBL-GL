import numpy
import os
import pickle
import sqlite3
import logging
import pandas as pd

import torch
from torch.utils.data import Dataset
from preprocess.lex.doc_sim import BowSimilarity


import zlib
from parser.DFG import DFG_java
from parser.parser_utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)


from preprocess.bl_datasamples import BLDataSample

from tree_sitter import Language, Parser
import torch
from torch_geometric.data import Data, Batch

import time
import numpy as np



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

dfg_function={
    'java':DFG_java
}

dfg_length = 1024

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

def source_code_decompress(compressed_sc):
    sc_code = zlib.decompress(bytes.fromhex(compressed_sc)).decode()
    return sc_code

#remove comments, tokenize code and extract dataflow                                        
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
    return code_tokens,dfg


def construct_dfg(code):
    code_tokens,dfg = extract_dataflow(code, parsers["java"], "java")
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)  
    dfg = dfg[:dfg_length]

    nodes = []

    edges = [[], []]
    for idx, items in enumerate(dfg):
        nodes.append(items[0])
        
        comes_from_nodes = items[4]
        
        for comes_from_node in comes_from_nodes:
            edges[0].append(comes_from_node)
            edges[1].append(idx)
    
    return nodes, edges


class BLGraphDataset(Dataset):

    def __init__(self, db_path, dataset_path, word_sim):
        logger.info(f"reading from db path in BLGraphDataset {db_path}")
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''SELECT query_max_size, code_max_size, core_term_size FROM conf''')
        self.query_max_size, self.code_max_size, self.core_term_size = self.cursor.fetchone()
        self.cursor.execute('''SELECT count(*) FROM samples''')
        self.data_set = pd.read_csv(dataset_path)
        self.len = self.cursor.fetchone()[0]
        self.word_sim = word_sim
    def __del__(self):
        self.conn.close()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        self.cursor.execute('''SELECT pkl FROM samples WHERE id = ?''', [idx])
        result = self.cursor.fetchone()
        if result:
            sample = pickle.loads(result[0])
            query_id = sample.id
            neg_samples = sample.neg_data_list[:2]
            pos_samples = [sample.pos_data]
            neg_matrix = numpy.asarray(
                [BLGraphDataset.pad_matrix(numpy.transpose(neg.matrix), self.code_max_size, self.query_max_size) for neg
                in neg_samples])
            pos_matrix = numpy.asarray(
                [BLGraphDataset.pad_matrix(numpy.transpose(pos.matrix), self.code_max_size, self.query_max_size) for pos
                in pos_samples])
            neg_lengths = numpy.asarray([len(neg.core_terms) for neg in neg_samples])
            pos_lengths = numpy.asarray([len(pos.core_terms) for pos in pos_samples])
            neg_core_terms = numpy.asarray(
                [BLGraphDataset.pad_terms(neg.core_terms, self.code_max_size) for neg in neg_samples])
            pos_core_terms = numpy.asarray(
                [BLGraphDataset.pad_terms(pos.core_terms, self.code_max_size) for pos in pos_samples])
            neg_ids = numpy.asarray([int(neg.code_id) for neg in neg_samples])
            
            
            pos_data_mm = pos_samples[0]
            bug_id = query_id
            code_id = pos_data_mm.code_id

            pos_data_point = self.data_set[(self.data_set.bug_id == bug_id) & (self.data_set.cid == code_id) & (self.data_set.match == 1)]
            project_name = None
            if pos_data_point.project_name_x.unique().shape[0] > 1:
                neg_data_points = self.data_set[(self.data_set.bug_id == bug_id) & (self.data_set.cid.isin(code_id)) & (self.data_set.match == 0)]
                project_name = neg_data_points.project_name_x.value_counts().idxmax()
            else:
                project_name = pos_data_point.project_name_x.iloc[0]
            
            positive_code_id = [code_id]
            code_id_list = [positive_code_id, neg_ids]
            
            nodes_and_edges = []
            # print(project_name, bug_id, )
            # print(",".join(neg_ids))
            start = time.time()
            uncompressed_sc_content_dict = dict(self.data_set[(self.data_set.project_name_x == project_name) \
                & (self.data_set.bug_id == bug_id)][["cid", "file_content"]].values)
            # print(uncompressed_sc_content_dict)
            nodes_to_pad = 0
            for code_ids in code_id_list:
                # node_vectors = []
                nodes_idxs = []
                edges_list = []
                datas = []
                # results = []
                for code_id in code_ids:
                    uncompressed_sc_content = uncompressed_sc_content_dict[code_id]
                    sc_content = source_code_decompress(uncompressed_sc_content)
                    nodes, edges = construct_dfg(sc_content)
                    if not nodes:
                        print(code_id)
                    # nodes_vector = [self.word_sim[token] for token in nodes]
                    nodes_idx = [self.word_sim.word_to_index(token) for token in nodes]
                    data = Data(x=torch.tensor(nodes_idx), edge_index=torch.tensor(edges), num_nodes=len(nodes_idx))
                    datas.append(data)
                    # if not nodes:
                    #     print(data)
                    nodes_to_pad = max(nodes_to_pad, len(nodes_idx))
                    # node_vectors.append(nodes_vector)
                    # nodes_idxs.append(nodes_idx)
                    # edges_list.append(edges)
                # nodes_and_edges.append((nodes_idxs, edges_list))
                nodes_and_edges.append(datas)

            a = self.word_sim.word_to_index("pad")

            end = time.time() - start
            print("Total time to vectorize and construct dfg", end)
            # (pos_node_idx, pos_edges), (neg_node_idx, neg_edges) = nodes_and_edges
            pos_data, neg_data = nodes_and_edges
            # nodes_to_pad=50
            pos_matrix, pos_core_terms, pos_lengths, neg_matrix, neg_core_terms, neg_lengths, neg_ids = \
                [torch.from_numpy(i) for i in [pos_matrix, pos_core_terms, pos_lengths, neg_matrix, neg_core_terms, neg_lengths, neg_ids]]
            
           
            return BLDataSample(
                query_id=query_id.item(),
                pos_matrix=pos_matrix,
                pos_core_terms=pos_core_terms,
                pos_lengths=pos_lengths,
                neg_matrix=neg_matrix,
                neg_core_terms=neg_core_terms,
                neg_lengths=neg_lengths,
                neg_ids=neg_ids,
                pos_data=pos_data,
                neg_data=neg_data
                # pos_node_idx=pos_node_idx,
                # pos_edges=pos_edges,
                # neg_node_idx=neg_node_idx,
                # neg_edges=neg_edges
                # negatives_node_vectors_and_edges=negatives_node_vectors_and_edges
            )
        else:
            return None

    @staticmethod
    def eval(model, data, word_sim, query_max_size, code_max_size, device):
        model.eval()
        data = [item for item in data if len(item[0]) <= query_max_size and len(item[1]) <= code_max_size]
        mrr = 0
        hit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(data)):
            items = []
            for j in range(len(data)):
                items.append(MatchingMatrix(data[i][0], data[j][1], data[i][2], word_sim, query_max_size))
            matrices = numpy.asarray(
                [[BLGraphDataset.pad_matrix(numpy.transpose(item.matrix), code_max_size, query_max_size)] for item in
                 items])
            lengths = [torch.LongTensor([len(item.core_terms)]).to(device) for item in items]
            core_terms = numpy.asarray(
                [[BLGraphDataset.pad_terms(item.core_terms, code_max_size)] for item in items])
            scores = model(torch.from_numpy(matrices).to(device), lengths,
                                  torch.from_numpy(core_terms).to(device)).data.cpu().numpy()
            l = []
            for j in range(len(data)):
                if scores[j] >= scores[i]:
                    l.append((int(data[j][2]), float(scores[j])))
            mrr += 1.0 / len(l)
            for k in range(len(hit)):
                if len(l) <= k + 1:
                    hit[k] += 1
            print(
                '#%d:' % int(data[i][2]), 'rank=%d' % len(l), 'MRR=%.4f' % (mrr / (i + 1)),
                ', '.join([('Hit@%d=%.4f' % (k + 1, (h / (i + 1)))) for k, h in enumerate(hit)])
                  )

    def get_sample(self, idx):
        self.cursor.execute('''SELECT pkl FROM samples WHERE id = ?''', [idx])
        sample = pickle.loads(self.cursor.fetchone()[0])
        return sample

    @staticmethod
    def pad_matrix(matrix, code_max_size, query_max_size):
        padded = numpy.zeros([code_max_size, query_max_size * 2])
        slen = len(matrix)
        assert slen <= code_max_size
        padded[:slen, :] = matrix
        return padded

    @staticmethod
    def pad_terms(terms, code_max_size):
        seq = [0] * code_max_size
        tlen = len(terms)
        assert tlen <= code_max_size
        seq[:tlen] = terms
        return seq


class CodeSearchDataSample:

    def __init__(self, id, pos_data, neg_data_list):
        self.id = id
        self.pos_data = pos_data
        self.neg_data_list = neg_data_list


class MatchingMatrix:

    def __init__(self, document_1, document_2, code_id, word_sim, query_max_size):
        self.code_id = code_id
        self.matrix = self.__matrix(document_1, document_2, word_sim, query_max_size)
        self.core_terms = self.__core_terms(document_2, word_sim)

    @staticmethod
    def __matrix(document_1, document_2, word_sim, query_max_size):
        ret = numpy.zeros([query_max_size * 2, len(document_2)])
        for i in range(len(document_1)):
            for j in range(len(document_2)):
                ret[i * 2][j] = word_sim.sim(document_1[i], document_2[j])
                ret[i * 2 + 1][j] = word_sim.idf(document_1[i])
        return ret

    @staticmethod
    def __core_terms(document, word_sim):
        return [(word_sim.core_term_dict[word] if word in word_sim.core_terms else 1) for word in document]
