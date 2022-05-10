from __future__ import absolute_import

import logging
import os
import time
import json

import numpy as np
import re
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from learning.model.rnn import RnnModel
from preprocess.dataset import CodeSearchDataset
from preprocess.dataset import MatchingMatrix
from preprocess.lex.token import Tokenizer
from preprocess.lex.word_sim import WordEmbeddings, WordSim

from sklearn.metrics import roc_auc_score, average_precision_score


import pandas as pd
from learning.model.gnn import RNNPlusGNNModel

from preprocess.bl_datasamples import BLDataBatch
from preprocess.graph_dataset import BLGraphDataset
from preprocess.lex import word_sim

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if batch:
        return torch.utils.data.dataloader.default_collate(batch)
    return None

def collate_wrapper(batch):
    return BLDataBatch(batch)

class CodeSearcher:
    def __init__(self, conf):
        self.conf = conf
        self.wkdir = self.conf['data']['wkdir']
        
        self.result_file_path = self.conf['data']['result']
        
        self.ranking_result_path = self.conf['data']['ranking_results']

        self.model_dir = self.conf['data']['modedir']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_type = self.conf['data']['model_type']
        
        if self.model_type == "base_model":
            train_data = CodeSearchDataset(os.path.join(self.wkdir, conf['data']['train_db_path']))
            self.model = RnnModel(int(conf['data']['query_max_len']), train_data.core_term_size, int(conf['model']['core_term_embedding_size']), int(conf['model']['lstm_hidden_size']), int(conf['model']['lstm_layers']), float(self.conf['train']['margin'])).to(self.device)
        else:
            self.word_emb = WordEmbeddings(pretrain=False, update=False, fasttext_corpus_path=self.conf['data']['word_emb_model_path'])
            train_data = BLGraphDataset(os.path.join(self.wkdir, conf['data']['train_db_path']), self.conf['data']['sc_datapath'], self.word_emb)
            self.model = RNNPlusGNNModel(int(conf['data']['query_max_len']), train_data.core_term_size, int(conf['model']['core_term_embedding_size']), int(conf['model']['lstm_hidden_size']), int(conf['model']['lstm_layers']), float(self.conf['train']['margin'])).to(self.device)
            
        self.batch_size = int(self.conf['train']['batch_size'])

    def save_model(self, epoch):
        model_dir = os.path.join(self.model_dir, 'models')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir, 'epoch%d.h5' % (epoch%3)))

    def load_model(self, epoch):
        model_path = os.path.join(self.model_dir, 'models/epoch%d.h5' % epoch)
        logger.info(f"Loading model from {model_path}")
        assert os.path.exists(model_path), 'Weights not found.'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def train(self):
        train_data = None
        if self.model_type == "base_model":
            train_data = CodeSearchDataset(os.path.join(self.wkdir, self.conf['data']['train_db_path']))
        else:
            train_data = BLGraphDataset(os.path.join(self.wkdir, self.conf['data']['train_db_path']), self.conf['data']['sc_datapath'], self.word_emb)
        # valid_data = CodeSearchDataset(os.path.join(self.wkdir, self.conf['data']['valid_db_path']))
        # test_data = CodeSearchDataset(os.path.join(self.wkdir, self.conf['data']['test_db_path']))
        train_size = len(train_data)
        if torch.cuda.device_count() > 1:
            logger.info(f"let's use {torch.cuda.device_count()} GPUs")

        save_round = int(self.conf['train']['save_round'])
        nb_epoch = int(self.conf['train']['nb_epoch'])
        batch_size = self.batch_size
        collate_wrapper_fn = collate_fn if self.mode_type == "base_model" else collate_wrapper
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper_fn)
        optimizer = optim.Adam(self.model.parameters(), lr=float(self.conf['train']['lr']))
        save_loss = float("inf")
        cool_off = 0
        for epoch in range(nb_epoch):
            self.model.train()
            epoch_loss = 0
            start = time.time()
            for _, dataitem in tqdm(dataloader):
                if self.mode_type == "base_model":
                    pos_matrix, pos_core_terms, pos_length, neg_matrix, neg_core_terms, neg_length, neg_ids = dataitem
                else:
                    pos_matrix, pos_core_terms, pos_length, neg_matrix, neg_core_terms, neg_length, neg_ids = \
                        dataitem.pos_matrix, \
                        dataitem.pos_core_terms, \
                        dataitem.pos_length, \
                        dataitem.neg_matrix, \
                        dataitem.neg_core_terms, \
                        dataitem.neg_length, \
                        dataitem.neg_ids 
                    pos_dfg = dataitem.pos_data.x
                    neg_dfg = dataitem.neg_data.x
                pos_length = [self.gVar(x) for x in pos_length]
                neg_length = [self.gVar(x) for x in neg_length]
                if self.mode_type == "base_model":
                    loss = self.model.loss(self.gVar(pos_matrix), self.gVar(pos_core_terms), pos_length,
                                  self.gVar(neg_matrix), self.gVar(neg_core_terms), neg_length)
                else:
                    loss = self.model.loss(self.gVar(pos_matrix), self.gVar(pos_core_terms), pos_length,
                                  self.gVar(neg_matrix), self.gVar(neg_core_terms), neg_length, pos_dfg, neg_dfg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            end = time.time() - start
            logger.info(f"epoch { epoch } Total Train Run Time  {end}")
            current_epoch_loss = epoch_loss / (train_size/batch_size)
            logger.info(f"epoch {epoch} : Loss = { epoch_loss / (train_size/batch_size) }")
            if current_epoch_loss < save_loss:
                self.save_model(epoch)
            else:
                cool_off += 1
                if cool_off > 10:
                    break
            # print('Validation...')
            # self.eval(valid_data)
            # start = time.time()
            # logger.info('Test...')
            # self.eval(test_data)
            # end = time.time() - start
            # logger.info(f"epoch {epoch} Total Test Run Time  {end}")

    def eval2(self):
        data = Tokenizer().parse(os.path.join(self.wkdir, self.conf['data']['test_nl_path']), os.path.join(self.wkdir, self.conf['data']['test_code_path']))
        fasttext_corpus_path = os.path.join(self.wkdir, re.sub(r'\.db$', '.txt', self.conf['data']['test_db_path']))
        core_term_path = os.path.join(self.wkdir, 'conf/core_terms.txt')
        word_sim = WordSim(core_term_path, pretrain=(self.conf['model']['pretrained_wordvec'] == str(True)), update=False, fasttext_corpus_path=fasttext_corpus_path)
        CodeSearchDataset.eval(self.model, data, word_sim, int(self.conf['data']['query_max_len']), int(self.conf['data']['code_max_len']), self.device)

    def test(self):
        self.model.eval()
        batch_size = 1
        if self.model_type == "base_model":
            test_data = CodeSearchDataset(os.path.join(self.wkdir, self.conf['data']['test_db_path']))
        else:
            test_data = BLGraphDataset(os.path.join(self.wkdir, self.conf['data']['test_db_path']), self.conf['data']['sc_datapath'], self.word_sim)

        collate_wrapper_fn = collate_fn if self.mode_type == "base_model" else collate_wrapper

        dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_wrapper_fn)

        all_results=[]
        mrr_score = []
        map_scores = []
        top_k_counter = [0] * 20
        total_bug_report = 0
        all_rank_result = []
        for  data_item in dataloader:
            if data_item:
                if self.model_type == "base_model":
                    q_id, pos_matrix, pos_core_terms, pos_length, pos_id, neg_matrix, neg_core_terms, neg_length, neg_ids = data_item
                else:
                    q_id, pos_matrix, pos_core_terms, pos_length, pos_id, neg_matrix, neg_core_terms, neg_length, neg_ids = \
                        data_item.query_id, \
                        data_item.pos_matrix, \
                        data_item.pos_core_terms, \
                        data_item.pos_length, \
                        data_item.neg_matrix, \
                        data_item.neg_core_terms, \
                        data_item.neg_length, \
                        data_item.neg_ids 
                    pos_dfg = data_item.pos_data.x
                    neg_dfg = data_item.neg_data.x
                cid = [pos_id.tolist()[0]] + neg_ids.tolist()[0]
                
                pos_length = [self.gVar(x) for x in pos_length]
                neg_length = [self.gVar(x) for x in neg_length]

                if self.model_type == "base_model":
                    pos_score = self.model(self.gVar(pos_matrix), pos_length, self.gVar(pos_core_terms)).data.cpu().numpy()
                    neg_score = self.model(self.gVar(neg_matrix), neg_length, self.gVar(neg_core_terms)).data.cpu().numpy()
                else:
                    pos_score = self.model(self.gVar(pos_matrix), \
                    pos_length, self.gVar(pos_core_terms), pos_dfg).data.cpu().numpy()
                    neg_score = self.model(self.gVar(neg_matrix), \
                    neg_length, self.gVar(neg_core_terms), neg_dfg).data.cpu().numpy()
                true_negative = [0] * len(neg_score)

                true_label = np.array([1] + true_negative)

                predicted = np.concatenate((pos_score,neg_score))  
        
                predicted = predicted.ravel()

                map_scores.append(average_precision_score(true_label, predicted))
                result_df = pd.DataFrame({"cid": cid, "match": true_label, "predicted": predicted})
                result_df["bug_id"] = q_id.tolist()[0]
                all_rank_result.append(result_df.copy())
                sorted_prediction_rank = np.argsort(-predicted)
                sorted_prediction_value = np.array([true_label[item] for item in sorted_prediction_rank])
                lowest_retrieval_rank = (sorted_prediction_value == 0).argsort(axis=0)
                mrr_score.append(1.0 / (lowest_retrieval_rank[0] + 1))
                
                sorted_label_rank = np.argsort(-true_label)
                for position_k in range(0, 20):
                    common_item = np.intersect1d(sorted_prediction_rank[:(position_k + 1)],
                                                sorted_label_rank[:(position_k + 1)])
                    if len(common_item) > 0:
                        top_k_counter[position_k] += 1
                
                total_bug_report += 1

        acc_dict = {}
        for i, counter in enumerate(top_k_counter):
            acc = counter / total_bug_report
            acc_dict[i + 1] = round(acc, 3)

        avg_map, avg_mrr, top_n = np.mean(map_scores),np.mean(mrr_score), acc_dict
        logger.info(f"Total Bug report processed {total_bug_report}")
        metrics = {
            "avg_map": avg_map,
            "avg_mrr": avg_mrr,
            "top_n": top_n,
            "total_bug_reports": total_bug_report
        }

        with open(self.result_file_path, "w") as f:
            json.dump(metrics, f)
            
        all_rank_result = pd.concat(all_rank_result)
        
        all_rank_result.to_csv(self.ranking_result_path, index=False)

    def eval(self, test_data):
        self.model.eval()
        batch_size = self.batch_size
        dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        
        def top_k_acc(pos_score, neg_score, k):
            ranks = compute_rank(pos_score, neg_score)
            result = [1 for r in ranks if r <= k]
            count = sum(result)
            return count/len(ranks)

        def mrr(pos_score, neg_score):
            ranks = compute_rank(pos_score, neg_score)
            reciprocal = [1/r for r in ranks]
            return sum(reciprocal)/len(ranks)

        def compute_rank(pos_score, neg_score):
            ranks = [len(neg_score[0])+1]*len(pos_score)
            for i, pos_ in enumerate(pos_score):
                sort_neg_score = sorted(neg_score[i], reverse=True)
                for j, neg_ in enumerate(sort_neg_score):
                    if pos_ > neg_:
                        ranks[i] = j + 1
                        break
            return ranks

        top_k = 20
        accs = [[] for _ in range(top_k)]
        mrrs = []
        for q_id, pos_matrix, pos_core_terms, pos_length, neg_matrix, neg_core_terms, neg_length, neg_ids in dataloader:
            pos_length = [self.gVar(x) for x in pos_length]
            neg_length = [self.gVar(x) for x in neg_length]
            pos_score = self.model(self.gVar(pos_matrix), pos_length, self.gVar(pos_core_terms)).data.cpu().numpy()
            neg_score = self.model(self.gVar(neg_matrix), neg_length, self.gVar(neg_core_terms)).data.cpu().numpy()
            neg_score = np.split(neg_score, len(pos_score))
            for i in range(top_k):
                accs[i].append(top_k_acc(pos_score, neg_score, i+1))
            mrrs.append(mrr(pos_score, neg_score))
        for i in range(top_k):
            print('Hit@{}: {}'.format(i+1, np.mean(accs[i])))
        print('MRR: {}'.format(np.mean(mrrs)))

    def gVar(self, tensor):
        return tensor.to(self.device)

    def predict(self, output_file):
        tokenizer = Tokenizer()
        # {str: list} ->  {label: tokens}
        nl_dict = tokenizer.parse_nl(os.path.join(self.wkdir, self.conf['data']['test_nl_path']))
        code_dict = tokenizer.parse_code(os.path.join(self.wkdir, self.conf['data']['test_code_path']))
        fasttext_corpus_path = os.path.join(self.wkdir, re.sub(r'\.db$', '.txt', self.conf['data']['test_db_path']))
        core_term_path = os.path.join(self.wkdir, 'conf/core_terms.txt')
        word_sim = WordSim(core_term_path, pretrain=(self.conf['model']['pretrained_wordvec'] == str(True)), 
                           update=False, fasttext_corpus_path=fasttext_corpus_path)

        query_max_size = int(self.conf['data']['query_max_len'])
        code_max_size = int(self.conf['data']['code_max_len'])
        device = self.device
        self.model.eval()
        
        nl_data = [(nid, tokens[: query_max_size]) for (nid, tokens) in nl_dict.items()]
        code_data = [(cid, tokens[: code_max_size]) for (cid, tokens) in code_dict.items()]
        print(f'nl size: {len(nl_data)}, code size: {len(code_data)}')

        fout = open(output_file, 'w')
        for nid, nl_token in nl_data:
            items = []
            for cid, code_token in code_data:
                items.append(MatchingMatrix(nl_token, code_token, cid,
                                            word_sim, query_max_size))
            matrices = np.asarray([
                            [CodeSearchDataset.pad_matrix(np.transpose(item.matrix),
                                                          code_max_size,
                                                          query_max_size)]
                            for item in items])
            lengths = [torch.LongTensor([len(item.core_terms)]).to(device) for item in items]
            core_terms = np.asarray([
                                [CodeSearchDataset.pad_terms(item.core_terms, code_max_size)]
                            for item in items])
            output = self.model(torch.from_numpy(matrices).to(device),
                                lengths,
                                torch.from_numpy(core_terms).to(device))
            scores = output.cpu().detach().numpy().squeeze()
            print(f'nl {nid} writing...')
            for i, s in enumerate(scores):
                fout.write(f'{nid} {i + 1} {s}\n')
        fout.close()

