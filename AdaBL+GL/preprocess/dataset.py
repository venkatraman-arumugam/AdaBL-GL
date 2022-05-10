import numpy
import os
import pickle
import sqlite3
import logging

import torch
from torch.utils.data import Dataset
from preprocess.lex.doc_sim import BowSimilarity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

class CodeSearchDataset(Dataset):



    def create_bl_dataset(bl_dataset_df, word_sim, db_path, query_max_size, code_max_size, top_k, sampling_size, print_log=True):

        new_bl_dataset_df = bl_dataset_df[(bl_dataset_df.reports_processed_len <= query_max_size) & (bl_dataset_df.file_content_processed_len <= code_max_size)].loc[::]
        core_term_size = len(word_sim.core_terms) + 2
        print(f"Writing DB to {db_path}")
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE conf (query_max_size INT, code_max_size INT, core_term_size INT)''')
        cursor.execute('''CREATE TABLE samples (id INT PRIMARY KEY, pkl TEXT)''')
        cursor.execute('''INSERT INTO conf VALUES (?,?,?)''', [query_max_size, code_max_size, core_term_size])
        conn.commit()


        # doc_sim = BowSimilarity(new_bl_dataset_df.reports_processed.to_list())
        samples_buffer = []
        index = 0
        skipped = 0
        for _, bug_id in enumerate(new_bl_dataset_df.bug_id.unique()):
            try:
                if print_log and index % 50 == 0:
                    print(index,  '/', new_bl_dataset_df.bug_id.unique().shape)

                positive_data_points = new_bl_dataset_df[(new_bl_dataset_df.bug_id == bug_id) & (new_bl_dataset_df.match == 1)].loc[::]
                negative_data_points =  new_bl_dataset_df[(new_bl_dataset_df.bug_id == bug_id) & (new_bl_dataset_df.match == 0)].loc[::]
                # print(positive_data_points, bug_id)
                pos_item = (positive_data_points.reports_processed.iloc[0], positive_data_points.file_content_processed.iloc[0], positive_data_points.cid.iloc[0])
                pos_data = MatchingMatrix(pos_item[0], pos_item[1], pos_item[2], word_sim, query_max_size)

                if "test" in db_path:
                    negative_items = negative_data_points.sample(n = 50)
                else:
                    negative_items = negative_data_points.sample(n = 10)

                neg_data_list = [MatchingMatrix(br, sc, cid, word_sim, query_max_size) for br, sc, cid in zip(negative_items.reports_processed, negative_items.file_content_processed, negative_items.cid)]
                pkl = pickle.dumps(CodeSearchDataSample(bug_id, pos_data, neg_data_list))
                samples_buffer.append([index, pkl])
                if index > 0 and (index % 1000 == 0 or index + 1 == new_bl_dataset_df.bug_id.unique().shape[0]):
                    cursor.executemany('''INSERT INTO samples VALUES (?,?)''', samples_buffer)
                    conn.commit()
                    samples_buffer.clear()
                    print("Total skipped for ", skipped)
                index += 1
            except Exception:
                skipped += 1
                index += 1
                # if index > 0 and (index % 1000 == 0 or index + 1 == new_bl_dataset_df.bug_id.unique().shape[0]):
                #     cursor.executemany('''INSERT INTO samples VALUES (?,?)''', samples_buffer)
                #     conn.commit()
                #     samples_buffer.clear()
                # index += 1
        if samples_buffer:
            cursor.executemany('''INSERT INTO samples VALUES (?,?)''', samples_buffer)
            conn.commit()
            samples_buffer.clear()
        print("Total skipped for ", skipped)
        conn.close()

            

    @staticmethod
    def create_dataset(data, word_sim, db_path, query_max_size, code_max_size, top_k, sampling_size, print_log=True):

        data = [item for item in data if len(item[0]) <= query_max_size and len(item[1]) <= code_max_size]
        core_term_size = len(word_sim.core_terms) + 2

        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE conf (query_max_size INT, code_max_size INT, core_term_size INT)''')
        cursor.execute('''CREATE TABLE samples (id INT PRIMARY KEY, pkl TEXT)''')
        cursor.execute('''INSERT INTO conf VALUES (?,?,?)''', [query_max_size, code_max_size, core_term_size])
        conn.commit()

        documents = [item[0] for item in data]
        doc_sim = BowSimilarity(documents)
        samples_buffer = []
        for i in range(len(data)):
            if print_log and i % 100 == 0:
                print(i, '/', len(data))
            item = data[i]
            pos_data = MatchingMatrix(item[0], item[1], item[2], word_sim, query_max_size)
            neg_idx_list = doc_sim.negative_sampling(i, top_k, sampling_size)
            neg_data_list = [MatchingMatrix(item[0], data[idx][1], data[idx][2], word_sim, query_max_size) for idx in
                             neg_idx_list]
            pkl = pickle.dumps(CodeSearchDataSample(item[2], pos_data, neg_data_list))
            samples_buffer.append([i, pkl])
            if i > 0 and (i % 1000 == 0 or i + 1 == len(data)):
                cursor.executemany('''INSERT INTO samples VALUES (?,?)''', samples_buffer)
                conn.commit()
                samples_buffer.clear()
        conn.close()

    def __init__(self, db_path):
        logger.info(f"reading from db path {db_path}")
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''SELECT query_max_size, code_max_size, core_term_size FROM conf''')
        self.query_max_size, self.code_max_size, self.core_term_size = self.cursor.fetchone()
        self.cursor.execute('''SELECT count(*) FROM samples''')
        self.len = self.cursor.fetchone()[0]

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
            neg_samples = sample.neg_data_list
            pos_samples = [sample.pos_data]
            neg_matrix = numpy.asarray(
                [CodeSearchDataset.pad_matrix(numpy.transpose(neg.matrix), self.code_max_size, self.query_max_size) for neg
                in neg_samples])
            pos_matrix = numpy.asarray(
                [CodeSearchDataset.pad_matrix(numpy.transpose(pos.matrix), self.code_max_size, self.query_max_size) for pos
                in pos_samples])
            neg_lengths = numpy.asarray([len(neg.core_terms) for neg in neg_samples])
            pos_lengths = numpy.asarray([len(pos.core_terms) for pos in pos_samples])
            neg_core_terms = numpy.asarray(
                [CodeSearchDataset.pad_terms(neg.core_terms, self.code_max_size) for neg in neg_samples])
            pos_core_terms = numpy.asarray(
                [CodeSearchDataset.pad_terms(pos.core_terms, self.code_max_size) for pos in pos_samples])
            neg_ids = numpy.asarray([int(neg.code_id) for neg in neg_samples])
            return query_id, pos_matrix, pos_core_terms, pos_lengths, sample.pos_data.code_id, neg_matrix, neg_core_terms, neg_lengths, neg_ids
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
                [[CodeSearchDataset.pad_matrix(numpy.transpose(item.matrix), code_max_size, query_max_size)] for item in
                 items])
            lengths = [torch.LongTensor([len(item.core_terms)]).to(device) for item in items]
            core_terms = numpy.asarray(
                [[CodeSearchDataset.pad_terms(item.core_terms, code_max_size)] for item in items])
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
