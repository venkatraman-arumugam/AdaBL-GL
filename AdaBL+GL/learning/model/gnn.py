from pydantic import NoneStrBytes
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv


from learning.model.rnn import LSTMModel

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, GINEConv, GATv2Conv, GatedGraphConv, GlobalAttention
import torch.nn.functional as F
from learning.model.embedding import CodeEmbeddingModule

from learning.model.rnn import RnnModel
import numpy as np

from preprocess.lex.word_sim import WordEmbeddings


class GraphConvEncoder(torch.nn.Module):
    """
    Kipf and Welling: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    (https://arxiv.org/pdf/1609.02907.pdf)
    """

    def __init__(self, emb_weights):
        super(GraphConvEncoder, self).__init__()
        self.__st_embedding = LSTMModel(emb_weights)
        size = 8
        self.input_GCL = GCNConv(size, size)

        self.input_GPL = TopKPooling(size,
                                     ratio=0.8)

        for i in range(3):
            setattr(self, f"hidden_GCL{i}",
                    GCNConv(size, size))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(size,
                            ratio=0.8))

        self.attpool = GlobalAttention(torch.nn.Linear(size, 64))

    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding, edge_index, _, batch, _, _ = self.input_GPL(node_embedding, edge_index, None,
                                                                    batch)
        # [n_XFG; XFG hidden dim]
        out = self.attpool(node_embedding, batch)
        for i in range(3):
            node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
            node_embedding, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, None, batch)
            out += self.attpool(node_embedding, batch)
        # [n_XFG; XFG hidden dim]
        return out


    
class RNNPlusGNNModel(nn.Module):

    def __init__(self, query_max_size, core_term_size, core_term_embedding_size, 
                 lstm_hidden_size=64, lstm_num_layers=2, margin=0.25, word_emb_weights=None):
        super(RnnModel, self).__init__()
        self.code_embedding = CodeEmbeddingModule(core_term_size, core_term_embedding_size)
        self.margin = margin
        self.rnn = nn.LSTM(
            input_size=query_max_size * 2 + core_term_embedding_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.05)
        self.reduce_features_rnn = nn.Linear(lstm_hidden_size, 8)
        self.gnn = GraphConvEncoder(word_emb_weights)
        self.fc = nn.Linear(lstm_hidden_size, 1)
        print('RNN model, count(parameters)=%d' % (sum([np.prod(list(p.size())) for p in self.parameters()])))

    def forward(self, matrix, length, core_terms, dfg):
        x, length, idx_unsort = self.code_embedding(matrix, length, core_terms)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, list(length), batch_first=True)
        # [batch_size * sample_size][time_steps][lstm_hidden_size * 2]
        x, _ = self.rnn(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # print(x)
        index = length.view(-1, 1, 1)
        index = index.expand(x.shape[0], 1, x.shape[2]) - 1
        x = torch.gather(x, 1, index)
        x = x[idx_unsort]
        rnn_x = self.reduce_features_rnn(x)
        gnn_x = self.gnn(dfg)
        combined = torch.cat((rnn_x, gnn_x), dim=1)
        out = self.fc(combined).squeeze(1)
         #.squeeze(1)
        return out

    def loss(self, pos_matrix, pos_core_terms, pos_lengths, neg_matrix, neg_core_terms, neg_lengths, pos_dfg, neg_dfg):
        pos_score = self.forward(pos_matrix, pos_lengths, pos_core_terms, pos_dfg)
        neg_score = self.forward(neg_matrix, neg_lengths, neg_core_terms, neg_dfg)
        k = int(neg_score.shape[0]/pos_score.shape[0])
        pos_score = pos_score.view(-1, 1).expand(-1, k).contiguous().view(-1, 1).squeeze()
        loss = (self.margin - pos_score + neg_score).clamp(min=1e-6).mean()
        return loss