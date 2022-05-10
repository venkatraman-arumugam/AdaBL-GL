from dataclasses import dataclass
from typing import List
from torch_geometric.data import Data, Batch
import torch



@dataclass
class BLDataSample:
    query_id: int
    pos_matrix: torch.Tensor
    pos_core_terms : torch.Tensor
    pos_lengths : torch.Tensor
    neg_matrix : torch.Tensor
    neg_core_terms : torch.Tensor
    neg_lengths : torch.Tensor
    neg_ids : torch.Tensor
    pos_data: Data
    neg_data: Data

class BLDataBatch:
    def __init__(self, BLDs: List[BLDataSample]):
        
        self.query_id = []
        self.pos_matrix = []
        self.pos_core_terms = []
        self.pos_lengths = []

        self.neg_matrix = []
        self.neg_core_terms = []
        self.neg_lengths = []

        self.neg_ids = []

        self.pos_data = []
        self.neg_data = []

        for bld in BLDs:
            self.query_id.append(bld.query_id)

            self.pos_matrix.append(bld.pos_matrix)
            self.pos_core_terms.append(bld.pos_core_terms)
            self.pos_lengths.append(bld.pos_lengths)

            self.neg_matrix.append(bld.neg_matrix)
            self.neg_core_terms.append(bld.neg_core_terms)
            self.neg_lengths.append(bld.neg_lengths)

            self.neg_ids.append(bld.neg_ids)

            self.pos_data.append(bld.pos_data)
            self.neg_data.append(bld.neg_data)


        self.query_id = torch.tensor(self.query_id, dtype=torch.int)
        self.pos_matrix = torch.concat(self.pos_matrix)
        self.pos_core_terms = torch.concat(self.pos_core_terms)
        self.pos_lengths = torch.concat(self.pos_lengths)

        self.neg_matrix = torch.concat(self.neg_matrix)
        self.neg_core_terms = torch.concat(self.neg_core_terms)
        self.neg_lengths = torch.concat(self.neg_lengths)

        self.neg_ids = torch.concat(self.neg_ids)

        self.pos_data = Batch.from_data_list(self.pos_data[0])
        self.neg_data = Batch.from_data_list(self.neg_data[0])
       
        self.sz = len(BLDs)

    def __len__(self):
        return self.sz

    def pin_memory(self):
        self.pos_data = self.pos_data.pin_memory()
        self.neg_data = self.neg_data.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.query_id = self.query_id.to(device)
        self.pos_matrix = self.pos_matrix.to(device)
        self.pos_core_terms = self.pos_core_terms.to(device)
        self.pos_lengths = self.pos_lengths.to(device)

        self.neg_matrix = self.neg_matrix.to(device)
        self.neg_core_terms = self.neg_core_terms.to(device)
        self.neg_lengths = self.neg_lengths.to(device)

        self.neg_ids = self.neg_id.to(device)

        self.pos_data = self.pos_data.to(device)
        self.neg_data = self.neg_data.to(device)