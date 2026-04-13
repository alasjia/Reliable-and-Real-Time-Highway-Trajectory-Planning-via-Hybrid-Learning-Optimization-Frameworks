import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Batch, Data

from global_graph import GlobalGraph
from subgraph import SubGraph
from basic_module import MLP


class VectorNetBackbone(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 in_channels,
                 device,
                 with_aux,
                 num_subgraph_layres,
                 num_global_graph_layer,
                 subgraph_width,
                 global_graph_width,
                 aux_mlp_width = 64
                 ):
        super(VectorNetBackbone, self).__init__()
        # some params
        self.num_subgraph_layres = num_subgraph_layres
        self.global_graph_width = global_graph_width

        self.device = device

        # subgraph feature extractor
        self.subgraph = SubGraph(in_channels, num_subgraph_layres, subgraph_width)

        # global graph
        self.global_graph = GlobalGraph(self.subgraph.out_channels + 2,
                                        self.global_graph_width,
                                        num_global_layers=num_global_graph_layer)

        # auxiliary recoverey mlp
        self.with_aux = with_aux
        if self.with_aux:
            # self.aux_mlp = nn.Sequential(
            #     nn.Linear(self.global_graph_width, aux_mlp_width),
            #     nn.LayerNorm(aux_mlp_width),
            #     nn.ReLU(),
            #     nn.Linear(aux_mlp_width, self.subgraph.out_channels)
            # )
            self.aux_mlp = nn.Sequential(
                MLP(self.global_graph_width, aux_mlp_width, aux_mlp_width),
                nn.Linear(aux_mlp_width, self.subgraph.out_channels)
            )

    def forward(self, data):
        batch_size = data.num_graphs
        valid_lens = data.valid_len
        max_valid_len = data.max_valid_len

        id_embedding = data.identifier

        # ---- FIX v2: rebuild batch-unique cluster ids (robust to padding) ----
        mv = int(max_valid_len[0].item())  # e.g., 13

        # IMPORTANT: use data.cluster (not data.x) because padding x is all zeros
        # original polyline ids are 1..mv, so convert to 0..mv-1
        local_poly = (data.cluster.long() - 1) % mv          # [num_nodes], in [0..12]
        data.cluster = local_poly + data.batch.long() * mv   # make unique per-graph inside the batch

        sub_graph_out = self.subgraph(data)  # should now be (batch_size*13, hidden)

        # 下面保持原来的逻辑（aux / torch.cat / global_graph）
        if self.training and self.with_aux:
            randoms = 1 + torch.rand((batch_size,), device=self.device) * (valid_lens - 2) + \
                    max_valid_len * torch.arange(batch_size, device=self.device)
            mask_polyline_indices = randoms.long()
            aux_gt = sub_graph_out[mask_polyline_indices]
            sub_graph_out[mask_polyline_indices] = 0.0


        assert sub_graph_out.size(0) == id_embedding.size(0), (sub_graph_out.size(), id_embedding.size())

        x = torch.cat([sub_graph_out, id_embedding], dim=1).view(
            batch_size, -1, self.subgraph.out_channels + 2
        )

        if self.training:
            global_graph_out = self.global_graph(x, valid_lens=valid_lens)
            if self.with_aux:
                aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices]
                aux_out = self.aux_mlp(aux_in)
                return global_graph_out, aux_out, aux_gt
            return global_graph_out, None, None
        else:
            global_graph_out = self.global_graph(x, valid_lens=valid_lens)
            return global_graph_out, None, None



