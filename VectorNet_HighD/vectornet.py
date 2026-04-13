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
# from HighD_datapre import data_pre


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
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
            
        假设你有一个批次的数据，其中每个序列的长度不同，但为了批量处理，
        你添加了填充使得它们具有相同的长度。valid_lens 会告诉模型每个序列实际的有效长度，
        从而确保在计算注意力权重时不会受到填充部分的影响。

        """
        batch_size = data.num_graphs     #RL 单次处理one track/ one graph     
        valid_lens = data.valid_len
        max_valid_len  = data.max_valid_len 
        
        id_embedding = data.identifier

        sub_graph_out = self.subgraph(data)   #（polylines_num, features_num)
        # #
        # x = sub_graph_out.view(batch_size, -1, self.subgraph.out_channels)      #这里好像不知道每个polyline所属的example吧？？？但一个example是一个Data？
        # global_graph_out = self.global_graph(x, valid_lens=valid_lens)  #(mini batch size, 9, 64)   9好像是子图(agent)的数量
        
        
        if self.training and self.with_aux:
            randoms = 1 + torch.rand((batch_size,), device=self.device) * (valid_lens - 2) + \
                      max_valid_len * torch.arange(batch_size, device=self.device)
            '''
            第一行随机选择出要mask的polyline，要mask真实存在的对象，所以是valid_lens
            第二行相当于在一个batch里生成一个global polyline id，采用的是对齐后的最大len( max_valid_len ) 
            tensor([   0,   13,   26,   39,   52,   65,   78,   91,  104,  117,  ...], device='cuda:0')
            所以每次相当于只mask一个polyline
            '''
            mask_polyline_indices = randoms.long()
            aux_gt = sub_graph_out[mask_polyline_indices]
            sub_graph_out[mask_polyline_indices] = 0.0

        # reconstruct the batch global interaction graph data
        x = torch.cat([sub_graph_out, id_embedding], dim=1).view(batch_size, -1, self.subgraph.out_channels + 2)  #第一维度1664代表polyline数量   （128，13，66）

        if self.training:
            # mask out the features for a random subset of polyline nodes
            # for one batch, we mask the same polyline features

            # global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            global_graph_out = self.global_graph(x, valid_lens=valid_lens)     #torch.Size([128, 13, 64])

            if self.with_aux:
                aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices]   #torch.Size([128*13, 64])
                aux_out = self.aux_mlp(aux_in)  #torch.Size([128, 64])

                return global_graph_out, aux_out, aux_gt

            return global_graph_out, None, None

        else:
            # global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            global_graph_out = self.global_graph(x, valid_lens=valid_lens)
            
        return global_graph_out, None, None


if __name__ == "__main__":
    device = torch.device('cuda:0')
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    pred_len = 30
    
    dataset = data_pre()  
    
    # data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
    '''
    DataLoader将dataset从examples维度进行分割，分割后一个iteration输入的examples number为batch_size；
    shuffle是先将expamples打乱，再进行batch包装；
    采用DataLoader前后数据形状的维度数没有变化，只是变化形状中代表examples num这一维度的数值。
    '''
    
    model = VectorNetBackbone(dataset.num_features, with_aux=False, device=device).to(device)
    
    model.train()
    out, aux_out, mask_feat_gt = model(dataset.to(device))
    print("Training Pass")
    
    # model.train()
    # for i, data in enumerate(tqdm(data_iter, total=len(data_iter), bar_format="{l_bar}{r_bar}")):
    #     out, aux_out, mask_feat_gt = model(data.to(device))
    #     print("Training Pass")

    # model.eval()
    # for i, data in enumerate(tqdm(data_iter, total=len(data_iter), bar_format="{l_bar}{r_bar}")):
    #     out, _, _ = model(data.to(device))
    #     print("Evaluation Pass")
