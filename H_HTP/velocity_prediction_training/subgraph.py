# The imlementation of subgraph encoding of VectorNet
# Written by: Jianbang LIU @ RPAI, CUHK
# Created: 2021.10.02


'''
constructing the polyline subgraphs:
无论车道线还是vehicle trajectory均采用MLP进行node encoding
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# source: https://github.com/xk-huang/yet-another-vectornet
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from basic_module import MLP
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, max_pool, avg_pool
from torch_geometric.utils import add_self_loops, remove_self_loops

class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layres = num_subgraph_layres
        self.hidden_unit = hidden_unit
        self.out_channels = hidden_unit

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', MLP(in_channels, hidden_unit, hidden_unit))
            in_channels = hidden_unit * 2

        self.linear = nn.Linear(hidden_unit * 2, hidden_unit)

    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data (Data): [x, y, cluster, edge_index, valid_len]
            
            x: 可能是每个vector的特征（*10）  (122376, 10),  共有122376个vector
            y:每个vector的标签？    (4860)
            cluster: 为了聚类而存在的张量，value为每个vector所属的polyline的id  (122376)     最大值是20330，说明polyline的数量应该是20330；
                            在后续max_pool(sub_data.cluster, sub_data)中实现各个polyline的分别pooling。
            edge_index:  可能存储node与node之间的关系（边 edge)，（2， 1220282）,value应该是点的id，2表示两个端点
            valid_len:  ？   (81)
            
        针对每一个轨迹，每一个车道线分别建模（一个polyline），
        对于轨迹，一个vector为一个时间维度上的采样点
        对于车道线，一个vector为一个空间维度上的采样点（由于车道线不会随时间变化，所以不存在时间维度）
            
        """
        sub_data.x = sub_data.x.float()
        x = sub_data.x
        sub_data.cluster = sub_data.cluster.long()
        
        #重新分配cluster,将全局index转换为递增的局部index
        # 获取唯一元素和它们的逆索引  
        unique_x, inverse_indices = torch.unique(sub_data.cluster, return_inverse=True)  
        # inverse_indices 就是我们要找的 x_  
        local_cluster = inverse_indices     #
                
        sub_data.edge_index = sub_data.edge_index.long()

        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                sub_data.x = x
                agg_data = max_pool(local_cluster, sub_data)

                x = torch.cat([x, agg_data.x[local_cluster]], dim=-1)  #按列进行concat
                
                '''
                Q: agg_data.x的形状是torch.Size([20331, 64])，sub_data.cluster是torch.Size([122376])，如何进行索引？
                A: 依据sub_data.cluster中的第i个值ci，将agg_data.x中第ci（polyline id)放在输出tensor的第i行
                '''

        x = self.linear(x)  #将concat后的64*2的node特征重新压缩回64维
        sub_data.x = x
        out_data = max_pool(local_cluster, sub_data)   #将node为单位的维度转为以polyline为单位
        x = out_data.x

        assert x.shape[0] % int(sub_data.max_valid_len[0]) == 0 

        return F.normalize(x, p=2.0, dim=1)      # L2 normalization    (1047, 64)   每一个轨迹学出一组特征值(64)
    '''
    Q: is this normalization for the goal in the paper:   ?
    "to make the inpuit node features invariant to the locations of target agents,
    we normalize the coordinates of all vectors to be centered around the location of target agent at its last observed time step"
    A: no. chatgpt3.5 said" 这行代码是为了对输入的向量进行L2范数归一化处理，即将向量的每个元素除以其L2范数，使得向量的长度（模）为1。"
        so i think the normal in paper should be done by ourselves.
        "To avoid trivial solutions for L_node by lowering the magnitude of node features" ?
        
        
        <torch.Tensor object at 0x7f8166193c00>
        tensor([450, 450, 450,  ..., 440, 440, 440], device='cuda:0')
    '''
# %%


if __name__ == "__main__":
    # #test example
    # #the first demension of edge_indexs must be same.    edge_indexs: (2, the edge number)     x(the node number, the feature number)
    # data1 = Data(x=torch.tensor([[1.0, 20], [2.0, 25] , [3.0, 28]]), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), cluster = torch.tensor( [0, 0, 0] ))    #the first polyline
    # data2 = Data(x=torch.tensor([[3.0, 28], [4.0, 29], [5.0, 30.1]]), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]) , cluster = torch.tensor( [1, 1, 1] ))   #the second polyline 
    
    # batch = Batch.from_data_list([data1, data2])
    batch_data = data_pre()
    
    input_channels = 2  #x和y
    model = SubGraph(input_channels)
    
    result = model(batch_data)
    
    # data = Data(x=torch.tensor( [[1.0], [7.0]]  ), edge_index=torch.tensor( [[0, 1], [1, 0]] )  )
    # print(data)
    # layer = GraphLayerProp(1, 1, True)
    # for k, v in layer.state_dict().items():
    #     if k.endswith('weight'):
    #         v[:] = torch.tensor([[1.0]])
    #     elif k.endswith('bias'):
    #         v[:] = torch.tensor([1.0])
    # y = layer(data.x, data.edge_index)
    pass
