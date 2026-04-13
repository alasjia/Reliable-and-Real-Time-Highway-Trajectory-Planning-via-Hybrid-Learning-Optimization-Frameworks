# source: https://github.com/xk-huang/yet-another-vectornet
import math

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalGraph(nn.Module):
    """
    Global graph that compute the global information
    """
    def __init__(self, in_channels,
                 global_graph_width,
                 num_global_layers=1,
                 need_scale=False):
        super(GlobalGraph, self).__init__()
        self.in_channels = in_channels
        self.global_graph_width = global_graph_width

        self.layers = nn.Sequential()

        in_channels = self.in_channels
        for i in range(num_global_layers):
            self.layers.add_module(
                f'glp_{i}', SelfAttentionFCLayer(in_channels,
                                                 self.global_graph_width,
                                                 need_scale)
            )

            in_channels = self.global_graph_width

    def forward(self, x, **kwargs):
        for name, layer in self.layers.named_modules():
            if isinstance(layer, SelfAttentionFCLayer):
                x = layer(x, **kwargs)
        return x


class SelfAttentionFCLayer(nn.Module):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self, in_channels, global_graph_width, need_scale=False):
        super(SelfAttentionFCLayer, self).__init__()
        self.in_channels = in_channels
        self.graph_width = global_graph_width
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)
        self.scale_factor_d = 1 + \
            int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, valid_lens):
        '''
        valid_lens:
        '''
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.graph_width)
        attention_weights = self.masked_softmax(scores, valid_lens)
        x = torch.bmm(attention_weights, value)  #批量矩阵乘法（batch matrix multiplication）
        return x
    '''
    torch.bmm: 批量矩阵乘法（Batch Matrix Multiplication）的操作：
    输入的两个张量形状分别为(batch_size, n, m)和(batch_size, m, p)
    torch.bmm函数将执行批量矩阵相乘的操作，计算每个批次中对应位置的两个矩阵的乘积
    返回一个具有形状(batch_size, n, p)的新张量
    '''

    @staticmethod
    def masked_softmax(X, valid_lens):
        """
        masked softmax for attention scores
        args:
            X: 3-D tensor, valid_len: 1-D or 2-D tensor
        """
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.shape[0] != shape[0]:
                valid_len = torch.repeat_interleave(valid_lens, repeats=shape[0], dim=0)
            else:
                valid_len = valid_lens.reshape(-1)

            # Fill masked elements with a large negative, whose exp is 0
            mask = torch.zeros_like(X, dtype=torch.bool)
            for batch_id, cnt in enumerate(valid_len):
                cnt = int(cnt.detach().cpu().numpy())
                mask[batch_id, :, cnt:] = True
                mask[batch_id, cnt:] = True
            X_masked = X.masked_fill(mask, -1e12)   #填充一个极大值
            return nn.functional.softmax(X_masked, dim=-1) * (1 - mask.float())   #mask部分变为0


if __name__ == "__main__":
    pass
