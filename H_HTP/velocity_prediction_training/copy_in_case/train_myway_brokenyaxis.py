import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR  
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset
from torch.utils.data import ConcatDataset, random_split
import time
import pandas as pd


from HighD_datapre import data_pre
from vectornet import VectorNetBackbone
from basic_module import MLP

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# from utils.VNloss import VectorLoss
from utils.visual_obs import visualize_graph
from config_nw import *



class VectorNet(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 in_channels,
                 horizon,
                 device ,
                 with_aux: bool,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width= 64,  
                 traj_pred_mlp_width=64    #64
                 ):
        super(VectorNet, self).__init__()
        # some params
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.out_channels = 1   #2: x
        self.horizon = horizon
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.k = 1

        self.device = device
        self.v_limit = MAX_SPEED   #m/s  
        self.delta_t = SPF  #sec

        # subgraph feature extractor
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layres=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            device=device
        )

        # pred mlp
        self.traj_pred_mlp = nn.Sequential(
            MLP(global_graph_width*13, traj_pred_mlp_width),
            nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels),
            nn.Sigmoid()
        )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        global_feat, aux_out, aux_gt = self.backbone(data)              # [batch_size, time_step_len, global_graph_width]   [128, 9, 64]
        # target_feat = global_feat[:, 0]    # [12, 64]

        global_feat = global_feat.view(global_feat.shape[0], -1)   #展平--> [128, 9*64]
        
        pred_ = self.traj_pred_mlp(global_feat) # [batch_size, 160]       #应该是 [128 9, 160]
        # 在sigmoid()输出[0,1]基础上扩大至[0, v_limit*delta t]
        # pred = pred_ *self.v_limit*self.delta_t  # output long. s
        pred = pred_ *self.v_limit  # output long. v
        
        return {"pred": pred, "aux_out": aux_out, "aux_gt":aux_gt}


def train_epoch(model, data_loader, device , criterion, optimizer, pred_len, output_features, scheduler):
    # training
    model.train()
    graph_num = len(data_loader.dataset)  # record the number of batches
    total_loss = 0
    batch_num = 0
    
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch)    
        y_hat = output['pred']    #(128, 30)     
        y_gt = batch.y.view(-1,  pred_len*output_features).to(torch.float32)    
        # y_hat = torch.cumsum(y_hat_, dim=1)
        # y_gt = torch.cumsum(y_gt_, dim=1)
        # Both use mse loss  
        loss = criterion(y_hat, y_gt)      #+ criterion(output['aux_out'], output['aux_gt'])   #nn.MSE default Mean(sum(l_gt - l-hat)**2)
        # loss = torch.sum((y_hat - y_gt)**2)   / (128*30) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().cpu().numpy().tolist()
        batch_num += 1 
    mean_loss  = total_loss / batch_num   # mean_loss is the squared error at each element (j-th time step in batch i)
    scheduler.step()
    return np.sqrt(mean_loss)    # Here, take the square root to convert MSE to RMSE, which is clearer

def eval_epoch(model, data_loader, device, criterion, optimizer, pred_len, output_features):
    model.eval()  # evaluation mode
    graph_num = len(data_loader.dataset)  # record the number of batches
    total_loss = 0
    total_loss_vx = 0
    total_loss_vy = 0
    batch_num = 0
    # In evaluation mode, we do not back propagate the loss, so there is no gradient descent
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)     #(128, 80*2)        #(128, 9, 2)                       #(128, 9, 160) 
            y_hat = output['pred']    #(128, 15)     
            y_gt = batch.y.view(-1,  pred_len*output_features).to(torch.float32)    
            # y_hat = torch.cumsum(y_hat_, dim=1)
            # y_gt = torch.cumsum(y_gt_, dim=1)
            # In evaluation mode, only calculate the loss of predicted trajectory
            loss = criterion(y_hat, y_gt)
            # loss = torch.sum((y_hat - y_gt)**2)   / (128*30) 

            total_loss += loss.detach().cpu().numpy().tolist()
            batch_num += 1 
        mean_loss  = total_loss / batch_num
        # mean_loss_vx  = total_loss_vx / batch_num
        # mean_loss_vy  = total_loss_vy / batch_num
        # return mean_loss, mean_loss_vx, mean_loss_vy
        return np.sqrt(mean_loss)
    
def get_indices_masks(tensor_2d):
    # Get the second dimension indices (i.e., column indices)
    column_indices = torch.arange(tensor_2d.size(1))
    # Create a boolean mask, only True when column index is odd
    odd_indices_mask = column_indices % 2 != 0
    eve_indices_mask = column_indices % 2 == 0
    return odd_indices_mask, eve_indices_mask



def loss_plot_broken_yaxis(train_loss_ls, dev_loss_ls, test_loss, train_loss, dev_loss, epochs,
                                  save_path=None, figsize=(8, 3), dpi=300):
    # Set academic style parameters
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'figure.dpi': dpi,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.linewidth': 0.8
    })

    import numpy as np
    train_loss_ls = np.asarray(train_loss_ls, dtype=float)
    dev_loss_ls   = np.asarray(dev_loss_ls, dtype=float)
    all_loss = np.concatenate([train_loss_ls, dev_loss_ls])
    n = len(train_loss_ls)

    # ---------- 自动决定是否启用 broken axis ----------
    # 用“后期”来估计收敛尺度：跳过最开始 10% 或至少 5 个 epoch
    tail_start = max(5, int(0.1 * n))
    tail_loss = np.concatenate([train_loss_ls[tail_start:], dev_loss_ls[tail_start:]])
    all_max = float(np.max(all_loss))
    tail_max = float(np.max(tail_loss)) if len(tail_loss) > 0 else float(np.max(all_loss))

    # 当 early spike 很夸张时启用断轴
    spike_ratio = all_max / max(tail_max, 1e-12)
    use_broken_axis = True #spike_ratio >= 5.0  # 你也可以改成 8.0/10.0 更“保守”

    # ---------- 画图 ----------
    if not use_broken_axis:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        ax.plot(train_loss_ls, color='#1f77b4', linewidth=2, label='Training')
        ax.plot(dev_loss_ls,   color='#d62728', linewidth=2, linestyle='--', label='Validation')

        ax.set_xlabel('Epoch', labelpad=10)
        ax.set_ylabel('RMSE (m/s)', labelpad=10)

        ax.tick_params(axis='both', which='both', direction='in')
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        ax.grid(True, which='major', linestyle='-', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)

        ax.legend(frameon=True, framealpha=0.8, loc='upper right', borderpad=0.5)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        ax.set_xlim(0, n - 1)
        y_padding = 0.1 * (np.max(all_loss) - np.min(all_loss))
        ax.set_ylim(np.min(all_loss) - y_padding, np.max(all_loss) + y_padding)

    else:
        # ===== broken y-axis: 两个上下子图共享 x =====
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True,
            figsize=(figsize[0], max(figsize[1] * 1.6, 4.6)),  # 断轴建议稍微高一点
            gridspec_kw={'height_ratios': [1, 2]},
            constrained_layout=True
        )

        # 两个轴都画相同曲线（视觉一致）
        for ax in (ax_top, ax_bot):
            ax.plot(train_loss_ls, color='#1f77b4', linewidth=2, label='Training')
            ax.plot(dev_loss_ls,   color='#d62728', linewidth=2, linestyle='--', label='Validation')

            ax.tick_params(axis='both', which='both', direction='in')
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

            ax.grid(True, which='major', linestyle='-', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.2)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        # 标签：下轴放 x label，上轴放 y label（避免重复拥挤）
        ax_bot.set_xlabel('Epoch', labelpad=10)
        ax_top.set_ylabel('RMSE (m/s)', labelpad=10)

        # ---- 自动设置上下 y 范围 ----
        # 下图：显示“后期细节”
        low_min = float(np.min(tail_loss)) if len(tail_loss) > 0 else float(np.min(all_loss))
        low_max = tail_max
        low_pad = 0.12 * (low_max - low_min + 1e-12)
        ax_bot.set_ylim(max(0.0, low_min - low_pad), low_max + low_pad)

        # 上图：显示“前期 spike”
        high_max = all_max
        # 上图下边界：至少比下图上边界高一点，否则断轴没有意义
        high_min = max(ax_bot.get_ylim()[1] * 1.05, high_max * 0.55)
        high_pad = 0.05 * (high_max - high_min + 1e-12)
        ax_top.set_ylim(high_min - high_pad, high_max + high_pad)

        # x 范围
        ax_bot.set_xlim(0, n - 1)

        # legend 放上图即可（避免重复）
        ax_top.legend(frameon=True, framealpha=0.8, loc='upper right', borderpad=0.5)

        # ---- 画断裂符号（斜杠）----
        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(labelbottom=False)  # 上图不显示 x tick label

        d = 0.008  # 斜杠大小（相对于轴坐标）
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.0)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)              # 左下
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)        # 右下

        kwargs = dict(transform=ax_bot.transAxes, color='k', clip_on=False, linewidth=1.0)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)        # 左上
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右上

    print('test loss: %.4f m, last train loss: %.4f, last dev loss: %.4f m ' % (test_loss, train_loss, dev_loss))

    if save_path is not None:
        plt.savefig(
            os.path.join(save_path, f"traj_pred_VN_{epochs}ep_brokeny.svg"),
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
    plt.close()



if __name__ == "__main__":
    # compute on gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # hyperparameters
    epochs = 100
    decay_lr_factor = 0.3  # 0.9
    decay_lr_every = 5
    lr = 0.001
    input_features = 15
    pred_len = 30    # 15
    output_features = 1  # longitudinal delta s



    sta_t = time.time()
    STA_REC_IDX, END_REC_IDX = 25, 45   # the index range of record files used for model training, validation, and testing
    pt_dataset_savepath = '/home/chwei/reliable_and_realtime_highway_trajectory_planning/velocity_predicition/PyG_DataSet'
    train_loader, dev_loader, test_loader  = data_pre(pt_dataset_savepath, STA_REC_IDX, END_REC_IDX)  

    end_t = time.time()
    print(f"Data processing runtime: {end_t - sta_t:.2f} seconds")

    # # Check the input graph data if it is correct by visualizing
    # for batch in train_loader:
    #     visualize_graph(batch[4])

    # get model
    model = VectorNet(input_features, pred_len, device, with_aux=False).to(device)
    # # #load parameters
    # model.load_state_dict(torch.load(os.path.join(save_dir_train, 'VN_parameters_ep100_vp.pth')))

    criterion = nn.MSELoss()
    # criterion = VectorLoss(aux_loss=True, reduction='mean')
    # criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Create learning rate scheduler
    scheduler = StepLR(optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

    start_t = time.time()

    train_loss_ls = []
    dev_loss_ls = []
    dev_loss_x_ls = []
    dev_loss_y_ls = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, device, criterion, optimizer, pred_len, output_features, scheduler)
        # validation on development set
        dev_loss = eval_epoch(model, dev_loader, device, criterion, optimizer, pred_len, output_features)  # dev_loss_x, dev_loss_y
        # save
        train_loss_ls.append(train_loss)
        dev_loss_ls.append(dev_loss)
        # dev_loss_x_ls.append(dev_loss_x)
        # dev_loss_y_ls.append(dev_loss_y)
        print(f"epoch {epoch}:  train_loss: {train_loss: .4f} \t"
              f"dev_loss: {dev_loss: .4f}   unit: m/s ",)  # f"vx: {dev_loss_x: .4f}, ", f"vy: {dev_loss_y: .4f}")
        # print( f"dev_loss: {dev_loss: .4f}, ", f"vx: {dev_loss_x: .4f}, ", f"vy: {dev_loss_y: .4f}")
    end_t = time.time()

    test_loss = eval_epoch(model, test_loader, device, criterion, optimizer, pred_len, output_features)    # test_loss_vx, test_loss_vy

    print(f"Training runtime: {end_t - start_t:.6f} seconds")


    fig = loss_plot_broken_yaxis(
        train_loss_ls, dev_loss_ls, test_loss, train_loss, dev_loss, epochs,save_path=save_dir_train, 
    )
    plt.close(fig)

    # Save neural network parameters after training ends
    torch.save(model.state_dict(), os.path.join(save_dir_train, f'VN_parameters_ep{epochs}_vp' + '.pth'))









