import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR  
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch.utils.data import ConcatDataset, random_split
import time

from vectornet import VectorNetBackbone
from basic_module import MLP

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# from utils.visual_obs import visualize_graph

from config_nw import *


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, idx=0, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False )   #, weights_only=False    #记得换回来！！！

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"record{rid:02d}.pt" for rid in REC_ID_LIST]

    def download(self):
        pass

    def process(self):
        pass


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
                 vel_pred_mlp_width=64    #64
                 ):
        super(VectorNet, self).__init__()
        # some params
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.out_channels = 1   #x
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
        self.vel_pred_mlp = nn.Sequential(
            MLP(global_graph_width*13, vel_pred_mlp_width),
            nn.Linear(vel_pred_mlp_width, self.horizon * self.out_channels),
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
        
        v_norm = self.vel_pred_mlp(global_feat) # [batch_size, 160]       #应该是 [128 9, 160]
        # 在sigmoid()输出[0,1]基础上扩大至[0, v_limit]
        v_hat = v_norm *self.v_limit
        
        return {"pred": v_hat, "aux_out": aux_out, "aux_gt":aux_gt}


def train_epoch(model, data_loader, device , criterion, optimizer, pred_len, scheduler):
    #training
    model.train()
    graph_num = len(data_loader.dataset)# record the number of the batch
    total_loss = 0
    batch_num = 0
    
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch)    
        y_hat = output['pred']    #(128, 30)   
        B = batch.num_graphs  # 128
        y_gt = batch.y.view(B, pred_len*1).to(torch.float32)   # [B,T]   output_features=1  
        #均采用mse loss  
        loss = criterion(y_hat, y_gt)      #+ criterion(output['aux_out'], output['aux_gt'])   #nn.MSE默认Mean(sum(l_gt - l-hat)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().cpu().numpy().tolist()
        batch_num += 1 
    mean_loss  = total_loss / batch_num   #mean_loss是每一个元素(batchi中第j时间步)位置上的误差的平方
    scheduler.step()
    return np.sqrt(mean_loss)    #此处通过开方将MSE转为RMSE，更清晰一点



def eval_epoch(model, data_loader, device, criterion, pred_len):
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    batch_num = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            output = model(batch)
            v_hat = output["pred"]  # [B,T]

            B = batch.num_graphs
            v_gt = batch.y.view(B, pred_len*1).to(torch.float32)  # [B,T] 

            loss = criterion(v_hat, v_gt)
            total_loss += loss.item()

            # === 速度 -> 位置(1D) ===
            # x_t = x0 + sum_{k<=t} v_k * dt
            # set x0 = 0 for all samples, since ADE/FDE only cares about relative displacement
            x_hat = torch.cumsum(v_hat * SPF, dim=1)    # [B,T]
            x_gtp = torch.cumsum(v_gt  * SPF, dim=1)    # [B,T]

            # 位置误差：1D 下“距离”就是 abs
            err = (x_hat - x_gtp).abs()              # [B,T]  unit: m
            total_ade += err.mean().item()
            total_fde += err[:, -1].mean().item()

            batch_num += 1

    mean_mse = total_loss / batch_num
    rmse_v = np.sqrt(mean_mse)  # 速度RMSE (m/s)
    ade_pos = total_ade / batch_num  # 位置ADE (m)
    fde_pos = total_fde / batch_num  # 位置FDE (m)
    return rmse_v, ade_pos, fde_pos



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
    use_broken_axis = spike_ratio >= 5.0  # 你也可以改成 8.0/10.0 更“保守”

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

    print('test loss: %.4f m/s, last train loss: %.4f m/s, last dev loss: %.4f m/s ' % (test_loss, train_loss, dev_loss))

    if save_path is not None:
        plt.savefig(
            os.path.join(save_path, f"vel_pred_VN_{epochs}ep_brokeny.svg"),
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
    plt.close()


if __name__ == "__main__":
    # compute on gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #hyperparameters
    epochs = 100     #100
    decay_lr_factor = 0.9  #0.3
    decay_lr_every = 5
    lr = 0.001
    input_features = 15
    pred_len = 30    #15    



    sta_t = time.time()
    REC_ID_LIST = list(range(26, 46))      #  list(range(26, 46))   !!!

    datasets_ls = []
    for i in tqdm(range(len(REC_ID_LIST)), desc="Reading Records"):
        datasets_ls.append(MyOwnDataset(root=read_dir_pyg, idx=i))
    final_dataset = ConcatDataset(datasets_ls)

    train_loc, vali_loc= int(0.7*len(final_dataset) ), int( 0.2*len(final_dataset) )
    test_loc = len(final_dataset) - train_loc - vali_loc
    print('train_num:valid_num:test_num = %d:%d:%d'%(train_loc, vali_loc, test_loc) )
    train_dataset, validation_dataset, test_dataset = random_split(final_dataset, [train_loc, vali_loc, test_loc], generator=torch.Generator().manual_seed(42))  #设置随机种子，方便复现
    

    mini_batch_size = 128
    train_loader = DataLoader( train_dataset, batch_size=mini_batch_size, shuffle=True )
    dev_loader = DataLoader( validation_dataset, batch_size=mini_batch_size, shuffle=True )
    test_loader = DataLoader( test_dataset, batch_size=mini_batch_size, shuffle=False )    

    
    print("the total number of trajectory: %d"%len(final_dataset))
    
    end_t = time.time()
    print(f"数据处理运行时间: {end_t - sta_t:.2f} 秒")
    
    
    # #check the input graph data is correct by visualizing
    # for batch in train_loader:
    #     visualize_graph(batch[4])


    # get model
    model = VectorNet(input_features, pred_len, device, with_aux=False ).to(device)
    # # #load parameters
    # model.load_state_dict(torch.load( os.path.join(save_dir_train, 'VN_parameters_3.pth')  ))
    
    criterion = nn.MSELoss()
    # criterion = VectorLoss( aux_loss=True, reduction='mean')
    # criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 创建学习率调度器  
    scheduler = StepLR(optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)  
  
    
    start_t = time.time()
    
    train_loss_ls = []
    dev_loss_ls = []
    for epoch in range(epochs):
        train_rmse = train_epoch(model, train_loader, device, criterion, optimizer, pred_len, scheduler)
        #validaiton on development set
        dev_rmse, dev_ade, dev_fde = eval_epoch(model, dev_loader, device, criterion, pred_len)
        #save
        train_loss_ls.append(train_rmse)
        dev_loss_ls.append(dev_rmse)
        
        print(f"epoch {epoch}: train_rmse {train_rmse:.4f} | dev_rmse {dev_rmse:.4f} m | ADE {dev_ade:.4f} m | FDE {dev_fde:.4f} m")

    end_t = time.time()
    
    #测试集
    test_rmse, test_ade, test_fde = eval_epoch(model, test_loader, device, criterion, pred_len)
    
    print(f"训练运行时间: {end_t - start_t:.6f} 秒")
    
    fig = loss_plot_broken_yaxis(
        train_loss_ls, dev_loss_ls, test_rmse, train_rmse, dev_rmse, epochs, save_path= save_dir_train
    )  
    plt.close(fig)
    
    #训练结束后保存神经网络参数
    torch.save(
        model.state_dict(),
        os.path.join(save_dir_train, f"vel_pred_VN_{epochs}ep_lrdf09.pth")
    )


'''
<<Training Log>>


[02/14/2026]

epoch 95: train_rmse 0.1353 | dev_rmse 0.1371 m | ADE 0.0529 m | FDE 0.1876 m
epoch 96: train_rmse 0.1353 | dev_rmse 0.1371 m | ADE 0.0529 m | FDE 0.1876 m
epoch 97: train_rmse 0.1353 | dev_rmse 0.1371 m | ADE 0.0529 m | FDE 0.1876 m
epoch 98: train_rmse 0.1353 | dev_rmse 0.1371 m | ADE 0.0529 m | FDE 0.1876 m
epoch 99: train_rmse 0.1353 | dev_rmse 0.1371 m | ADE 0.0529 m | FDE 0.1876 m
训练运行时间: 16226.662360 秒
test loss: 0.1349 m/s, last train loss: 0.1353 m/s, last dev loss: 0.1371 m/s 


[03/10/2026]
epoch 0: train_rmse 1.1857 | dev_rmse 0.5740 m | ADE 0.5821 m | FDE 1.1163 m
epoch 1: train_rmse 1.2116 | dev_rmse 0.4779 m | ADE 0.4812 m | FDE 0.9201 m
epoch 2: train_rmse 0.3750 | dev_rmse 0.2819 m | ADE 0.2341 m | FDE 0.4618 m
epoch 3: train_rmse 0.2992 | dev_rmse 0.3899 m | ADE 0.4690 m | FDE 0.9765 m
epoch 4: train_rmse 0.2737 | dev_rmse 0.2072 m | ADE 0.1288 m | FDE 0.3011 m
...
epoch 95: train_rmse 0.1006 | dev_rmse 0.1223 m | ADE 0.0589 m | FDE 0.1937 m
epoch 96: train_rmse 0.1004 | dev_rmse 0.1205 m | ADE 0.0689 m | FDE 0.2019 m
epoch 97: train_rmse 0.1003 | dev_rmse 0.1052 m | ADE 0.0469 m | FDE 0.1474 m
epoch 98: train_rmse 0.1002 | dev_rmse 0.1058 m | ADE 0.0473 m | FDE 0.1482 m
epoch 99: train_rmse 0.1003 | dev_rmse 0.1240 m | ADE 0.0797 m | FDE 0.2180 m
训练运行时间: 39976.024463 秒
test loss: 0.1198 m/s, last train loss: 0.1003 m/s, last dev loss: 0.1240 m/s 



'''




