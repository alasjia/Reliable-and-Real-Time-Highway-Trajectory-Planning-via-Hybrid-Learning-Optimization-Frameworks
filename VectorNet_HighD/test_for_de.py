import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import time

from vectornet import VectorNetBackbone
from basic_module import MLP

from utils.visual_obs import visualize_graph
from config_nw import *

from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader  # 替代 data.DataLoader
from torch_geometric.data import InMemoryDataset

class MyOwnDataset(InMemoryDataset):
    """
    Load ONE recordXX.pt as an InMemoryDataset.
    """
    def __init__(self, root: str, rec_id: int, transform=None, pre_transform=None, pre_filter=None):
        self.rec_id = int(rec_id)
        super().__init__(root, transform, pre_transform, pre_filter)
        # only one processed file for this dataset
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # must match the saved pt filename
        return [f"record{self.rec_id:02d}.pt"]

    def download(self):
        pass

    def process(self):
        pass

def build_test_loader_from_rec_list(
    pyg_root: str,
    rec_id_list,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
):
    """
    Align with train_myway.py style: loop REC_ID_LIST -> ConcatDataset -> DataLoader
    """
    datasets = [MyOwnDataset(root=pyg_root, rec_id=rid) for rid in rec_id_list]
    final_dataset = ConcatDataset(datasets)
    loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, final_dataset

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
        self.out_channels = 2  # (x, y)
        self.horizon = horizon
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.k = 1

        self.device = device
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
            nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels)
        )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        global_feat, aux_out, aux_gt = self.backbone(data)              # [batch_size, time_step_len, global_graph_width]   [128, 9, 64]

        global_feat = global_feat.view(global_feat.shape[0], -1)   #展平--> [128, 9*64]
        
        pred = self.traj_pred_mlp(global_feat)          # [B, T*2]
        pred = pred.view(-1, self.horizon, 2)           # [B, T, 2]
        return {"pred": pred, "aux_out": aux_out, "aux_gt": aux_gt}


def test_for_de(model, data_loader, device,  pred_len, save_path_record, if_plot_bev = False, spf = 0.1):
    model.eval()  #evaluation mode
    graph_num = len(data_loader.dataset)# record the number of the batch
    ade_total = 0
    fde_total = 0
    mde_global = 0
    total_traj_num = 0
    batch_num = 0

    #in evaluation mode, we do not back propagate the loss, so there is no gradient descent
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)     

            y_hat = output["pred"].detach().cpu()         # (B, T, 2)
            y_gt  = batch.pos_xy_gt.view(-1, pred_len, 2).detach().cpu()

            #批量绘图保存
            if if_plot_bev:
                lane_flat = batch.lane_ys_nonorm.detach().cpu().numpy()
                # 推断每个样本的 lane 数（highway 通常稳定）
                n_lane = len(np.unique(lane_flat))
                # reshape 成 (B, n_lane)
                lane_markings = lane_flat.reshape(-1, n_lane)

                plot_bev_traj_batch_2d(
                    y_hat=y_hat,
                    y_gt=y_gt,
                    lane_markings=lane_markings,
                    save_path=save_path_record,
                    batch_id=batch_num
                )

            
            for tra in range(y_gt.shape[0]):
                ade_tra, fde_tra, mde_tra, _, _ = get_ade_and_fde_traj(y_gt[tra], y_hat[tra])
                
                ade_total += ade_tra
                fde_total += fde_tra
                # print("traj %d ade and fde: %.2f, %.2f m"%(tra, ade_tra, fde_tra))
                
                if mde_tra > mde_global:
                    mde_global = mde_tra
                
                total_traj_num += 1
            batch_num += 1
        mean_ade  = ade_total / total_traj_num  #平均到每一个trajectory
        mean_fde  = fde_total / total_traj_num
        print( "ADE: %.2f m    FDE: %.2f m    MDE: %.2f m "%(mean_ade, mean_fde, mde_global))
        return mean_ade, mean_fde, total_traj_num
    
    return 0, 0
                


def get_ade_and_fde_traj(pos_gt: torch.Tensor, pos_hat: torch.Tensor):
    """
    pos_gt:  (T, 2) torch tensor 
    pos_hat: (T, 2) torch tensor 
    return: ade, fde, mde, pos_hat_np, pos_gt_np
    """
    gt = pos_gt.numpy()
    hat = pos_hat.numpy()

    # per-step euclidean displacement error
    des = np.linalg.norm(hat - gt, axis=-1)   # (T,)
    ade = float(des.mean())
    fde = float(des[-1])
    mde = float(des.max())
    return ade, fde, mde, hat, gt

def plot_bev_traj_batch_2d(
    y_hat, y_gt, lane_markings,
    save_path, batch_id,
    nrows=5, ncols=1,
    margin_x=10.0,
    margin_y=3.0
):
    """
    y_hat: (B, T, 2)  predicted trajectories
    y_gt : (B, T, 2)  ground truth trajectories
    lane_markings: (B, N_lane) or list of lists, each element is y = const
    """

    # ---- to numpy
    if torch.is_tensor(y_hat):
        y_hat = y_hat.detach().cpu().numpy()
    if torch.is_tensor(y_gt):
        y_gt = y_gt.detach().cpu().numpy()

    B, T, _ = y_hat.shape
    plot_num = -(-B // (nrows * ncols))  # ceil


    samples_per_fig = nrows * ncols - 1  # 因为 j==0 留空

    for i in range(plot_num):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
        axs = axs.flatten()

        for j in range(nrows * ncols):
            if j == 0:
                axs[j].axis("off")
                continue

            idx = i * samples_per_fig + (j - 1)  

            if idx >= B:
                axs[j].remove()
                continue

            ax = axs[j]
            pos_pre = y_hat[idx]
            pos_gt = y_gt[idx]

            for y in lane_markings[idx]:
                ax.axhline(y=float(y), color="gray", linestyle="--", linewidth=1)

    # for i in range(plot_num):
    #     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
    #     axs = axs.flatten()

    #     for j in range(nrows * ncols):
    #         idx = i * (nrows * ncols) + j

    #         # 你原来第 0 个子图是空白区，保留这个习惯
    #         if j == 0:
    #             axs[j].axis("off")
    #             continue

    #         if idx >= B:
    #             axs[j].remove()
    #             continue

    #         ax = axs[j]

    #         pos_pre = y_hat[idx]   # (T,2)
    #         pos_gt  = y_gt[idx]

    #         # ---------- lane markings (horizontal lines)
    #         for y in lane_markings[idx]:
    #             ax.axhline(y=y, color="gray", linestyle="--", linewidth=1)

            # ---------- ego vehicle (at origin)
            ego_w, ego_h = 4.8, 1.8
            ego_rect = patches.Rectangle(
                (-ego_w / 2, -ego_h / 2),
                ego_w, ego_h,
                linewidth=1.5,
                edgecolor="black",
                facecolor="grey",
                alpha=0.5,
                label="Ego"
            )
            ax.add_patch(ego_rect)

            # ---------- trajectories
            ax.plot(pos_pre[:, 0], pos_pre[:, 1],
                    color="orangered", linewidth=1.5, label="Prediction")
            ax.plot(pos_gt[:, 0], pos_gt[:, 1],
                    color="green", linewidth=1.5, linestyle="--", label="Ground Truth")

            ax.scatter(pos_pre[:, 0], pos_pre[:, 1],
                       color="orangered", s=20)
            ax.scatter(pos_gt[:, 0], pos_gt[:, 1],
                       color="green", s=20, alpha=0.6)

            # ---------- axis range (auto but highway-aware)
            xs = np.concatenate([pos_pre[:, 0], pos_gt[:, 0], [0.0]])
            ys = np.concatenate([pos_pre[:, 1], pos_gt[:, 1], [0.0]])

            ax.set_xlim(xs.min() - margin_x, xs.max() + margin_x)
            ax.set_ylim(ys.min() - margin_y, ys.max() + margin_y)

            # BEV 必须等比例
            ax.set_aspect("equal", adjustable="box")

            ax.set_title(f"Case {idx}")

        # global legend
        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.04, 0.8), fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"batch{batch_id}_fig{i}.svg"))
        plt.close()



if __name__ == "__main__":
    # compute on gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #if visualize the specific distance errors for each trajectory
    if_plot_bev = False

    #hyperparameters
    input_features = 15
    pred_len = 30
    sec_per_frame =0.1
 

    # ===== data preparation: build total_loader by reading processed .pt only =====
    REC_ID_LIST = [53, 54, 55]  # [53, 54, 55] 
    test_loader, test_dataset = build_test_loader_from_rec_list(
        pyg_root=read_dir_pyg,
        rec_id_list=REC_ID_LIST,
        batch_size=128,
        shuffle=False
    )

    # get model
    model = VectorNet(input_features, pred_len, device, with_aux= False).to(device)   
    #load parameters        
    model.load_state_dict(torch.load( os.path.join( read_dir_train, 'traj_pred_VN_100ep_lrdf09.pth')  ))   #!!! 
    start_t = time.time()
    
    #输出测试结果，if_plot_bev选择是否绘制BEV视角的对比图
    ade_tras, fde_tras, traj_num = test_for_de(model, test_loader, device, pred_len, save_dir_bev,if_plot_bev, spf=sec_per_frame)

    end_t = time.time()
    
    
    print(f"测试运行时间: {end_t - start_t:.6f} 秒")
    
    
    '''
    01/2026
    [rec(53,54,55), 100ep.pth]
    ADE: 0.66 m    FDE: 1.52 m    MDE: 21.06 m 
    测试运行时间: 32.711430 秒

    14/02/2026
    [rec(53,54,55), 100ep.pth]
    ADE: 0.66 m    FDE: 1.52 m    MDE: 20.58 m 
    测试运行时间: 59.082692 秒

    11/03/2026
    [rec(53,54,55), 100ep.pth]
    ADE: 0.66 m    FDE: 1.52 m    MDE: 20.27 m 
    测试运行时间: 91.812812 秒

    '''

    