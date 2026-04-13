import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import time

# from HighD_datapre import data_pre
from vectornet import VectorNetBackbone
from basic_module import MLP

from utils.visual_obs import visualize_graph
from config_nw import *


from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch.utils.data import ConcatDataset

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



def test_for_de(model, data_loader, device,  pred_len, output_features, save_path_record, if_plot_bev = False, spf = 0.1):
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
            output = model(batch)     #(128, 80*2)        #(128, 9, 2)                       #(128, 9, 160) 
            y_hat = output['pred'].detach().cpu()

            #用于结果展示，放置在cpu上
            y_gt = batch.pos_x_gt.view(-1, pred_len).detach().cpu()  
            x_t0 = batch.x_t0.view(y_gt.shape[0], -1).detach().cpu()
            # lane_mkings = batch.lane_mk.view(y_gt.shape[0],  -1).detach().cpu()

            #批量绘图保存
            if if_plot_bev:
                plot_bev_traj_batch(y_hat, y_gt, save_path_record, batch_num, x_t0, spf)
            
            for tra in range(y_gt.shape[0]):
                ade_tra, fde_tra, mde_tra, _, _ = get_ade_and_fde(y_gt[tra], y_hat[tra], x_t0[tra], spf)
                # # 计算纵向轨迹的预测精度
                # loss = criterion(y_gt, y_hat)
                # plot_bev_traj(lane_mkings[tra], y_hat , np.array(y_gt[tra]), tra, ade_tra, fde_tra)
                
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
                

def get_indices_masks(tensor_2d):
    # Get the second dimension indices (i.e., column indices)
    column_indices = torch.arange(tensor_2d.size(1))
    # Create a boolean mask, only True when column index is odd
    odd_indices_mask = column_indices % 2 != 0
    eve_indices_mask = column_indices % 2 == 0
    return odd_indices_mask, eve_indices_mask

def get_ade_and_fde(pos_gt, out_hat, xva_t0, spf, out_content = 'vx'):

    x0, v0, a0 = xva_t0[0].item(),  xva_t0[1].item(),  xva_t0[2].item()
    if  out_content == 'vx':
        pos_pre = get_restored_pos_v(out_hat, x0, spf)
    elif  out_content == 'delatx':
        pos_pre = get_restored_pos_ds(out_hat, x0)
    elif out_content == 'ax':
        pos_pre = get_restored_pos_acc(out_hat, x0, v0, a0, spf)
    
    if isinstance(pos_gt, torch.Tensor):
        pos_gt = pos_gt.detach().cpu().numpy()
    if isinstance(pos_pre, torch.Tensor):
        pos_pre = pos_pre.detach().cpu().numpy()

    single_des = []
    for i in range(pos_gt.shape[0]):
        #get_euclidean_distance
        # single_des.append(torch.sqrt(  (pos_pre[i, 0] - pos_gt[i, 0])**2 + (pos_pre[i, 1] - pos_gt[i, 1])**2  )  )#0 and 1 are longitudinal and lateral positions respectively
        single_des.append(np.abs(pos_pre[i] - pos_gt[i])  )  #for only longitudinal positions
    ade = sum(single_des) / len(single_des)
    fde = single_des[-1]
    maxde = np.max(single_des)
    return ade.item(), fde.item(), maxde.item(), pos_pre, pos_gt


def plot_bev_traj(lane_markings, pos_ev_pre, pos_ev_gt, case_id, ade, fde):
    fig, ax = plt.subplots(figsize=(17, 2))  

    #------------------------Draw lane lines as reference
    for y in lane_markings :  
        plt.axhline(y=y.item(), color='gray', linestyle='--')  # Use plt.axhline() to draw horizontal lines

    #-----------------------Draw the latest ego vehicle
    rectangles = []  # Used to store created rectangle objects  
    labels = []  # Used to store rectangle labels  
    r_wid = 4.8
    r_height = 1.8
    r_x =  0 - 0.5*r_wid          # After normalization, the ego vehicle's coordinate at the latest moment is (0,0)
    r_y =  0 - 0.5*r_height      
    # Draw rectangle, set border width, fill color, and transparency  
    rect = patches.Rectangle((r_x, r_y), r_wid, r_height,   
                            linewidth=0.7,  # Set border width  
                            facecolor='none',  # Set fill color  
                            edgecolor='grey',  # Set border color  
                            alpha=0.9)  # Set transparency  
    rectangles.append(rect)
    # Add rectangle to the plot
    for rect in rectangles:
        ax.add_patch(rect)
        
    # #-----------------------Draw historical trajectory
    # for coor_x, coor_y in pos_svs_hist:
    #     plt.scatter(coor_x, coor_y, s=10, c='navy', marker='o',  label='History', alpha=0.8)          # Draw discrete points   # Set point size to 10, color to blue

    #-----------------------Draw future predicted trajectory and ground truth trajectory
    plt.scatter([coor_x  for  coor_x, _ in pos_ev_pre], [coor_y  for  _, coor_y in pos_ev_pre], s=10, c='orangered', marker='o',  label='Prediction', alpha=0.8)          # Prediction trajectory in orange
    plt.scatter([coor_x  for  coor_x, _ in pos_ev_gt], [coor_y  for  _, coor_y in pos_ev_gt], s=10, c='green', marker='o',  label='Ground Truth', alpha=0.5)          # Ground truth trajectory in green
    

    #-----------------------Plot settings and display
    # Add X and Y axis labels  
    plt.xlabel('X-axis', fontsize = 20)  
    plt.ylabel('Y-axis', fontsize = 20)      
    # Set axis display range
    plt.xlim(pos_ev_pre[0, 0] - 5,   pos_ev_pre[0, 0]+ 200)    # Show full range
    plt.ylim(pos_ev_pre[0, 1]  - 9,  pos_ev_pre[0, 1] + 9  )
    plt.title("Case Id: %d"%case_id)
    # Get the current axis maximum value  
    x_max, y_max = plt.xlim()[1], plt.ylim()[1]  
    plt.text(x_max*0.5, y_max *0.5, "ADE: %.2f m\nFDE: %.2f m " % (ade, fde)  )
    # Show legend  
    # plt.legend(handles=[prediction_scatter, ground_truth_scatter], labels=['Prediction', 'Ground Truth'])  
    plt.legend()
  
    
    plt.show()
    
    return 0


def plot_bev_traj_batch(y_hat, y_gt, save_path, batch_id, x_t0, spf, lane_markings= None):
    nrows = 5
    ncols = 1
    plot_num =  -(- y_hat.shape[0]// (nrows*ncols)  )   #Ceiling division
    
    for i in range(plot_num):
        # Create a figure and several subplots    
        fig = plt.figure(figsize=(15, 8) ) 
        axs = fig.subplots(nrows=nrows, ncols=ncols)    #        fig._constrained_layout_pads[0] = 1
        
        # fig, axs = plt.subplots(nrows=nrows, conls = ncols, figsize = (15,  5) )  # 5 rows, 1 column of subplots  
        for j in range(nrows*ncols):
            # Current batch trajectory id
            case_id = i*(nrows*ncols)  + j
            #
            if j == 0:
                axs[j].axis('off')  #Remove axis
                continue
            if case_id < y_hat.shape[0]:
                # pos_pre_tra =  y_hat[case_id]  #.view(-1, 2)    
                ade_tra, fde_tra, mde_tra, pos_pre, pos_gt = get_ade_and_fde(y_gt[case_id], y_hat[case_id], x_t0[case_id], spf)
                
                # #------------------------Draw lane lines as reference
                # for y in lane_markings[case_id] :  
                #     axs[j].axhline(y=y.item(), color='gray', linestyle='--')  # Use plt.axhline() to draw horizontal lines

                #-----------------------Draw the latest ego vehicle
                rectangles = []  # Used to store created rectangle objects  
                labels = []  # Used to store rectangle labels  
                r_wid = 4.8
                r_height = 1.8
                r_x =  0 - 0.5*r_wid          # After normalization, the ego vehicle's coordinate at the latest moment is (0,0)
                r_y =  0 - 0.5*r_height      
                # Draw rectangle, set border width, fill color, and transparency  
                rect = patches.Rectangle((r_x, r_y), r_wid, r_height,   
                                        linewidth=1.5,  # Set border width  
                                        facecolor='grey',  # Set fill color  
                                        edgecolor='black',  # Set border color  
                                        alpha=0.5,
                                        label='Ego Vehicle')  # Set transparency  
                rectangles.append(rect)
                # Add rectangle to the plot
                for rect in rectangles:
                    axs[j].add_patch(rect)
                    
                # #-----------------------Draw historical trajectory
                # for coor_x, coor_y in pos_svs_hist:
                #     axs[j].scatter(coor_x, coor_y, s=10, c='navy', marker='o',  label='History', alpha=0.8)          # Draw discrete points   # Set point size to 10, color to blue

                #-----------------------Draw future predicted trajectory and ground truth trajectory
                axs[j].scatter([coor_x  for  coor_x in np.array(pos_pre) ], [0  for  _ in np.array(pos_pre) ], s=20, c='orangered',  marker='o',  label='Prediction', alpha=0.8)          # Prediction trajectory in orange   facecolor='orangered',  edgecolor='black', 
                axs[j].scatter([coor_x  for  coor_x in np.array(pos_gt) ], [0  for  _  in np.array(pos_gt)], s=20, c='green',marker='v',  label='Ground Truth', alpha=0.5)          # Ground truth trajectory in green
                
                #-----------------------Plot settings and display
                # # Add X and Y axis labels  
                # axs[j].xlabel('X-axis', fontsize = 20)  
                # axs[j].ylabel('Y-axis', fontsize = 20)      
                # Set axis display range
                axs[j].set_xlim(pos_pre[0] - 5,   pos_pre[0]+ 180)    # Show full range
                axs[j].set_ylim(0  - 6,  0 + 6  )
                axs[j].set_title("Case Id: %d"%j)   #  case_id
                # Record error results
                axs[j].text(165, -3, "ADE: %.2f m\nFDE: %.2f m\nMDE: %.2f m " % (ade_tra, fde_tra, mde_tra)  , fontsize = 12)
            # If data has been plotted, you can choose to hide or remove the subplot  
            else:
                axs[j].remove()  # Or axs[i, j].axis('off') to hide the subplot  
        
        #Set global legend   Q: It may overlap, current version seems not to support layout = "constrained"
        objects, labels = axs[1].get_legend_handles_labels() 
        fig.legend(objects, labels, loc=(0.04, 0.8), fontsize = 12) 
        
        # # Draw data and set legend on the first subplot  
        # axs[0].legend(loc=(0.7, 0.3), fontsize = 12)  
        
        # Show figure  
        plt.tight_layout()  # This will automatically adjust subplot parameters to fill the entire image area  
        plt.savefig(os.path.join(save_path, 'batch'+str(batch_id)+'_fig'+str(i)+'.svg') )
        # plt.show()
        plt.close()
    
    return 0

def get_restored_pos_acc(label_acc, x0, v0, a0, spf):
    restored_data, restored_v = np.zeros_like(label_acc), np.zeros_like(label_acc)
    restored_data[0] = x0 + v0 * spf + 0.5 * a0 * spf **2
    restored_v[0] = v0 + a0 * spf 
    for i in range(1, restored_data.shape[-1]):
        restored_data[i] = restored_data[i-1] +  restored_v[i-1]* spf + 0.5 * label_acc[i-1] * spf **2   #认为当前时刻的（瞬时）v和a决定下一时刻的position
        restored_v[i] = restored_v[i-1] + label_acc[i-1] * spf 
    return restored_data

def get_restored_pos_ds(label_ds, x0):
    return x0 + np.cumsum(label_ds)

def get_restored_pos_v(label_v, x0, spf):
    dss = label_v * spf
    return x0 + np.cumsum(dss)   #!!!



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_features = 15
    pred_len = 30
    output_features = 1
    sec_per_frame = 0.1


    # ✅ New: explicit record list (same spirit as train_myway.py)
    REC_ID_LIST = [47, 48, 49]         # or [54, 55, 56], or any arbitrary list

    # drawing the bird-eye-view trajectory to visually check the displacement error
    if_plot_bev = False

    test_loader, test_dataset = build_test_loader_from_rec_list(
        pyg_root=read_dir_pyg,
        rec_id_list=REC_ID_LIST,
        batch_size=128,
        shuffle=False
    )
    print("Test REC_ID_LIST:", REC_ID_LIST)
    print("Total trajectories:", len(test_dataset))

    model = VectorNet(input_features, pred_len, device, with_aux=False).to(device)
    model.load_state_dict(torch.load(os.path.join(read_dir_train, 'vel_pred_VN_100ep.pth')))

    start_t = time.time()
    ade_tras, fde_tras, traj_num = test_for_de(
        model, test_loader, device, pred_len, output_features, save_dir_bev,
        if_plot_bev=if_plot_bev, spf=sec_per_frame
    )
    end_t = time.time()
    print(f"Test runtime: {end_t - start_t:.6f} seconds")


    

    '''
    【20250620】the input features have x, y, vx, vy, ax, ay; the output is vx:
    【 ALL TRIANING ON RECORD 26-45    initial lr=0.001   decaying lr*=0.9 per 5 eps     100epochs    time sliding by 1 sec】   
    A.[VN_parameters_4_display.pth]:       ON RECORD 54, 55, 56 (三车道, 109780条轨迹):  ADE: 0.63 m    FDE: 1.47 m    MDE: 19.47 m
                                           ON RECORD 53, 54, 55 (三车道, 121934条轨迹):  ADE: 0.63 m    FDE: 1.46 m    MDE: 19.48 m 
    B.[VN_parameters_4_display_25ep.pth]:  ON RECORD 54, 55, 56 (三车道, 109780条轨迹):  ADE: 0.62 m    FDE: 1.45 m    MDE: 20.14 m 
                                           ON RECORD 53, 54, 55 (三车道, 121934条轨迹):  ADE: 0.62 m    FDE: 1.45 m    MDE: 20.14 m 

    
    【20250623】same training setting as above:
    ON RECORD 54, 55, 56 (Total trajectories: 56550 ??):   ADE: 0.06 m    FDE: 0.16 m    MDE: 6.17 m  
                                           
    '''