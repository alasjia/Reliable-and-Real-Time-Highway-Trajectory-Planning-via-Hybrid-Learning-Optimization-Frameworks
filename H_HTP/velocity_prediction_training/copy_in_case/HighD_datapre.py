'''
created on 12th July 2024
@author: yujia lu

[v] Change the time interval between each element in the time series from 0.2 to 0.1 sec
after 2024.12.03: 
[v] Modify the output to longitudinal position delta s, to facilitate speed limit settings
[v] Optimize the input format for the graph neural network: conversion between unstructured graph data and structured 2D arrays; iteration of the cluster parameter for subgraph aggregation in the graph network
[v] Modify the output again to discrete velocity.  2024.12.19
[v] Expand the dataset by splitting each scenario into multiple trajectory samples (sliding along the time axis).  2024.12.29
[v] Since the expanded dataset exceeds the storage capacity of a list, change the data reading and storage format to PyG's built-in DataSet   2025.1.2
'''

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset
from torch.utils.data import ConcatDataset, random_split
import os
import re
import time
import random
from tqdm import tqdm

from config_nw import *

# from utils.visual_obs import visualize_graph



#torch-geometric     v =1.7.2
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, cur_rec_idx, sta_rec_idx, end_rec_idx, transform=None, pre_transform=None, pre_filter=None):
        self.cur_rec_idx = cur_rec_idx
        self.sta_rec_idx = sta_rec_idx
        self.end_rec_idx = end_rec_idx
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(
            self.processed_paths[self.cur_rec_idx - self.sta_rec_idx],
            weights_only=False
        )
        # This step is executed after the .pt file is stored. Note: if a completely identical processed_file_names is detected (as long as it is included), process() will not be executed.

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # Generate a list of file names dynamically
        file_names = []
        for i in range(self.sta_rec_idx, self.end_rec_idx):  # Change the range to the desired number of files.  
            file_name = f"record{str(i+1).zfill(2)}.pt"
            file_names.append(file_name)
        return file_names

    def download(self):
        pass

    def process(self):
        #------------- Read data into a huge `Data` list.
        #
        max_sv_num = 8
        max_lane_num = 4
        # Location of the original data
        read_dir1 = '/home/chwei/reliable_and_realtime_highway_trajectory_planning/data_processing/raw_highd_data'
            
        # Read processed data
        files_ = os.listdir(read_dir2)  #60*4
        # Use regular expressions to match files starting with '.~lock.' and ending with '#', filter out temporary files
        pattern = r'^\.~lock.*#$'  
        files = [file for file in files_ if not re.match(pattern, file)]
        files.sort()  

        cluster_global_ptr = 0  # Global parameter
        for record in tqdm( range(self.sta_rec_idx, self.end_rec_idx), desc='Processing Records'):  #range(3, 53),
            # Clear at the start of reading each recording
            graph_ls = []
            # Read the corresponding original data according to the order of processed data
            rMetafile_name = files[record][-6:-4] + '_recordingMeta.csv'
            tMetafile_name = files[record][-6:-4] + '_tracksMeta.csv'
            recordingMeta = pd.read_csv(os.path.join(read_dir1, rMetafile_name))
            tracksMeta = pd.read_csv(os.path.join(read_dir1, tMetafile_name))   # To determine the driving direction of the vehicle
            #
            tracks_df = pd.read_csv(os.path.join(read_dir2, str(files[record])),  index_col=0)
            # Accumulate all record data
            graph_ls, cluster_global_ptr = get_single_record_data(tracks_df, graph_ls, recordingMeta, tracksMeta, max_sv_num, max_lane_num,  cluster_global_ptr, divide_prop, if_test4DE = False)
            
            assert cluster_global_ptr % 13 == 0  # After alignment, the number of polylines in each graph is 13
            
            # For PyG<2.4:
            torch.save(self.collate(graph_ls), self.processed_paths[record - self.sta_rec_idx])  
            #self.processed_paths returns  ['root/data1.pt'], ['root/data2.pt'],...
            
        

def data_pre(pt_dataset_savepath, sta_rec_idx, end_rec_idx, if_test4DE=False):
    sta_t = time.time()

    datasets_ls = []
    for cur_rec_idx in range(sta_rec_idx, end_rec_idx):
        datasets_ls.append(MyOwnDataset(root=pt_dataset_savepath, cur_rec_idx=cur_rec_idx, sta_rec_idx=sta_rec_idx, end_rec_idx=end_rec_idx))
        
    final_dataset = ConcatDataset(datasets_ls)
    train_loc, vali_loc= int(0.7*len(final_dataset) ), int( 0.2*len(final_dataset) )
    test_loc = len(final_dataset) - train_loc - vali_loc
    # train_valid_prop = 0.9
    train_dataset, validation_dataset, test_dataset = random_split(final_dataset, [train_loc, vali_loc, test_loc], generator=torch.Generator().manual_seed(42))  # Set random seed for reproducibility
    

    mini_batch_size = 128
    train_loader = DataLoader( train_dataset, batch_size=mini_batch_size, shuffle=True )
    dev_loader = DataLoader( validation_dataset, batch_size=mini_batch_size, shuffle=True )
    test_loader = DataLoader( test_dataset, batch_size=mini_batch_size, shuffle=False )    

    
    print("the total number of trajectory: %d"%len(final_dataset))
    
    end_t = time.time()
    print(f"Data processing runtime: {end_t - sta_t:.2f} seconds")
    
    if if_test4DE:
        total_loader = DataLoader(final_dataset, batch_size=mini_batch_size, shuffle=False ) 
        return total_loader   

    return train_loader, dev_loader, test_loader   


def get_single_record_data(tracks_df, graph_ls, recordingMeta, tracksMeta, max_sv_num, max_lane_num,  cluster_global_ptr, divide_prop, if_test4DE = False):
    # Group by 'id'
    grouped = tracks_df .groupby('id')

    # Initialize an empty list to store each group's data
    group_data = []
    group_id = []
    # Iterate over each group
    for tra_id, group in grouped:
        group_id.append(tra_id)
        
        total_frame_len = int(0.04*group.shape[0]/SPF)  # The maximum duration of the EV trajectory in the current scenario, frequency converted from 0.04 to 0.1, 0.04s is the original sequence data frequency. Floor division.
        fixed_len = F_HIS+F_FUR  # The number of frames corresponding to the required EV trajectory duration
        
        for t in range(0, total_frame_len - fixed_len + 1 , 10):  # Number of sliding steps
            if t+fixed_len > total_frame_len:
                break    # Will not enter
                
            features_we_want = get_features_we_want(group)
            
            frequency_we_want = get_frequency_we_want(features_we_want)
            
            fixedlen_time_we_want = get_fixedlen_time_we_want(frequency_we_want, sta = t, end = t+fixed_len)  # 5 sec

            # Convert Dataframes to NumPy arrays
            result_array_ =  fixedlen_time_we_want.to_numpy()      #(number of frames, 54 features):
            
            #input data
            # The current vector features include start/end coordinates, etc. (x_sta, y_sta, x_end, y_end, time_stamp, polyline index)
            frame_len = result_array_.shape[0]   #25(2.5 s)      50(5 s)
            divide_loc = int(frame_len * divide_prop)  #10(2.5 s)      20(5 s)
            
            # Normalize the coordinates to be centered around the location of the target agent at its last observed time step
            result_array = normalized_traj(result_array_, max_sv_num, divide_loc) 
            
            # Replace NaN values with 0; be careful to replace after normalization
            result_array = np.nan_to_num(result_array, nan=0)  

            single_exa_ls = []
            edge_idx_sou_ls = []
            edge_idx_tar_ls = []
            valid_veh_ls = []
            tra_num = 0
            polyline_num = 0
            # Total number of vectors for all trajectories
            total_tra_vec_len = 0
            # Total number of vectors for all trajectories + lane lines
            total_vec_len = 0
            # The dividing indices for storing vehicle and lane information, and for lane and padding information
            divide_row_idx = [0, 0]
            #----------trajectories
            for veh in range(MAX_SV+1):
                single_tra_input = result_array[:F_HIS, 6*veh:6*veh+6].copy()  #(60, 6)
                # Take all valid rows and keep row indices: remove empty parts in SV data with incomplete information for the whole period
                non_zero_rows = np.array( [row for index, row in enumerate(single_tra_input) if not np.all(row == 0)] )
                non_zero_rows_indices = np.array([index for index, row in enumerate(single_tra_input) if not np.all(row == 0)] )
                if non_zero_rows.shape[0] < 2:   # If all are 0, skip to the next loop, i.e., this SV does not exist; set to 2 because at least two rows are needed to generate a vector
                    continue
                #******** Generate vector/node features
                feat_coor1 = non_zero_rows[:-1]  #(29,6 )
                feat_coor2 = non_zero_rows[1: ]
                feat_coor = np.concatenate((feat_coor1, feat_coor2), axis = 1)     #(x -1, 12)
                feat_timeid = non_zero_rows_indices[1:].reshape(-1, 1)    #  (8,  1)  Use the local frame id of the target node in the vector as a timestamp
                feat_vehid = np.ones((feat_coor.shape[0], 1)) if veh == 0 else np.ones((feat_coor.shape[0], 1))*2    #ev is 1, other sv are 2, lane line is 3
                feat_polyid = np.ones((feat_coor.shape[0], 1)) * (polyline_num+1)  # Accumulated relationship
                single_tra_feats = np.concatenate((feat_coor, feat_timeid, feat_vehid, feat_polyid), axis = 1)   #(x - 1, 12+3)
                
                #******** Generate edge_idx
                # For an agent's trajectory, here we do not use the approach in the original paper of connecting every point, but only connect adjacent trajectory points
                edge_idx_sou = np.array([ i for i in range(feat_coor.shape[0] - 1)]) + total_tra_vec_len

                edge_idx_tar = np.array([ j+1 for j in range(feat_coor.shape[0] - 1)]) + total_tra_vec_len
                edge_idx_sou_ls.append(edge_idx_sou)
                edge_idx_tar_ls.append(edge_idx_tar)
                
                #update
                single_exa_ls.append(single_tra_feats)
                tra_num += 1
                polyline_num += 1
                total_tra_vec_len += feat_coor.shape[0]
                total_vec_len += feat_coor.shape[0]
                divide_row_idx[0] += single_tra_feats.shape[0]
                valid_veh_ls.append(veh)
                    
            #----------lane lines
            # Lane line sampling vectors, features need to be aligned with the vector feature length in agent trajectories: (x_sta, y_sta, x_end, y_end, time_stamp = 0, polyid = lane index, polyline type = 1)      polyline type: 0 for tra, 1 for lane line, 2..
            lane_array, lane_ys_nonorm  = get_lane_samples(recordingMeta, result_array_[F_HIS-1, 0], result_array_[F_HIS-1, 1])
            # Combine the sampling points to the source-end vector
            for line in range(lane_array.shape[0]):
                #******** Generate vector/node features
                lane_feat_coor1 = lane_array[line, :-1, :]   #(99, 6)
                lane_feat_coor2 = lane_array[line, 1:, :]
                lane_feat_coor =  np.concatenate((lane_feat_coor1, lane_feat_coor2), axis=1)    #(99, 4)  
                lane_timeid =  np.zeros(( lane_feat_coor.shape[0], 1) )    #(99, 1)
                lane_lineid =  np.ones((lane_feat_coor.shape[0], 1) )  * 3        #(line+1)   #local id:  1, 2, 3,...  from left to right
                lane_polyid =  np.ones((lane_feat_coor.shape[0], 1)) * (polyline_num+1)   #object index
                single_lane_feats = np.concatenate( (lane_feat_coor, lane_timeid, lane_lineid, lane_polyid), axis=1 )  #(99, 7)  
                
                #******** Generate edge_idx
                edge_idx_sou = np.array([ i for i in range(lane_feat_coor.shape[0] - 1)]) + total_vec_len
                edge_idx_tar = np.array([ j+1 for j in range(lane_feat_coor.shape[0] - 1)]) + total_vec_len
                edge_idx_sou_ls.append(edge_idx_sou)
                edge_idx_tar_ls.append(edge_idx_tar)
                
                #update
                single_exa_ls.append(single_lane_feats)
                polyline_num += 1
                total_vec_len += lane_feat_coor.shape[0]    #graph global

            #graph edge data
            edge_idx = np.array([np.concatenate(edge_idx_sou_ls, axis = 0), np.concatenate(edge_idx_tar_ls, axis = 0) ] )
            #graph node data and prediction gt
            single_data = np.concatenate(single_exa_ls, axis = 0)      # (batch_size, 15)
            # single_label = result_array[divide_loc-1:, 2]    #(16, ）
            # single_label = np.array([ single_label_[i] - single_label_[i-1] for i in range(1, single_label_.shape[0]) ])    #(15, ）
            single_label = result_array[divide_loc:, 2]    #(16, ）
            
            # ev_fur_pos_gt = result_array[ F_HIS:, :2]   #(30, 2）  
            divide_row_idx[1] = single_data.shape[0]
            
            #******** Generate identifier, used to give a marker to masked out polylines for the graph completion task
            # Each value in identifier represents: draw a rectangle for each polyline, the minimum coordinate of the rectangle
            identifier = np.empty((0,2))
            for pl in np.unique(single_data[:,-1]):
                [indices] = np.where(single_data[:, -1] == pl)
                identifier = np.vstack([identifier,  np.min(single_data[indices, :2], axis = 0) ]  )  # axis = 0 means find the minimum in the first dimension, note np.min() finds the minimum in each column separately, not together
                
            #******** Generate cluster, cluster is a custom Data attribute for clustering.   
            # cluster_minib = single_data[:, -1] + cluster_global_ptr    # (379,)
            
            # Convert to graph data
            single_graph = Data(x = torch.tensor([
                                                [x_sta, y_sta, vx_sta, vy_sta, ax_sta, ay_sta, 
                                                    x_end, y_end, vx_end, vy_end, ax_end, ay_end,
                                                    time_id, loc_polyid, glo_polyid] for 
                                                    x_sta, y_sta, vx_sta, vy_sta, ax_sta, ay_sta, 
                                                    x_end, y_end, vx_end, vy_end, ax_end, ay_end, 
                                                    time_id, loc_polyid, glo_polyid in single_data]).float(),    #torch.Size([vector_num, feature_num])   
                                            edge_index = torch.from_numpy(edge_idx ).long(), 
                                            identifier = torch.from_numpy(identifier).float(),   #(valid_len, 2)
                                            cluster = torch.from_numpy(single_data[:, -1].copy() ).long(),   #short()/int16() max value 2^15 - 1   
                                            traj_len = torch.tensor([tra_num]).int(),   #int() converts data type to torch.int32
                                            valid_len = torch.tensor([polyline_num]).int(),
                                            max_valid_len = torch.tensor([MAX_SV+1+MAX_LINE]).int(),  #13
                                            y = torch.tensor([v_fut for v_fut in single_label]).float(),   #[40, 2]
                                            #divide_loc-1 represents the last frame in historical data; 0/2/4 represent x, vx, ax
                                            x_t0 = torch.tensor([result_array[divide_loc-1, 0], result_array[divide_loc-1, 2], result_array[divide_loc-1, 4]]),
                                            pos_x_gt  = torch.tensor(np.array( [result_array[divide_loc:, 0] - result_array[divide_loc-1, 0] ] ) ),   # In fact, result_array[divide_loc-1, 0] is all 0 after centering
                                            
                                            valid_sv_idxs = torch.tensor(valid_veh_ls).int(),  # Store the SV ids present in this graph data (including EV)
                                            divide_row_idx = torch.tensor(divide_row_idx).int(),  # The row index in the graph data for the first row of lane line data and the first row of padding data
                                            # Record the y-axis values of lane lines in the original coordinate system (global) in the current record
                                            lane_ys_nonorm = torch.tensor( lane_ys_nonorm ),
                                            # The coordinates of the EV at the current moment during actual execution, i.e., the origin of the local coordinate system in the global coordinate system
                                            xys_t0_nonorm = torch.tensor([result_array_[F_HIS-1, 0], result_array_[F_HIS-1, 1]])
                                            )   
            
            # Padding for aligning the number of polylines in the same graph
            if single_graph.max_valid_len[0].item() > polyline_num:
                single_graph = get_padding_graph(single_graph) # Zero padding
                
            #******** Generate cluster, cluster is a custom Data attribute for clustering.   
            single_graph.cluster += cluster_global_ptr
            # Update the cluster_global_ptr
            cluster_global_ptr += (1+MAX_SV+MAX_LINE)
                
            # #check the representation of input data
            # if tra_id%17 == 0 and t > 50 and t < 70:
            #     visualize_graph(single_graph)
                
            graph_ls.append(single_graph)

    return graph_ls, cluster_global_ptr


def get_features_we_want(pd_df):
    res = pd_df[[
            'x', 'y', 'xVelocity',  'yVelocity',  'xAcceleration', 'yAcceleration', 
            'preced_x', 'preced_y', 'preced_vx', 'preced_vy', 'preced_ax', 'preced_ay', 
            'follow_x', 'follow_y','follow_vx', 'follow_vy', 'follow_ax', 'follow_ay', 
            'leftPreced_x','leftPreced_y', 'leftPreced_vx', 'leftPreced_vy', 'leftPreced_ax','leftPreced_ay', 
            'leftAlongs_x', 'leftAlongs_y', 'leftAlongs_vx','leftAlongs_vy', 'leftAlongs_ax', 'leftAlongs_ay', 
            'leftFollow_x','leftFollow_y', 'leftFollow_vx', 'leftFollow_vy', 'leftFollow_ax','leftFollow_ay', 
            'rightPreced_x', 'rightPreced_y', 'rightPreced_vx', 'rightPreced_vy', 'rightPreced_ax', 'rightPreced_ay',
            'rightAlongs_x', 'rightAlongs_y', 'rightAlongs_vx', 'rightAlongs_vy', 'rightAlongs_ax', 'rightAlongs_ay', 
            'rightFollow_x', 'rightFollow_y', 'rightFollow_vx', 'rightFollow_vy', 'rightFollow_ax', 'rightFollow_ay'
                                        ]]
    
    return res
    
def get_frequency_we_want(pd_df):
    # Use pandas to rebuild the timestamp; starting time is created randomly
    time_index = pd.date_range(start='2024-01-01 00:00:00', periods=pd_df.shape[0], freq='0.04s')    # Divide row indices into two groups, step size 5; need to reorder from 0, otherwise the first batch may not be 5 frames
    # Set as the index of the original dataframe
    pd_df.index = time_index
    # Downsample  0.04 s --> 0.1 s
    res = pd_df.resample('0.1s').mean()
    return res

def get_fixedlen_time_we_want(pd_df, sta, end):
    return pd_df.iloc[sta: end]


def get_padding_graph(data):
    feature_len = data.x.shape[1]
    prt_incre = data.max_valid_len[0].item() - data.valid_len[0].item()

    # pad feature with zero nodes
    data.x = torch.cat([data.x, torch.zeros((prt_incre, feature_len), dtype=data.x.dtype)])
    data.cluster = torch.cat([data.cluster, torch.arange( data.valid_len[0].item()+1 , data.max_valid_len[0].item()+1, dtype=data.cluster.dtype)]).long()
    data.identifier = torch.cat([data.identifier, torch.zeros((prt_incre, 2), dtype=data.identifier.dtype)])

    assert data.cluster.shape[0] == data.x.shape[0], "[ERROR]: Loader error!"

    return data


def get_lane_samples(rdMeta_df, EV_x, EV_y):  # Forward sampling distance = N_LINE_SAM*INTVAL_LINE_SAM
    lane_markings = get_lane_markings(rdMeta_df)
    # Sample x-axis length 200m, sampling interval 1m, i.e., each lane includes 200 sampling points
    # The scenario in highd data is simple, the number of lanes in each example is fixed, e.g., 3 lanes for 2-lane roads, 4 lanes for 3-lane roads
    lane_samples = []
    # According to the parameter norm, decide whether to normalize the coordinates of the line sampling points
    EV_x = 0 if IF_NORM else EV_x
    EV_y = EV_y if IF_NORM else 0
    for lane_y in lane_markings:
        lane_samples.append([ [x for x in range(int(EV_x), int(EV_x) + N_LINE_SAM*INTVAL_LINE_SAM, INTVAL_LINE_SAM)],  
                                                            [lane_y - EV_y for _ in range(N_LINE_SAM)], 
                                                            [0 for _ in range(N_LINE_SAM)], [0 for _ in range(N_LINE_SAM)], 
                                                            [0 for _ in range(N_LINE_SAM)], [0 for _ in range(N_LINE_SAM)]   ])   # Except for x and y, other v/a are 0
    
    res = np.array( lane_samples )   
    res = res.transpose(0, 2, 1)  # Swap the last two dimensions
    return res, lane_markings      #(lane_num, 100, 6)


def get_lane_markings(recordingMeta_df):
    lo_markings = recordingMeta_df['lowerLaneMarkings']
    # Use str.split() to split by semicolon into a list  
    split_lists_lo = lo_markings.str.split(';')  
    # Convert each string to float  
    float_lists_lo = split_lists_lo.apply(lambda x: [float(i) for i in x]).iloc[0]  
    lo_markings_ls = float_lists_lo
    return  lo_markings_ls
    
def normalized_traj(raw_array, sv_num, divide_loc ):  #( 50, 54)
    res_array = raw_array.copy()
    center_coord = res_array[divide_loc-1, :2]  #(2,)
    cols_need_norm_x = np.array([6*i  for i in range(sv_num+1)])
    cols_need_norm_y = np.array([6*i+1  for i in range(sv_num+1)])
    for frame in range(raw_array.shape[1]):
        res_array[:, cols_need_norm_x] = res_array[:, cols_need_norm_x]  - center_coord[0]
        res_array[:, cols_need_norm_y] = res_array[:, cols_need_norm_y]  - center_coord[1]
    return res_array  #( 50, 54)



if __name__ == "__main__":
    # The proportion of input historical data in a complete trajectory
    divide_prop = 0.4    # 0.375---predict  5s    0.6----predict  3.2s     0.4----predict  4.8s

    sta_t = time.time()

    pt_dataset_savepath = 'velocity_predicition/test_PyG'
    # If the data has already been processed and saved as .pt files, you can directly use the MyOwnDataset class and do not need to run the following line
    STA_REC_IDX, END_REC_IDX = 4,5     #the starting index and ending index of recording in the "data_pre" function.
    train_loader, dev_loader, test_loader  = data_pre(pt_dataset_savepath, STA_REC_IDX, END_REC_IDX)  
    
    
    end_t = time.time()
    print(f"Runtime: {end_t - sta_t:.2f} 秒")



