'''
created on 12th July 2024
@author: yujia lu

[v]更改时间序列中每个元素之间的时间间隔0.2-->0.1 sec
after 2024.12.03: 
[v]修改输出内容为纵向位置间，delta s，为了方便设置限速
[v]优化图神经网络的输入格式：unstructured的图数据和structured的二维数组之间的转换; 图网络子图聚集参数cluster的迭代
[v]重新修改输出内容为离散的速度。  2024.12.19
[v]扩展数据集，将每个场景拆分为多条轨迹数据（沿时间轴滑动）。  2024.12.29
[v]由于数据集扩展后超出list存储体量，修改数据读取存储格式为PyG自带的DataSet   2025.1.2
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
    def __init__(self, root, idx=0, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False)
        #是在存储完毕.pt文件后执行这一步，注意如果检查到完全同名的processed_file_names（只要包括）可以就不会执行process()了
        #weights_only=False: PyTorch 2.6 开始默认更严格了，不允许直接反序列化这些对象，除非显式允许

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"record{rid:02d}.pt" for rid in REC_ID_LIST]

    def download(self):
        pass

    def process(self):
        #------------- Read data into huge `Data` list.
        #
        max_sv_num = 8
        max_lane_num = 4
            
        # 读取处理后数据
        files_ = os.listdir(read_dir_dp)  #60*4
        # 使用正则表达式匹配以'.~lock.'开头，并且以'#'结尾的文件  ,过滤掉临时文件
        pattern = r'^\.~lock.*#$'  
        files = [file for file in files_ if not re.match(pattern, file)]
        files.sort()  

        cluster_global_ptr = 0  # 全局 polyline 指针

        for idx, rid in enumerate(REC_ID_LIST):
            print(f"Processing recording {rid:02d}")

            graph_ls = []   # ✅ 必须初始化

            # ---------- 找对应的 csv ----------
            csv_name = [f for f in files if f.endswith(f"_{int(rid):02d}.csv")][0]
            tracks_df = pd.read_csv(os.path.join(read_dir_dp, csv_name), index_col=0)

            # ---------- 原始 highD 元数据 ----------
            rMetafile_name = f"{rid:02d}_recordingMeta.csv"
            tMetafile_name = f"{rid:02d}_tracksMeta.csv"

            recordingMeta = pd.read_csv(os.path.join(read_dir_hd, rMetafile_name))
            tracksMeta = pd.read_csv(os.path.join(read_dir_hd, tMetafile_name))

            # ---------- 核心：构图 ----------
            graph_ls, cluster_global_ptr = get_single_record_data(
                tracks_df,
                graph_ls,
                recordingMeta,
                tracksMeta,
                max_sv_num,
                max_lane_num,
                cluster_global_ptr,
                divide_prop,
                if_test4DE=False
            )

            assert cluster_global_ptr % 13 == 0

            # ---------- 保存 ----------
            torch.save(
                self.collate(graph_ls),
                self.processed_paths[idx]
            )
        

# def data_pre(divide_prop, read_dir2, if_test4DE=False):
#     sta_t = time.time()
#     # STA_REC_IDX, END_REC_IDX = 53, 54
#     dataset_input_path = "E:/UTC_PHD/Compared1_VN_trajectory_prediction/DataSets/PyG_DataSet_demo"    #记得换回来！！！
#     # dataset_input_path = '/home/chwei/AutoVehicle_DataAndOther/myData/VectorNetResults/VN5/results1219/PyG_DataSet_3_54'
#     datasets_ls = []
#     for i in range(STA_REC_IDX, END_REC_IDX):
#         datasets_ls.append(MyOwnDataset(root=dataset_input_path)  )
        
#     final_dataset = ConcatDataset(datasets_ls)
#     train_loc, vali_loc= int(0.7*len(final_dataset) ), int( 0.2*len(final_dataset) )
#     test_loc = len(final_dataset) - train_loc - vali_loc
#     # train_valid_prop = 0.9
#     train_dataset, validation_dataset, test_dataset = random_split(final_dataset, [train_loc, vali_loc, test_loc], generator=torch.Generator().manual_seed(42))  #设置随机种子，方便复现
    

#     mini_batch_size = 128
#     train_loader = DataLoader( train_dataset, batch_size=mini_batch_size, shuffle=True )
#     dev_loader = DataLoader( validation_dataset, batch_size=mini_batch_size, shuffle=True )
#     test_loader = DataLoader( test_dataset, batch_size=mini_batch_size, shuffle=False )    
    
#     # for batch in batch_iter:
#     #     print(batch)
        
    
#     print("the total number of trajectory: %d"%len(final_dataset))
    
#     end_t = time.time()
#     print(f"数据处理运行时间: {end_t - sta_t:.2f} 秒")
    
#     if if_test4DE:
#         total_loader = DataLoader(final_dataset, batch_size=mini_batch_size, shuffle=False ) 
#         return total_loader   

#     return train_loader, dev_loader, test_loader   


def get_single_record_data(tracks_df, graph_ls, recordingMeta, tracksMeta, max_sv_num, max_lane_num,  cluster_global_ptr, divide_prop, if_test4DE = False):
    # 按 'id' 分组
    grouped = tracks_df .groupby('id')

    # 初始化一个空列表来存储每个分组的数据
    group_data = []
    group_id = []
    # 遍历每个分组
    for tra_id, group in grouped:
        group_id.append(tra_id)
        
        total_frame_len = int(0.04*group.shape[0]/SPF)  #当前场景EV轨迹最大时长, 频率从0.04转换为0.1，0.04s是原始时序数据的频率。向下取整
        fixed_len = F_HIS+F_FUR  #我们需要的EV轨迹时长对应的帧数
        
        for t in range(0, total_frame_len - fixed_len + 1 , 10):  #滑动次数
            if t+fixed_len > total_frame_len:
                break    #不会进去
                
            features_we_want = get_features_we_want(group)
            
            frequency_we_want = get_frequency_we_want(features_we_want)
            
            fixedlen_time_we_want = get_fixedlen_time_we_want(frequency_we_want, sta = t, end = t+fixed_len)  # 5 sec

            # 将Dataframes转换为NumPy 数组
            result_array_ =  fixedlen_time_we_want.to_numpy()      #(number of frames, 54 features):
            
            #input data
            #当前vector的特征包括起终点坐标等(x_sta, y_sta, x_end, y_end, time_stamp, polyline index)
            frame_len = result_array_.shape[0]   #25(2.5 s)      50(5 s)
            divide_loc = int(frame_len * divide_prop)  #10(2.5 s)      20(5 s)
            
            #normalize the coordinates to be centered around the location of target agent at its last observed time step
            result_array = normalized_traj(result_array_, max_sv_num, divide_loc) 
            
            # Replace NaN values with 0; Cared to replace after the normalization
            result_array = np.nan_to_num(result_array, nan=0)  

            
            
            single_exa_ls = []
            edge_idx_sou_ls = []
            edge_idx_tar_ls = []
            valid_veh_ls = []
            tra_num = 0
            polyline_num = 0
            #所有轨迹的vector总数
            total_tra_vec_len = 0
            #所有轨迹+车道线的vector总数
            total_vec_len = 0
            #分别存储vehicle信息和lane信息的分界处、lane信息与padding信息的分界处
            divide_row_idx = [0, 0]
            #----------trajectories
            for veh in range(MAX_SV+1):
                single_tra_input = result_array[:F_HIS, 6*veh:6*veh+6].copy()  #(60, 6)
                #取所有有效的行数据并保留行索引  ：删去全时段信息不全的SV数据中的空值部分
                non_zero_rows = np.array( [row for index, row in enumerate(single_tra_input) if not np.all(row == 0)] )
                non_zero_rows_indices = np.array([index for index, row in enumerate(single_tra_input) if not np.all(row == 0)] )
                if non_zero_rows.shape[0] < 2:   #如果全部为0，则直接跳入下次循环，即不存在该位置的SV；设置为2是因为至少需要两行信息来生成一个vector
                    continue
                #********生成vector/node features
                feat_coor1 = non_zero_rows[:-1]  #(29,6 )
                feat_coor2 = non_zero_rows[1: ]
                feat_coor = np.concatenate((feat_coor1, feat_coor2), axis = 1)     #(x -1, 12)
                feat_timeid = non_zero_rows_indices[1:].reshape(-1, 1)    #  (8,  1)  取vector中target node的local frame id作为时间戳标记
                feat_vehid = np.ones((feat_coor.shape[0], 1)) if veh == 0 else np.ones((feat_coor.shape[0], 1))*2    #ev是1  其他sv都是2  lane line 是3
                feat_polyid = np.ones((feat_coor.shape[0], 1)) * (polyline_num+1)  #累加关系
                single_tra_feats = np.concatenate((feat_coor, feat_timeid, feat_vehid, feat_polyid), axis = 1)   #(x - 1, 12+3)
                
                #********生成edge_idx
                #对于一个agent的轨迹，这里没有采用原文中对每个点间都建立边的做法，而是相邻轨迹点进行联系
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
            #车道线采样vector，特征需与agent轨迹中vector特征长度对齐：(x_sta, y_sta, x_end, y_end, time_stamp = 0, polyid = lane index, polyline type = 1)      polyline type：0为tra，1为lane line，2..
            lane_array, lane_ys_nonorm  = get_lane_samples(recordingMeta, result_array_[F_HIS-1, 0], result_array_[F_HIS-1, 1])
            #combine the sampling points to the source-end vector
            for line in range(lane_array.shape[0]):
                #********生成vector/node features
                lane_feat_coor1 = lane_array[line, :-1, :]   #(99, 6)
                lane_feat_coor2 = lane_array[line, 1:, :]
                lane_feat_coor =  np.concatenate((lane_feat_coor1, lane_feat_coor2), axis=1)    #(99, 4)  
                lane_timeid =  np.zeros(( lane_feat_coor.shape[0], 1) )    #(99, 1)
                lane_lineid =  np.ones((lane_feat_coor.shape[0], 1) )  * 3        #(line+1)   #local id:  1, 2, 3,...  从左至右
                lane_polyid =  np.ones((lane_feat_coor.shape[0], 1)) * (polyline_num+1)   #object index
                single_lane_feats = np.concatenate( (lane_feat_coor, lane_timeid, lane_lineid, lane_polyid), axis=1 )  #(99, 7)  
                
                #********生成edge_idx
                edge_idx_sou = np.array([ i for i in range(lane_feat_coor.shape[0] - 1)]) + total_vec_len
                edge_idx_tar = np.array([ j+1 for j in range(lane_feat_coor.shape[0] - 1)]) + total_vec_len
                edge_idx_sou_ls.append(edge_idx_sou)
                edge_idx_tar_ls.append(edge_idx_tar)
                
                #update
                single_exa_ls.append(single_lane_feats)
                polyline_num += 1
                total_vec_len += lane_feat_coor.shape[0]    #graph global

            #graph 边数据
            edge_idx = np.array([np.concatenate(edge_idx_sou_ls, axis = 0), np.concatenate(edge_idx_tar_ls, axis = 0) ] )
            #graph 节点数据和预测gt
            single_data = np.concatenate(single_exa_ls, axis = 0)      # (batch_size, 15)
            # single_label = result_array[divide_loc-1:, 2]    #(16, ）
            # single_label = np.array([ single_label_[i] - single_label_[i-1] for i in range(1, single_label_.shape[0]) ])    #(15, ）
            single_label = result_array[divide_loc:, 2]    #(16, ）
            
            # ev_fur_pos_gt = result_array[ F_HIS:, :2]   #(30, 2）  
            divide_row_idx[1] = single_data.shape[0]
            
            #********生成identifier，用于完成graph completion task时给被masked out的polyline一个标记
            #identifier中每个值代表：包含一个polyline画一个矩形， 矩形的最小坐标
            identifier = np.empty((0,2))
            for pl in np.unique(single_data[:,-1]):
                [indices] = np.where(single_data[:, -1] == pl)
                identifier = np.vstack([identifier,  np.min(single_data[indices, :2], axis = 0) ]  )  # axis = 0表示在第一维度中找出最小的，注意np.min()分别找出两列中的最小值，并没有捆绑在一起
                
            #********生成cluster，cluster是自定义的图Data属性，用于聚类。   
            # cluster_minib = single_data[:, -1] + cluster_global_ptr    # (379,)
            
            #转为graph data
            single_graph = Data(x = torch.tensor([
                                                [x_sta, y_sta, vx_sta, vy_sta, ax_sta, ay_sta, 
                                                    x_end, y_end, vx_end, vy_end, ax_end, ay_end,
                                                    time_id, loc_polyid, glo_polyid] for 
                                                    x_sta, y_sta, vx_sta, vy_sta, ax_sta, ay_sta, 
                                                    x_end, y_end, vx_end, vy_end, ax_end, ay_end, 
                                                    time_id, loc_polyid, glo_polyid in single_data]).float(),    #torch.Size([vector_num, feature_num])   
                                            edge_index = torch.from_numpy(edge_idx ).long(), 
                                            identifier = torch.from_numpy(identifier).float(),   #(valid_len, 2)
                                            cluster = torch.from_numpy(single_data[:, -1].copy() ).long(),   #short()/int16()最大值2^15 - 1   
                                            traj_len = torch.tensor([tra_num]).int(),   #int()将数据类型转为torch.int32
                                            valid_len = torch.tensor([polyline_num]).int(),
                                            max_valid_len = torch.tensor([MAX_SV+1+MAX_LINE]).int(),  #13
                                            y = torch.tensor([v_fut for v_fut in single_label]).float(),   #[40, 2]
                                            #divide_loc-1代表historical数据中最后一帧；0/2/4分别表示x，vx，ax
                                            x_t0 = torch.tensor([result_array[divide_loc-1, 0], result_array[divide_loc-1, 2], result_array[divide_loc-1, 4]]),
                                            pos_x_gt  = torch.tensor(np.array( [result_array[divide_loc:, 0] - result_array[divide_loc-1, 0] ] ) ),   #事实上result_array[divide_loc-1, 0]经过中心化全是0
                                            
                                            valid_sv_idxs = torch.tensor(valid_veh_ls).int(),  #存储该图数据中存在的SV对应的SV id（包含EV）
                                            divide_row_idx = torch.tensor(divide_row_idx).int(),  #第一行lane line数据在图数据中的行索引、第一行padding数据在图数据中的行索引
                                            #记录当前record中原始坐标系(global)下的车道线们的y轴坐标值
                                            lane_ys_nonorm = torch.tensor( lane_ys_nonorm ),
                                            #EV实际执行过程中 当前时刻的坐标，是local坐标系在global坐标系中的原点坐标
                                            xys_t0_nonorm = torch.tensor([result_array_[F_HIS-1, 0], result_array_[F_HIS-1, 1]])
                                            )   
            
            #padding for aligning the number of polylines in a same graph
            if single_graph.max_valid_len[0].item() > polyline_num:
                single_graph = get_padding_graph(single_graph) #全零填充
                
            #********生成cluster，cluster是自定义的图Data属性，用于聚类。   
            single_graph.cluster += cluster_global_ptr
            #update the cluster_global_ptr
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
    #利用pandas库重建时间戳；起始时间随机创建
    time_index = pd.date_range(start='2024-01-01 00:00:00', periods=pd_df.shape[0], freq='0.04s')    # 将行索引分为两组，步长为5；需要重新从0开始排序，否则第一批次可能不是5 frames
    #设置为原始dataframe的索引
    pd_df.index = time_index
    #进行下采样  0.04 s --> 0.1 s
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


def get_lane_samples(rdMeta_df, EV_x, EV_y):  #向前采样距离 = N_LINE_SAM*INTVAL_LINE_SAM
    lane_markings = get_lane_markings(rdMeta_df)
    #采样x轴长度200m，采样间距为1m，即每个lane包括200个采样点
    #highd数据的场景简单，每个example里lane的数量是固定的，比如2车道为3 lanes，3车道为4 lanes
    lane_samples = []
    #依据参数norm决定line 采样点坐标是否需要归一化处理
    EV_x = 0 if IF_NORM else EV_x
    EV_y = EV_y if IF_NORM else 0
    for lane_y in lane_markings:
        lane_samples.append([ [x for x in range(int(EV_x), int(EV_x) + N_LINE_SAM*INTVAL_LINE_SAM, INTVAL_LINE_SAM)],  
                                                            [lane_y - EV_y for _ in range(N_LINE_SAM)], 
                                                            [0 for _ in range(N_LINE_SAM)], [0 for _ in range(N_LINE_SAM)], 
                                                            [0 for _ in range(N_LINE_SAM)], [0 for _ in range(N_LINE_SAM)]   ])   #除了x和y，其他v/a都是0
    
    res = np.array( lane_samples )   
    res = res.transpose(0, 2, 1)  #交换后两维的顺序
    return res, lane_markings      #(lane_num, 100, 6)


def get_lane_markings(recordingMeta_df):
    lo_markings = recordingMeta_df['lowerLaneMarkings']
    # 使用str.split()按分号拆分为列表  
    split_lists_lo = lo_markings.str.split(';')  
    # 转换每个字符串为浮点数  
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
    #the proportion of input historical data in a complete trajectory
    divide_prop = 0.4    # 0.375---predict  5s    0.6----predict  3.2s     0.4----predict  4.8s
    
    STA_REC_IDX, END_REC_IDX = 0, 2  #全局常量 必须写在main里   3, 53

    sta_t = time.time()

    REC_ID_LIST = list(range(55, 60))    # 防止序号错乱      #!!!

    datasets_ls = []
    for i in range(len(REC_ID_LIST)):
        datasets_ls.append(MyOwnDataset(root=read_dir_pyg, idx=i))
    final_dataset = ConcatDataset(datasets_ls)
        
    # train_loc, vali_loc= int(0.7*len(final_dataset) ), int( 0.2*len(final_dataset) )
    # test_loc = len(final_dataset) - train_loc - vali_loc
    # # train_valid_prop = 0.9
    # train_dataset, validation_dataset, test_dataset = random_split(final_dataset, [train_loc, vali_loc, test_loc], generator=torch.Generator().manual_seed(42))  #设置随机种子，方便复现
    

    # mini_batch_size = 128
    # train_loader = DataLoader( train_dataset, batch_size=mini_batch_size, shuffle=True )
    # dev_loader = DataLoader( validation_dataset, batch_size=mini_batch_size, shuffle=True )
    # test_loader = DataLoader( test_dataset, batch_size=mini_batch_size, shuffle=False )    
    
    # for batch in batch_iter:
    #     print(batch)
        
    
    print("the total number of trajectory: %d"%len(final_dataset))
    
    end_t = time.time()
    print(f"数据处理运行时间: {end_t - sta_t:.2f} 秒")
    
    # if if_test4DE:
    #     total_loader = DataLoader(final_dataset, batch_size=mini_batch_size, shuffle=False ) 
    #     return total_loader   

    # return train_loader, dev_loader, test_loader   
    
    
    
'''
array([[ -43.385     ,   46.44166667, -101.66333333,  -10.215     ,
                  nan, -185.59833333,           nan,           nan,
                  nan],
       [ -41.185     ,   48.745     ,  -99.44      ,   -7.67      ,
                  nan, -182.405     ,           nan,           nan,
                  nan],
       [ -38.915     ,   51.155     ,  -97.17333333,   -4.33833333,
                  nan, -179.07166667,           nan,           nan,
                  nan],
       [ -36.635     ,   53.59      ,  -94.855     ,   -0.945     ,
                  nan, -175.72      ,           nan,           nan,
                  nan],
       [ -34.35166667,   55.99833333,  -92.52      ,    2.455     ,
                  nan, -172.365     ,           nan,           nan,
                  nan],
       [ -32.065     ,   58.395     ,  -90.21      ,    5.855     ,
                  nan, -169.01      ,           nan,           nan,
                  nan],
       [ -29.77833333,   60.80166667,  -87.90666667,    9.245     ,
                  nan, -165.645     ,           nan,           nan,
                  nan],
       [ -27.495     ,   63.22      ,  -85.61      ,   12.63      ,
                  nan, -162.295     ,           nan,           nan,
                  nan],
       [ -25.19833333,   65.64166667,  -83.29      ,   16.03833333,
                  nan, -158.945     ,           nan,           nan,
                  nan],
       [ -22.915     ,   68.035     ,  -80.985     ,   19.45      ,
                  nan, -155.61      ,           nan,           nan,
                  nan],
       [ -20.62166667,   70.43833333,  -78.69      ,   22.83833333,
                  nan, -152.265     ,           nan,           nan,
                  nan],
       [ -18.315     ,   72.87      ,  -76.395     ,   26.22      ,
                  nan, -148.905     ,           nan,           nan,
                  nan],
       [ -16.015     ,   75.30166667,  -74.09      ,   29.60833333,
                  nan, -145.55833333,           nan,           nan,
                  nan],
       [ -13.725     ,   77.705     ,  -71.78      ,   33.01      ,
                  nan, -142.215     ,           nan,           nan,
                  nan],
       [ -11.43833333,   80.10833333,  -69.5       ,   36.38833333,
                  nan, -138.895     ,           nan,           nan,
                  nan],
       [  -9.15      ,   82.525     ,  -67.22      ,   39.785     ,
                  nan, -135.58      ,           nan,           nan,
                  nan],
       [  -6.85833333,   84.93833333,  -64.94333333,   43.19166667,
                  nan, -132.265     ,           nan,           nan,
                  nan],
       [  -4.58      ,   87.35      ,  -62.68      ,   46.585     ,
                  nan, -128.945     ,           nan,           nan,
                  nan],
       [  -2.295     ,   89.78833333,  -60.39333333,   49.93166667,
                  nan, -125.62166667,           nan,           nan,
                  nan],
       [   0.        ,   92.23      ,  -58.115     ,   53.295     ,
                  nan, -122.29      ,           nan,           nan,
                  nan],
       [   2.29166667,   94.665     ,  -55.85      ,   56.69166667,
                  nan, -118.95833333,           nan,           nan,
                  nan],
       [   4.565     ,   97.1       ,  -53.595     ,   60.08      ,
                  nan, -115.64      ,           nan,           nan,
                  nan],
       [   6.82833333,   99.535     ,  -51.37      ,   63.435     ,
                  nan, -112.31833333,           nan,           nan,
                  nan],
       [   9.11      ,  101.965     ,  -49.145     ,   66.805     ,
                  nan, -108.995     ,           nan,           nan,
                  nan],
       [  11.40833333,  104.395     ,  -46.91      ,   70.15166667,
                  nan, -105.68166667,           nan,           nan,
                  nan],
       [  13.7       ,  106.82      ,  -44.655     ,   73.545     ,
                  nan, -102.375     ,           nan,           nan,
                  nan],
       [  15.97166667,  109.235     ,  -42.38      ,   76.955     ,
                  nan,  -99.085     ,           nan,           nan,
                  nan],
       [  18.25      ,  111.655     ,  -40.105     ,   80.34      ,
                  nan,  -95.8       ,           nan,           nan,
                  nan],
       [  20.52833333,  114.065     ,  -37.86333333,   83.715     ,
                  nan,  -92.49166667,           nan,           nan,
                  nan],
       [  22.785     ,  116.43      ,  -35.615     ,   87.075     ,
                  nan,  -89.17      ,           nan,           nan,
                  nan],
       [  25.04833333,  118.69166667,  -33.34      ,   90.455     ,
                  nan,  -85.875     ,           nan,           nan,
                  nan],
       [  27.325     ,  121.045     ,  -31.07      ,   93.845     ,
                  nan,  -82.57      ,           nan,           nan,
                  nan],
       [  29.615     ,  123.405     ,  -28.85      ,   97.22833333,
                  nan,  -79.29166667,           nan,           nan,
                  nan],
       [  31.9       ,  125.775     ,  -26.625     ,  100.61      ,
                  nan,  -76.025     ,           nan,           nan,
                  nan],
       [  34.17166667,  128.13833333,  -24.38333333,  103.99166667,
                  nan,  -72.75833333,           nan,           nan,
                  nan],
       [  36.43      ,  130.495     ,  -22.125     ,  107.375     ,
                  nan,  -69.48      ,           nan,           nan,
                  nan],
       [  38.695     ,  132.855     ,  -19.87333333,  110.735     ,
                  nan,  -66.17833333,           nan,           nan,
                  nan],
       [  40.96      ,  135.205     ,  -17.65      ,  114.085     ,
                  nan,  -62.875     ,           nan,           nan,
                  nan],
       [  43.23166667,  137.54833333,  -15.43      ,  117.46166667,
                  nan,  -59.585     ,           nan,           nan,
                  nan],
       [  45.495     ,  139.89      ,  -13.205     ,  120.815     ,
                  nan,  -56.31      ,           nan,           nan,
                  nan],
       [  47.765     ,  142.225     ,  -10.96      ,  124.19833333,
                  nan,  -53.02833333,           nan,           nan,
                  nan],
       [  50.05      ,  144.56      ,   -8.705     ,  127.6       ,
                  nan,  -49.75      ,           nan,           nan,
                  nan],
       [  52.33166667,  146.885     ,   -6.48666667,  130.96833333,
                  nan,  -46.48166667,           nan,           nan,
                  nan],
       [  54.595     ,           nan,   -4.24      ,  134.34      ,
                  nan,  -43.215     ,           nan,           nan,
                  nan],
       [  56.83833333,           nan,   -1.99333333,  137.715     ,
                  nan,  -39.955     ,           nan,           nan,
                  nan],
       [  59.075     ,           nan,    0.225     ,  140.415     ,
                  nan,  -36.68      ,           nan,           nan,
                  nan],
       [  61.34833333,           nan,    2.43333333,           nan,
                  nan,  -33.405     ,           nan,           nan,
                  nan],
       [  63.61      ,           nan,    4.675     ,           nan,
                  nan,  -30.14      ,           nan,           nan,
                  nan],
       [  65.86166667,           nan,    6.91      ,           nan,
                  nan,  -26.88833333,           nan,           nan,
                  nan],
       [  68.105     ,           nan,    9.135     ,           nan,
                  nan,  -23.615     ,           nan,           nan,
                  nan]])
'''
'''
array([[ -43.385     ,   46.44166667, -101.66333333,  -10.215     ,
                  nan, -185.59833333,           nan,           nan,
                  nan],
       [ -41.185     ,   48.745     ,  -99.44      ,   -7.67      ,
                  nan, -182.405     ,           nan,           nan,
                  nan],
       [ -38.915     ,   51.155     ,  -97.17333333,   -4.33833333,
                  nan, -179.07166667,           nan,           nan,
                  nan],
       [ -36.635     ,   53.59      ,  -94.855     ,   -0.945     ,
                  nan, -175.72      ,           nan,           nan,
                  nan],
       [ -34.35166667,   55.99833333,  -92.52      ,    2.455     ,
                  nan, -172.365     ,           nan,           nan,
                  nan],
       [ -32.065     ,   58.395     ,  -90.21      ,    5.855     ,
                  nan, -169.01      ,           nan,           nan,
                  nan],
       [ -29.77833333,   60.80166667,  -87.90666667,    9.245     ,
                  nan, -165.645     ,           nan,           nan,
                  nan],
       [ -27.495     ,   63.22      ,  -85.61      ,   12.63      ,
                  nan, -162.295     ,           nan,           nan,
                  nan],
       [ -25.19833333,   65.64166667,  -83.29      ,   16.03833333,
                  nan, -158.945     ,           nan,           nan,
                  nan],
       [ -22.915     ,   68.035     ,  -80.985     ,   19.45      ,
                  nan, -155.61      ,           nan,           nan,
                  nan],
       [ -20.62166667,   70.43833333,  -78.69      ,   22.83833333,
                  nan, -152.265     ,           nan,           nan,
                  nan],
       [ -18.315     ,   72.87      ,  -76.395     ,   26.22      ,
                  nan, -148.905     ,           nan,           nan,
                  nan],
       [ -16.015     ,   75.30166667,  -74.09      ,   29.60833333,
                  nan, -145.55833333,           nan,           nan,
                  nan],
       [ -13.725     ,   77.705     ,  -71.78      ,   33.01      ,
                  nan, -142.215     ,           nan,           nan,
                  nan],
       [ -11.43833333,   80.10833333,  -69.5       ,   36.38833333,
                  nan, -138.895     ,           nan,           nan,
                  nan],
       [  -9.15      ,   82.525     ,  -67.22      ,   39.785     ,
                  nan, -135.58      ,           nan,           nan,
                  nan],
       [  -6.85833333,   84.93833333,  -64.94333333,   43.19166667,
                  nan, -132.265     ,           nan,           nan,
                  nan],
       [  -4.58      ,   87.35      ,  -62.68      ,   46.585     ,
                  nan, -128.945     ,           nan,           nan,
                  nan],
       [  -2.295     ,   89.78833333,  -60.39333333,   49.93166667,
                  nan, -125.62166667,           nan,           nan,
                  nan],
       [   0.        ,   92.23      ,  -58.115     ,   53.295     ,
                  nan, -122.29      ,           nan,           nan,
                  nan],
       [   2.29166667,   94.665     ,  -55.85      ,   56.69166667,
                  nan, -118.95833333,           nan,           nan,
                  nan],
       [   4.565     ,   97.1       ,  -53.595     ,   60.08      ,
                  nan, -115.64      ,           nan,           nan,
                  nan],
       [   6.82833333,   99.535     ,  -51.37      ,   63.435     ,
                  nan, -112.31833333,           nan,           nan,
                  nan],
       [   9.11      ,  101.965     ,  -49.145     ,   66.805     ,
                  nan, -108.995     ,           nan,           nan,
                  nan],
       [  11.40833333,  104.395     ,  -46.91      ,   70.15166667,
                  nan, -105.68166667,           nan,           nan,
                  nan],
       [  13.7       ,  106.82      ,  -44.655     ,   73.545     ,
                  nan, -102.375     ,           nan,           nan,
                  nan],
       [  15.97166667,  109.235     ,  -42.38      ,   76.955     ,
                  nan,  -99.085     ,           nan,           nan,
                  nan],
       [  18.25      ,  111.655     ,  -40.105     ,   80.34      ,
                  nan,  -95.8       ,           nan,           nan,
                  nan],
       [  20.52833333,  114.065     ,  -37.86333333,   83.715     ,
                  nan,  -92.49166667,           nan,           nan,
                  nan],
       [  22.785     ,  116.43      ,  -35.615     ,   87.075     ,
                  nan,  -89.17      ,           nan,           nan,
                  nan],
       [  25.04833333,  118.69166667,  -33.34      ,   90.455     ,
                  nan,  -85.875     ,           nan,           nan,
                  nan],
       [  27.325     ,  121.045     ,  -31.07      ,   93.845     ,
                  nan,  -82.57      ,           nan,           nan,
                  nan],
       [  29.615     ,  123.405     ,  -28.85      ,   97.22833333,
                  nan,  -79.29166667,           nan,           nan,
                  nan],
       [  31.9       ,  125.775     ,  -26.625     ,  100.61      ,
                  nan,  -76.025     ,           nan,           nan,
                  nan],
       [  34.17166667,  128.13833333,  -24.38333333,  103.99166667,
                  nan,  -72.75833333,           nan,           nan,
                  nan],
       [  36.43      ,  130.495     ,  -22.125     ,  107.375     ,
                  nan,  -69.48      ,           nan,           nan,
                  nan],
       [  38.695     ,  132.855     ,  -19.87333333,  110.735     ,
                  nan,  -66.17833333,           nan,           nan,
                  nan],
       [  40.96      ,  135.205     ,  -17.65      ,  114.085     ,
                  nan,  -62.875     ,           nan,           nan,
                  nan],
       [  43.23166667,  137.54833333,  -15.43      ,  117.46166667,
                  nan,  -59.585     ,           nan,           nan,
                  nan],
       [  45.495     ,  139.89      ,  -13.205     ,  120.815     ,
                  nan,  -56.31      ,           nan,           nan,
                  nan],
       [  47.765     ,  142.225     ,  -10.96      ,  124.19833333,
                  nan,  -53.02833333,           nan,           nan,
                  nan],
       [  50.05      ,  144.56      ,   -8.705     ,  127.6       ,
                  nan,  -49.75      ,           nan,           nan,
                  nan],
       [  52.33166667,  146.885     ,   -6.48666667,  130.96833333,
                  nan,  -46.48166667,           nan,           nan,
                  nan],
       [  54.595     ,           nan,   -4.24      ,  134.34      ,
                  nan,  -43.215     ,           nan,           nan,
                  nan],
       [  56.83833333,           nan,   -1.99333333,  137.715     ,
                  nan,  -39.955     ,           nan,           nan,
                  nan],
       [  59.075     ,           nan,    0.225     ,  140.415     ,
                  nan,  -36.68      ,           nan,           nan,
                  nan],
       [  61.34833333,           nan,    2.43333333,           nan,
                  nan,  -33.405     ,           nan,           nan,
                  nan],
       [  63.61      ,           nan,    4.675     ,           nan,
                  nan,  -30.14      ,           nan,           nan,
                  nan],
       [  65.86166667,           nan,    6.91      ,           nan,
                  nan,  -26.88833333,           nan,           nan,
                  nan],
       [  68.105     ,           nan,    9.135     ,           nan,
                  nan,  -23.615     ,           nan,           nan,
                  nan]])

'''

'''
array([[ -43.385     ,   46.44166667, -101.66333333,  -10.215     ,
                  nan, -185.59833333,           nan,           nan,
                  nan],
       [ -41.185     ,   48.745     ,  -99.44      ,   -7.67      ,
                  nan, -182.405     ,           nan,           nan,
                  nan],
       [ -38.915     ,   51.155     ,  -97.17333333,   -4.33833333,
                  nan, -179.07166667,           nan,           nan,
                  nan],
       [ -36.635     ,   53.59      ,  -94.855     ,   -0.945     ,
                  nan, -175.72      ,           nan,           nan,
                  nan],
       [ -34.35166667,   55.99833333,  -92.52      ,    2.455     ,
                  nan, -172.365     ,           nan,           nan,
                  nan],
       [ -32.065     ,   58.395     ,  -90.21      ,    5.855     ,
                  nan, -169.01      ,           nan,           nan,
                  nan],
       [ -29.77833333,   60.80166667,  -87.90666667,    9.245     ,
                  nan, -165.645     ,           nan,           nan,
                  nan],
       [ -27.495     ,   63.22      ,  -85.61      ,   12.63      ,
                  nan, -162.295     ,           nan,           nan,
                  nan],
       [ -25.19833333,   65.64166667,  -83.29      ,   16.03833333,
                  nan, -158.945     ,           nan,           nan,
                  nan],
       [ -22.915     ,   68.035     ,  -80.985     ,   19.45      ,
                  nan, -155.61      ,           nan,           nan,
                  nan],
       [ -20.62166667,   70.43833333,  -78.69      ,   22.83833333,
                  nan, -152.265     ,           nan,           nan,
                  nan],
       [ -18.315     ,   72.87      ,  -76.395     ,   26.22      ,
                  nan, -148.905     ,           nan,           nan,
                  nan],
       [ -16.015     ,   75.30166667,  -74.09      ,   29.60833333,
                  nan, -145.55833333,           nan,           nan,
                  nan],
       [ -13.725     ,   77.705     ,  -71.78      ,   33.01      ,
                  nan, -142.215     ,           nan,           nan,
                  nan],
       [ -11.43833333,   80.10833333,  -69.5       ,   36.38833333,
                  nan, -138.895     ,           nan,           nan,
                  nan],
       [  -9.15      ,   82.525     ,  -67.22      ,   39.785     ,
                  nan, -135.58      ,           nan,           nan,
                  nan],
       [  -6.85833333,   84.93833333,  -64.94333333,   43.19166667,
                  nan, -132.265     ,           nan,           nan,
                  nan],
       [  -4.58      ,   87.35      ,  -62.68      ,   46.585     ,
                  nan, -128.945     ,           nan,           nan,
                  nan],
       [  -2.295     ,   89.78833333,  -60.39333333,   49.93166667,
                  nan, -125.62166667,           nan,           nan,
                  nan],
       [   0.        ,   92.23      ,  -58.115     ,   53.295     ,
                  nan, -122.29      ,           nan,           nan,
                  nan],
       [   2.29166667,   94.665     ,  -55.85      ,   56.69166667,
                  nan, -118.95833333,           nan,           nan,
                  nan],
       [   4.565     ,   97.1       ,  -53.595     ,   60.08      ,
                  nan, -115.64      ,           nan,           nan,
                  nan],
       [   6.82833333,   99.535     ,  -51.37      ,   63.435     ,
                  nan, -112.31833333,           nan,           nan,
                  nan],
       [   9.11      ,  101.965     ,  -49.145     ,   66.805     ,
                  nan, -108.995     ,           nan,           nan,
                  nan],
       [  11.40833333,  104.395     ,  -46.91      ,   70.15166667,
                  nan, -105.68166667,           nan,           nan,
                  nan],
       [  13.7       ,  106.82      ,  -44.655     ,   73.545     ,
                  nan, -102.375     ,           nan,           nan,
                  nan],
       [  15.97166667,  109.235     ,  -42.38      ,   76.955     ,
                  nan,  -99.085     ,           nan,           nan,
                  nan],
       [  18.25      ,  111.655     ,  -40.105     ,   80.34      ,
                  nan,  -95.8       ,           nan,           nan,
                  nan],
       [  20.52833333,  114.065     ,  -37.86333333,   83.715     ,
                  nan,  -92.49166667,           nan,           nan,
                  nan],
       [  22.785     ,  116.43      ,  -35.615     ,   87.075     ,
                  nan,  -89.17      ,           nan,           nan,
                  nan],
       [  25.04833333,  118.69166667,  -33.34      ,   90.455     ,
                  nan,  -85.875     ,           nan,           nan,
                  nan],
       [  27.325     ,  121.045     ,  -31.07      ,   93.845     ,
                  nan,  -82.57      ,           nan,           nan,
                  nan],
       [  29.615     ,  123.405     ,  -28.85      ,   97.22833333,
                  nan,  -79.29166667,           nan,           nan,
                  nan],
       [  31.9       ,  125.775     ,  -26.625     ,  100.61      ,
                  nan,  -76.025     ,           nan,           nan,
                  nan],
       [  34.17166667,  128.13833333,  -24.38333333,  103.99166667,
                  nan,  -72.75833333,           nan,           nan,
                  nan],
       [  36.43      ,  130.495     ,  -22.125     ,  107.375     ,
                  nan,  -69.48      ,           nan,           nan,
                  nan],
       [  38.695     ,  132.855     ,  -19.87333333,  110.735     ,
                  nan,  -66.17833333,           nan,           nan,
                  nan],
       [  40.96      ,  135.205     ,  -17.65      ,  114.085     ,
                  nan,  -62.875     ,           nan,           nan,
                  nan],
       [  43.23166667,  137.54833333,  -15.43      ,  117.46166667,
                  nan,  -59.585     ,           nan,           nan,
                  nan],
       [  45.495     ,  139.89      ,  -13.205     ,  120.815     ,
                  nan,  -56.31      ,           nan,           nan,
                  nan],
       [  47.765     ,  142.225     ,  -10.96      ,  124.19833333,
                  nan,  -53.02833333,           nan,           nan,
                  nan],
       [  50.05      ,  144.56      ,   -8.705     ,  127.6       ,
                  nan,  -49.75      ,           nan,           nan,
                  nan],
       [  52.33166667,  146.885     ,   -6.48666667,  130.96833333,
                  nan,  -46.48166667,           nan,           nan,
                  nan],
       [  54.595     ,           nan,   -4.24      ,  134.34      ,
                  nan,  -43.215     ,           nan,           nan,
                  nan],
       [  56.83833333,           nan,   -1.99333333,  137.715     ,
                  nan,  -39.955     ,           nan,           nan,
                  nan],
       [  59.075     ,           nan,    0.225     ,  140.415     ,
                  nan,  -36.68      ,           nan,           nan,
                  nan],
       [  61.34833333,           nan,    2.43333333,           nan,
                  nan,  -33.405     ,           nan,           nan,
                  nan],
       [  63.61      ,           nan,    4.675     ,           nan,
                  nan,  -30.14      ,           nan,           nan,
                  nan],
       [  65.86166667,           nan,    6.91      ,           nan,
                  nan,  -26.88833333,           nan,           nan,
                  nan],
       [  68.105     ,           nan,    9.135     ,           nan,
                  nan,  -23.615     ,           nan,           nan,
                  nan]])
'''

