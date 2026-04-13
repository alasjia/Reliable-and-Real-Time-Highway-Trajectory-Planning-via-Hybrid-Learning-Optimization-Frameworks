'''
[v]修改DNN输出内容从longitudinal position到acceleration
[v]在DNN input中加入所有车辆（ev和sv）的v和a信息
'''
'''

[v]更改sv搜索方式，采用"op+drl202406_dataset"内regenerated处理后数据集
[v]更改时间序列中每个元素之间的时间间隔0.2-->0.1 sec

'''

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader

import sys
sys.path.append("/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training")  #!!!
from config_nw import *

# def normalized_traj(raw_data):  #(60,54)
#     center_coord = raw_data[F_HIS-1, :2].copy()  #(2,)
#     cols_need_norm_x = np.array([6*i  for i in range(MAX_SV+1)])
#     cols_need_norm_y = np.array([6*i+1  for i in range(MAX_SV+1)])
#     single_exa = raw_data.copy()   #在函数内改变传入的array会改变外部array（传址）
#     single_exa[:, cols_need_norm_x] = single_exa[:, cols_need_norm_x]  - center_coord[0]
#     single_exa[:, cols_need_norm_y] = single_exa[:, cols_need_norm_y]  - center_coord[1]
#     return single_exa

def normalized_traj(raw_array, sv_num, divide_loc ):  #( 50, 54)
    res_array = raw_array.copy()
    center_coord = res_array[divide_loc-1, :2]  #(2,)
    cols_need_norm_x = np.array([6*i  for i in range(sv_num+1)])
    cols_need_norm_y = np.array([6*i+1  for i in range(sv_num+1)])
    res_array[:, cols_need_norm_x] = res_array[:, cols_need_norm_x]  - center_coord[0]
    res_array[:, cols_need_norm_y] = res_array[:, cols_need_norm_y]  - center_coord[1]
    return res_array  #( 50, 54)


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
    time_index = pd.date_range(start='2024-01-01 00:00:00', periods=pd_df.shape[0], freq='0.04s')   
    #设置为原始dataframe的索引
    pd_df.index = time_index
    #进行下采样  0.04 s --> 0.1 s
    res = pd_df.resample('0.1s').mean()
    return res

# def get_fixedlen_time_we_want(pd_df):
#     fixed_len = int(SPF*(F_HIS+F_FUR)/0.04)  #0.04 IS ORIGINAL FREQUENCY OF DATA
#     return pd_df.iloc[:fixed_len]

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



def get_lower_kin(action, x0):   #action.size 应该为(2,)
    #速度规划模型输出内容为速度
    restored_x = x0 + np.cumsum(action * SPF)  
    restored_v = action.copy()
    return restored_x,  restored_v


def get_sv_gt(track_df):
    features_we_want  = track_df[[
            'preced_x', 'preced_y', 'preced_vx', 'preced_vy', 'preced_ax', 'preced_ay', 
            'follow_x', 'follow_y','follow_vx', 'follow_vy', 'follow_ax', 'follow_ay', 
            'leftPreced_x','leftPreced_y', 'leftPreced_vx', 'leftPreced_vy', 'leftPreced_ax','leftPreced_ay', 
            'leftAlongs_x', 'leftAlongs_y', 'leftAlongs_vx','leftAlongs_vy', 'leftAlongs_ax', 'leftAlongs_ay', 
            'leftFollow_x','leftFollow_y', 'leftFollow_vx', 'leftFollow_vy', 'leftFollow_ax','leftFollow_ay', 
            'rightPreced_x', 'rightPreced_y', 'rightPreced_vx', 'rightPreced_vy', 'rightPreced_ax', 'rightPreced_ay',
            'rightAlongs_x', 'rightAlongs_y', 'rightAlongs_vx', 'rightAlongs_vy', 'rightAlongs_ax', 'rightAlongs_ay', 
            'rightFollow_x', 'rightFollow_y', 'rightFollow_vx', 'rightFollow_vy', 'rightFollow_ax', 'rightFollow_ay'
                                        ]]
    frequency_we_want = get_frequency_we_want(features_we_want)
    sv_gt = frequency_we_want.to_numpy()
    return sv_gt
    
def get_single_track_data(track_df, cluster_global_ptr, recordingMeta = None, tracksMeta  = None, gt_draw = False):
    #track_df.shape : e.g. (218, 81), 81 is the number of column, which is fixed; 218 means timestep, which is changed.
    features_we_want = get_features_we_want(track_df)
    frequency_we_want = get_frequency_we_want(features_we_want)
    fixedlen_time_we_want = get_fixedlen_time_we_want(frequency_we_want, sta = 0, end = F_HIS+F_FUR)  # 5 sec

    # 将Dataframes转换为NumPy 数组
    result_array_ = fixedlen_time_we_want.to_numpy()  #(50, 54)

    #input data
    #当前vector的特征包括起终点坐标等(x_sta, y_sta, x_end, y_end, time_stamp, polyline index)
    frame_len = result_array_.shape[0]   #25(2.5 s)      50(5 s)
    divide_loc = int(frame_len * divide_prop)  #10(2.5 s)      20(5 s)
    
    #normalize the coordinates to be centered around the location of target agent at its last observed time step
    result_array = normalized_traj(result_array_, MAX_SV, divide_loc) 
    
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
        feat_vehid = np.ones((feat_coor.shape[0], 1)) if veh == 0 else np.ones((feat_coor.shape[0], 1))*2    #ev是1  其他sv都是2  车道线是3
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
        lane_lineid =  np.ones((lane_feat_coor.shape[0], 1) )  * 3        #(line+1)  
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
    single_data = np.concatenate(single_exa_ls, axis = 0)      #(558, 15)
    # ev_fur_pos_gt = result_array[F_HIS:, :2]   #(30, 2）  
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
    
    return single_graph, cluster_global_ptr, lane_array.shape[0]
