import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import sys
sys.path.append("/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/VectorNet_HighD") 
from config_nw import *


def get_single_track_data(track_df, cluster_global_ptr, recordingMeta):

    features_we_want = get_features_we_want(track_df)
    
    frequency_we_want = get_frequency_we_want(features_we_want)
    
    fixedlen_time_we_want = get_fixedlen_time_we_want(frequency_we_want, sta = 0, end = F_HIS+F_FUR)  # 5 sec

    # 将Dataframes转换为NumPy 数组
    result_array_ =  fixedlen_time_we_want.to_numpy()      #(number of frames, 54 features):
    
    #input data
    #当前vector的特征包括起终点坐标等(x_sta, y_sta, x_end, y_end, time_stamp, polyline index)
    frame_len = result_array_.shape[0]   #(50, 54)
    divide_loc = int(frame_len * divide_prop)    #50是包含历史和未来的总帧数,divided_loc划分出历史和未来/是历史序列的帧数
    
    #normalize the coordinates to be centered around the location of target agent at its last observed time step
    result_array = normalized_traj(result_array_, divide_loc) 
    # Replace NaN values with 0; Cared to replace after the normalization
    result_array = np.nan_to_num(result_array, nan=0)  

    
    # ====== trajectories ======
    single_exa_ls, edge_idx_sou_ls, edge_idx_tar_ls, valid_veh_ls = [], [], [], []
    tra_num, polyline_num = 0, 0
    #所有轨迹的vector总数
    total_tra_vec_len = 0
    #所有轨迹+车道线的vector总数
    total_vec_len = 0
    #分别存储vehicle信息和lane信息的分界处、lane信息与padding信息的分界处
    divide_row_idx = [0, 0]
    
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

    # 未来轨迹 GT：相对最后观测时刻的 (x,y).  相对坐标还是全局坐标???
    gt_xy = result_array[divide_loc:, :2] - result_array[divide_loc-1, :2]    # shape [F_FUR, 2]
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

                                    y = torch.tensor(gt_xy).float(),                 # shape [F_FUR, 2]
                                    x_t0 = torch.tensor(result_array[divide_loc-1, :2]).float(),  # (x0,y0) 以后画图/还原用
                                    pos_xy_gt = torch.tensor(gt_xy).float(),         # 直接存一份，评估更方便
                                    
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

['id', 'x', 'y', 'width', 'height', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', 'frontSightDistance', 'backSightDistance', 'dhw', 'thw', 'ttc', 'precedingXVelocity', 'precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId', 'leftFollowingId', 'rightPrecedingId', 'rightAlongsideId', 'rightFollowingId', 'laneId']
    
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

def normalized_traj(raw_array, divide_loc ):  #( 50, 54)
    res_array = raw_array.copy()
    center_coord = res_array[divide_loc-1, :2]  #(2,)

    cols_need_norm_x = np.array([6*i  for i in range(MAX_SV+1)])
    cols_need_norm_y = np.array([6*i+1  for i in range(MAX_SV+1)])

    res_array[:, cols_need_norm_x] -= center_coord[0]
    res_array[:, cols_need_norm_y] -= center_coord[1]

    return res_array  #( 50, 54)


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

def get_obs_2D(obs_graph):
    #提取图数据中的vehicle部分信息
    vehicle_array = np.array(obs_graph.x[: obs_graph.divide_row_idx[0].item()] )  #(133, 15)
    #每个车辆的信息分别存储在一个array中
    vehicle_data_ls = [vehicle_array[np.where(vehicle_array[:, -1] == i)] for i in range(1, obs_graph.traj_len.item()+1) ] 
    
    obs_2D = np.zeros(((MAX_SV+1)*F_HIS, N_FEA))   
    idx = 0
    for veh in range((MAX_SV+1)):
        if veh in obs_graph.valid_sv_idxs:  #车辆id对齐 
            for t_i, t_v in enumerate(  (vehicle_data_ls[idx][:, -3] - 1).astype(int)  ):#时间戳对齐     -1是因为图数据中时间戳索引从1开始
                obs_2D[veh*F_HIS+ t_v] = vehicle_data_ls[idx][t_i, :N_FEA]
            obs_2D[veh*F_HIS+ t_v+1] = vehicle_data_ls[idx][t_i, N_FEA:2*N_FEA] #从最后一个向量中提取最后一个时间戳
            idx += 1
    return obs_2D
    
def get_localCS_sv_gt(sv_gt, xys_t0_nonorm):
    sv_gt_local = sv_gt.copy()   #(N, 48)
    sv_gt_local[:, [i for i in range(0, 48, 6)] ] -=  xys_t0_nonorm[0]
    sv_gt_local[:, [i for i in range(1, 49, 6)]] -=  xys_t0_nonorm[1]
    return sv_gt_local

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

def get_next_obs(obs, next_obs_2D_, cluster_global_ptr, line_num):
    '''在标准化格式下，无需更新index信息(3/15)，只更新kinematics信息(12/15)
    修改10.14：next_obs.cluster不再和next_obs.x[-1]相等，单独更新; 图数据变得更加unstructured
    '''
    
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
    #随着EV实际执行一个时间步，建立新的local coordinate system 
    next_obs_2D, xy_incre  = normalize_next_obs_2D(next_obs_2D_)
    for veh in range(MAX_SV+1):
        single_tra_input = next_obs_2D[F_HIS*veh: F_HIS*(veh+1), :].copy()  #(30, 6)
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
        feat_vehid = np.ones((feat_coor.shape[0], 1)) if veh == 0 else np.ones((feat_coor.shape[0], 1))*2    #ev是1  其他sv都是2
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
    ''' 依据divide_row_idx 提取出原先lane line信息，依据EV在上一时间步坐标系下的位置更新至当前步的坐标系
        更新object index'''
    all_lane_feats_ = np.array(obs.x[obs.divide_row_idx[0].item(): obs.divide_row_idx[1].item()].clone())  #从上一时刻的输入中将lane line部分copy过来   (297, 15)
    all_lane_feats = normalize_lane_line(all_lane_feats_, xy_incre)
    for line in range(line_num):
        single_lane_feats = all_lane_feats[(N_LINE_SAM - 1)*line: (N_LINE_SAM - 1)*(line+1)]  #(99, 15)  
        single_lane_feats[:, -1] = polyline_num+1  #由于SV的数量可能变化，所以需要更新object index!
        #********生成edge_idx
        edge_idx_sou = np.array([ i for i in range(N_LINE_SAM - 2)]) + total_vec_len
        edge_idx_tar = np.array([ j+1 for j in range(N_LINE_SAM - 2)]) + total_vec_len
        edge_idx_sou_ls.append(edge_idx_sou)
        edge_idx_tar_ls.append(edge_idx_tar)
        
        #update
        single_exa_ls.append(single_lane_feats)
        polyline_num += 1
        total_vec_len += N_LINE_SAM - 1   #graph global

    #graph 边数据
    edge_idx = np.array([np.concatenate(edge_idx_sou_ls, axis = 0), np.concatenate(edge_idx_tar_ls, axis = 0) ] )
    #graph 节点数据和预测gt
    single_data = np.concatenate(single_exa_ls, axis = 0)      #(558, 15)
    # ev_fur_pos_gt = result_array[F_HIS:, :2]   #(30, 2）  
    divide_row_idx[1] = single_data.shape[0]
    
    #********生成identifier，用于完成graph completion task时给被masked out的polyline一个标记
    #identifier中每个值代表：包含一个polyline画一个矩形， 矩形的最小坐标       #(identifier 暂时不用，因为with_aux=False)
    identifier = np.empty((0,2))
    for pl in np.unique(single_data[:,-1]):
        [indices] = np.where(single_data[:, -1] == pl)
        identifier = np.vstack([identifier,  np.min(single_data[indices, :2], axis = 0) ]  )  # axis = 0表示在第一维度中找出最小的，注意np.min()分别找出两列中的最小值，并没有捆绑在一起
        
    
    #转为graph data
    next_obs = Data(x = torch.tensor([
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
                                    lane_ys_nonorm = obs.lane_ys_nonorm.clone(),  
                                     #EV实际执行过程中 当前时刻的坐标，是local坐标系在global坐标系中的原点坐标
                                    xys_t0_nonorm =  torch.tensor([xy_incre[0] + obs.xys_t0_nonorm[0].item(),  xy_incre[1] + obs.xys_t0_nonorm[1].item()])
                                    )   
    
    #padding for aligning the number of polylines in a same graph
    if next_obs.max_valid_len[0].item() > polyline_num:
        next_obs = get_padding_graph(next_obs) #全零填充
        
    #********生成cluster，cluster是自定义的图Data属性，用于聚类。   
    next_obs.cluster += cluster_global_ptr
    #update the cluster_global_ptr
    cluster_global_ptr += (1+MAX_SV+MAX_LINE)

    return next_obs, cluster_global_ptr

#因为next_obs = None无法通过sample batch函数， next_obs = obs无法通过q值计算，所以选择生成一个“假的next_obs”即仅修改obs的cluster index信息
def get_fake_next_obs(obs, cluster_global_ptr, line_num = None):
    next_obs = obs.clone()#深复制
    
    
    # #vehicles
    # for veh in range((MAX_SV+1)):
    #     next_obs.x[veh*(F_HIS-1):(veh+1)*(F_HIS-1), -1] = cluster_global_ptr  #此处选择更新
    #     cluster_global_ptr += 1
    # #lane line 
    # for line in range(line_num):
    #     next_obs.x[(F_HIS-1)*(MAX_SV+1)+line*(N_LINE_SAM-1): (F_HIS-1)*(MAX_SV+1)+ (line+1)*(N_LINE_SAM-1), -1] = cluster_global_ptr
    #     cluster_global_ptr += 1
    # #多余的那一条data需要重置cluter（不然就是0）
    # prt_incre = next_obs.max_valid_len[0].item() - next_obs.valid_len[0].item()
    # if prt_incre>0:
    #     next_obs.x[-prt_incre:, -1] = cluster_global_ptr
    #     cluster_global_ptr += 1
    
    
    
    #更新global_idx\cluster等graph参数，不然buffer中sample出的batch在subgraph中无法正常聚类！
    next_obs.cluster += (1+MAX_SV+MAX_LINE) # (973,)
    #update the cluster_global_ptr
    cluster_global_ptr += (1+MAX_SV+MAX_LINE)
    
    return next_obs, cluster_global_ptr


def get_padding_graph(data):
    feature_len = data.x.shape[1]
    prt_incre = data.max_valid_len[0].item() - data.valid_len[0].item()

    # pad feature with zero nodes
    data.x = torch.cat([data.x, torch.zeros((prt_incre, feature_len), dtype=data.x.dtype)])
    data.cluster = torch.cat([data.cluster, torch.arange( data.valid_len[0].item()+1 , data.max_valid_len[0].item()+1, dtype=data.cluster.dtype)]).long()
    data.identifier = torch.cat([data.identifier, torch.zeros((prt_incre, 2), dtype=data.identifier.dtype)])

    assert data.cluster.shape[0] == data.x.shape[0], "[ERROR]: Loader error!"

    return data


def normalize_next_obs_2D(raw_array):   #(270, 6)
    xy_incre = [raw_array[F_HIS-1, 0], raw_array[F_HIS-1, 1] ]
    target_array = raw_array.copy()
    for i in range(MAX_SV + 1):
        for j in range(F_HIS):
            if not np.all(target_array[F_HIS*i+j] == 0 ):
                target_array[F_HIS*i+j, 0] = raw_array[F_HIS*i+j, 0] - xy_incre[0]
                target_array[F_HIS*i+j, 1] = raw_array[F_HIS*i+j, 1] - xy_incre[1]
    return target_array, xy_incre

def normalize_lane_line(raw_array, xy_incre):  # (297, 15)
    #无需更新纵向坐标，因为HighD全部为直线道路，不更新相当于动态向前取值
    #需要更新横向坐标
    target_array = raw_array.copy()
    target_array[:, [1, 7]] -=  xy_incre[1]
    return target_array




# def get_fv(ev_traj, ev_cur_xyva, sv_pos_fur, i, ev_y_init):
#     #判断是否换道                 
#     '''注意:
#     1.这里对是否换道的判断依据不精确，但由于有运动学约束所以理论上可以
#     2.这里对换道后的前车判断存在漏洞，一是仅支持搜索初始8辆SV（可能存在别的前车！），二是数假设SV不换道！
#     （但这个应该也能解释，因为规划时间只有3s，而且是滚动规划   （真实数据中很少有SV换道情况，但存在的SV换道cases可能带来误差
#     3.没有考虑other lane factor，仅考虑same lane的前车的跟驰模型，尤其对于换道过程中忽略其他车辆的影响可能明显'''
#     '''
#     首先判断所在车道，得到可能的前车列表
#     其次重新搜索前车'''
#     if (ev_traj[1][i] - ev_y_init > 3.75/2) :  #向右换道（y值增加
#         fv_candi = [5,6,7]
#     elif ev_y_init - ev_traj[1][i] > 3.75/2:  #向左换道
#         fv_candi = [2,3,4]
#     else:     #in the initial lane
#         fv_candi = [0, 1]

#     for candi in fv_candi:  #sv indexes in left lane
#         if ev_traj[0][i] < sv_pos_fur[i, N_FEA*candi]:  #EV换入间隙位于目标前车之后
#             return True, candi
#     return False, None  #False有两种情况：1. EV换入间隙位于目标前车之前（包括超车）  2.本来就不存在目标前车(初始 8 SV)

# def uniform_obs_format(obs, x_restored, sv_gt_local, t_ptr, lane_num):
#     '''    可以优化的点2：
#     对特定位置的SV是否存在的判断条件存在漏洞：
#     “obs[F_HIS*(veh+1) - 1]”仅依据历史数据最后一个时间步的信息进行判断
#     虽然如果在最后时刻都没有出现的SV理论上距离EV很远，可能不会造成影响，但还是不算严谨
#     '''
#     #获取EV的current信息与furture信息
#     EV_curr = obs[F_HIS -1].copy()  #['x', 'y']  历史数据中最后一条数据就是当前时刻数据
#     Ev_xPos_fut = x_restored.copy()
#     #获取SV的current信息与furture信息
#     DynObs_curr =[ []  for _ in range(lane_num)]  
#     DynObs_xPos_fut = [ []  for _ in range(lane_num)]  
    
#     if lane_num == 2:   #如果为二车道，需要判断一下ev所在车道
#         #array.any()：只要array中有一个True(非零)就返回True   array.all()：array中所有都为True才返回True
#         if  obs[F_HIS*3:F_HIS*(3+1)].any() or  obs[F_HIS*4:F_HIS*(4+1)].any() or obs[F_HIS*5:F_HIS*(5+1)].any():
#             #如果左侧有sv，说明ev在右车道
#             for veh in range(1, MAX_SV+1):
#                 if obs[F_HIS*(veh+1) - 1][0] == 0 and obs[F_HIS*(veh+1) - 1][1] == 0 and obs[F_HIS*(veh+1) - 1][3] == 0 and obs[F_HIS*(veh+1) - 1][4] == 0:
#                     continue   #如果该位置无veh，则退出循环
#                 if veh in [1,2]:
#                     DynObs_curr[1].append(obs[F_HIS*(veh+1) - 1].copy())
#                     DynObs_xPos_fut[1].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
#                 elif veh in [3,4,5]:
#                     DynObs_curr[0].append(obs[F_HIS*(veh+1) - 1].copy())
#                     DynObs_xPos_fut[0].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
#                 elif veh in [6,7,8]:
#                     sys.exit("Invalid vehicle")
#         else:
#             #如果左侧无sv，说明ev在左车道，或者无左侧或右侧sv
#             for veh in range(1, MAX_SV+1):
#                 if obs[F_HIS*(veh+1) - 1][0] == 0 and obs[F_HIS*(veh+1) - 1][1] == 0 and obs[F_HIS*(veh+1) - 1][3] == 0 and obs[F_HIS*(veh+1) - 1][4] == 0:
#                     continue   #如果该位置无veh，则退出循环
#                 if veh in [1,2]:
#                     DynObs_curr[0].append(obs[F_HIS*(veh+1) - 1].copy())
#                     DynObs_xPos_fut[0].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
#                 elif veh in [6,7,8]:
#                     DynObs_curr[1].append(obs[F_HIS*(veh+1) - 1].copy())
#                     DynObs_xPos_fut[1].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
#                 elif veh in [3,4,5]:
#                     sys.exit("Invalid vehicle")
        
    
#     if lane_num == 3:
#         #获取SV的current信息
#         for veh in range(1, MAX_SV+1):
#             if obs[F_HIS*(veh+1) - 1][0] == 0 and obs[F_HIS*(veh+1) - 1][1] == 0 and obs[F_HIS*(veh+1) - 1][3] == 0 and obs[F_HIS*(veh+1) - 1][4] == 0:
#                 continue   #如果该位置无veh，则退出循环；为了防止出现巧合，多加几个条件
#             if veh in [3,4,5]:
#                 DynObs_curr[0].append(obs[F_HIS*(veh+1) - 1].copy())
#                 DynObs_xPos_fut[0].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
#             elif veh in [1,2]:
#                 DynObs_curr[1].append(obs[F_HIS*(veh+1) - 1].copy())
#                 DynObs_xPos_fut[1].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
#             elif veh in [6,7,8]:
#                 DynObs_curr[2].append(obs[F_HIS*(veh+1) - 1].copy())
#                 DynObs_xPos_fut[2].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
#             else:
#                 sys.exit("Invalid vehicle")
#     '''
#     注意：此处"if obs[F_HIS*(veh+1) - 1][0] == 0 and..."用"F_HIS*(veh+1) - 1"即每个SV在obs中最后一时刻的信息有无来判断SV有无是合理的
#                 因为如果obs中最后时刻都没有SV，之后（规划时段）更不会有该SV
#     '''
#     return DynObs_curr, EV_curr, DynObs_xPos_fut, Ev_xPos_fut 
    
# def get_uplo_markings(file_name  ):
#     directory = 'data_processing/raw_highd_data'
#     # file_name = "03_recordingMeta.csv"  #2 LANE
#     file_path = os.path.join(directory, file_name)
#     recordingMeta_df = pd.read_csv(file_path, index_col=0)
#     up_markings = recordingMeta_df['upperLaneMarkings']
#     lo_markings = recordingMeta_df['lowerLaneMarkings']
    
#     # 使用str.split()按分号拆分为列表  
#     split_lists_up = up_markings.str.split(';')  
#     # 转换每个字符串为浮点数  
#     float_lists_up = split_lists_up.apply(lambda x: [float(i) for i in x]).iloc[0]  
    
#     # 使用str.split()按分号拆分为列表  
#     split_lists_lo = lo_markings.str.split(';')  
#     # 转换每个字符串为浮点数  
#     float_lists_lo = split_lists_lo.apply(lambda x: [float(i) for i in x]).iloc[0]  
    
#     up_markings_ls = [round(30 - val, 2) for val in float_lists_up]
#     up_markings_ls = up_markings_ls[::-1]
#     lo_markings_ls = float_lists_lo
    
#     return up_markings_ls, lo_markings_ls

