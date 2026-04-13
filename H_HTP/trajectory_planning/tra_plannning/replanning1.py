
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from config import *
from lateral_planning.path_planning import path_qp_model_modi
if not old_version:
    from tra_plannning.data4VN import get_single_track_data, get_sv_gt, get_lower_kin, get_padding_graph
else:
    from tra_plannning.data4VN_old import get_single_track_data, get_sv_gt, get_lower_kin, get_padding_graph

import sys
sys.path.append("/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training")  #!!!
from config_nw import *



def reset(track_df, recordingMeta, cluster_global_ptr):
    obs_t0, cluster_global_ptr, line_num = get_single_track_data(track_df, cluster_global_ptr, recordingMeta=recordingMeta, gt_draw =True)
    #从原始数据集中获取global坐标系下的周围车辆运动信息
    sv_gt = get_sv_gt( track_df)  #(frame_num, feature_num)  (68, 48)

    return obs_t0, sv_gt, cluster_global_ptr, line_num


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

def step(PathModel, Vars_ls, obs, action, sv_gt, cluster_global_ptr, line_num, t_ptr, EV_curr_ykin, case_id = 0, time_id = 0):   #time_pointer应该是更新后的指
    '''
    obs:   graph object
    action:  (30,)
    '''
    '''三种情况结束游戏(done is True)： 
    1. 当前轨迹gt data的时长不足以提供sv状态（因为采用gt data代替仿真）
    2. path planning模型无解：意味着第一个时间步就是infeasible
    3. EV倒退
    '''
    
    #结束的情况一：到达sv真实轨迹数据的时间末端；注意此种情况不存入replay buffer，因为结束的原因与模型变量无关
    track_max_f = sv_gt.shape[0]      #sv_gt最大时长，实际上是ev在数据集中的轨迹时长
    if t_ptr > (track_max_f - F_FUR):  
        return None, True, t_ptr, cluster_global_ptr, None, None, None, None, None, 0

    #---------------------------part 1: execute the action, get planning result, a trajectory-------------------------------
    #将原始数据中的周围车辆位置信息转换为local坐标系下
    sv_gt_local = get_localCS_sv_gt(sv_gt, np.array(obs.xys_t0_nonorm).copy() )  #(155, 48)
    #为了适应已测试好的数据结构，将graph形式的obs转换为size(270, 6)
    obs_2D = get_obs_2D(obs)    #(270, 6)
    #将action中的高阶运动学信息转换为低阶的位移和速度  
    x_restored, vx_restored = get_lower_kin(action, obs_2D[F_HIS-1, 0])    #(30,)
    
    #结束的情况二：倒退    
    if (x_restored[1] - x_restored[0]) < 0:   #只判断第一时间步
        fake_next_obs, cluster_global_ptr = get_fake_next_obs(obs, cluster_global_ptr)
        return fake_next_obs, True, t_ptr, cluster_global_ptr, None, None, None, None, None, 0
    
    #按照不同车道数获取车道线y坐标值
    lane_ys = [obs.x[obs.divide_row_idx[0].item():][99*i, 1].item() for i in range(line_num)]#此处采用标准化之后的车道线坐标
    #执行横向规划的优化模型，完成trajectory planning
    #note: 如果不想绘制global坐标系下的轨迹，删掉ev_global_x0即可
    traj,  DynObs_xPos_fut,  la_derivative1, EV_curr_ykin, DynObs_curr, solving_time = op_planner(PathModel, Vars_ls, obs_2D, x_restored, sv_gt_local, t_ptr, lane_ys, EV_curr_ykin, 
                                                        ev_global_x0 = obs.xys_t0_nonorm[0].item(), case_id=case_id, time_id=time_id)  
    #获取规划模型结果(a trajectory)对应时段内的surrounding vehicles信息
    sv_pos_fur = sv_gt_local[t_ptr: t_ptr+F_FUR].copy()    #(30, 48)
    

    #结束的情况三：path planning无解
    if len(traj[0]) < 1:
        fake_next_obs, cluster_global_ptr = get_fake_next_obs(obs, cluster_global_ptr)
        return fake_next_obs, True, t_ptr, cluster_global_ptr, None, None, None, None, None, 0
    
    
    # double check
    if not (x_restored[:len(traj[0])]==traj[0]).all():
        sys.exit("Something Wrong with longitudinal trajectory!")
        
    #---------------------------part 2: update the observation of the environment, according to the planning result-------------------------------
    '''注意：对于未来横向状态：
    1.由于op模型结果没有直接给出横向加速度/速度，所以需要反推；
    2.反推计算公式输出该时间间隔内的平均速度/加速度，区别于纵向结果中的瞬时速度/加速度
    
    第1帧（t=0)加速度很大的原因：obs_2D[F_HIS-1, 1] = 0, traj[1][0]即使值很小，例如0.01 m，vy = 0.1，ax就可能大于1了
    '''
    vy_next = (traj[1][0] - obs_2D[F_HIS-1, 1]) / SPF  #F_HIS-1: obs_2D第1维代表时间维度，大小为F_HIS*9  F_HIS-1是EV最后一帧信息
    ay_next = (vy_next - obs_2D[F_HIS-1, 3]) / SPF  
    ax_next = (vx_restored[0] - obs_2D[F_HIS-1, 2]) / SPF  
    ev_kin_next = np.concatenate( (  obs_2D[1:F_HIS], 
                                   np.array([traj[0][0], traj[1][0], vx_restored[0], vy_next, ax_next, ay_next]).reshape((1, -1))  )    )#(30,6)

    t_ptr += 1
    
    sv_on_ls = []
    for sv in range(1, (MAX_SV+1)):
        sv_on_ls.append(  sv_gt_local[t_ptr - F_HIS: t_ptr, N_FEA*(sv-1): N_FEA*(sv-1)+N_FEA].copy()   ) 
    sv_kin_next  = np.concatenate(sv_on_ls, axis=0)   #(240, 6)
    sv_kin_next  =np.array( [np.nan_to_num(_, nan=0) for _ in sv_kin_next ] )     # Replace NaN values with 0
    next_obs_2D_ = np.concatenate( (ev_kin_next, sv_kin_next) )   #(270,6)
    
    #将（90，6）size next_obs 转换为graph形式的next_obs     
    next_obs, cluster_global_ptr = get_next_obs(obs, next_obs_2D_, cluster_global_ptr, line_num)
    
    #由于DynObs_curr是local coordinate system下，所以查找y值在0值所在车道范围内的前车即可
    for idx, val in enumerate( lane_ys[1:]):  #查找车道范围
        if val > 0:
            ev_line_lb, ev_line_ub = lane_ys[idx], lane_ys[idx+1]
            break
    fv_x, fv_v = np.nan, np.nan  #如果FV不存在，则ttc为nan
    for i in range(len(DynObs_curr)):  # 锁定前车位置
        for j in range(len(DynObs_curr[i])):
            if DynObs_curr[i][j][1] > ev_line_lb and DynObs_curr[i][j][1] < ev_line_ub and DynObs_curr[i][j][0] > 0:
                fv_x, fv_v =  DynObs_curr[i][j][0], DynObs_curr[i][j][2]  #由于DynObs_curr中单个车道的SV搜索机制为沿x轴减小方向，所以无需担心重复赋值（重复意味更近）
    ttc = (fv_x - 0 - VEH_LEN) / (obs_2D[F_HIS-1, 2] - fv_v - 1e-6) if obs_2D[F_HIS-1, 2] > fv_v else np.nan#time to collision: 车头间距/纵向速度差   注意如果ev车速低于FV车速，则TTC为nan

    exec_ev_kin = [traj[0][0] + obs.xys_t0_nonorm[0].item(), traj[1][0] + obs.xys_t0_nonorm[1].item(),
                   vx_restored[0], vy_next, 
                   ax_next, ay_next, 
                   ttc, la_derivative1[0]]   #图长的一样的原因是每次都更新坐标系！
    
    #获取exec_ev_kin对应时刻的SVs的运动学参数：第一维分别是x, y, vx, vy, ax, ay; 第二维分别对应8辆SV（不存在的为nan）
    exec_svs_kin = [[sv_gt[t_ptr-1, sv*N_FEA] for sv in range(MAX_SV)],
                    [sv_gt[t_ptr-1, sv*N_FEA+1] for sv in range(MAX_SV)],
                    [sv_gt[t_ptr-1, sv*N_FEA+2] for sv in range(MAX_SV)],
                    [sv_gt[t_ptr-1, sv*N_FEA+3] for sv in range(MAX_SV)],
                    [sv_gt[t_ptr-1, sv*N_FEA+4] for sv in range(MAX_SV)],
                    [sv_gt[t_ptr-1, sv*N_FEA+5] for sv in range(MAX_SV)]  ]
    
    sx_plan = traj[0]
    sy_plan = traj[1]
    
    return next_obs, False, t_ptr, cluster_global_ptr, exec_ev_kin, EV_curr_ykin, exec_svs_kin, sx_plan, sy_plan, solving_time  #done == fasle
    # return next_obs, False, t_ptr, cluster_global_ptr, [], EV_curr_ykin, [], [], [], solving_time


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


'''
Q：case 121中，ev在一段时间后完全处于另一车道，所以8SV的序号不再符合初始时刻的，如何修改？
A：传入ev_y_init参数用于搜索前车，ev_y_init参数在一个case是固定值
'''
def get_fv(ev_traj, ev_cur_xyva, sv_pos_fur, i, ev_y_init):
    #判断是否换道                 
    '''注意:
    1.这里对是否换道的判断依据不精确，但由于有运动学约束所以理论上可以
    2.这里对换道后的前车判断存在漏洞，一是仅支持搜索初始8辆SV（可能存在别的前车！），二是数假设SV不换道！
    （但这个应该也能解释，因为规划时间只有3s，而且是滚动规划   （真实数据中很少有SV换道情况，但存在的SV换道cases可能带来误差
    3.没有考虑other lane factor，仅考虑same lane的前车的跟驰模型，尤其对于换道过程中忽略其他车辆的影响可能明显'''
    '''
    首先判断所在车道，得到可能的前车列表
    其次重新搜索前车'''
    if (ev_traj[1][i] - ev_y_init > 3.75/2) :  #向右换道（y值增加
        fv_candi = [5,6,7]
    elif ev_y_init - ev_traj[1][i] > 3.75/2:  #向左换道
        fv_candi = [2,3,4]
    else:     #in the initial lane
        fv_candi = [0, 1]

    for candi in fv_candi:  #sv indexes in left lane
        if ev_traj[0][i] < sv_pos_fur[i, N_FEA*candi]:  #EV换入间隙位于目标前车之后
            return True, candi
    return False, None  #False有两种情况：1. EV换入间隙位于目标前车之前（包括超车）  2.本来就不存在目标前车(初始 8 SV)



def op_planner(PathModel, Vars_ls, obs, x_restored, sv_gt_local, t_ptr, lane_markings, EV_curr_ykin, ev_global_x0 = 0, case_id = 0, time_id = 0):
    ''' 可以优化的点1：
    当前的数据结构可以输入任何数量的周边车辆，但仍然需要建立/转化为structured data的原因是需要时间对齐：动态物体在不同时刻具有不同空间状态
    目前数据结构的处理过程为：graph(unstructured data) => 2D_array(structured data) =>  按车道分类的array。
    而最后一步是多余的。
    HighD数据已将SV按照与EV的相对位置分类为8类，这省去了本次编程中对特定SV（前车等）的搜索工作。
    '''
    #将observation和action转换为统一的数据格式
    DynObs_curr, EV_curr, DynObs_xPos_fut, Ev_xPos_fut = uniform_obs_format(obs, x_restored, sv_gt_local,  t_ptr, len(lane_markings)-1)   #a single case, the len is the vehicle number.

    # #判断当前case的车道线位置
    # lane_markings = get_lane_markings(up_markings , lo_markings, EV_curr)
    
    # # 绘图展示BEV驾驶场景
    # for case in range(len(EV_curr)):
    #     scenario_plot(DynObs_curr[case], StaObs[case],  EV_curr[case], up_markings, lo_markings )
    
    # path planning
    start_time_path = time.time()
    flag_path, lateral_locs, la_derivative1, longitudinal_locs, func_path, EV_curr_ykin, solving_time = path_qp_model_modi(PathModel, Vars_ls, DynObs_curr, EV_curr, EV_curr_ykin, DynObs_xPos_fut, Ev_xPos_fut, lane_markings, ev_global_x0, case_id, time_id)
    # print("路径规划运行时间：%.4f sec" % (time.time() - start_time_path))  
    
    if flag_path == 1:
        #横向规划时长<=纵向规划时长，需要按照横向规划结果统一长度
        traj_res = [longitudinal_locs[:len(lateral_locs)], np.array(lateral_locs) ]
        return traj_res,  DynObs_xPos_fut, la_derivative1, EV_curr_ykin, DynObs_curr, solving_time
    #横向规划无解
    return [[], []],  DynObs_xPos_fut, la_derivative1, EV_curr_ykin, DynObs_curr, 0


def uniform_obs_format(obs, x_restored, sv_gt_local, t_ptr, lane_num):
    '''    可以优化的点2：
    对特定位置的SV是否存在的判断条件存在漏洞：
    “obs[F_HIS*(veh+1) - 1]”仅依据历史数据最后一个时间步的信息进行判断
    虽然如果在最后时刻都没有出现的SV理论上距离EV很远，可能不会造成影响，但还是不算严谨
    '''
    #获取EV的current信息与furture信息
    EV_curr = obs[F_HIS -1].copy()  #['x', 'y']  历史数据中最后一条数据就是当前时刻数据
    Ev_xPos_fut = x_restored.copy()
    #获取SV的current信息与furture信息
    DynObs_curr =[ []  for _ in range(lane_num)]  
    DynObs_xPos_fut = [ []  for _ in range(lane_num)]  
    
    if lane_num == 2:   #如果为二车道，需要判断一下ev所在车道
        #array.any()：只要array中有一个True(非零)就返回True   array.all()：array中所有都为True才返回True
        if  obs[F_HIS*3:F_HIS*(3+1)].any() or  obs[F_HIS*4:F_HIS*(4+1)].any() or obs[F_HIS*5:F_HIS*(5+1)].any():
            #如果左侧有sv，说明ev在右车道
            for veh in range(1, MAX_SV+1):
                if obs[F_HIS*(veh+1) - 1][0] == 0 and obs[F_HIS*(veh+1) - 1][1] == 0 and obs[F_HIS*(veh+1) - 1][3] == 0 and obs[F_HIS*(veh+1) - 1][4] == 0:
                    continue   #如果该位置无veh，则退出循环
                if veh in [1,2]:
                    DynObs_curr[1].append(obs[F_HIS*(veh+1) - 1].copy())
                    DynObs_xPos_fut[1].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
                elif veh in [3,4,5]:
                    DynObs_curr[0].append(obs[F_HIS*(veh+1) - 1].copy())
                    DynObs_xPos_fut[0].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
                elif veh in [6,7,8]:
                    sys.exit("Invalid vehicle")
        else:
            #如果左侧无sv，说明ev在左车道，或者无左侧或右侧sv
            for veh in range(1, MAX_SV+1):
                if obs[F_HIS*(veh+1) - 1][0] == 0 and obs[F_HIS*(veh+1) - 1][1] == 0 and obs[F_HIS*(veh+1) - 1][3] == 0 and obs[F_HIS*(veh+1) - 1][4] == 0:
                    continue   #如果该位置无veh，则退出循环
                if veh in [1,2]:
                    DynObs_curr[0].append(obs[F_HIS*(veh+1) - 1].copy())
                    DynObs_xPos_fut[0].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
                elif veh in [6,7,8]:
                    DynObs_curr[1].append(obs[F_HIS*(veh+1) - 1].copy())
                    DynObs_xPos_fut[1].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
                elif veh in [3,4,5]:
                    sys.exit("Invalid vehicle")
        
    
    if lane_num == 3:
        #获取SV的current信息
        for veh in range(1, MAX_SV+1):
            if obs[F_HIS*(veh+1) - 1][0] == 0 and obs[F_HIS*(veh+1) - 1][1] == 0 and obs[F_HIS*(veh+1) - 1][3] == 0 and obs[F_HIS*(veh+1) - 1][4] == 0:
                continue   #如果该位置无veh，则退出循环；为了防止出现巧合，多加几个条件
            if veh in [3,4,5]:
                DynObs_curr[0].append(obs[F_HIS*(veh+1) - 1].copy())
                DynObs_xPos_fut[0].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
            elif veh in [1,2]:
                DynObs_curr[1].append(obs[F_HIS*(veh+1) - 1].copy())
                DynObs_xPos_fut[1].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
            elif veh in [6,7,8]:
                DynObs_curr[2].append(obs[F_HIS*(veh+1) - 1].copy())
                DynObs_xPos_fut[2].append(sv_gt_local[t_ptr: t_ptr+F_FUR, 6*(veh-1): 6*veh].copy())
            else:
                sys.exit("Invalid vehicle")
    '''
    注意：此处"if obs[F_HIS*(veh+1) - 1][0] == 0 and..."用"F_HIS*(veh+1) - 1"即每个SV在obs中最后一时刻的信息有无来判断SV有无是合理的
                因为如果obs中最后时刻都没有SV，之后（规划时段）更不会有该SV
    '''
    return DynObs_curr, EV_curr, DynObs_xPos_fut, Ev_xPos_fut 
    
def get_uplo_markings(file_name  ):
    directory = 'data_processing/raw_highd_data'
    # file_name = "03_recordingMeta.csv"  #2 LANE
    file_path = os.path.join(directory, file_name)
    recordingMeta_df = pd.read_csv(file_path, index_col=0)
    up_markings = recordingMeta_df['upperLaneMarkings']
    lo_markings = recordingMeta_df['lowerLaneMarkings']
    
    # 使用str.split()按分号拆分为列表  
    split_lists_up = up_markings.str.split(';')  
    # 转换每个字符串为浮点数  
    float_lists_up = split_lists_up.apply(lambda x: [float(i) for i in x]).iloc[0]  
    
    # 使用str.split()按分号拆分为列表  
    split_lists_lo = lo_markings.str.split(';')  
    # 转换每个字符串为浮点数  
    float_lists_lo = split_lists_lo.apply(lambda x: [float(i) for i in x]).iloc[0]  
    
    up_markings_ls = [round(30 - val, 2) for val in float_lists_up]
    up_markings_ls = up_markings_ls[::-1]
    lo_markings_ls = float_lists_lo
    
    return up_markings_ls, lo_markings_ls

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


if __name__ == '__main__':                  
    reset()