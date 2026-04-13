import os, re
import torch
from torch_geometric.loader import DataLoader  #Data, Batch,
import numpy as np
import pandas as pd
import time
import gurobipy as gp
import array

import tra_plannning.replanning1 as env
from utils.res_exec_dynamically import *
from utils.res_kine import *
from utils.res_exec_shots import *
from utils.gt_data_visual import ev_gt_visualization
from config import *

import sys
sys.path.append("/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training")  
# from HighD_datapre import data_pre
if not old_version:
    from train_myway import VectorNet
else:
    from train_myway_before_changing_name import VectorNet   

from config_nw import *
from utils.visual_obs import visualize_graph


def longitudinal_planning(o, model):#m/s
    o_dl= DataLoader( [o], batch_size=1 )  #Dataloader里只有一条数据，由于VN需要Batch输入，所以就先这样笨笨的处理了...
    for batch in o_dl:
        o_bc = batch    
        
    res = model(o_bc)['pred']
    return np.array(res.detach().cpu()).reshape(-1)

def execute_replan(read_dir1, read_dir2, read_dir3, device, rec_sta, rec_end, case_sta, case_end):
    # global parameters
    cluster_global_ptr = 0   
    maxsteps_per_epoch  =1000

    files_ = os.listdir(read_dir2)  #60*4
    # 使用正则表达式匹配以'.~lock.'开头，并且以'#'结尾的文件  ,过滤掉临时文件
    pattern = r'^\.~lock.*#$'  
    files = [file for file in files_ if not re.match(pattern, file)]
    files.sort()   
    
    #初始化速度规划模型，由于是开环测试所以模型参数不会变化
    input_features = 15
    pred_len = F_FUR
    # get model and load parameters     
    lo_model = VectorNet(input_features, pred_len, device, with_aux= False).to(device)   
    if not old_version:
        lo_model.load_state_dict(torch.load( os.path.join( read_dir3, 'vel_pred_VN_100ep_lrdf09.pth')  ))    #output long. v  
    else:
        lo_model.load_state_dict(torch.load( os.path.join( read_dir3, 'VN_parameters_4_display.pth')  ))      #'VN_parameters_4.pth'
    
    #初始化路径规划优化模型，一次运行仅建立一次模型
    PathModel_glo, Vars_ls = lateral_op_model(pred_len)   #0.02 sec
    # PathModel_glo.write("model1.lp")

    
    
    #结果展示之——求解率
    num_succ_cases, num_total_cases = 0, 0
    num_succ_steps, num_total_steps = 0, 0
    fail_cases_arr, fail_steps_bycase_arr  = array.array('i') , array.array('i')  #所有失败案例的case index   类型码i为有符号32位整数；d为双精度浮点数

    #结果展示之——运行时间
    ave_planning_time_arr = array.array('f')    #每个case中，单步规划的运行时间的平均值
    planning_time_arr = array.array('f')  # 所有单次规划的运行时间的存储
    solving_time_arr = array.array('f') #所有单次规划优化模型的求解时间

    
    for rec in range(rec_sta, rec_end):  #13和14是3车道，15-18是2车道        12
        rMetafile_name = files[rec][-6:-4] + '_recordingMeta.csv'
        # tMetafile_name = files[rec][-6:-4] + '_tracksMeta.csv'
        recordingMeta = pd.read_csv(os.path.join(read_dir1, rMetafile_name))
        # tracksMeta = pd.read_csv(os.path.join(read_dir1, tMetafile_name))   #为了判断车辆行车方向
        tracks_df = pd.read_csv(os.path.join(read_dir2, str(files[rec])),  index_col=0)


        # 按 'id' 分组
        grouped = tracks_df.groupby('id')
        group_data = [group  for _, group in grouped]
        group_id =  [tra_id  for tra_id, _ in grouped]
        case_num =  len(group_id)
        case_end_ = case_end if  case_end < case_num else case_num
        

        for ep in range(case_sta, case_end_):    #多车/复杂的场景：109 119 267 348 200    epochs
            #initialize
            sx_ls, vx_ls, ax_ls, jx_ls, ttc_ls = [], [], [], [], []
            sy_ls, vy_ls, ay_ls, fai_ls = [], [], [], []            
            svs_kin_ls = [ [], [], [], [], [], []  ]
            ope_time_ep = 0   #该case的运行时间统计清零
            sx_plan_ls, sy_plan_ls = [], []   #存储所有时间步的规划结果，每个元素为一次的规划结果
            
            track_df = group_data[ep]

            #episode parameters
            t_ptr = F_HIS  #时间指针：观察序列中最后一个数据在整个cas时长中的时间索引，初始取值由观察序列时长参数决定
            
            # Prepare for interaction with environment
            o, sv_gt, cluster_global_ptr, line_num = env.reset(track_df, recordingMeta,  cluster_global_ptr)  
            '''attention abt sv_gt:
            每个case中的sv_gt依据初始时刻的EV最新坐标进行归一化处理
            '''
            '''维护一个数组存储当前时刻lateral planning结果，包括位置y和y关于x的一阶导、二阶导、三阶导，初始时刻设置为0
            实际上不是严格的运动学参数
            '''
            EV_curr_ykin = [0, 0, 0, 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ]  
            gt_total_t =  sv_gt.shape[0] - F_FUR #ground truth中ego vehicle具有多少个时间步的数据
            
            # #可视化真实数据中的运动学信息
            # if CONTINUE_SWITCH_GT:
            #     ev_gt_visualization(track_df, rec, ep,  SAVE_DIR1 )
                
            # #设置为匀速！！！！
            # o.to(device)   #gpu
            # o_a = o.clone()  #由于VN会改变input obs的内容，所以进行深复制
            # lo_res_ = longitudinal_planning(o_a, lo_model)
            # lo_res = np.array([lo_res_[0] + 2 for i in lo_res_] )
                
            for t in range(maxsteps_per_epoch):    
                # visualize_graph(o)
                
                #计时开始
                t_s = time.time()
                
                # o.to(device)   #gpu
                o_a = o.clone()  #由于VN会改变input obs的内容，所以进行深复制
                
                # # Step the env
                # o = o.to('cpu')  #cpu
                
                # #计时开始
                # t_s = time.time()
                #不设置为匀速！！！！
                lo_res = longitudinal_planning(o_a, lo_model)

                # if t==1:
                #     print("debugging point")
                
                o2, d, t_ptr, cluster_global_ptr, exec_ev_kin, EV_curr_ykin, exec_svs_kin, sx_plan, sy_plan, solving_time   = env.step(PathModel_glo, Vars_ls, o, lo_res, sv_gt, cluster_global_ptr, line_num, t_ptr, EV_curr_ykin, ep, t)  
                if_update = False  
                # #可视化VectorNet输入的图数据 
                # visualize_graph(o2)

                #优化模型清零
                # PathModel_glo.write("model2.lp")
                PathModel_glo.reset() #no warm start
                # PathModel_glo.write("model3.lp")
                PathModel_glo.remove(PathModel_glo.getConstrs()) 
                # PathModel_glo.remove(PathModel_glo.getGenConstrs() ) 
                # PathModel_glo.write("model4.lp")
                PathModel_glo.update() #
                # PathModel_glo.write("model5.lp")
                
                if not d:
                    # print("单步规划运行时间：%.4f sec" % (time.time() - t_s))  
                    ope_time_ep += time.time() - t_s  #sec
                    planning_time_arr.append(time.time() - t_s)  #extend将多个元素一次性添加到末尾
                    solving_time_arr.append(solving_time)
                    
                    if t > 0:   #不绘制第1帧图像
                        #纵向运动
                        sx_ls.append(exec_ev_kin[0])   #exec_ev_kin: (8,)   includes x, y, vx, vy, ax, ay, jx, steering angle
                        vx_ls.append(exec_ev_kin[2])
                        ax_ls.append(exec_ev_kin[4])
                        ttc_ls.append(exec_ev_kin[6])
                        
                        #横向运动
                        sy_ls.append(exec_ev_kin[1])
                        vy_ls.append(exec_ev_kin[3])
                        ay_ls.append(exec_ev_kin[5])
                        fai_ls.append(np.degrees( np.arctan( exec_ev_kin[7])  )  )  #np.degrees(): 弧度转为角度     np.arctan(): 正切值转为弧度
                        
                        sx_plan_ls.append(sx_plan + (exec_ev_kin[0] - sx_plan[0])) 
                        sy_plan_ls.append(sy_plan + (exec_ev_kin[1] - sy_plan[0]))
                        
                        #SV的运动学信息
                        for i in range(len(exec_svs_kin)):
                            svs_kin_ls[i].append(exec_svs_kin[i])
                    
                    num_succ_steps += 1
                    
                # End of epoch handling: 一个场景轨迹推演结束
                else:
                    t_sta_display, t_end_display = 45, 110   #需要手动更改！
                    '''
                    参数记录：
                    54_699: 30, 90
                    53_1189:  30, 80 
                    53_1091: 45, 110 

                    10_579:  35, 95
                    10_1424:  40, 100 
                    '''
                    t_sta_display2, t_end_display2 = 10, 1000    #需要手动更改！
                    '''
                    参数记录：others are 0,1000
                    53_1189:  10, 1000 
                    53_1091: 10, 1000 
                    '''
                    # 绘制一个场景中规划轨迹的真实执行结果      
                    if len(sx_ls) > 0:   #如果当前从0时刻就无法求解，则直接跳过绘图，否则引发bug
                        #For observing the executing process step by step
                        if CONTINUE_SWITCH_E:
                            gif_path = save_executing_gif(
                                o, sx_ls, sy_ls, fai_ls, sv_gt[1:, :],
                                rec, ep, SAVE_DIR3,
                                fps=10
                            )
                            print("GIF saved to:", gif_path)
                        else:
                            draw_trajectory_executing(
                                o, sx_ls, sy_ls, fai_ls, sv_gt[1:, :],
                                t_ptr, line_num, rec, ep
                            )

                        draw_trajectory_multi_shots(o, sx_ls, sy_ls, fai_ls,  sv_gt[1:, :], t_ptr, line_num, rec, ep)
                        draw_kine_res_merged(sx_ls, vx_ls, sy_ls, vy_ls,  fai_ls, svs_kin_ls, rec, ep, sample_rate = 0.1)
                        
                        # # 展示绘图用
                        draw_trajectory_replanning(o, sx_ls, sy_ls, fai_ls, sx_plan_ls, sy_plan_ls, rec, ep)
                        # draw_trajectory_multi_shots(o, sx_ls[t_sta_display: t_end_display], sy_ls[t_sta_display: t_end_display], 
                        #                              fai_ls[t_sta_display: t_end_display],  sv_gt[1+t_sta_display:, :], 
                        #                              t_ptr, line_num, rec, ep, t_sta_display)
                        # draw_kine_res_merged(sx_ls, vx_ls, sy_ls, vy_ls,  fai_ls, svs_kin_ls, rec, ep, sample_rate = 0.1, 
                        #                         t_sta=t_sta_display2, t_end=t_end_display2)

                        # '''
                        # svs_kin_ls = [svs_sx_ls, svs_vx_ls, svs_ax_ls, svs_sy_ls, svs_vy_ls, svs_ay_ls]'''
                        
                    
                    #统计求解率：ground truth中所有时刻均有可行解，定义为求解成功；其余为求解失败。
                    #优化模型无解的情况有两种：1、无可行安全走廊；2、模型无可行解。
                    if t_ptr > gt_total_t:   #成功求解
                        num_succ_cases += 1
                    else:    #无解情况    #注意idx均从0开始
                        fail_cases_arr.append( ep)
                        fail_steps_bycase_arr.append( (gt_total_t + 1) - t_ptr  )   #最大可能的步数-当前已规划的步数
                    num_total_steps += gt_total_t + 1 - F_HIS
                        
                    #统计运行时间
                    if t > 0:   #忽略完全无解情况，因此列表ave_ope_time的序号不对应每个case的索引！
                        ave_planning_time_arr.append(ope_time_ep/ t) #t+1-1

                    break  #退出内层循环

                # Super critical, easy to overlook step: make sure to update 
                o = o2
        
        num_total_cases += case_end_ - case_sta
    return num_succ_cases, num_total_cases, [num_succ_cases/num_total_cases, num_succ_steps/num_total_steps], [fail_cases_arr, fail_steps_bycase_arr], \
                    [ave_planning_time_arr, planning_time_arr, solving_time_arr]


def lateral_op_model(pred_len):
    PathModel_glo = gp.Model('MIQP') 
    
#---------------------------设置变量  0.001 sec
    l_ls = list(range(pred_len))
    ll_ls = list(range(pred_len))
    lll_ls = list(range(pred_len))
    llll_ls = list(range(pred_len))
    llabs_ls = list(range(pred_len))
    
    #此处上下界的约束包括了运动学约束
    #所有的运动学变量代表【瞬时】元素
    lateral_positions = PathModel_glo.addVars(l_ls, lb = -1000, ub = +1000, name = 'lp')     #横向位移   （m）,后续会修改
    lp_derivative1 = PathModel_glo.addVars(ll_ls, lb = -np.tan(MAX_ORI_ANGLE), ub = np.tan(MAX_ORI_ANGLE), name = 'lpd1')     #横向位移关于纵向位移的一阶导数      m/m     heading约束！ tangent 60 
    lp_derivative2 = PathModel_glo.addVars(lll_ls, lb = -3, ub = 3, name = 'lpd2')     #横向位置二阶导数      m/m^2
    lp_derivative3 = PathModel_glo.addVars(llll_ls, lb = -3, ub = 3, name = 'lpd3')     #横向位置三阶导数   用于提升平稳性
    lp1_for_abs = PathModel_glo.addVars(llabs_ls,  lb = 0, ub = 1.732, name = 'lp1_for_abs')   #用于将斜率转换为绝对值
    
    # ------------------------------构建用于构建绝对值函数的约束
    for i  in range(pred_len):   
        PathModel_glo.addGenConstrAbs(lp1_for_abs[i], lp_derivative1[i] )

    #------------------------------模型更新
    PathModel_glo.update()             
    
    return PathModel_glo, [lateral_positions, lp_derivative1, lp_derivative2, lp_derivative3, lp1_for_abs]



