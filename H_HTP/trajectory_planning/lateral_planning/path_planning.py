#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-02-21 11:00
# @Author  : Yujia Lu
# @Email   : 23111256@bjtu.edu.cn
'''
经验参数：
safety_lat_dis ：低速场景下应该变小，比如收费站，排队蠕动；高速场景下应该变大
fg_weight
fixed_ds：安全格的纵向长度

[v]去除SV横向位置不变的假设  07.15
[v]优化可行安全区间(s轴)的搜索方法，可应对l轴障碍物重叠的情况 07.15
[v]10.27 修正两个错误！: a.在实现l和l对各阶导数的运动学关系等式约束时，ds_NS计算为相邻离散点距离的2倍，相当于放大各变量；
                                                 b.建立运动学等式约束时，忽略与初始时刻的运动学关系约束，这与一开始三阶导数的错误等式约束有关。
                                                 
'''

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
import time
from scipy.interpolate import interp1d, UnivariateSpline
# import matplotlib.pyplot as plt

from config import *
from lateral_planning.visualization import path_plot_hd, kine_res_plot

#路径优化模型的建立与求解
def path_qp_model_modi(PathModel, Vars_ls, DynObs_curr, EV_curr, EV_curr_ykin, DynObs_xPos_fut, Ev_xPos_fut, lane_markings , ev_global_x0 = 0, case_id = 0, time_id = 0):
    start_t = time.time()
    
    flag = 0       #是否具有可行解的标记
    solutions = []
    res_lateral_positions =[]
    res_lp_derivative1 = []
    res_lp_derivative2 = []
    res_lp_derivative3 = []
    run_time_pa = -99999
    path_func = []
    
    
    #---车辆参数设置
    max_veh_slength = np.sqrt(VEH_LEN**2 + VEH_WID**2)  #车身在x轴方向上可能最大长度

    #---纵向空间离散化参数设置
    n_xPos = Ev_xPos_fut.shape[0]
    #---纵向规划结果的离散点和将要进行的侧向规划的离散点一一对应
    SafetySpace1_sloc = Ev_xPos_fut.copy()  
    
    '''
    #优化模型中EV 边界的建模：
    纵向：以纵向规划结果中的离散点为中心，分别向两侧延伸half_ds，这意味着安全走廊搜索中的安全格的纵向长度为ds。
                注意，ds的取值和速度为正相关关系，并且具有一定取值范围[EV沿纵轴最大长度，一个经验最大值]
    横向：从变量l出发，分别向两侧延伸epslion+safe_distance+考虑转向角的进一步离散化
                注意，这里用到了dds，所以其实ds和dds不是针对同一方向边界而设置的变量，但它们之间有乘法联系，所以总会引起迷惑。
    '''
    # ds-v_EV线性函数的斜率k取值为0.4。这样处理下的车头间距还是偏小的，但由于该方法是被动换道。
    #fixed_ds的经验取值对求解率的影响较大！谨慎改动
    half_ds_NS = np.zeros(n_xPos)
    dds_NS = np.zeros(n_xPos)
    for ds_id in range(n_xPos):
        fixed_ds = 0.5*EV_curr[2]  #    应该改为每一个离散点对应的fixed_ds！！！  0.4*v_EV[ds_id]  
        ds = fixed_ds if fixed_ds > max_veh_slength else max_veh_slength
        half_ds_NS[ds_id] = 0.5*ds
        dds_NS[ds_id] = ds/LAMBDA_N     

    #-------------------------安全走廊+避撞约束中上下界
    #搜索安全走廊，并确定安全走廊的纵向长度(last_grid_idx+1)*ds
    #---确定每个安全格的纵向长度
    SafetySpace1_lub, SafetySpace1_llb, SafetySpace1_sloc, last_grid_idx= safety_grids_extraction(DynObs_curr, EV_curr, DynObs_xPos_fut, Ev_xPos_fut, lane_markings, n_xPos, half_ds_NS)
    ds_idx_max = last_grid_idx + 1  #安全走廊的纵向长度
    
    #无安全走廊的判定
    if ds_idx_max < 2:    #至少3个样本点才可以进行样条插值
        print("当前无可行的安全走廊！")
        return flag, res_lateral_positions, res_lp_derivative1, SafetySpace1_sloc, path_func, EV_curr_ykin,run_time_pa
        
        # sys.exit()
    
        
    #获取避撞约束中上下界
    SafetySpace2_lub, SafetySpace2_llb, SafetySpace2_sloc = obs_lbound_extraction(dds_NS, half_ds_NS, n_xPos, SafetySpace1_lub, SafetySpace1_llb, SafetySpace1_sloc, ds_idx_max, EV_curr[0])


    # ************************参考轨迹生成
    #搜索推荐车道，返回车道中线ys
    guide_lane_ls = get_guide_lane(ds_idx_max, SafetySpace1_lub, SafetySpace1_llb, EV_curr, DynObs_curr, lane_markings)

    
    # ************************最终轨迹生成
    t_bm_s = time.time()
    
    # #---------------------------设置变量  0.001 sec
    # l_ls = list(range(ds_idx_max))
    # ll_ls = list(range(ds_idx_max))
    # lll_ls = list(range(ds_idx_max))
    # llll_ls = list(range(ds_idx_max))
    # llabs_ls = list(range(ds_idx_max))
    
    # #此处上下界的约束包括了运动学约束
    # #所有的运动学变量代表【瞬时】元素
    # lateral_positions = PathModel.addVars(l_ls, lb = lane_markings[0], ub = lane_markings[-1], name = 'lp')     #横向位移   （m）
    # lp_derivative1 = PathModel.addVars(ll_ls, lb = -1.732, ub = 1.732, name = 'lpd1')     #横向位移关于纵向位移的一阶导数      m/m     heading约束！ tangent 60 
    # lp_derivative2 = PathModel.addVars(lll_ls, lb = -3, ub = 3, name = 'lpd2')     #横向位置二阶导数      m/m^2
    # lp_derivative3 = PathModel.addVars(llll_ls, lb = -3, ub = 3, name = 'lpd3')     #横向位置三阶导数   用于提升平稳性
    # lp1_for_abs = PathModel.addVars(llabs_ls,  lb = 0, ub = 1.732, name = 'lp1_for_abs')   #用于将斜率转换为绝对值
    
    #修改横向位移变量的上下界
    for i in range(ds_idx_max):
        Vars_ls[0][i].LB = lane_markings[0]
        Vars_ls[0][i].UB = lane_markings[-1]
    
    '''
    创建条件用去0.02-0.03 sec左右！
    '''
    #------------------------------构建光滑连续性约束
    '''    
    EV_curr_ykin = [l, ld, ld2, ld3]
    '''
    #每个离散点之间的纵向距离计算
    ds_xPos = np.zeros(n_xPos)
    for ds_id in range(n_xPos):
        ds_xPos[ds_id] = SafetySpace1_sloc[ds_id] - EV_curr[0] if ds_id == 0 else SafetySpace1_sloc[ds_id] - SafetySpace1_sloc[ds_id - 1]
    #第一个时间步/纵向离散点处的条件需要用到已知信息（每次规划中初始时刻的运动学信息）
    #在第一个时间步，只有lp_derivative3[0]是变量，其他都是常量！！！
    PathModel.addConstr(Vars_ls[2][0] == EV_curr_ykin[2] + Vars_ls[3][0]*ds_xPos[0])    
    PathModel.addConstr(Vars_ls[1][0] == EV_curr_ykin[1] + Vars_ls[2][0]*ds_xPos[0] + 0.5*Vars_ls[3][0]*(ds_xPos[0]**2)  )
    PathModel.addConstr(Vars_ls[0][0] == EV_curr_ykin[0] + Vars_ls[1][0]*ds_xPos[0] + 0.5*Vars_ls[2][0]*(ds_xPos[0]**2) + (1/6)*Vars_ls[3][0]*(ds_xPos[0]**3)  )

    #后续时间步：注意ds_NS(纵向离散点之间distance)的索引和模型变量数组索引不一致
    #事实上所有变量都可以转换为仅包含lp_derivative3的表达式？
    for i in range(1, ds_idx_max):
        PathModel.addConstr(Vars_ls[2][i] == Vars_ls[2][i-1] + Vars_ls[3][i]*ds_xPos[i])    
    for i in range(1, ds_idx_max):
        PathModel.addConstr(Vars_ls[1][i] == Vars_ls[1][i-1] + Vars_ls[2][i]*ds_xPos[i] + 0.5*Vars_ls[3][i]*(ds_xPos[i]**2) )
    for i in range(1, ds_idx_max ):
        PathModel.addConstr(Vars_ls[0][i] == Vars_ls[0][i-1] + Vars_ls[1][i]*ds_xPos[i] + 0.5*Vars_ls[2][i]*(ds_xPos[i]**2) + (1/6)*Vars_ls[3][i]*(ds_xPos[i]**3) )

    
    # # ------------------------------构建用于构建绝对值函数的约束
    # for i  in range(ds_idx_max):   
    #     PathModel.addGenConstrAbs(lp1_for_abs[i], lp_derivative1[i] )
        
        
    #------------------------------构建防止碰撞约束：从safety_grids_extraction获取l取值范围
    #简单线性化方法获得线性函数表达式
    slope,  intercept = epsilon_linearization()   
    #在表示车身的上下界中加入一个横向安全距离参数，以保证对车辆边界的完全覆盖
    expressions_lub, expressions_llb, evSpace_sloc= ev_lbound_generation(SafetySpace1_sloc, dds_NS, ds_idx_max, Vars_ls[4], Vars_ls[1],  Vars_ls[0], slope, intercept)

    safety_lat_dis = 0.3  #+EV_curr[2]*0.01   #与障碍物至少保持一定横向距离      单位：m     收费站宽度大概3.2m，一般车宽不超过2m，0.3m可以通过收费站

    for ds_idx  in range(ds_idx_max):   
        for j in range(LAMBDA_N):
            #所有车身区间的上界<=安全区间上界
            PathModel.addConstr(  expressions_lub[j + LAMBDA_N*ds_idx] + safety_lat_dis <= SafetySpace2_lub[ds_idx, j])  
            #所有车身区间的下界>=安全区间下界
            PathModel.addConstr(  expressions_llb[j + LAMBDA_N*ds_idx] - safety_lat_dis >= SafetySpace2_llb[ds_idx, j])

    
    #------------------------------设置目标函数
    
    '''
    设置目标函数用0.005 sec左右
    '''
    coefficient1 = 1
    coefficient2 = 500   #200可以勉强抵达道路中线   300无法抵达
    coefficient3 = 500
    coefficient4 = 500
    coefficient5 = 500
    coefficient6 = 500
    ob_epr1 = gp.QuadExpr(sum(   (Vars_ls[0][i] - guide_lane_ls[i]  )**2 for i in range(ds_idx_max)  ))        #traffic rules  
    ob_epr2 = gp.QuadExpr(sum(Vars_ls[1][i]**2 for i in range(ds_idx_max)   ))      
    ob_epr3 = gp.QuadExpr(sum(Vars_ls[2][i]**2 for i in range(ds_idx_max)   ))     
    ob_epr4 = gp.QuadExpr(sum(Vars_ls[3][i]**2 for i in range(ds_idx_max)   ))     
    ob_epr5 = gp.QuadExpr(sum((Vars_ls[1][i] - Vars_ls[1][i-1] )**2 for i in range(1, ds_idx_max) ) + (Vars_ls[1][0] - EV_curr_ykin[1] )**2 )  
    ob_epr6 = gp.QuadExpr(sum((Vars_ls[2][i] - Vars_ls[2][i-1] )**2 for i in range(1, ds_idx_max) ) + (Vars_ls[2][0] - EV_curr_ykin[2] )**2 )  
    PathModel.setObjective(coefficient1 * ob_epr1 + coefficient2 * ob_epr2 + coefficient3 * ob_epr3 + coefficient4 *ob_epr4 + coefficient5 *ob_epr5 + coefficient6 *ob_epr6, GRB.MINIMIZE)  # + coefficient5 *ob_epr5 + coefficient6 *ob_epr6

 
    '''
    模型求解  0.01 sec 左右 
    '''
    #------------------------------模型求解
    PathModel.optimize()             
    
    # #建立array用于结果存储
    # EVSpace_lub = np.zeros((n_xPos, LAMBDA_N))   #l轴   横轴   上界
    # EVSpace_llb = np.zeros((n_xPos, LAMBDA_N))   #l轴   横轴   下界
    

    #********存储与可视化计算结果
    if PathModel.Status == GRB.OPTIMAL :
        flag = 1  
        solutions = PathModel.getAttr ('X')        #自变量值
        # obj_res = PathModel.getObjective().getValue()   #目标函数值
        run_time_gb = PathModel.Runtime
    
    
        #分别存储
        res_lateral_positions =  solutions[: ds_idx_max] 
        res_lp_derivative1 =  solutions[ds_idx_max : ds_idx_max*2 ] 
        res_lp_derivative2 =  solutions[ds_idx_max* 2 : ds_idx_max*3 ] 
        res_lp_derivative3  =  solutions[ds_idx_max * 3 : ds_idx_max*4  ] 
        res_lp1_abs  =  solutions[ds_idx_max * 4  : ds_idx_max*5  ] 
        
        # #存储优化得到的考虑车身尺寸的安全走廊
        # for ds_idx in range(ds_idx_max):
        #     res_extra_dis = res_lp1_abs[ds_idx] *slope + intercept+ res_lp1_abs[ds_idx] * dds_NS[ds_idx]*0.5   #safety_dis_cf 
        #     res_lub_dn =  res_lateral_positions[ds_idx] + res_extra_dis
        #     res_llb_dn =  res_lateral_positions[ds_idx] - res_extra_dis
            
        #     EVSpace_lub[ds_idx, 0] = res_lub_dn - (dds_NS[ds_idx]*0.5 + 2*dds_NS[ds_idx]) *  res_lp_derivative1 [ds_idx]
        #     EVSpace_lub[ds_idx, 1] = res_lub_dn - (dds_NS[ds_idx]*0.5 + dds_NS[ds_idx]) *  res_lp_derivative1 [ds_idx]
        #     EVSpace_lub[ds_idx, 2] = res_lub_dn - 0.5 * dds_NS[ds_idx] *  res_lp_derivative1 [ds_idx]
        #     EVSpace_lub[ds_idx, 3] = res_lub_dn + 0.5 * dds_NS[ds_idx] *  res_lp_derivative1 [ds_idx]
        #     EVSpace_lub[ds_idx, 4] = res_lub_dn + (dds_NS[ds_idx]*0.5 + dds_NS[ds_idx]) *  res_lp_derivative1 [ds_idx]
        #     EVSpace_lub[ds_idx, 5] = res_lub_dn + (dds_NS[ds_idx]*0.5 + 2*dds_NS[ds_idx]) *  res_lp_derivative1 [ds_idx]
            
        #     EVSpace_llb[ds_idx, 0] = res_llb_dn - (dds_NS[ds_idx]*0.5 + 2*dds_NS[ds_idx]) *  res_lp_derivative1 [ds_idx]
        #     EVSpace_llb[ds_idx, 1] = res_llb_dn - (dds_NS[ds_idx]*0.5 + dds_NS[ds_idx]) *  res_lp_derivative1 [ds_idx]
        #     EVSpace_llb[ds_idx, 2] = res_llb_dn -  0.5 * dds_NS[ds_idx] *  res_lp_derivative1 [ds_idx]
        #     EVSpace_llb[ds_idx, 3] = res_llb_dn + 0.5 * dds_NS[ds_idx] *  res_lp_derivative1 [ds_idx]
        #     EVSpace_llb[ds_idx, 4] = res_llb_dn + (dds_NS[ds_idx]*0.5 + dds_NS[ds_idx]) *  res_lp_derivative1 [ds_idx]
        #     EVSpace_llb[ds_idx, 5] = res_llb_dn + (dds_NS[ds_idx]*0.5 + 2*dds_NS[ds_idx]) *  res_lp_derivative1 [ds_idx]
        
    
    '''
    lateral_positions :   solutions[:n_xPos*LAMBDA_N] 
    lp_derivative1:   solutions[n_xPos*LAMBDA_N : n_xPos*LAMBDA_N*2 ] 
    lp_derivative2:   solutions[n_xPos*LAMBDA_N * 2 : n_xPos*LAMBDA_N*3 ] 
    lp_derivative3 :  solutions[n_xPos*LAMBDA_N * 3 : n_xPos*LAMBDA_N*4 - 1 ] 
    lp1_abs :  solutions[n_xPos*LAMBDA_N * 4 -1 : n_xPos*LAMBDA_N*4 -1 + n_xPos] 
    '''

    # #绘图检查安全走廊
    # safety_corridor_plot(SafetySpace1_lub, SafetySpace1_llb, SafetySpace1_sloc, StaObs, DynObs_curr, EV_curr,  Ave_xVels, Ave_yVels, Ave_xVels_ev, ds_idx_max, lane_markings)
    
    # run_time_pa = time.time() - start_t
    # print(f"path model求解时间: {run_time_pa:.4f} 秒")
    
    #依据采样点生成拟合曲线
    if flag == 1:
        # path_func = spine_path_generation(res_lateral_positions, SafetySpace1_sloc, EV_curr)  #没有用
        
        # if time_id >40 and time_id % 10 == 4:   # time_id % 10 == 1:
        #     #绘图检查优化结果
        #     path_plot_hd(SafetySpace1_lub, SafetySpace1_llb, SafetySpace1_sloc,  half_ds_NS, dds_NS, 
        #                 EVSpace_lub, EVSpace_llb, SafetySpace2_sloc, 
        #                 DynObs_curr, EV_curr,  DynObs_xPos_fut, Ev_xPos_fut, 
        #                 lane_markings, ev_global_x0, ds_idx_max, 
        #                 res_lateral_positions, res_lp_derivative1, run_time_pa, path_func, guide_lane_ls, 
        #                 case_id, time_id)      #          
        #     kine_res_plot(case_id, time_id, EV_curr, SafetySpace1_sloc, res_lateral_positions, res_lp_derivative1)

        
        #更新最新时刻的横向运动规划结果
        EV_curr_ykin = [EV_curr[1], res_lp_derivative1[0], res_lp_derivative2[0], res_lp_derivative3[0] ]
        
        return flag, res_lateral_positions, res_lp_derivative1, SafetySpace1_sloc, path_func, EV_curr_ykin, run_time_gb
            
    print("当前无可行的path！")
    return flag, None, None, None, None, None, 0


#构建安全走廊：输入周边环境感知信息，输出安全走廊中各个安全格的位置（SafetySpace1_lub, SafetySpace1_llb, SafetySpace1_sloc）以及走廊长度（last_grid_idx）
'''
说明：
(1)每个安全格的纵向长度由NS决定，即path 采样点数量与安全格的数量一致
(2)在搜索安全区间的过程中，动态障碍物和静态障碍物的处理是不一样的，动态障碍物需要考虑在EV行进过程中其位置随时间的变化
'''

def safety_grids_extraction(DynObs_curr, EV_curr, DynObs_xPos_fut, Ev_xPos_fut, lane_markings, n_xPos, half_ds_NS):   
    #-------------------------------初始化参数
    lane_num = len(DynObs_curr)
    h_min = lane_markings[0]    #假设EV初始时刻l轴位置(l = 0)为车道中心线，下边界为车道边线
    h_max =lane_markings[-1] #假设环境为两车道
    

    SafetySpace1_sloc = Ev_xPos_fut.copy()  #s轴  纵轴
    SafetySpace1_lub = np.zeros(n_xPos)   #l轴   横轴   上界
    SafetySpace1_llb = np.zeros(n_xPos)   #l轴   横轴   下界
    
    fg_weight = 0.5
            
    #-----搜索每个区间内的动态障碍物index +  前后侧边界坐标
    # dy_obstacle_idxs = [[] for _ in range(n_xPos)]  #索引 
    dy_obstacle_sub = [[] for _ in range(n_xPos)]  #纵向上界
    dy_obstacle_slb = [[] for _ in range(n_xPos)]  #纵向下界
    dy_obstacle_lbs = [[] for _ in range(n_xPos)]  #横向上下界
    
    
    for ds_id in range(n_xPos):
        ds_half = half_ds_NS[ds_id]
        start_line = SafetySpace1_sloc[ds_id] - ds_half
        end_line = SafetySpace1_sloc[ds_id] + ds_half
        
            
        for lane in range(lane_num):
            for veh in range(len(DynObs_xPos_fut[lane])):
                #判断条件为三种情形：前侧边界在该区间内  / 后侧边界在该区间内  / 整个车身横跨该区间 。 满足之一则认为该车是该区间的障碍物
                font_line = DynObs_xPos_fut[lane][veh][ds_id][0] + 0.5*VEH_LEN
                back_line = DynObs_xPos_fut[lane][veh][ds_id][0] - 0.5*VEH_LEN 
                if ( (font_line <= end_line) and (font_line  >= start_line) )  or   ( (back_line <= end_line) and (back_line  >= start_line) ) or   ( (font_line >= end_line) and (back_line  <= start_line)  ) :   #如果该环境车经过时间time_in_this_ds，在这个区间内
                    dy_obstacle_slb[ds_id].append(back_line)
                    dy_obstacle_sub[ds_id].append(font_line)
                    dy_obstacle_lbs[ds_id].append(  (DynObs_xPos_fut[lane][veh][ds_id][1] - 0.5 * VEH_WID, DynObs_xPos_fut[lane][veh][ds_id][1] + 0.5 * VEH_WID)  )
                    '''啊！忽然发现这里忽略了SV的在换道时的steering angle，问题不大。。。有时间再改'''

        
    #-------------------------------开始搜索安全走廊的范围    
    last_grid_idx = -1      #最后一个唯一可行空间格的id，如果无安全走廊，则安全走廊长度(last_grid_idx+1)=0
    for ds_idx in range(n_xPos):
        # srange_start = EV_curr[0] + ds_idx * ds         
        # srange_end = EV_curr[0] + (ds_idx+1) * ds

        
         #如果该区间内不存在障碍物
        if len(dy_obstacle_lbs[ds_idx]) == 0 : 
            SafetySpace1_lub[ds_idx] = h_max
            SafetySpace1_llb[ds_idx] = h_min
                
        #如果该区间内存在障碍物
        else:         
            #step1：初始可行安全区间
            #采用扫瞄线算法，算法复杂度为O(nlogn); 当前只考虑Dynamic Obstacles
            # Step 1: Initialize the result list
            uncovered_intervals = []
            # Step 2: Sort the intervals based on the start point, and then by the end point
            sorted_intervals = sorted(dy_obstacle_lbs[ds_idx].copy(), key=lambda x: (x[0], x[1]))
            # Step 3: Initialize the scanning line
            line = h_min
            # Step 4: Process each interval
            for a_i, b_i in sorted_intervals:
                # If there is an uncovered interval before the current interval
                if line < a_i:
                    uncovered_intervals.append((line, a_i))
                # Move the scanning line to the end of the current interval
                line = max(line, b_i)
            # Step 5: Check for the last uncovered interval after the last sorted interval
            if line < h_max:
                uncovered_intervals.append((line, h_max))


            #step2：有效安全区间：符合>vehcle length + delta l
            '''
            当前假设所有车沿中心线行驶情景下，两个相邻车道车辆之间的空隙= 1.95 m > 车宽1.9 m，所以会出现走廊图中细小的缝隙的安全空间
            '''
            feasible_dis_ls = [uncovered_intervals[i][1] - uncovered_intervals[i][0] for i in range(len(uncovered_intervals))]

            validdis_ls = [dis for idx, dis in  enumerate(feasible_dis_ls) if dis > (VEH_WID + 1) ]  #有效安全区间对应的横向距离        #修改1224：加入经验安全距离系数
            valididx_ls = [idx for idx, dis in  enumerate(feasible_dis_ls) if dis > (VEH_WID + 1) ]  #有效安全区间在可行区间集中对应的id
            #如果该s区间内不存在有效安全栅格，则中止当前时刻对sn长度内安全走廊的搜索
            if len(valididx_ls) == 0:  
                SafetySpace1_lub[ds_idx] = np.nan         
                SafetySpace1_llb[ds_idx] = np.nan #外界执行时可用np.isnan()函数判断是否到走廊终点
                break   
        
            #step3：唯一安全区间
            final_lub, final_llb = -999999, -999999          #该ds内最终的安全区间格l轴上下界
            nearest_dis = 10000000  #存储按照就近原则计算出的最短距离和对应的区间id
            nearest_idx = 10000000
            #对于首个ds区域，选择与EV最近的有效安全区间
            if ds_idx == 0:  
                for i in valididx_ls:
                    #计算EV所在l值与第i个格子中点y值的距离
                    nearest_dis_candi = abs( EV_curr[1] - 0.5*(uncovered_intervals[i][0] + uncovered_intervals[i][1]) )  
                    if nearest_dis_candi < nearest_dis:    
                        nearest_dis = nearest_dis_candi
                        nearest_idx = i
                final_lub, final_llb = uncovered_intervals[nearest_idx][1], uncovered_intervals[nearest_idx][0]
                #将ds内所有dds存储为相同上下界l取值
                SafetySpace1_lub[ds_idx] = final_lub
                SafetySpace1_llb[ds_idx] = final_llb
                    
            #后续安全区间，按照就近原则取唯一区间
            else:     
                for i in valididx_ls:
                    #注意SafetySpace1_llb[ds_idx-1, -1]中的-1表示用ds中二次离散中的最后一个dds小区间的中点l值
                    nearest_dis_candi = ( 
                                         (1 - fg_weight) * abs( 0.5*(SafetySpace1_llb[ds_idx-1] + SafetySpace1_lub[ds_idx-1]) - 0.5*(uncovered_intervals[i][0] + uncovered_intervals[i][1]) ) +
                                         fg_weight * abs( EV_curr[1] - 0.5*(uncovered_intervals[i][0] + uncovered_intervals[i][1]) )
                                         )  #每个离散区间（栅格）中心之间的纵向（s轴）距离是相同的，因此中心之间的距离仅用计算横向距离
                    if nearest_dis_candi < nearest_dis:    
                        nearest_dis = nearest_dis_candi
                        nearest_idx = i
                final_lub = uncovered_intervals[nearest_idx][1]
                final_llb  = uncovered_intervals[nearest_idx][0]
    
                #以下三行输入是为了初步绘图检查（未考虑车身尺寸的安全走廊）
                SafetySpace1_lub[ds_idx] = final_lub
                SafetySpace1_llb[ds_idx] = final_llb
                
        #存储安全走廊中最后一个格子的index
        last_grid_idx = ds_idx
        
    return SafetySpace1_lub, SafetySpace1_llb, SafetySpace1_sloc, last_grid_idx


#基于安全走廊获取避撞约束对应安全区间的上下界：输入代表车身分割次数的系数lambda（知道该系数就可以推算获得区间dds长度），输出所有避撞约束空间对应的安全区间上下界l值（障碍物的边界）
def obs_lbound_extraction(dds_NS, half_ds_NS, n_xPos, SafetySpace1_lub, SafetySpace1_llb, SafetySpace1_sloc, ds_idx_max, ev_x0):
    #纵向空间离散化参数设置
    SafetySpace2_sloc = np.zeros( (n_xPos, LAMBDA_N) )  #s轴  注意是区间的中点位置
    SafetySpace2_lub = np.ones( (n_xPos, LAMBDA_N) ) * 10000 #l轴     上界
    SafetySpace2_llb = np.ones( (n_xPos, LAMBDA_N) ) * (-10000)   #l轴      下界
    #刚开始和最后个别dds可能不属于任何ds，则初始值相当于没约束
    
    #s轴位置
    data = list(range(LAMBDA_N))     #[0,1,2,3]
    mid = LAMBDA_N // 2  
    left = data[:mid][::-1]           #[0, 1]         [::-1]  是将list元素反转
    right = data[mid:]                 #[2,3]
    i = 0
    for ds_id in range(ds_idx_max):
        for pair in zip(left, right):
            SafetySpace2_sloc[ds_id, pair[0]] = SafetySpace1_sloc[ds_id] - dds_NS[ds_id] * 0.5 - dds_NS[ds_id]*i              
            SafetySpace2_sloc[ds_id, pair[1]] = SafetySpace1_sloc[ds_id] + dds_NS[ds_id] * 0.5 + dds_NS[ds_id]*i
            i +=1
        i = 0
    
    #l轴上界约束值
    '''
    half_ds = SafetySpace1_sloc[0] - ev_x0
    for ds_id in range(ds_idx_max):
        #由于后来按照时间长度而不是空间长度分组，所以每个区间的空间长度不同需要依次判断，否则会出现区间之间的空白区域
        half_ds = (SafetySpace1_sloc[ds_id] - SafetySpace1_sloc[ds_id -1])*0.5 if ds_id > 0 else half_ds   
        for dds_id in range(LAMBDA_N):  #对于每一个dds区间
            #以当前ds_id为依据，生成包括左、右id的两个列表（不包括当前ds_id）
            data = list(range(ds_idx_max))
            position  = ds_id
            left = data[:position][::-1]
            right = data[position+1:]
            
            #先从EV质心所在的ds位置判断一下，如果在ds内就退出循环
            if SafetySpace2_sloc[ds_id, dds_id] < (SafetySpace1_sloc[position] + half_ds) and SafetySpace2_sloc[ds_id, dds_id] > (SafetySpace1_sloc[position] - half_ds):
                SafetySpace2_lub[ds_id, dds_id] = SafetySpace1_lub[position]
                SafetySpace2_llb[ds_id, dds_id] = SafetySpace1_llb[position]
            else:
                #如果该dds不属于ds区间内，则以该ds位置为起始分别向两侧搜索
                if dds_id < (LAMBDA_N /2):     #0，1向前搜索，2，3向后搜索
                    for l_id in left:
                        if SafetySpace2_sloc[ds_id, dds_id] < (SafetySpace1_sloc[l_id] + half_ds) and SafetySpace2_sloc[ds_id, dds_id] > (SafetySpace1_sloc[l_id] - half_ds):
                            SafetySpace2_lub[ds_id, dds_id] = SafetySpace1_lub[l_id]
                            SafetySpace2_llb[ds_id, dds_id] = SafetySpace1_llb[l_id]
                            break
                else:
                    for r_id in right:
                        if SafetySpace2_sloc[ds_id, dds_id] < (SafetySpace1_sloc[r_id] + half_ds) and SafetySpace2_sloc[ds_id, dds_id] > (SafetySpace1_sloc[r_id] - half_ds):
                            SafetySpace2_lub[ds_id, dds_id] = SafetySpace1_lub[r_id]
                            SafetySpace2_llb[ds_id, dds_id] = SafetySpace1_llb[r_id]
                            break
    '''
    #由于2024/12修改后的安全格纵向长度一定大于EV最大车长，所以不必分条件讨论
    for ds_id in range(ds_idx_max):
        for dds_id in range(LAMBDA_N):  #对于每一个dds区间):
            SafetySpace2_lub[ds_id, dds_id] = SafetySpace1_lub[ds_id]
            SafetySpace2_llb[ds_id, dds_id] = SafetySpace1_llb[ds_id]
            
    return SafetySpace2_lub, SafetySpace2_llb, SafetySpace2_sloc

#第二步：输入初步的安全走廊，输出gurobi线性表达式形式的上下界关于斜率与质心横向位置的函数值（ev_lub_exprs, ev_llb_exprs, evSpace_sloc）
def ev_lbound_generation(SafetySpace1_sloc, dds_NS, ds_idx_max, path_snk_abs, path_snk, centroid_lp, slope,  intercept):   
    #进一步纵向离散的array及表达式存储
    evSpace_sloc = np.zeros( (ds_idx_max, LAMBDA_N) )  
    ev_llb_exprs = []
    ev_lub_exprs = []
    

    for ds_idx in range(ds_idx_max):

        # slope_abs = PathModel.addGenConstrAbs(slope)
        epsilon_n = path_snk_abs[ds_idx] *slope + intercept#此处path_snk_abs是斜率的绝对值
        safety_dis_cf = path_snk_abs[ds_idx] * dds_NS[ds_idx]*0.5               #使得上下边界完全包裹车身
        lub_dn =  centroid_lp[ds_idx] + epsilon_n + safety_dis_cf                         #epsilon_n大概变化0.9~1.7m左右
        llb_dn =  centroid_lp[ds_idx] - epsilon_n - safety_dis_cf
        
        #step5：考虑航向角的最终安全区间（将安全区间n进一步离散为4个，并沿道路法线进行平移）
        #每个大的纵向空间的中点s轴值，以该点为中点向两侧延伸，获得进一步纵向离散的四个空间位置
        # sloc0 =  SafetySpace1_sloc[ds_idx]  #注意sloc0是大空间格中间位置，但不是小空间格中点位置
        # #纵向s轴，记录为每个小空间格中点位置，代表样本点的s位置
        # evSpace_sloc[ds_idx, 0] = sloc0 - dds_NS[ds_idx]*0.5 - dds_NS[ds_idx]
        # evSpace_sloc[ds_idx, 1] = sloc0 - dds_NS[ds_idx]*0.5
        # evSpace_sloc[ds_idx, 2] = sloc0 + dds_NS[ds_idx]*0.5
        # evSpace_sloc[ds_idx, 3] = sloc0 + dds_NS[ds_idx]*0.5 + dds_NS[ds_idx]
        
        #横向l轴上界
        ev_lub_exprs.append(lub_dn - (dds_NS[ds_idx]*0.5 + 2*dds_NS[ds_idx]) *  path_snk[ds_idx])
        ev_lub_exprs.append(lub_dn - (dds_NS[ds_idx]*0.5 + dds_NS[ds_idx]) *  path_snk[ds_idx])
        ev_lub_exprs.append(lub_dn - 0.5 * dds_NS[ds_idx] *  path_snk[ds_idx])
        ev_lub_exprs.append(lub_dn + 0.5 * dds_NS[ds_idx] *  path_snk[ds_idx])
        ev_lub_exprs.append(lub_dn + (dds_NS[ds_idx]*0.5 + dds_NS[ds_idx]) *  path_snk[ds_idx])
        ev_lub_exprs.append(lub_dn + (dds_NS[ds_idx]*0.5 + 2*dds_NS[ds_idx]) *  path_snk[ds_idx])
        #横向l轴下界
        ev_llb_exprs.append(llb_dn - (dds_NS[ds_idx]*0.5 + 2*dds_NS[ds_idx]) *  path_snk[ds_idx])
        ev_llb_exprs.append(llb_dn - (dds_NS[ds_idx]*0.5 + dds_NS[ds_idx]) *  path_snk[ds_idx])
        ev_llb_exprs.append(llb_dn -  0.5 * dds_NS[ds_idx] *  path_snk[ds_idx])
        ev_llb_exprs.append(llb_dn + 0.5 * dds_NS[ds_idx] *  path_snk[ds_idx])
        ev_llb_exprs.append(llb_dn + (dds_NS[ds_idx]*0.5 + dds_NS[ds_idx]) *  path_snk[ds_idx])
        ev_llb_exprs.append(llb_dn + (dds_NS[ds_idx]*0.5 + 2*dds_NS[ds_idx]) *  path_snk[ds_idx])
            
            
    
    return  ev_lub_exprs, ev_llb_exprs, evSpace_sloc
        
#精细化安全区间的线性化
def epsilon_linearization():
    '''
    斜率k对应的角度是航向角（车辆中线与坐标轴X轴夹角）
    车辆换道过程中的航向角sita_n有一定取值范围[sita_min, MAX_ORI_ANGLE]
    相应地,斜率k=tan(sita_n)也有取值范围[tan(sita_min), tan(MAX_ORI_ANGLE)]
    
    此处建立斜率k到epsilon的函数映射：epsilon = f1(k) = np.sqrt(1 + k**2) * veh_width * 0.5  ==>     epsilon = f2(k) =[ m * abs(k) + n ]
    线性化方法采用两个端点连线的函数表达式（获得m*k+n，由于f1(k)为对称函数)，两个端点分别为原函数在x=0和x=tan(MAX_ORI_ANGLE)]处的边界点

    '''
    
    funx  = lambda x: np.sqrt(1 + x**2) * VEH_WID * 0.5 
    
    x1, y1 = 0, funx(0)
    x2, y2 = np.tan(MAX_ORI_ANGLE), funx(  np.tan(MAX_ORI_ANGLE)  )

    slope = (y2 - y1) / (x2 - x1)    #斜率
    intercept = y1 - slope * x1    #常数项
    
    return slope, intercept
        
        
def get_guide_lane(ds_idx_max, SafetySpace1_lub, SafetySpace1_llb, EV_curr, DynObs_curr, lane_markings):
    '''
    获取目标函数中的guideline:
    if ev所在车道属于安全走廊：
        guide lane保持上一时刻所在车道
    else：
        guide lane = 最靠近安全走廊中心的车道中线
    '''
    # 参数设置
    if DO_SMOOTH==1:
        guide_lane_ls = [EV_curr[1]]
    else:
        guide_lane_ls = []
    guide_lane_last = -9999
    lane_id_last  = -9999
    #判断EV起始所在车道
    for lane in range(len(DynObs_curr)):
        if EV_curr[1] > lane_markings[lane] and EV_curr[1] < lane_markings[lane+1] :
            lane_id_last = lane
            guide_lane_last = 0.5*(lane_markings[lane] + lane_markings[lane + 1]) #所在车道中线y值
            break
    if guide_lane_last < -10:
        print("ev 横轴坐标值不在任一车道内！")
        sys.exit()
    #开始搜索每个安全空间格对应的guide line
    for i in range(ds_idx_max):
        safety_grid_centery  = (SafetySpace1_lub[i] + SafetySpace1_llb[i])*0.5
        #---如果安全格的宽度小于车道宽时guide line采用安全格中点
        if SafetySpace1_lub[i] -  SafetySpace1_llb[i] < (lane_markings[1] - lane_markings[0]  ): #假设所有车道的宽度一样
            guide_lane_ls.append(safety_grid_centery)
            continue
        #---否则，判断最佳车道中线作为guide line
        emp_coff  = 0.2*LANE_WID  #emp_coff 放宽一点这个约束
        if lane_markings[lane_id_last] + emp_coff >= SafetySpace1_llb[i] and  lane_markings[lane_id_last + 1] - emp_coff <= SafetySpace1_lub[i]:
            pass
        else:
            #搜索最靠近安全走廊中心的车道线：
            lane_candi = -9999
            min_value = 9999
            for lane in range(len(DynObs_curr)):
                differ = abs(safety_grid_centery - 0.5*(lane_markings[lane] + lane_markings[lane + 1]) )
                if differ < min_value:
                    min_value = differ
                    lane_candi = lane
            lane_id_last = lane_candi
            guide_lane_last = 0.5*(lane_markings[lane_id_last] + lane_markings[lane_id_last + 1])
        guide_lane_ls.append(guide_lane_last)
    return guide_lane_ls
        
#基于控制点生成平滑曲线，多种曲线拟合方法
def spine_path_generation(lateral_positions, SafetySpace1_sloc, EV_curr):
    #第一个采样点是EV在当前时刻（已知的）的信息
    if len(lateral_positions) > 2:
        # 采样点，需要增加EV初始时刻的采样点
        x = np.zeros(len(lateral_positions) + 1)
        y = np.zeros(len(lateral_positions) + 1)
        x[0] = EV_curr[0]
        y[0] = EV_curr[1]     
        x[1: ] = SafetySpace1_sloc[:len(lateral_positions)]  
        y[1: ] = np.array(lateral_positions)

    else:
        #如果path results中采样点少于两个，则进行插值以满足样条曲线拟合最低样本点数量要求
        x = np.zeros(len(lateral_positions) + 1 + len(lateral_positions) + 1)
        y = np.zeros(len(lateral_positions) + 1 + len(lateral_positions) + 1)
        x[0] = EV_curr[0]
        y[0] = EV_curr[1]     
        for i in range(len(lateral_positions) ):
            x[i*2+2] = SafetySpace1_sloc[i]  
            y[i*2+2] = np.array(lateral_positions)[i]
        #
        for i in range(len(lateral_positions) ):
            x[i*2+1] = (x[(i-1)*2+2] + x[i*2+2])/2
            y[i*2+1] = (y[(i-1)*2+2] + y[i*2+2])/2
        #last sample point
        x[-1] = x[-2] + 0.001
        y[-1] = y[-2]
    
    # 第二种方法：UnivariateSpline样条函数拟合
    try:
        spline = UnivariateSpline(x, y, s=1)   #spline: 样条函数    spline(x)返回array类型  正参数s：拟合的平滑程度（相对于紧密度），s越大曲线越平滑
    except:
        print('UnivariateSpline error')
        return None

    return spline            #决定采用拟合方法，更加平滑

