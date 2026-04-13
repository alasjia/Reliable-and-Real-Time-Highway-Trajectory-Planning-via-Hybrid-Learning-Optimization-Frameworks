#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-02-21 11:00
# @Author  : Yujia Lu
# @Email   : 23111256@bjtu.edu.cn

import numpy as np
from scipy.integrate import simpson
from config import *

# 获得如图所示的局部四辆环境车（静态搜索）
'''
车辆编号： 0—ego   1—原车道后车  2—原车道前车  3—目标车道后车  4—目标车道前车
--------------------------------------------------------------------------------
                   3                                                                      4                              lane2
--------------------------------------------------------------------------------
                               1                       0                            2                                   lane1
--------------------------------------------------------------------------------
'''

#暴力搜索 数值转换                     
def s_to_x(s_val, func_path, veh0x0):
    error2 = 1000
    step = 0.01  #步长为1cm
    x1 = s_val + step            #为了加速数值转换,改变思路为向前搜索：因为曲线的长度一定比x的长度大,因此设置为从s_val开始向前搜索
    if s_val == 0:
        return 0
    while error2 >= 0.01:  #误差为1cm
        x1 -= step
        res  = get_curve_length(func_path, 0, x1, veh0x0)
        error2 = abs(s_val - res)#.evalf()
        # error1 = res[1]    #error1是scipy计算误差，error2是与s真实值误差
    return x1

'''
曲线长度的计算：scipy库的数值积分方法simpson()， simpson()函数采用辛普森法则，输入被积函数的值和对应的自变量，然后函数会返回对应区间内的数值积分结果。
'''
def get_curve_length(func, x0_local, x1_local, veh0x0):
    x0 = x0_local + veh0x0
    x1 = x1_local + veh0x0
    if x1_local == x0_local:
        sf = 0
    else:
        sf= simpson(np.sqrt( 1 + np.square( func.derivative()(np.linspace(x0, x1,100) ) ) ) , np.linspace(x0, x1, 100)) 
    return sf
    
    
    
# def extract_average_xvels( DynObs_uplane, DynObs_lowlane, EV_xyva, vels):
def extract_average_xvels( veh_curr_xs, vels):
    '''
    input:
    DynObs_uplane, DynObs_lowlane, EV_xyva: current vehcle status
    vels:  the longitudinal velocity of  the ego vehicle and surrounding vehicles( sv numver is n , n >= 0)          
                    p is the number of velocity prediction frames as the output of dnn model.
                    size (1+n, p)    
    output: the average velocities over every spatial interval in the path optimal model.
                    size (1+n, q)
    '''
    
    search_interval = SN/ NS
    vehs_num = vels.shape[0]
    # pred_frames = vels.shape[2]
    pred_frames = vels.shape[1]
    
    delta_t = 0.04    #second,  Time interval corresponding to shooting frequency(frameRate = 25 Hz) in highD dataset

    '''
    the first dimension  in x_coords and vels
    EV
    low f
    low p
    up f
    up p
    '''
    # veh_curr_xs = [EV_xyva[0], DynObs_lowlane[0, 0], DynObs_lowlane[1, 0], DynObs_uplane[0, 0], DynObs_uplane[1, 0] ]
    ave_vels_all = []
    for veh in range(vehs_num):
        #initialization
        x_coords = np.zeros_like(vels[0])
        x_coords[0] = veh_curr_xs[veh]+ delta_t * vels[veh, 0]
        #implementation
        for i in range(1, pred_frames):
            # for veh in range(vehs_num ):
            x_coords[i] = x_coords[i-1]  + delta_t * vels[veh, i]
        
        ave_vels = []
        start_spot = veh_curr_xs[veh] #initial x 
        start_id = 0
        avev_idx = 0
        for i in range(pred_frames ):
            if abs(x_coords[i] - start_spot) > search_interval:
                ave_vels.append( np.mean(vels[veh, start_id:i])  ) 
                #update
                avev_idx += 1
                start_spot = x_coords[i]  #注意，spatial interval划分所需的时间长度是以ego veh为准的
                start_id = i
                
        #transform
        ave_vels_all.append(np.abs(np.array(ave_vels) ) )
                
    return ave_vels_all


def extract_average_xvels_failone(ev_curr_x, vels):
    '''
    input:
    DynObs_uplane, DynObs_lowlane, EV_xyva: current vehcle status
    vels:  the longitudinal velocity of  the ego vehicle and surrounding vehicles( sv numver is n , n >= 0)          
                    p is the number of velocity prediction frames as the output of dnn model.
                    size (examples, 1+n, p)    
    output: the average velocities over every spatial interval in the path optimal model.
                    size (examples, 1+n, q)
    '''
    
    search_interval = SN/ NS
    examples_num = vels.shape[0]
    pred_frames = vels.shape[1]
    
    delta_t = 0.04    #second,  Time interval corresponding to shooting frequency(frameRate = 25 Hz) in highD dataset
    
    ave_vels_all = []
    for exam in range(examples_num):
        #initialization
        x_coords = np.zeros_like(vels[0])
        x_coords[0] = ev_curr_x[exam]+ delta_t * vels[exam, 0]
        #implementation
        for i in range(1, pred_frames):
            # for veh in range(vehs_num ):
            x_coords[i] = x_coords[i-1]  + delta_t * vels[exam, i]
        
        ave_vels = []
        start_spot = ev_curr_x[exam] #initial x 
        start_id = 0
        avev_idx = 0
        for i in range(pred_frames ):
            if abs(x_coords[i] - start_spot) > search_interval:
                ave_vels.append( np.mean(vels[exam, start_id:i])  ) 
                #update
                avev_idx += 1
                start_spot = x_coords[i]  #注意，spatial interval划分所需的时间长度是以ego veh为准的
                start_id = i
                
        #transform
        ave_vels_all.append(np.abs(np.array(ave_vels) ) )
                
    return ave_vels_all



def get_lane_markings(up_markings , lo_markings, EV_curr):
    #判断行车方向以绘制车道线
    if EV_curr[1] >  lo_markings[0]:
        lane_markings = lo_markings
    else:
        lane_markings = up_markings
    
    return lane_markings


def get_dynobs_status(lane_num, true_NS, ds, DynObs_curr, Ave_xVels, Ave_yVels, Ave_xVels_ev):
    #-----维护一个数组，存储NS个时刻 动态障碍物分别的位置（仅是横轴 s值）
    #(lane_num, veh_num, NS, 2 )               2:x and y
    DynObs_xs_NS = [ [ ] for lane in range(lane_num)]
    for lane in range(lane_num):
        for veh in range(len(DynObs_curr[lane])):
            DynObs_xs_NS[lane].append([])
            
    #initialization 
    time_in_this_ds = ds / Ave_xVels_ev[0]
    for lane in range(lane_num):
        for veh in range(len(DynObs_curr[lane])):
            sv_x = DynObs_curr[lane][veh][0] + Ave_xVels[lane][veh][0]*time_in_this_ds    #first [0] means coordinate x, second [0] means the time index.
            sv_y = DynObs_curr[lane][veh][1] + Ave_yVels[lane][veh][0]*time_in_this_ds
            DynObs_xs_NS[lane][veh].append([sv_x, sv_y])
    #
    for ds_id in range(1, NS):
        if ds_id < true_NS:
            time_in_this_ds = ds / Ave_xVels_ev[ds_id]    #EV在该ds中点位置时，其他车在哪
            for lane in range(lane_num):
                for veh in range(len(DynObs_curr[lane])):
                    sv_x = DynObs_xs_NS[lane][veh][ds_id - 1][0] + Ave_xVels[lane][veh][ds_id]*time_in_this_ds    
                    sv_y = DynObs_xs_NS[lane][veh][ds_id - 1][1] + Ave_yVels[lane][veh][ds_id]*time_in_this_ds
                    DynObs_xs_NS[lane][veh].append([sv_x, sv_y])
        else:
            #在超出具有真实数据的区间，x采用匀速运动假设，速度为最后的速度
            time_in_this_ds = ds / Ave_xVels_ev[-1]    #EV在该ds中点位置时，其他车在哪
            for lane in range(lane_num):
                for veh in range(len(DynObs_curr[lane])):
                    sv_x = DynObs_xs_NS[lane][veh][ds_id - 1][0] + Ave_xVels[lane][veh][-1]*time_in_this_ds    
                    sv_y = DynObs_xs_NS[lane][veh][ds_id - 1][1] #y假设不变
                    DynObs_xs_NS[lane][veh].append([sv_x, sv_y])
                    
    return DynObs_xs_NS

def is_biobs_avoidance(switch,  StaObs , DynObs_curr):
    if not switch:
        StaObs = np.zeros_like(StaObs) 

    return  StaObs 
        
        
def get_global_pos(DynObs_Pos_fut_, EV_curr, DynObs_curr, DynObs_vvid, valid_vehids):
    DynObs_xPos_fut = [ []  for _ in range(len(DynObs_curr))]  
    for lane in range(len(DynObs_curr) ):
        for veh in range(len(DynObs_curr[lane])  ):
            idx_in_vv = valid_vehids.index(DynObs_vvid[lane][veh])
            DynObs_xPos_fut[lane].append((DynObs_Pos_fut_[idx_in_vv] + DynObs_curr[lane][veh][0]).copy()   )
    
    Ev_xPos_fut = (DynObs_Pos_fut_[0] +  EV_curr[0]).copy()
    return DynObs_xPos_fut, Ev_xPos_fut 