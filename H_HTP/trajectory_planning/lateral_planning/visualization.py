#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-02-21 11:00
# @Author  : Yujia Lu
# @Email   : 23111256@bjtu.edu.cn


'''
10.22: 增加ev_global_x0，使得x轴处于global coordinate system，便于观察。
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import matplotlib.ticker as ticker
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb  
import matplotlib.colors as mcolors
import os

from config import *
from lateral_planning.utils import get_dynobs_status


#可视化基于双态障碍物的安全走廊
def safety_corridor_plot(SafetySpace1_lub, SafetySpace1_llb, SafetySpace1_sloc, StaObs, DynObs_curr, EV_curr,  Ave_xVels, Ave_yVels, Ave_xVels_ev, ds_idx_max, lane_markings):
    # 创建图表  
    fig, ax = plt.subplots(figsize=(17, 2))  
    
    ds = SafetySpace1_sloc[1] - SafetySpace1_sloc[0]
    true_NS = len(Ave_xVels_ev)
    lane_num = len(DynObs_curr)
    DynObs_xs_NS = get_dynobs_status(lane_num, true_NS, ds, DynObs_curr, Ave_xVels, Ave_yVels, Ave_xVels_ev)
        
    for time_id in range(ds_idx_max):   #时间长度---->空间长度
        # 清空上一时刻的图
        ax.clear()      #不清空的话还挺酷的，有轨迹延迟效果
        
        #------------------------绘制车道线作为参考
        for y in lane_markings :  
            plt.axhline(y=y, color='gray', linestyle='--')  # 使用plt.axhline()函数绘制横线

        
        #-----------------------处理EV与周边车辆信息
        rectangles1 = []  # 用于存储创建的矩形对象  
        labels = []  # 用于存储矩形的标签  
        #sv
        for lane in range(lane_num):
            for veh in range(len(DynObs_xs_NS[lane])):
                x = DynObs_xs_NS[lane][veh][time_id][0] - VEH_LEN/2  
                y = DynObs_xs_NS[lane][veh][time_id][1] - VEH_WID/2
                rect = patches.Rectangle((x, y), VEH_LEN, VEH_WID, linewidth=1, facecolor='cornflowerblue')   
                rectangles1.append(rect)                    # 将矩形对象添加到列表中 
        
        # ev
        ev_global_xt = EV_curr[0] + ds*(time_id+1)
        ev_y_t =  EV_curr[1] 
        ev_angle_t = 0  #斜率需要转为角度
        rect = patches.Rectangle((ev_global_xt - VEH_LEN/2 , ev_y_t - VEH_WID/2), VEH_LEN, VEH_WID, linewidth=1, facecolor='orangered', angle = ev_angle_t)  
        rectangles1.append(rect)
        
        #static obs
        for obs in range(StaObs.shape[0]):
            x = StaObs[obs, 0] - StaObs[obs, 2]/2  
            y = StaObs[obs, 1] - StaObs[obs, 3]/2
            rect = patches.Rectangle((x, y), StaObs[obs, 2], StaObs[obs, 3], linewidth=1, facecolor='pink')   
            rectangles1.append(rect)                    # 将矩形对象添加到列表中 
            
        
        #-------------------------绘制安全区间/安全走廊
        rectangles2 = []
        r2_wid =  SafetySpace1_sloc[1] - SafetySpace1_sloc[0]   #ds = 200/20 = 10
        
        for i in range(ds_idx_max):  #<=20
            r2_x =  SafetySpace1_sloc[i] - r2_wid*0.5           #中点位置需要转为左下角位置
            r2_height = SafetySpace1_lub[i] - SafetySpace1_llb[i]
            # 绘制矩形，并设置边框宽度、填充色和透明度  
            rect2 = patches.Rectangle((r2_x, SafetySpace1_llb[i]), r2_wid, r2_height,   
                                    linewidth=1,  # 设置边框宽度  
                                    facecolor='green',  # 设置填充色  
                                    edgecolor='grey',  # 设置边框颜色  
                                    alpha=0.2)  # 设置透明度  
            rectangles2.append(rect2)

        
        #-----------------------添加矩形到图中
        rectangles_all = [rectangles2,  rectangles1]    #先画走廊再画车
        for rectangles in rectangles_all:
            for rect in rectangles:
                ax.add_patch(rect)

        
        #-----------------------绘图参数设置及显示
        # 添加X和Y轴标签  
        plt.xlabel('X-axis', fontsize = 20)  
        plt.ylabel('Y-axis', fontsize = 20)      
        #设置坐标轴显示范围
        plt.xlim(ev_global_xt - 100,   ev_global_xt + 100)    #显示全部范围
        plt.ylim(EV_curr[1] - 10, EV_curr[1] + 10  )
        
        #显示
        plt.show(block = False)
        plt.pause(0.5)  #停顿1s再绘制下一时刻
        # # plt.draw()  # 绘制图形   加了好像没啥区别
        # input("Press [enter] to continue.")  # 暂 停直到用户按下回车键

        
        
        # #存图
        # dirpath = "/home/chwei/AutoVehicle/optimization_algorithm/tra_path_planning_pngs"
        # plt.savefig(os.path.join(dirpath, 'path_0109_lambda6_NS30_after''.svg'))      #plt.savefig(os.path.join(dirpath, 'path_case'+str(png_id+1)+'_fig''.svg'))     path_1224_NS100_O2_9
        # # 关闭图
        # plt.close(fig)
        
        
    #关闭所有窗口
    plt.close()
    
    
    return 1


#可视化安全走廊与path，检查模型优化结果
def path_plot_hd(SafetySpace1_lub, SafetySpace1_llb, SafetySpace1_sloc,  half_ds_NS, dds_NS,
                 EVSpace_lub, EVSpace_llb, SafetySpace2_sloc, DynObs_curr, EV_curr,  DynObs_xPos_fut, Ev_xPos_fut,  
                 lane_markings, ev_global_x0, ds_idx_max, res_lateral_positions, res_lp_derivative1, run_time, path_func, guide_lane_ls,
                 case_id = 0, glo_t_id = 0):
    # 创建图表  
    fig, ax = plt.subplots(figsize=(17, 3))  
    
    lane_num = len(DynObs_curr)
        
    for time_id in range(ds_idx_max):   #时间长度---->空间长度
        
        # 清空上一时刻的图
        ax.clear()      #不清空的话还挺酷的，有轨迹延迟效果
        
        #------------------------绘制车道线作为参考
        for y in lane_markings :  
            plt.axhline(y=y, color='gray', linestyle='--')  # 使用plt.axhline()函数绘制横线


        #-----------------------处理EV与周边车辆信息
        rectangles1 = []  # 用于存储创建的矩形对象  
        labels = []  # 用于存储矩形的标签  
        #sv
        for lane in range(lane_num):
            for veh in range(len(DynObs_xPos_fut[lane])):
                x = DynObs_xPos_fut[lane][veh][time_id][0] - VEH_LEN/2  + ev_global_x0
                y = DynObs_xPos_fut[lane][veh][time_id][1] - VEH_WID/2
                rect = patches.Rectangle((x, y), VEH_LEN, VEH_WID, linewidth=2,  facecolor='none', edgecolor='#638DEE')   
                rectangles1.append(rect)                    # 将矩形对象添加到列表中 
        
        # ev
        ev_local_xt = SafetySpace1_sloc[time_id] + ev_global_x0
        ev_y_t =  res_lateral_positions[time_id] 
        ev_angle_t = np.degrees( np.arctan(  res_lp_derivative1[time_id] )  )   #斜率需要转为角度，转向角
        #亲爱的大聪明，np.arctan()返回弧度，需要转换成角度
        rect = patches.Rectangle((ev_local_xt - VEH_LEN/2 , ev_y_t - VEH_WID/2), VEH_LEN, VEH_WID, linewidth=2, facecolor='none', edgecolor='#ED746A', angle = ev_angle_t, )  
        rectangles1.append(rect)
        '''rotation_point= {'xy', 'center', (number, number)}, default: 'xy'
            If 'xy', rotate around the anchor point. If 'center' rotate around the center. If 2-tuple of number, rotate around this coordinate.
            '''
                
        if CONTINUE_SWITCH_S:
            #-------------------------绘制安全区间/安全走廊
            rectangles2 = []
            r2_wid = half_ds_NS[time_id]* 2
            r2_x =  SafetySpace1_sloc[time_id] - r2_wid*0.5  + ev_global_x0         #中点位置需要转为左下角位置       
            r2_height = SafetySpace1_lub[time_id] - SafetySpace1_llb[time_id]
            # 绘制矩形，并设置边框宽度、填充色和透明度  
            rect2 = patches.Rectangle((r2_x, SafetySpace1_llb[time_id]), r2_wid, r2_height,   
                                    linewidth=2,  # 设置边框宽度  
                                    facecolor='#66C999'  ,  # 设置填充色  
                                    edgecolor='#66C999'  ,  # 设置边框颜色  
                                    alpha=0.5)  # 设置透明度  
            rectangles2.append(rect2)


            #-------------------------绘制考虑车身尺寸的上下边界
            rectangles3 = []
            for j in range(EVSpace_lub.shape[1]):  #6个sub grid
                r3_wid = dds_NS[time_id]    #0.2    #采用一个小的固定值，因为按照建模原理应该是画一条线（x=a, not  x in [a, b]）
                r3_x =  SafetySpace2_sloc[time_id,j] - 0.5*r3_wid  + ev_global_x0         #中点位置转为左下角位置
                r3_height = EVSpace_lub[time_id, j] - EVSpace_llb[time_id, j]
                
                # 绘制矩形，并设置边框宽度、填充色和透明度  
                rect3 = patches.Rectangle((r3_x, EVSpace_llb[time_id, j]), r3_wid, r3_height,   
                                        linewidth=2,  # 设置边框宽度  
                                        facecolor='#DBAA77',  # 设置填充色  
                                        edgecolor='#DBAA77',  # 设置边框颜色  
                                        alpha=0.3)  # 设置透明度  
                rectangles3.append(rect3)
                
                
        else:
            #-------------------------绘制安全区间/安全走廊
            rectangles2 = []
            for i in range(ds_idx_max):  #<=20
                r2_wid = half_ds_NS[i]* 2
                r2_x =  SafetySpace1_sloc[i] - r2_wid*0.5  + ev_global_x0         #中点位置需要转为左下角位置       
                r2_height = SafetySpace1_lub[i] - SafetySpace1_llb[i]
                # 绘制矩形，并设置边框宽度、填充色和透明度  
                rect2 = patches.Rectangle((r2_x, SafetySpace1_llb[i]), r2_wid, r2_height,   
                                        linewidth=2,  # 设置边框宽度  
                                        facecolor='#66C999',  # 设置填充色  
                                        edgecolor='#66C999',  # 设置边框颜色  
                                        alpha=0.1)  # 设置透明度  
                rectangles2.append(rect2)


            #-------------------------绘制考虑车身尺寸的上下边界
            rectangles3 = []
            # 参数定义，实现颜色渐变
            gradients = create_gradient(mcolors.to_rgb('#DBAA77'), num_steps=ds_idx_max) #填充的渐变色
            for i in range(ds_idx_max):  #<=40
                for j in range(EVSpace_lub.shape[1]):  #6个sub grid
                    r3_wid = dds_NS[i]    #0.2    #采用一个小的固定值，因为按照建模原理应该是画一条线（x=a, not  x in [a, b]）
                    r3_x =  SafetySpace2_sloc[i,j] - 0.5*r3_wid  + ev_global_x0         #中点位置转为左下角位置
                    r3_height = EVSpace_lub[i, j] - EVSpace_llb[i, j]

                    
                    # 绘制矩形，并设置边框宽度、填充色和透明度  
                    rect3 = patches.Rectangle((r3_x, EVSpace_llb[i, j]), r3_wid, r3_height,   
                                            linewidth=2,  # 设置边框宽度  
                                            facecolor=gradients[i],  # 设置填充色  
                                            edgecolor=gradients[i],  # 设置边框颜色  
                                            alpha=0.1)  # 设置透明度  
                    rectangles3.append(rect3)
        
        #-----------------------添加矩形到图中
        rectangles_all = [rectangles2, rectangles3, rectangles1]    #先画走廊再画车
        for rectangles in rectangles_all:
            for rect in rectangles:
                ax.add_patch(rect)
            
        #-----------------------绘制优化结果中的path离散点位置
        #绘制EV当前的位置（已知信息）
        plt.scatter(EV_curr[0]+ev_global_x0, EV_curr[1], s=10, c='#ED746A', marker='o',  label='Discrete Points', alpha=1)  
        for i in range(ds_idx_max):
            path_s = SafetySpace1_sloc[i]  + ev_global_x0
            path_l = res_lateral_positions[i]
            # 绘制离散点
            plt.scatter(path_s, path_l, s=10, c='black', marker='o',  label='Discrete Points', alpha=1)  # 设置点的大小为10，颜色为蓝色

        # #-----------------------绘制path拟合曲线
        # if path_func is not None:
        #     x_curve = np.linspace(EV_curr[0], np.max(SafetySpace1_sloc[:ds_idx_max]), 1000)
        #     plt.plot(x_curve + ev_global_x0, path_func(x_curve), '--', c = 'orangered')
        
        #------------------------绘制reference line
        plt.scatter(SafetySpace1_sloc + ev_global_x0, guide_lane_ls, s=20, c='red', marker='*',  label='Reference Line')  
        
        
        #-----------------------绘图参数设置及显示
        # 添加X和Y轴标签  
        plt.xlabel('X-axis', fontsize = 20)  
        plt.ylabel('Y-axis', fontsize = 20)      
        #设置坐标轴显示范围
        plt.xlim(ev_local_xt - 60,   ev_local_xt + 140)    #显示全部范围
        plt.ylim(EV_curr[1] - 15, EV_curr[1] + 10  )
        #添加必要的文本说明
        # plt.text(ev_local_xt - 20, lane_markings[0] -3,  "Speed PLanning Time: %.4f  sec"%run_time_sp)
        plt.text(ev_local_xt - 20, lane_markings[0] -5,  "Path PLanning Time: %.4f  sec"%run_time)  #计算机运行时间，而不是gurobi计算时间
        # plt.text(ev_local_xt - 20, lane_markings[0] -7,  "Speed Model Output: acc:%.2f m/s2   jerk:%.2f m/s3"%(action[0], action[1]) )

        plt.title("Case Id: %d  Time Id: %d"%(case_id, glo_t_id) )

        
        if CONTINUE_SWITCH_S:
            plt.show(block = False)
            plt.pause(1)  #停顿1s再绘制下一时刻
        else:
            #显示
            if time_id == 0:
                plt.show()
                break
        # plt.draw()  # 绘制图形   加了好像没啥区别
        # input("Press [enter] to continue.")  # 暂 停直到用户按下回车键
        
        # if time_id == (ds_idx_max - 1):
        #     #存图
        #     dirpath = "/home/chwei/AutoVehicle/optimization_algorithm/optimization202402/results"
        #     plt.savefig(os.path.join(dirpath, 'path_record1_track46_v2''.svg'))      #plt.savefig(os.path.join(dirpath, 'path_case'+str(png_id+1)+'_fig''.svg'))     path_1224_NS100_O2_9
        #     # 关闭图
        #     plt.close(fig)
        #     break
        
        
    #关闭所有窗口
    plt.close()
    
    
    return 1


def kine_res_plot(case_id, time_id,  EV_curr, SafetySpace1_sloc, res_lateral_positions, res_lp_derivative1):
    
    sx_res =  SafetySpace1_sloc.copy()
    vx_res = np.array([sx_res[i] - sx_res[i-1] for i in range(1, sx_res.shape[0])]) / 0.1
    vx_res = np.insert(vx_res, 0, (sx_res[0] - EV_curr[0])/0.1  )
    ax_res = np.array([vx_res[i] - vx_res[i-1] for i in range(1, sx_res.shape[0])]) / 0.1
    ax_res = np.insert(ax_res, 0, (vx_res[0] - EV_curr[2])/0.1  )
    
    sy_res = np.array(res_lateral_positions).copy()
    vy_res = np.array([sy_res[i] - sy_res[i-1] for i in range(1, sy_res.shape[0])]) / 0.1
    vy_res = np.insert(vy_res, 0, (sy_res[0] - EV_curr[1])/0.1  )
    ay_res = np.array([vy_res[i] - vy_res[i-1] for i in range(1, vy_res.shape[0])]) / 0.1
    ay_res = np.insert(ay_res, 0, (vy_res[0] - EV_curr[3])/0.1  )
    fai_res =  np.degrees( np.arctan(res_lp_derivative1.copy() )  )
    
    steps = [i for i in range(sx_res.shape[0]) ]

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, 4, figsize=(22, 8))

    # Plot the longitudinal planning results in the first row
    axs[0, 0].plot(steps, sx_res, label='Longitudial displacement')
    axs[0, 1].plot(steps, vx_res, label='Longitudial speed')
    axs[0, 2].plot(steps, ax_res, label='Longitudial acceleration')
    # axs[0, 3].plot(steps, jx_res, label='Longitudial jerk')

    # Plot the lateral planning results in the second row
    axs[1, 0].plot(steps, sy_res, label='Lateral displacement')
    axs[1, 1].plot(steps, vy_res, label='Lateral speed')
    axs[1, 2].plot(steps, ay_res, label='Lateral acceleration')
    axs[1, 3].plot(steps, fai_res, label='Steering angle')


    # Set the title and labels for both subplots
    for i in range(2):
        for j in range(4):
            axs[i, j].set_xlabel('Step')
            
    for i in range(2):
        for j in range(3):
                    axs[i, j].set_ylabel(f'{["Displacement (m)","Speed (m/s)","Acceleration (m/s^2)"][j]}') 
    axs[0, 3].set_ylabel("Longitudial jerk (m/s^3)") 
    axs[1, 3].set_ylabel("Steering Angle (degree)")

    # Add a legend for all subplots
    for i in range(2):
        for j in range(4):
            axs[i, j].legend()

    # Set the title for the figure
    fig.suptitle(f'Case {case_id} Time {time_id}: Kinematics Results')

    # Display the plot
    plt.show()
    
    # # Save the figure
    # rec_id = 35
    # file_name = f'rec{rec_id+1}_case{case_id+1}'  # index starts from 1
    # plt.savefig(os.path.join(SAVE_DIR2, file_name+'.jpg'))
    return 1



def create_gradient(color, num_steps=5):
    """为给定颜色生成渐变序列"""
    gradient = []
    for i in range(num_steps):
        # 渐变公式：浅色 = 深色 * (i / (num_steps - 1))
        grad_color = tuple(c * (i / (num_steps - 1)) + (1 - i / (num_steps - 1)) for c in color)
        gradient.append(grad_color)
    return gradient

