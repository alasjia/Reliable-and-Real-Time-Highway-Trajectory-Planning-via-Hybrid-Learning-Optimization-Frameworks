import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import os 
import numpy as np

from config import *
import sys
sys.path.append("/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training")

from config_nw import *

def draw_kine_res_merged(sx_ls, vx_ls, sy_ls, vy_ls,  fai_ls, svs_kin_ls,  rec_id,  case_id, sample_rate = 0.1, t_sta=0, t_end=9999):     #vx0_vy0,
    # 将速度和加速度转换为沿行驶方向的耦合值
    # Q：包括sv吗？  a如何处理？
    
    # Calculate the average of every 10 elements in sx_ls
    interv = int(sample_rate / 0.1)
    kin_ls_avg = []
    for kin_ls in [sx_ls, sy_ls, vx_ls,  vy_ls,  fai_ls]:
        kin_ls_avg.append( [np.mean(kin_ls[i:i+interv ]) for i in range(0, len(sx_ls), interv  )] )
    v_merged_ls = [np.sqrt(kin_ls_avg[2][i]**2 + kin_ls_avg[3][i]**2) for i in range(len(kin_ls_avg[2])) ]
    a_merged_ls =[(v_merged_ls[i] - v_merged_ls[i-1])/0.1 for i in range(1, len(v_merged_ls)) ]
    a_merged_ls.insert(0, (v_merged_ls[0] -  np.sqrt(vx_ls[0]**2 + vy_ls[0]**2) )/0.1  )    #懒得添加vx0_vy0了。。先这样
    kin_ls_avg_merged = [kin_ls_avg[0], kin_ls_avg[1], v_merged_ls, a_merged_ls, kin_ls_avg[4]]
        
        
    
    svs_kin_ls_avg =  [ [], [], [], [], [], []  ]
    for j in range(0, len(svs_kin_ls)):  #遍历每个运动学指标
        for i in range(0, len(svs_kin_ls[0]), interv):   #遍历每个时间步
            svs_kin_ls_avg[j].append(np.mean(np.array(svs_kin_ls[j][i:i+interv]), axis=0) )   #convert the frequency
            '''注意对于ttc_ls，plt仅绘制非nan值部分，但x是对应的'''
    
    steps = [i*sample_rate for i in range(t_sta,  min(len(kin_ls_avg_merged[0]), t_end)   ) ]   
    labels = ['EV']
    color_ev = 'black'     #color与BEV图统一     '#ED746A' 
    colors_sv = [ '#638DEE', '#66C999', '#F49568', '#82C61E', '#E0A4DD', '#AA66EB', '#77DCDD', '#DBAA77']

    # Create a figure with 2 subplots
    num_figs = 5
    fig, axs = plt.subplots(num_figs, figsize=(7, 6))  #一共5个图：s-t, l-t, v-t, a-t, fai-t

    # -----Plot results of  longitudinal planning and  lateral planning  of ego vehicle
    for i in range(num_figs):
        if i == 2:
            axs[i].plot(steps, np.hstack(kin_ls_avg_merged[i])[t_sta:t_end],  linewidth=1.5, color = color_ev, label = 'Velocity profile of EV')   #, marker='o'  
        else:
            axs[i].plot(steps, np.hstack(kin_ls_avg_merged[i][t_sta:t_end]),  linewidth=1.5, color = color_ev)   #, marker='o'  
    
    
    #----------在速度-时间图中绘制SVs平均速度 ：但速度规划并没有学的很好，就不画了吧
    #无限速信息，所以不画
    # avg_v_sv_ls = []
    # for i in range(len(v_merged_ls)):
    #     avg_v_sv_ls.append(np.nanmean(svs_kin_ls_avg[2][i]))  # 当前时间步所有SV的平均速度
    # axs[2].plot(steps, avg_v_sv_ls[t_sta:t_end],  linewidth=1.5, linestyle = 'dotted', color = color_ev, label = 'Average velocity of SVs')   #    dashed
    # 画危险车辆和前方慢速车辆velocity profile，先计算轨迹速度
    tar_sv_idx = 0  #目标车辆索引
    vp_sv1_x = np.array(svs_kin_ls_avg[2])[:, tar_sv_idx]   #如果SV1不是目标车辆，需要手动更改！
    vp_sv1_y = np.array(svs_kin_ls_avg[3])[:, tar_sv_idx]
    vp_sv1 = np.sqrt( vp_sv1_x**2 + vp_sv1_y**2)
    axs[2].plot(steps, vp_sv1[t_sta:t_end],  linewidth=1.5, linestyle = 'dashed', color = color_ev, label = 'Velocity profile of SV1') 
    #画所有SV在所有时间步的平均值     要算EV吗？？？
    # avg_v_sv = [np.nanmean(np.sqrt(np.array(svs_kin_ls_avg[2])**2 + np.array(svs_kin_ls_avg[3])**2))   \
    #                           for  i in range(len(v_merged_ls)) ]   #不算EV 
    avg_v_sv = np.nanmean([ np.nanmean(  np.sqrt(np.array(svs_kin_ls_avg[2])[:,i]**2 + np.array(svs_kin_ls_avg[3])[:,i]**2)   ) \
                           for i in range(8)]  +   [np.mean(kin_ls_avg_merged[2])])  ##算EV   好像差的不大
    avg_v_sv_ls = [avg_v_sv for i in range(len(v_merged_ls))]
    axs[2].plot(steps, avg_v_sv_ls[t_sta:t_end],  linewidth=1.5, linestyle = 'dotted', color = color_ev, label = 'Average velocity')   
    '''
    目标车辆索引记录：
    54_699: 0
    10_1424:  0
    53_1189:   5
    53_1091: 3
    10_579: 0
    '''

    # Auto y-axis range with padding
    for i in range(num_figs):
        # Collect all y-data plotted on this axis
        y_all = []
        for line in axs[i].get_lines():
            ydata = line.get_ydata()
            finite = ydata[np.isfinite(ydata)]
            if len(finite) > 0:
                y_all.append(finite)
        if y_all:
            y_cat = np.concatenate(y_all)
            ymin, ymax = np.min(y_cat), np.max(y_cat)
            margin = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
            axs[i].set_ylim(ymin - margin, ymax + margin)


    # Set the title and labels for both subplots
    for i in range(num_figs):
        axs[i].set_xlabel('Time(s)', fontsize =10)   # loc = 'right'
        axs[i].xaxis.set_major_locator(MaxNLocator(nbins=max(1, int((len(vx_ls) * 0.1) / 1))))
        axs[i].set_xlim(t_sta*0.1, min(int(len(vx_ls)), t_end) *0.1)     #t轴从0开始
        axs[i].tick_params(axis="x", labelsize=10)   #调节坐标轴刻度标签的字体大小
            
    for i in range(num_figs):  
        axs[i].set_ylabel( ["s (m)", "l (m)", "Velocity\n(m/s)","Acceleration\n(m/$s^2$)", "Heading Angle\n(degree)"][i] , fontsize = 10 )   # loc = 'top'    # 使用 LaTeX 语法显示平方
        axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5))    #  设置刻度数量为5
        axs[i].tick_params(axis="y", labelsize=10)   #调节坐标轴刻度标签的字体大小
                    
    # for i in range(num_figs):
    #     axs[i].set_title([   "Longitudinal Displacement", "Lateral Displacement",
    #                          "Velocity", "Acceleration", "Heading Angle"][i], fontsize = 10, fontweight='bold')

    # Add a legend for figure
    axs[2].legend(loc = 'best', fontsize = 8, ncol=3 )  # 设置图例内容为 3列，即横向排列   

    # # Set the title for the figure
    # fig.suptitle(f'Case {case_id}: Kinematics Results', fontsize = 10)
    
    # Adjust spacing
    plt.subplots_adjust(left=0.14, right=0.96, top = 0.97, bottom=0.1, wspace = 0.2, hspace=0.6)

    # # Display the plot
    # plt.show()
    
    #存图
    plt.savefig(os.path.join(SAVE_DIR2, 'kin_res_merged'+'_rec'+str(rec_id+1)+'_case'+str(case_id+1)+'.svg'))   
    plt.close(fig)
    return 1



#include longitudinal和lateral的结果在一张图中展示，便于对比分析
def draw_kine_res_separated(sx_ls, vx_ls, ax_ls, jx_ls, ttc_ls, sy_ls, vy_ls, ay_ls, fai_ls, svs_kin_ls, rec_id, case_id, sample_rate = 0.1):
    # Extract epochs and rewards from ep_ret_ls
    
    # Calculate the average of every 10 elements in sx_ls
    interv = int(sample_rate / 0.1)
    kin_ls_avg = []
    for kin_ls in [sx_ls, vx_ls, ax_ls, ttc_ls, sy_ls, vy_ls, ay_ls, fai_ls]:
        kin_ls_avg.append( [np.mean(kin_ls[i:i+interv ]) for i in range(0, len(sx_ls), interv  )] )
    
    svs_kin_ls_avg =  [ [], [], [], [], [], []  ]
    for j in range(0, len(svs_kin_ls)):  #遍历每个运动学指标
        for i in range(0, len(svs_kin_ls[0]), interv):   #遍历每个时间步
            svs_kin_ls_avg[j].append(np.mean(np.array(svs_kin_ls[j][i:i+interv]), axis=0) )   #convert the frequency
            '''注意对于ttc_ls，plt仅绘制非nan值部分，但x是对应的'''
    
    steps = [i*sample_rate for i in range(len(kin_ls_avg[0])) ]
    labels = ['EV']
    color_ev = '#ED746A' #color与BEV图统一
    colors_sv = [ '#638DEE', '#66C999', '#F49568', '#82C61E', '#E0A4DD', '#AA66EB', '#77DCDD', '#DBAA77']

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))

    # -----Plot results of  longitudinal planning and  lateral planning  of ego vehicle
    for j in range(2):
        for i in range(4):
            axs[j, i].plot(steps, np.hstack(kin_ls_avg[j*4+i]),  linewidth=2, color = color_ev)   #, marker='o'  
    
    
    #plot the gt information of surounding vehicles
    sv_idx_real = 1
    for i in range(3):  #遍历每行（运动学指标）
        for j in range(2):  #遍历每列（纵、横）
            for sv_idx in range(8):  #遍历每辆车
                arr_svs = np.array(svs_kin_ls_avg[i*2+j]).T  #convert dimensions. 1st dimension is the sv index, 2nd dimension is the time index
                if np.isnan(arr_svs[sv_idx]).all():   #全部为True才是True
                    continue
                axs[j, i].plot(steps, arr_svs[sv_idx], linestyle='--', linewidth=2, color = colors_sv[sv_idx])  # label=f'SV{sv_idx+1} vel', 
                if i==0 and j==0:  #仅输出一次
                    labels.append(f'SV{sv_idx_real }')
                    sv_idx_real += 1

 
    # Set the y-axis range
    axs[0, 1].set_ylim(bottom=kin_ls_avg[1][0]-15, top=kin_ls_avg[1][0]+15)
    axs[0, 2].set_ylim(bottom=-3, top=3)
    axs[0, 3].set_ylim(bottom= 0, top=10)
    axs[1, 1].set_ylim(bottom=-5, top=5)
    axs[1, 2].set_ylim(bottom=-3, top=3)
    axs[1, 3].set_ylim(bottom=-8, top=8)


    # Set the title and labels for both subplots
    for i in range(2):
        for j in range(4):
            axs[i, j].set_xlabel('Time(s)')   # loc = 'right'
            
    for i in range(2):
        for j in range(3):
                    axs[i, j].set_ylabel( ["Displacement (m)","Velocity (m/s)","Acceleration (m/$s^2$)"][j]  )   # loc = 'top'
    axs[0, 3].set_ylabel("Time to collision (s)")  
    axs[1, 3].set_ylabel("Steering Angle (degree)")
                    
    for i in range(2):
        for j in range(4):
            axs[i, j].set_title([   ["Longitudinal Displacement", "Longitudinal Velocity", "Longitudinal Acceleration","Time to Collision"],
                                ["Lateral Displacement", "Lateral Velocity", "Lateral Acceleration", "Steering Angle"]
                ][i][j])

    # Add a legend for figure
    fig.legend(axs[0, 1].lines, labels=labels, loc='right')   

    # Set the title for the figure
    fig.suptitle(f'Case {case_id}: Kinematics Results')
    
    # Adjust spacing
    plt.subplots_adjust(left=0.05, right=0.93, top = 0.85, bottom=0.1, wspace = 0.2, hspace=0.4)

    # # Display the plot
    # plt.show()
    
    #存图
    plt.savefig(os.path.join(SAVE_DIR2, 'kin_res_separated'+'_rec'+str(rec_id+1)+'_case'+str(case_id+1)+'.svg'))   
    plt.close(fig)
    return 1
