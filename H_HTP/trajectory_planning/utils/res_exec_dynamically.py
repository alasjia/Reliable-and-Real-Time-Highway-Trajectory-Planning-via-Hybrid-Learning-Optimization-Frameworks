import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import os 
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import math

from config import *
import sys
sys.path.append("/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training")
from config_nw import *

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.transforms import Affine2D

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.transforms import Affine2D

def save_executing_gif(graph_data, sx_ls, sy_ls, fai_ls, sv_gt,
                              rec_id, case_id, out_dir,
                              fps=10, dpi=120):

    sx_ls = np.asarray(sx_ls)
    sy_ls = np.asarray(sy_ls)
    fai_ls = np.asarray(fai_ls)
    T = len(sx_ls)

    lane_markings = [i.item() for i in graph_data.lane_ys_nonorm]

    fig, ax = plt.subplots(figsize=(25, 3))

    # static lanes
    for y in lane_markings:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('X-axis', fontsize=20)
    ax.set_ylabel('Y-axis', fontsize=20)
    ax.set_xlim(sx_ls[0] - 20, sx_ls[-1] + 20)
    ax.set_ylim(sy_ls[0] - 15, sy_ls[0] + 10)

    title = ax.set_title("", fontsize=14)

    # pre-create patches (important: do NOT clear axis per frame)
    sv_rects = []
    for _ in range(MAX_SV):
        r = patches.Rectangle((0, 0), VEH_LEN, VEH_WID,
                              linewidth=2, facecolor='none', edgecolor='cornflowerblue')
        r.set_visible(False)
        ax.add_patch(r)
        sv_rects.append(r)

    ev_rect = patches.Rectangle((0, 0), VEH_LEN, VEH_WID,
                                linewidth=2, facecolor='none', edgecolor='orangered')
    ax.add_patch(ev_rect)

    def update(t):
        title.set_text(f"Record Id: {rec_id}  Case Id: {case_id}  Time Id: {t}")

        # SVs
        for sv_id, r in enumerate(sv_rects):
            x = sv_gt[F_HIS + t, 6*sv_id + 0]
            y = sv_gt[F_HIS + t, 6*sv_id + 1]
            if np.isnan(x) or np.isnan(y):
                r.set_visible(False)
            else:
                r.set_visible(True)
                r.set_xy((x - VEH_LEN/2, y - VEH_WID/2))

        # EV (rotate around center)
        cx, cy = float(sx_ls[t]), float(sy_ls[t])
        yaw = float(fai_ls[t])

        # If your fai_ls is degrees, uncomment this:
        yaw = np.deg2rad(yaw)

        ev_rect.set_xy((cx - VEH_LEN/2, cy - VEH_WID/2))
        ev_rect.set_transform(Affine2D().rotate_around(cx, cy, yaw) + ax.transData)

        return [title, ev_rect, *sv_rects]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)

    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f"executing_path_rec{rec_id+1}_case{case_id+1}.gif")

    ani.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return gif_path

def draw_trajectory_executing(graph_data, sx_ls, sy_ls, fai_ls, sv_gt, t_ptr, line_num, rec_id, case_id):
    # 创建图表  
    fig, ax = plt.subplots(figsize=(25, 3))  
    
    #未中心化的坐标系（原始坐标系）
    lane_num = line_num
    lane_markings =  [i.item() for i in graph_data.lane_ys_nonorm]
    #------------------------绘制车道线作为参考
    for y in lane_markings :  
        plt.axhline(y=y, color='gray', linestyle='--')  # 使用plt.axhline()函数绘制横线
        
    # Create a colormap
    cmap = plt.get_cmap('jet')  # You can choose any colormap
        
    for t in range(len(sx_ls)):   #时间长度---->空间长度
        if CONTINUE_SWITCH_E:
            # 清空上一时刻的图
            ax.clear()      
            #------------------------绘制车道线作为参考
            for y in lane_markings:  
                plt.axhline(y=y, color='gray', linestyle='--')  # 使用plt.axhline()函数绘制横线
        
        #-----------------------处理EV与周边车辆信息
        rectangles1 = []  # 用于存储创建的矩形对象  
        labels = []  # 用于存储矩形的标签  
        # #当前时刻的颜色设置
        # color = cmap(t / len(sx_ls))  # Normalize the time index to the range [0, 1]
        
        #sv  
        for sv_id in range(MAX_SV):
            x = sv_gt[F_HIS+t, 6*sv_id+0] - VEH_LEN/2  
            y = sv_gt[F_HIS+t, 6*sv_id+1] - VEH_WID/2
            if not np.isnan(x):
                # rect = patches.Rectangle((x, y), VEH_LEN, VEH_WID, linewidth=1, facecolor='none', edgecolor=color)
                rect = patches.Rectangle((x, y), VEH_LEN, VEH_WID, linewidth=2, facecolor='none', edgecolor='cornflowerblue')
                rectangles1.append(rect)                    # 将矩形对象添加到列表中 
        
        # ev
        ev_xt = sx_ls[t]
        ev_yt = sy_ls[t]
        ev_angle_t = fai_ls[t]
       # Assign a color to the EV rectangle based on its time index
        # rect = patches.Rectangle((ev_xt - VEH_LEN/2, ev_yt - VEH_WID/2), VEH_LEN, VEH_WID, linewidth=1, facecolor='none', edgecolor=color, angle=ev_angle_t)
        rect = patches.Rectangle((ev_xt - VEH_LEN/2, ev_yt - VEH_WID/2), VEH_LEN, VEH_WID, linewidth=2, facecolor='none', edgecolor='orangered', angle=ev_angle_t)
        rectangles1.append(rect)
            
        
        #-----------------------添加矩形到图中    
        for rect in rectangles1:
            ax.add_patch(rect)
            
        # if not CONTINUE_SWITCH_E:   #仅在静态绘图设置下绘制轨迹线条
        #     #-----------------------绘制实际执行的path的拟合曲线
        #     spline = interp1d(sx_ls, sy_ls)
        #     # spline = UnivariateSpline(sx_ls, sy_ls, s=1)   #spline: 样条函数    spline(x)返回array类型  正参数s：拟合的平滑程度（相对于紧密度），s越大曲线越平滑
        #     x_curve = np.linspace(sx_ls[0],sx_ls[-1], 500)
        #     plt.plot(x_curve, spline(x_curve), '-', c = 'orangered', linewidth=1)
        
        if CONTINUE_SWITCH_E:
            #-----------------------绘图参数设置及显示
            # 添加X和Y轴标签  
            plt.xlabel('X-axis', fontsize = 20)  
            plt.ylabel('Y-axis', fontsize = 20)      
            #设置坐标轴显示范围
            plt.xlim(sx_ls[0] - 20,   sx_ls[-1] + 20)    #显示全部范围
            plt.ylim(sy_ls[0] - 15, sy_ls[0] + 10  )
            #标题
            plt.title("Record Id: %d  Case Id: %d  Time Id: %d"%(rec_id, case_id, t) )
            
            plt.show(block = False)
            plt.pause(0.1)  #停顿1s再绘制下一时刻
        
    if not CONTINUE_SWITCH_E:
        #-----------------------绘图参数设置及显示
        # 添加X和Y轴标签  
        plt.xlabel('X-axis', fontsize = 20)  
        plt.ylabel('Y-axis', fontsize = 20)      
        #设置坐标轴显示范围
        plt.xlim(sx_ls[0] - 20,   sx_ls[-1] + 20)    #显示全部范围
        plt.ylim(sy_ls[0] - 15, sy_ls[0] + 10  )
        #添加必要的文本说明
        # plt.text(ev_xt - 20, lane_markings[0] -3,  "Speed PLanning Time: %.4f  sec"%run_time_sp)
        # plt.text(ev_xt - 20, lane_markings[0] -5,  "Path PLanning Time: %.4f  sec"%run_time)  #计算机运行时间，而不是gurobi计算时间
        # plt.text(ev_xt - 20, lane_markings[0] -7,  "Speed Model Output: acc:%.2f m/s2   jerk:%.2f m/s3"%(action[0], action[1]) )
        #标题
        plt.title("Record Id: %d  Case Id: %d  Time Length: %d"%(rec_id, case_id, len(sx_ls)) )


        # plt.show()

    #存图
    plt.savefig(os.path.join(SAVE_DIR3, 'executing_path_'+'rec'+str(rec_id+1)+'_case'+str(case_id+1)+'.svg'))   
    # 关闭图
    plt.close(fig)
    
    return 1


#绘制EV的完整绘画轨迹，shots of discrete timesteps
def draw_trajectory_replanning(graph_data, sx_ls, sy_ls, fai_ls, sx_plan_ls, sy_plan_ls, rec_id, case_id):
    color_plan, color_real = '#638DEE', '#ED746A'
    
    lane_markings =  [i.item() for i in graph_data.lane_ys_nonorm]
        
    # 创建图表  
    fig, ax = plt.subplots(figsize=(12, 3))  
    
    #------------------------绘制线条
    dra_loc_sta, dra_loc_end = 25, 80
    dra_loc_sta = min(dra_loc_sta, len(sx_plan_ls) - 1, len(sx_ls) - 1)
    dra_loc_end = min(dra_loc_end, len(sx_plan_ls), len(sy_plan_ls), len(sx_ls), len(sy_ls)) 
    '''
    参数记录：
    54_699: 30, 80  
    10_1424: 60,  len(sx_plan_ls)
    53_1189:    30, 70
    53_1091:  60, 100
    10_579: 40, 100
    '''
    #绘制车道线
    for y in lane_markings :  
        # ax.axhline(y=y - sy_ls[dra_loc_sta] , color='gray', linestyle='--')  # 使用plt.axhline()函数绘制横线
        ax.axhline(y=y, color='gray', linestyle='--', label = 'Road marking')  # 使用plt.axhline()函数绘制横线
    
    # #绘制真实轨迹  为了图例所以画两遍
    # # ax.plot(sx_ls[dra_loc_sta: dra_loc_end], sy_ls[dra_loc_sta: dra_loc_end] - sy_ls[dra_loc_sta] , color=color_real, linewidth=3)
    # ax.plot(sx_ls[dra_loc_sta: dra_loc_end], sy_ls[dra_loc_sta: dra_loc_end], color=color_real, linewidth=4, label = 'Real trajectory')
    
    #绘制所有时间步的规划轨迹
    for t in range(dra_loc_sta, dra_loc_end):
        # ax.plot(sx_plan_ls[t], sy_plan_ls[t] - sy_ls[dra_loc_sta] , '--', color=color_plan, linewidth=1.5, alpha = 0.8)
        ax.plot(sx_plan_ls[t], sy_plan_ls[t] , '--', color=color_plan, linewidth=1.5, alpha = 0.8, label = 'Dynamical pLanning trajectory')
        
    #绘制真实轨迹
    # ax.plot(sx_ls[dra_loc_sta: dra_loc_end], sy_ls[dra_loc_sta: dra_loc_end] - sy_ls[dra_loc_sta] , color=color_real, linewidth=3)
    ax.plot(sx_ls[dra_loc_sta: dra_loc_end], sy_ls[dra_loc_sta: dra_loc_end], color=color_real, linewidth=2.5, label = 'Real trajectory')
    


                
    #-----------------------坐标轴标签及范围
    # 添加X和Y轴标签  
    ax.set_xlabel('s-axis', fontsize = 10)  
    ax.set_ylabel('l-aixs', fontsize = 10)  
    
    # ax.set_xlim(left=sx_ls[dra_loc_sta], right= sx_ls[dra_loc_end-1] + 10 )
    # # ax.set_ylim(bottom=lane_markings[0]-0.5 - sy_ls[dra_loc_sta] , top=lane_markings[2]+0.5 - sy_ls[dra_loc_sta] )  #道路索引手动更改！
    # ax.set_ylim(bottom=lane_markings[0]- 0.5, top=lane_markings[2]+1.5 )  #道路索引手动更改！

    ax.set_xlim(left=sx_ls[dra_loc_sta], right=sx_ls[dra_loc_end-1] + 10)
    # Auto y-limits: cover all lane markings + actual trajectory data with padding
    y_refs = list(lane_markings) + list(sy_ls[dra_loc_sta:dra_loc_end])
    for t in range(dra_loc_sta, dra_loc_end):
        y_refs.extend(sy_plan_ls[t])
    y_min, y_max = np.nanmin(y_refs), np.nanmax(y_refs)
    margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
    ax.set_ylim(bottom=y_min - margin, top=y_max + margin)

    # -----------------------标题及图例
    fig.suptitle("Record Id: %d  Case Id: %d"%(rec_id, case_id) )
    #图例
    labels = ['Real trajectory', 'Dynamical pLanning trajectory']
    ax.legend(ax.lines[:2], loc = 'upper left', labels = labels, ncol = 2)   # 'center left',
    # fig.legend(ax.lines[:2], labels=labels, loc='left')   
                
    # Adjust spacing
    plt.subplots_adjust(left=0.04, right=0.98, top = 0.94, bottom=0.15, wspace = 0.2, hspace=0.6)
    
    # plt.show()

    #存图
    plt.savefig(os.path.join(SAVE_DIR5, 'replanning_'+'rec'+str(rec_id+1)+'_case'+str(case_id+1)+'.svg'))   
    # 关闭图
    plt.close(fig)    #由于当前的设置不会超过3张子图，故仅用一个figure就可以表示
    
    return 1
