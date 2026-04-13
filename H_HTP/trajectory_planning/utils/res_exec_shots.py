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



def draw_trajectory_multi_shots(graph_data, sx_ls, sy_ls, fai_ls, sv_gt, t_ptr, line_num, rec_id, case_id, t_sta = 0):
    bevs_per_sub = 3#每个subplot画几个bev
    # subs_per_fig = 3  #每个figure画几个subplot
    
    fre_bev = 0.5 #second, 每隔多长时间绘制一次BEV
    num_bev = int((len(sx_ls)*0.1)/fre_bev)  #当前case总共需要绘制多少次BEV，向下取整
    num_sub = math.ceil(num_bev / bevs_per_sub)    #当前case需要绘制几张subplot，向上取整
    num_fig = math.ceil(num_sub/num_bev)  if num_bev > 0 else 0 #当前case需要绘制几张figure，向上取整
    
    subs_per_fig = num_sub  #每个figure画几个subplot，每个case仅画一张figure！
    if subs_per_fig < 2:
        return 0
    
    color_ev = '#ED746A'
    color_ev_rgb = mcolors.to_rgb(color_ev)
    colors_sv = [ '#638DEE', '#66C999', '#F49568', '#82C61E', '#E0A4DD', '#AA66EB', '#77DCDD', '#DBAA77']
    colors_sv_rgb = [mcolors.to_rgb(color) for color in colors_sv]  # 将颜色从十六进制转换为 RGB 格式
    # 注意，实际上SV颜色不是随机的，是和位置对应的，比如原本车道的FV固定为'#638DEE'
    

    for fig_idx in range(num_fig):
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(subs_per_fig, 1, figsize=(10, 8))   #(10, 6)
        
        #未中心化的坐标系（原始坐标系）
        lane_num = line_num
        lane_markings =  [i.item() for i in graph_data.lane_ys_nonorm]
        #统计当前case中存在的SV数量，用于图例
        num_sv = 0 
        
        #*************绘制每个子图
        for i in range(subs_per_fig):
            if (fig_idx*subs_per_fig + i) >= num_sub:
                break
            
        
            #------------------------绘制多个时刻下的BEV
            # # Create a colormap
            # cmap_ev = plt.get_cmap('jet')  # You can choose any colormap
            # cmap_sv = plt.get_cmap('jet')  # 需要区分开ev和sv！采用单色渐变色
        
            #依次处理每个子图中EV与周边车辆信息
            rectangles1 = []  # 用于存储创建的矩形对象  
            labels = []  # 用于存储矩形的标签  
            t_start = fig_idx*subs_per_fig*bevs_per_sub + bevs_per_sub*i
            t_end = fig_idx*subs_per_fig*bevs_per_sub + bevs_per_sub*(i+1)
            for t in range(t_start, t_end):   
                if t >= num_bev:  #t索引从0开始，所以是>=
                    break
                
                t_raw = t * (int( fre_bev / 0.1) )  #将目标频率转换为原始频率 对应的index。注意fre_bev的取值！
                # #当前时刻的颜色设置
                # color = cmap( / bevs_per_sub)  # Normalize the time index to the range [0, 1]
                
                # ev
                ev_xt = sx_ls[t_raw]
                ev_yt = sy_ls[t_raw]
                ev_angle_t = fai_ls[t_raw]
                gradients = create_gradient(color_ev_rgb, num_steps=bevs_per_sub) #填充的渐变色
                # Assign a color to the EV rectangle based on its time index
                rect = patches.Rectangle((ev_xt - VEH_LEN/2, ev_yt - VEH_WID/2), VEH_LEN, VEH_WID, linewidth=1.5, facecolor=gradients[t - t_start], edgecolor=color_ev, angle=ev_angle_t)
                rectangles1.append(rect)
                
                #sv  
                num_sv_ = 0 #用于图例
                for sv_id in range(MAX_SV):
                    x = sv_gt[F_HIS+t_raw, 6*sv_id+0] - VEH_LEN/2  
                    y = sv_gt[F_HIS+t_raw, 6*sv_id+1] - VEH_WID/2
                    sv_angle = np.degrees( np.arctan( sv_gt[F_HIS+t_raw, 6*sv_id+3] / sv_gt[F_HIS+t_raw, 6*sv_id+2]  ))
                    gradients = create_gradient(colors_sv_rgb[sv_id],  num_steps=bevs_per_sub) #填充的渐变色
                    if not np.isnan(x):
                        rect = patches.Rectangle((x, y), VEH_LEN, VEH_WID, linewidth=1.5, facecolor=gradients[t - t_start], edgecolor=colors_sv[sv_id], angle=sv_angle)
                        rectangles1.append(rect)                    # 将矩形对象添加到列表中 
                        num_sv_ += 1
                num_sv = max(num_sv, num_sv_)  #获取当前case中存在的SV数量，用于图例
            
                #时间标签
                axs[i].text(ev_xt , ev_yt - 2.5, 't = %.1f s' % (t_raw*0.1+t_sta*0.1), ha = 'center' )  #居中  字体？fontstyle = 'italic'         color = color_ev,     weight = 'bold'
                
            
            #-----------------------添加矩形到图中    
            for rect in rectangles1:
                axs[i].add_patch(rect)
                

                    
            #-----------------------坐标轴标签及范围
            # # 添加X和Y轴标签  
            # axs[i].set_xlabel('X-axis', fontsize = 10)  
            # axs[i].set_ylabel('Y-axis', fontsize = 10)  
            
            t_raw_1 = bevs_per_sub*i * (int( fre_bev / 0.1) )  #将目标频率转换为原始频率 对应的index。注意fre_bev的取值！
            t_raw_2 = min(( bevs_per_sub*(i+1) -1 ) * (int( fre_bev / 0.1) ), len(sx_ls)-1)
            axs[i].set_xlim(left=sx_ls[t_raw_1] - 40, right= sx_ls[t_raw_2] + 40 )
            axs[i].set_ylim(bottom=lane_markings[0]-0.5, top=lane_markings[-1]+0.5)

        # -----------------------添加必要的文本说明
        #时间标签：见上述BEV代码
        # # 总标题
        # fig.suptitle("Record Id: %d  Case Id: %d"%(rec_id, case_id) )
        # #图例
        # labels = ['SV%d'%(sv_idx+1) for sv_idx in range(num_sv)]    
        # labels.insert(0, 'EV')      
        # fig.legend(axs[0].patches[:len(labels)], labels=labels, loc='right')   
        # '''
        # axs[0].patches包含该子图中绘制的一切对象；
        # 但不知道为什么，如果同时绘制了车道线和轨迹线，无论怎么定位都从线开始显示？
        # 所以把线单独放在后续绘制。
        # 这种方法不稳定，在确定展示的case后需要手动调整！
        # '''
        
        
        
        for i in range(subs_per_fig):
            if (fig_idx*subs_per_fig + i) >= num_sub:
                break   
            # #-----------------------绘制轨迹线条：实际执行的path的插值拟合曲线
            # t_raw_1 = bevs_per_sub*i * (int( fre_bev / 0.1) )  #将目标频率转换为原始频率 对应的index。注意fre_bev的取值！
            # t_raw_2 = min(( bevs_per_sub*(i+1) -1 ) * (int( fre_bev / 0.1) ), len(sx_ls)-1)
            
            # spline = interp1d(sx_ls[t_raw_1: t_raw_2+1], sy_ls[t_raw_1: t_raw_2+1])
            # x_curve = np.linspace(sx_ls[t_raw_1],sx_ls[t_raw_2], 100)
            # axs[i].plot(x_curve, spline(x_curve), '-', c = color_ev, linewidth=1)
            
            #------------------------绘制车道线作为参考
            for y in lane_markings :  
                axs[i].axhline(y=y, color='gray', linestyle='--')  # 使用plt.axhline()函数绘制横线
                
        plt.subplots_adjust(left=0.06, right=0.96, top = 0.95, bottom=0.05, wspace = 0.2, hspace=0.2)
        
        # plt.show()

        #存图
        plt.savefig(os.path.join(SAVE_DIR4, 'shots_'+'rec'+str(rec_id+1)+'_case'+str(case_id+1)+'.svg'))   
        # 关闭图
        plt.close(fig)    #由于当前的设置不会超过3张子图，故仅用一个figure就可以表示
    
    return 1


def create_gradient(color, num_steps=5):
    """为给定颜色生成渐变序列"""
    gradient = []
    for i in range(num_steps):
        # 渐变公式：浅色 = 深色 * (i / (num_steps - 1))
        grad_color = tuple(c * (i / (num_steps - 1)) + (1 - i / (num_steps - 1)) for c in color)
        gradient.append(grad_color)
    return gradient
