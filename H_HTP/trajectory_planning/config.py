#----------------全局常量设置
import numpy as np

#*****the optimazation section*****
VEH_LEN = 4.8       #车长
VEH_WID = 1.8      #车宽
WHEEL_BASE  = 2.8   #固定的车辆轴距，网上随便查的
LANE_WID = 3.75    #车道宽度


#path planning 参数设置
MAX_ORI_ANGLE = np.pi/3   # max_orientation_angle   弧度制 rad
LAMBDA_N = 6        #避撞约束中考虑车身的区间个数            注意：Lambda_N必须为偶数！    note: 不要轻易改变，因为有一些fixed location操作


#绘图设置
CONTINUE_SWITCH_S = False   #对于单次规划，是否绘制动态图
CONTINUE_SWITCH_E = True  #executing BEV
CONTINUE_SWITCH_GT = False #绘制真实数据

DO_SMOOTH = 0    #是否要进行guide_lane的平滑处理   0不处理  1样条曲线拟合  2简单优化模型

SAMPLE_RATE = 0.5

old_version = True   #是否使用旧版本的NN model，如果是False则使用最新版本的Model


#directories
read_dir_hd = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/DataSets/HighD_Dataset" 
read_dir_dp = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/DataSets/Dataset_after_Processing"       
read_dir_pyg = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/DataSets/PyG_DataSet_4_HHTP" 
if not old_version:
    read_dir_pth = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training/Training_Results/Results20260310"
else:
    read_dir_pth = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training/Training_Results/Results_important_old"   



#存储路径
if not old_version:
    save_dir_htp = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/trajectory_planning/Results_HHTP/Results_20260311"
else:
    save_dir_htp= "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/trajectory_planning/Results_HHTP/Results_20260309"

SAVE_DIR2 = save_dir_htp + '/kinematic_res' #save kinematic results of   [[single]]   prediction
SAVE_DIR3 = save_dir_htp + '/executing_gif'  #存储执行路径图
SAVE_DIR4 = save_dir_htp + '/executing_BEV_shots'  #存储执行路径图
SAVE_DIR5 = save_dir_htp + '/replanning_details'  #存储replanning