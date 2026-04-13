'''
Store some constants in the input and output of the Neural Network
'''
#global constants
SPF = 0.1   #sec_per_fra         时间序列中每个元素之间的时间间隔长度
F_HIS = 20    #num_his_frames   观察序列时长
F_FUR = 30    #num_fur_frames   规划时长为30 steps = 3 sec
IF_NORM = True #是否以EV当前时刻位置为中心建立坐标系
MAX_SV = 8   #  maximum number of surrounding vehicles
N_FEA = 6      #kinematic features in the observation: x, y, vx, vy, ax, ay        note: 不要轻易改变，因为有一些fixed location操作

MAX_LINE = 4      #  maximum number of lane lines
N_LINE_SAM = 100   #the number of lane lines' sampling
INTVAL_LINE_SAM = 2   #the space length of lane lines' sampling inteval , unit: meter

MAX_SPEED = 50   #当前case的交通规则中道路最大速度   m/s   equals to 180 km/h

#车辆几和参数
VEH_WID, VEH_LEN = 1.8, 4.8

#the proportion of input historical data in a complete trajectory
divide_prop = 0.4   # 0.375---predict  5s    0.6----predict  3.2s     0.4----predict  4.8s


#for data processing
read_dir_hd = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/DataSets/HighD_Dataset" #!!!
read_dir_dp = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/DataSets/Dataset_after_Processing"       #!!!
read_dir_pyg = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/DataSets/PyG_DataSet_4_VectorNet_HighD"  #!!!

#for training
save_dir_train = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/VectorNet_HighD/Training_Results/Results20260310"  #trained model parameters storage path
#save_dir_pic = ""      #drawing results storage path 

#for testing
read_dir_train = save_dir_train
save_dir_rpl = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/VectorNet_HighD/replanning/paper_pics_generation/Results/Results20260311"
