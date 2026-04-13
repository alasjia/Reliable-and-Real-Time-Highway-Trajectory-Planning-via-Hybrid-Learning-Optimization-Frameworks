'''
Store some constants in the input and output of the Neural Network
'''
# global constants
SPF = 0.1   # sec_per_frame         The time interval length between each element in the time series
F_HIS = 20    # num_his_frames   Length of the observed sequence
F_FUR = 30    # num_fur_frames   Planning length is 30 steps = 3 sec
IF_NORM = True # Whether to establish the coordinate system centered at the EV's current position
MAX_SV = 8   #  maximum number of surrounding vehicles
N_FEA = 6      # kinematic features in the observation: x, y, vx, vy, ax, ay        note: Do not change easily, as there are some fixed location operations

MAX_LINE = 4      #  maximum number of lane lines
N_LINE_SAM = 100   # the number of lane line sampling points
INTVAL_LINE_SAM = 2   # the spatial length of lane line sampling interval, unit: meter

MAX_SPEED = 50   # The maximum speed in the current case's traffic rules   m/s   equals to 180 km/h

#车辆几和参数
VEH_WID, VEH_LEN = 1.8, 4.8

#the proportion of input historical data in a complete trajectory
divide_prop = 0.4  


#for data processing
read_dir_hd = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/DataSets/HighD_Dataset" #!!!
read_dir_dp = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/DataSets/Dataset_after_Processing"       #!!!
read_dir_pyg = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/DataSets/PyG_DataSet_4_HHTP"  #!!!
               
#for training
# save_dir_train = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training/Training_Results/Results20260213"  #trained model parameters storage path
save_dir_train = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training/Training_Results/Results20260310"  #trained model parameters storage path 

#for testing
read_dir_train = save_dir_train
save_dir_bev = "/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/H_HTP/velocity_prediction_training/Testing_Results/DE_BEVfigs_0309"