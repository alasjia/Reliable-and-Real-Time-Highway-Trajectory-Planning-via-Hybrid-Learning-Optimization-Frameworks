20251228:

重新生成一遍PyG数据（HighD_datapre_pyg.py）


2026.01:
(1)基于HighD数据集（PyG格式）训练trajectory predication的E2E模型（train_VN.py），神经网路结构及其他超参数与Hybrid方法完全保持一致。
(2)基于训练完成的模型参数进行ADE与FDE的测试（test_for_de.py）。
(3)实现轨迹规划方法的online rolling/replanning（replanning文件夹）。
(4)实现论文中展示case的shots图和kinematic图的绘制，格式与Hybrid保持一致，并实现规划成功率的统计。


2026.02:
补丁patches：
（1）E2E方法shots图的绘制中，rec序号编码和Hybrid方法存在错位：rec_E2E = rec_Hyb + 1  (?)
（2）E2E方法shots图的绘制中，图中EV box下方的时间t标注存在错位：t_sta_display_traj(_E2E) = t_sta_display_traj(_Hyb) - 5
（3）E2E方法kinematics图的绘制中（kinematics_EV.py），由于将tar_sv_idx搜索简单化（手动指定TV/SV1），需要注意论文实验部分sceneC的tar_sv_idx = 3(A和B的均为默认的0)

论文中展示的case信息（按照Hybrid的索引）：
scene A: rec 55    ep 699    t = 4.0s -> 9.5s   t_sta_display_traj, t_end_display_traj = 35, 90  (in h-htp, t_sta_display_traj, t_end_display_traj = 25, 80)
scene B: rec 54    ep 1189   t = 2.0s -> 7.5s   t_sta_display_traj, t_end_display_traj = 15, 70  (in h-htp, t_sta_display_traj, t_end_display_traj = 10, 65)
scene C: rec 54    ep 1091   t = 4.5s -> 10.0s  t_sta_display_traj, t_end_display_traj = 40, 96


