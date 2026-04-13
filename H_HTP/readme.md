
2026.03:
论文中展示的case信息（按照Hybrid的索引）：
scene A: rec 54    ep 699    t = 3.0s -> 8.5s      t_sta_display, t_end_display = 30, 90
scene B: rec 53    ep 1189   t = 1.5s -> 7.0s      t_sta_display, t_end_display = 30, 80
scene C: rec 53    ep 1091   t = 4.5s -> 10.0s     t_sta_display, t_end_display = 45, 110


2026.03.09:
it is very weird, i don't know somehow the new trained model of velocity prediction can't have similar resluts on the above cases for displaying, acturely it works worse.
maybe it's because different order of data in the dataset are used in the training(e.g. "random_split()", "shuffle"), it might be a little difference for parameters of neural network. (the velocity profile looks too bad...)

2026.03.10
fixed: i made the "decay_lr_factor" wrong value(should be 0.9, but used 0.3). after correct to right lr decay factor (=0.9), update the results of Vectornet-HighD.