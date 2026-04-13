此次训练目的是作为速度规划结果输入优化模型（第一篇成果）

输出元素之间的时间间隔为0.1 sec(0.2-->0.1)
较于1203的修改：
输出内容重新改为纵向速度，每个元素之间的时间间隔为0.1 sec；
输出层的激活函数sigmoid,通过乘[V_limit]设置速度约束。


Note:
VN_XX_1: 50 records(4-53) and 100 epoches + decaying lr(*0.9 / 20 eps)
VN_XX_2: 50 records(4-53) and 25 epoches + decaying lr(*0.3 / 5 eps)

将数据集处理为一个场景包括多条轨迹（沿时间轴滑动），原先为一个场景对应一条轨迹：
VN_XX_3: 3 records(4,5,6) and 100 epoches + decaying lr(*0.9 / 20 eps) + sliding step=0.1 sec
VN_XX_4: 20 records(26-45, all three-lane cases) and 100 epoches + decaying lr(*0.9 / 5 eps) + sliding step=1 sec
VN_XX_4_2: 20 records(26-45, all three-lane cases) and 100 epoches + decaying lr(*0.9 / 5 eps) + sliding step=1 sec  (using InMemoryDataset methold)(453698 trajectories, 0.7:0.2:0.1, [317588, 90739, 45370]) 
VN_XX_5: 20 records(26-45, all three-lane cases) and 100 epoches + decaying lr(*0.9 / 5 eps) + sliding step=1 sec  (using InMemoryDataset methold)(453698 trajectories, 0.9:0.07:0.03,[408328, 31758, 13611]) 
