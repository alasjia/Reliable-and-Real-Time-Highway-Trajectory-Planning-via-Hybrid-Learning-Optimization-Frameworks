import torch
import time
import numpy as np
from config import *

from tra_plannning.replanning2 import execute_replan


if __name__ == "__main__":
    t_sta = time.time()
    


    device = torch.device('cpu')
    rec_sta, rec_end = 54, 55  
    case_sta, case_end = 698, 701

    print(torch.cuda.is_available())
    
    num_succ_cases, num_total_cases, succ_rates, fail_cases_res, running_time = execute_replan(read_dir_hd, read_dir_dp, read_dir_pth, device, rec_sta, rec_end, case_sta, case_end)  
    
    
    # 数据存储
    # save_dir1 = save_dir_htp + '/running_time/'
    # np.savetxt(save_dir1 + 'planning_time_'+'rec%d_%dto%d'%(rec_sta+1, case_sta, case_end) + '.csv', running_time[1] , delimiter=',', fmt='%.6f')  # fmt 控制精度
    # np.savetxt(save_dir1 + 'solving_time_'+'rec%d_%dto%d'%(rec_sta+1, case_sta, case_end) + '.csv',  running_time[2] , delimiter=',', fmt='%.6f')  
    # np.savetxt(save_dir1 + 'ave_planning_time_'+'rec%d_%dto%d'%(rec_sta+1, case_sta, case_end) + '.csv', running_time[0] , delimiter=',', fmt='%.6f')  
    # save_dir2 = save_dir_htp + 'i/success_rate/'
    # np.savetxt(save_dir2 + 'fail_cases_idx_'+'rec%d_%dto%d'%(rec_sta+1, case_sta, case_end) + '.csv', fail_cases_res[0], delimiter=',', fmt='%d')  
    # np.savetxt(save_dir2 + 'fail_steps_by_case_'+'rec%d_%dto%d'%(rec_sta+1, case_sta, case_end) + '.csv', fail_cases_res[1], delimiter=',', fmt='%d')  
    
    print(f"执行{num_total_cases}个场景，其中{num_succ_cases}个场景成功求解，两种求解率分别为{succ_rates[0]*100: .6f}%和{succ_rates[1]*100: .6f}%")
    ave_ope_time = sum(running_time[0]) / len(running_time[0])   
    print(f"平均每次单步规划用时 {ave_ope_time:.4f} 秒, 最大{max(running_time[1]):.4f}秒，最小{min(running_time[1]):.4f}秒。") 
    
    total_t =  time.time() - t_sta
    print(f"总共用时 {total_t:.2f} 秒") 
    
    
    '''    
    为什么ob_epr2_2和ob_epr3_1的约束力不同？
    
    当前path planning模型对reference path的敏感性很高，一方面是模型变量缺少关于时间的直接约束，无法提供侧向加速度的硬约束；另一方面是时间间隔为0.1s，意味着加速度接近瞬时加速度，一旦速度出现微小波动加速度就会突变
    尝试多种平滑reference path的方法仍然存在加速度的突变（最小可以控制在5左右？）一般都发生在换道开始和换道结束，（尤其是当EV要进入一个缩窄安全区间的时刻，如果目标函数中平滑度部分的权重过大反而会带来先大幅转弯再大幅会正的现象）。
    
    '''
