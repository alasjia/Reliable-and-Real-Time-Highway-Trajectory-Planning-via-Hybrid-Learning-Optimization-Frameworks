'''
补丁patches：
（1）E2E方法shots图的绘制中，ep序号编码和Hybrid方法存在错位：ep_E2E = ep_Hyb + 1
（2）E2E方法shots图的绘制中，图中EV box下方的时间t标注存在错位：t_sta_display_traj(_E2E) = t_sta_display_traj(_Hyb) - 5
（3）E2E方法kinematics图的绘制中（kinematics_EV.py），由于将tar_sv_idx搜索简单化（手动指定TV/SV1），需要注意论文实验部分sceneC的tar_sv_idx = 3(A和B的均为默认的0)

论文中展示的case信息（按照Hybrid的索引）：
scene A: rec 55    ep 700    t = 3.0s -> 8.5s
scene B: rec 54    ep 1190   t = 1.5s -> 7.0s
scene C: rec 54    ep 1092   t = 4.5s -> 10.0s
'''



import sys
sys.path.append("/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/VectorNet_HighD") 
from config_nw import *


import numpy as np
import os, re
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from online_rolling_lower import get_single_track_data, get_sv_gt, get_localCS_sv_gt, get_obs_2D, get_next_obs, get_fake_next_obs

import time
import array

from paper_pics_generation.trajectory_multi_BEV import draw_trajectory_multi_BEV_e2e
from paper_pics_generation.kinematics_EV import draw_kine_res_merged_e2e


def env_reset(track_df, recordingMeta, cluster_global_ptr):
    obs_t0, cluster_global_ptr, line_num = get_single_track_data(track_df, cluster_global_ptr, recordingMeta)
    #从原始数据集中获取global坐标系下的周围车辆运动信息
    sv_gt = get_sv_gt( track_df)  #(frame_num, feature_num)  (68, 48)

    return obs_t0, sv_gt, cluster_global_ptr, line_num


def step_e2e(
    obs,
    sv_gt,
    cluster_global_ptr,
    line_num,
    t_ptr,
    pred_xy=None,              # (T,2) in local CS, 推荐传这个
    exec_xy_global=None,       # (2,) in global CS, 兼容你原始接口
):
    """
    E2E online rolling 的一步环境更新（仿照混合框架 env.step）：
    - 用 gt 的 SV 作为环境演化（和混合框架一致）
    - EV 的“动作”来自 E2E 网络预测轨迹的第 1 个点

    Returns:
        next_obs, done, t_ptr, cluster_global_ptr, exec_ev_global_xy, exec_ev_local_xy
    """

    # ---------- 0) 终止条件：sv_gt 时长不够提供未来 F_FUR（与混合框架一致） ----------
    track_max_f = sv_gt.shape[0]
    if t_ptr > (track_max_f - F_FUR):
        return None, True, t_ptr, cluster_global_ptr, None, None

    # ---------- 1) 准备：把 SV gt 转到当前 local CS ----------
    # obs.xys_t0_nonorm = 当前 local 坐标系原点在 global 下的位置
    origin_global = np.array(obs.xys_t0_nonorm).copy()  # shape (2,)
    sv_gt_local = get_localCS_sv_gt(sv_gt, origin_global)  # (N,48)

    # obs -> structured (270,6)，拿到 EV 当前时刻运动学（local CS）
    obs_2D = get_obs_2D(obs)  # ((MAX_SV+1)*F_HIS, 6)
    ev_curr = obs_2D[F_HIS - 1].copy()  # [x, y, vx, vy, ax, ay] in local CS

    # ---------- 2) 执行“动作”：确定下一帧 EV 的位置 ----------
    if pred_xy is None and exec_xy_global is None:
        raise ValueError("step_e2e needs pred_xy (local) or exec_xy_global (global).")

    if pred_xy is not None:
        pred_xy = np.asarray(pred_xy).reshape(-1, 2)
        exec_ev_local_xy = pred_xy[0].copy()  # 下一帧位置（local CS）
        exec_ev_global_xy = exec_ev_local_xy + origin_global
    else:
        exec_xy_global = np.asarray(exec_xy_global).reshape(2,)
        exec_ev_global_xy = exec_xy_global.copy()
        exec_ev_local_xy = exec_ev_global_xy - origin_global

    # 安全/合理性：防止倒退（可以和混合框架一样只看第一步）
    # 注意：local CS 下 EV 当前 x 通常 ~0，所以判断 exec_ev_local_xy[0] < 0 等价“倒退”
    if (exec_ev_local_xy[0] - ev_curr[0]) < 0:
        # 生成 fake next obs：只更新 cluster，保证 batch/subgraph 不炸
        next_obs, cluster_global_ptr = get_fake_next_obs(obs, cluster_global_ptr, line_num)
        return next_obs, True, t_ptr, cluster_global_ptr, exec_ev_global_xy, exec_ev_local_xy

    # ---------- 3) 用“执行的下一帧位置”反推 vx, vy, ax, ay（与混合框架一致的差分思想） ----------
    vx_next = (exec_ev_local_xy[0] - ev_curr[0]) / SPF
    vy_next = (exec_ev_local_xy[1] - ev_curr[1]) / SPF
    ax_next = (vx_next - ev_curr[2]) / SPF
    ay_next = (vy_next - ev_curr[3]) / SPF

    # 构造 EV 的 next history window（滑窗）
    ev_kin_next = np.concatenate(
        (obs_2D[1:F_HIS], np.array([exec_ev_local_xy[0], exec_ev_local_xy[1], vx_next, vy_next, ax_next, ay_next]).reshape(1, -1)),
        axis=0
    )  # (F_HIS, 6)

    # 时间推进一帧（这一点很关键：SV 的历史窗口也要同步推进）
    t_ptr += 1

    # ---------- 4) 用 gt 的 SV 填充 next history window（与混合框架一致） ----------
    sv_on_ls = []
    for sv in range(1, MAX_SV + 1):
        sv_on_ls.append(
            sv_gt_local[t_ptr - F_HIS: t_ptr, N_FEA*(sv-1): N_FEA*(sv-1) + N_FEA].copy()
        )
    sv_kin_next = np.concatenate(sv_on_ls, axis=0)  # (MAX_SV*F_HIS, 6)
    sv_kin_next = np.array([np.nan_to_num(_, nan=0) for _ in sv_kin_next])

    next_obs_2D_ = np.concatenate((ev_kin_next, sv_kin_next), axis=0)  # ((MAX_SV+1)*F_HIS,6)

    # ---------- 5) structured -> graph，并重建 local CS（你 lower.py 已实现） ----------
    next_obs, cluster_global_ptr = get_next_obs(obs, next_obs_2D_, cluster_global_ptr, line_num)

    return next_obs, False, t_ptr, cluster_global_ptr, exec_ev_global_xy, exec_ev_local_xy



from dataclasses import dataclass, field

@dataclass
class RollingLogger:
    dt: float
    # executed EV global positions
    xy_g: list = field(default_factory=list)

    # kinematics (global)
    sx: list = field(default_factory=list)
    sy: list = field(default_factory=list)
    vx: list = field(default_factory=list)
    vy: list = field(default_factory=list)
    ax: list = field(default_factory=list)
    ay: list = field(default_factory=list)
    yaw: list = field(default_factory=list)

    # optional: predicted trajectories (global), per step
    pred_traj_glo: list = field(default_factory=list)

    # optional: per-step runtime
    step_time: list = field(default_factory=list)

    def push_xy(self, xy_g):
        """Push executed EV position (global) and update v/a/yaw by finite difference."""
        xy_g = np.asarray(xy_g).reshape(2,)
        self.xy_g.append(xy_g)

        # position
        self.sx.append(float(xy_g[0]))
        self.sy.append(float(xy_g[1]))

        # velocity / yaw / acceleration
        if len(self.xy_g) == 1:
            vx = vy = ax = ay = yaw = 0.0
        else:
            (x1, y1) = self.xy_g[-2]
            (x2, y2) = self.xy_g[-1]
            vx = (x2 - x1) / self.dt
            vy = (y2 - y1) / self.dt
            yaw = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            pvx = self.vx[-1] if len(self.vx) else 0.0
            pvy = self.vy[-1] if len(self.vy) else 0.0
            ax = (vx - pvx) / self.dt
            ay = (vy - pvy) / self.dt

        self.vx.append(float(vx))
        self.vy.append(float(vy))
        self.ax.append(float(ax))
        self.ay.append(float(ay))
        self.yaw.append(float(yaw))

    def push_pred_traj_local(self, pred_xy_local, origin_global_xy):
        """
        Convert pred trajectory from current local CS to GLOBAL and store.
        pred_xy_local: (T,2)
        origin_global_xy: (2,) = o.xys_t0_nonorm
        """
        pred = np.asarray(pred_xy_local).reshape(-1, 2).copy()
        org = np.asarray(origin_global_xy).reshape(2,)
        pred[:, 0] += org[0]
        pred[:, 1] += org[1]
        self.pred_traj_glo.append(pred)  # rolling 执行得到的 EV global 位置序列

    def push_step_time(self, sec: float):
        self.step_time.append(float(sec))


if __name__ == "__main__":

    # ========= 0) 全局参数/路径  =========
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_features = 15
    pred_len = 30
    maxsteps_per_epoch  =1000

    # (B) E2E 模型建立与参数加载：
    from train_VN import VectorNet  

    e2e_pth_path = os.path.join(read_dir_train, "traj_pred_VN_100ep_lrdf09.pth")   #!!!

    model = VectorNet(input_features, pred_len, device, with_aux= False).to(device)   
    model.load_state_dict(torch.load(e2e_pth_path) )
    model.eval()

    # (C) 选择要画的场景（rec = 文件索引，ep = case 索引）并初始化存储空间
    rec = 55     #!!!
    ep  = 699   #!!!
    svs_kin_ls = [ [], [], [], [], [], []  ]
    ope_time_ep = 0   #该case的运行时间统计清零

    # (D) 绘制选项
    # 如果你只想展示某一段（用于对齐混合框架论文图），打开se_seg
    t_sta_display_traj, t_end_display_traj = 35, 90   # 0.1s 间隔                 #!!!
    t_sta_display_kine, t_end_display_kine = 2, 999   # 0.1s 间隔  开始用第2帧，避免v/a突变  #!!!
    use_seg = True  #True  False

    # ========= 1) 准备 env.reset 所需：挑选 rec 文件、挑选 ep 车辆轨迹 =========
    # 复用 replanning2.py 的文件过滤 & 排序逻辑 :contentReference[oaicite:4]{index=4}
    files_ = os.listdir(read_dir_dp)
    pattern = r'^\.~lock.*#$'
    files = [f for f in files_ if not re.match(pattern, f)]
    files.sort()

    # 拼 recordingMeta 名字 :contentReference[oaicite:5]{index=5}
    rMetafile_name =  f"{rec:02d}_recordingMeta.csv" 
    recordingMeta = pd.read_csv(os.path.join(read_dir_hd, rMetafile_name))
    csv_name = [f for f in files if f.endswith(f"_{int(rec):02d}.csv")][0]
    tracks_df = pd.read_csv(os.path.join(read_dir_dp, csv_name), index_col=0)

    grouped = tracks_df.groupby("id")
    group_data = [g for _, g in grouped]   # 每个 id 一个 case
    case_num = len(group_data)
    assert ep < case_num, f"ep={ep} 超出该 rec 的 case 数量 {case_num}"

    track_df = group_data[ep]


    # ========= 2) env.reset 环境初始化 =========
    o, sv_gt, cluster_global_ptr, line_num = env_reset(track_df, recordingMeta,  cluster_global_ptr = 0)
    o0 = o.clone() #不重要,绘图使用
    #时间指针：观察序列中最后一个数据在整个cas时长中的时间索引，初始取值由观察序列时长参数决定
    t_ptr = F_HIS 
    #初始化运动学参数存储
    logger = RollingLogger(dt=SPF)

    # ========= 3) online rolling 开始 =========

    # 记录初始 global 原点（用于把 rolling 的 global 轨迹统一转回初始 local CS 画图）
    origin0_global = np.array(o.xys_t0_nonorm).copy()

    # 统计与缓存
    plan_time_arr = array.array('f')
    # num_succ_steps, num_total_steps = 0, 0  #用于统计成功率

    # ground truth 中可用的最大时间步（对齐 replanning2 的成功定义）
    gt_total_t = sv_gt.shape[0] - F_FUR

    done = False
    fail_reason = None

    for t in range(maxsteps_per_epoch):

        t_s = time.time()

        # 1) VN 推理（注意：你之前做了 o_cpu clone，是对的）
        o_cpu = o.clone().to("cpu")   # VN 可能改 input，所以 clone
        o_dl = DataLoader([o_cpu], batch_size=1)
        batch = next(iter(o_dl)).to(device)

        with torch.no_grad():
            out = model(batch)

        # 2) 取 pred_xy (T,2)
        y_hat = out["pred"].detach().cpu().numpy()
        if y_hat.ndim == 3:
            y_hat = y_hat[0]
        pred_xy = y_hat.reshape(-1, 2)     # 当前 local CS 下：相对 t0 的未来位置序列（不是增量）

        #存储每步的预测轨迹
        logger.push_pred_traj_local(pred_xy, origin_global_xy=np.array(o.xys_t0_nonorm))


        # 3) 环境 step：执行 pred_xy[0] 
        o2, done, t_ptr, cluster_global_ptr, exec_xy_g, exec_xy_l = step_e2e(
            obs=o,
            sv_gt=sv_gt,
            cluster_global_ptr=cluster_global_ptr,
            line_num=line_num,
            t_ptr=t_ptr,
            pred_xy=pred_xy,   # 推荐直接传 pred_xy
        )

        plan_time_arr.append(time.time() - t_s)

        # 4) 记录执行结果(全局坐标系)
        if exec_xy_g is not None:
            logger.push_xy(exec_xy_g)

            # 4.5) 记录当前时刻 SV 的运动学信息（用于 v-t 子图画 SV1 和 Average velocity）
            sv_t = t_ptr - 1  # 和 Hybrid 对齐：执行后的“当前帧”
            if 0 <= sv_t < sv_gt.shape[0]:
                row = sv_gt[sv_t]  # shape (48,) = 8 SV * 6 features
                # 每个都是 shape (8,)
                sv_x  = row[0::6]
                sv_y  = row[1::6]
                sv_vx = row[2::6]
                sv_vy = row[3::6]
                sv_ax = row[4::6]
                sv_ay = row[5::6]

                svs_kin_ls[0].append(sv_x)
                svs_kin_ls[1].append(sv_y)
                svs_kin_ls[2].append(sv_vx)
                svs_kin_ls[3].append(sv_vy)
                svs_kin_ls[4].append(sv_ax)
                svs_kin_ls[5].append(sv_ay)



        # 5) done 处理
        if done:
            # 这里 done 可能来自：
            # (a) 正常滚到结尾（t_ptr > track_max_f - F_FUR）
            # (b) 不合理动作（例如倒退）导致 fake next_obs 并提前终止
            if t_ptr > gt_total_t:
                fail_reason = None  # success
            else:
                fail_reason = f"terminated early at t_ptr={t_ptr}, gt_total_t={gt_total_t}"
            break


        # num_succ_steps += 1
        # num_total_steps += 1

        # 6) 非 done：更新观测
        o = o2

    # ========= 4) rolling 结果整理：把 global 轨迹转成初始 local CS，才能一次性画 =========
    if len(logger.xy_g) == 0:
        print("[WARN] No executed steps recorded; skip drawing.")
    else:
        # ========= 5) 绘图（把 rolling 执行轨迹当作 “pred_xy” 传入画图函数） =========
        #绘制时间间隔为100ms的shots
        draw_trajectory_multi_BEV_e2e(
            graph_data=o0,
            sx_ls=logger.sx, sy_ls=logger.sy, fai_ls=logger.yaw,
            sv_gt=sv_gt[1:, :],
            line_num=line_num,
            rec_id=rec, case_id=ep,
            t_sta=(t_sta_display_traj if use_seg else 0),
            t_end=(t_end_display_traj if use_seg else None),
            subs_per_fig=4
        )



        #绘制EV实际执行轨迹的运动学信息
        draw_kine_res_merged_e2e(
            sx_ls=logger.sx, vx_ls=logger.vx,
            sy_ls=logger.sy, vy_ls=logger.vy,
            fai_ls=logger.yaw,
            svs_kin_ls=svs_kin_ls,
            rec_id=rec, case_id=ep,
            sample_rate=0.1,
            t_sta=(t_sta_display_kine if use_seg else 0),
            t_end=(t_end_display_kine if use_seg else 9999),
            tar_sv_idx=0,   #0   手动改？检查一下最终图里tv v profile，应该是与Hybrid保持一致（average velocity也是！）
            save=True, show=False
        )


    # ========= 7) 打印统计信息 =========
    print("========== Online Rolling (E2E) Summary ==========")
    print(f"rec={rec}, ep={ep}")
    print(f"executed_steps = {len(logger.xy_g)}")
    print(f"t_ptr_end      = {t_ptr}, gt_total_t = {gt_total_t}")
    print(f"success        = {fail_reason is None and len(logger.xy_g)>0}")
    if fail_reason is not None:
        print(f"fail_reason    = {fail_reason}")
    if len(plan_time_arr) > 0:
        print(f"avg_step_time  = {float(np.mean(plan_time_arr)):.4f} s")
        print(f"max_step_time  = {float(np.max(plan_time_arr)):.4f} s")
