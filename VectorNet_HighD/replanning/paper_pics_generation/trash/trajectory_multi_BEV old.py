'''
    #调用
    draw_trajectory_multi_BEV_e2e(
        graph_data=o0,
        sx_ls=logger.sx, sy_ls=logger.sy, fai_ls=logger.yaw,
        sv_gt=sv_gt[1:, :],
        pred_traj_glo_ls=logger.pred_traj_glo,   # 直接用 logger 里存的
        line_num=line_num,
        rec_id=rec, case_id=ep,
        t_sta=(t_sta_display_traj if use_seg else 0),
        t_end=(t_end_display_traj if use_seg else None),
        n_shots=4
    )
'''



import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
# sys.path.append("E:/UTC_PHD/Compared1_VN_trajectory_prediction")  #记得改！！！  
sys.path.append("/home/chwei/Compared1_VN_trajectory_prediction") 
from config_nw import *

def draw_trajectory_multi_BEV_e2e(
    graph_data,
    sx_ls, sy_ls, fai_ls,
    sv_gt,                 # shape: (T_total, 6*MAX_SV) —— 建议传 sv_gt[1:,:] 与混合框架一致
    line_num,
    rec_id, case_id,
    pred_traj_glo_ls=None, # list，每个元素 (F_FUR,2) in GLOBAL；允许为 None
    t_sta=0,
    t_end=None,
    n_shots=6,
    save_dir=SAVE_DIR4,
    fname_prefix="multi_bev_e2e",
    show=False,
):
    """
    多帧 BEV 合成图（与混合框架 draw_trajectory_BEV 相同风格）

    - 每个 shot：画 lane markings + SV rectangles + EV rectangle（带 yaw）
    - 可选：叠加该时刻的 E2E 预测轨迹 pred_traj_glo_ls[t]（虚线）
    """
    if t_end is None:
        t_end = len(sx_ls)
    t_end = min(t_end, len(sx_ls))
    if t_end <= t_sta + 1:
        return 0

    # lane markings（与混合框架一致：直接用 lane_ys_nonorm，不做中心化）
    lane_markings_all = [i.item() for i in graph_data.lane_ys_nonorm]
    lane_markings = lane_markings_all[:line_num] if line_num is not None else lane_markings_all

    # 选取要截图的时刻：均匀采样
    idxs = np.linspace(t_sta, t_end - 1, n_shots).astype(int).tolist()
    idxs = sorted(list(dict.fromkeys(idxs)))  # 去重保序

    fig, axs = plt.subplots(len(idxs), 1, figsize=(25, 3.2 * len(idxs)), sharex=True)
    if len(idxs) == 1:
        axs = [axs]

    for ax, t in zip(axs, idxs):
        # ---------- lane ----------
        for y in lane_markings:
            ax.axhline(y=y, color="gray", linestyle="--")

        # ---------- SV rectangles ----------
        # 混合框架里是 sv_gt[F_HIS + t, ...]，这里我们假设传入的 sv_gt 已经对齐好
        # 如果你传的是 sv_gt[1:,:]，那就直接用 idx = F_HIS + t（与混合框架完全一致）
        idx = F_HIS + t
        if idx >= sv_gt.shape[0]:
            # 越界就跳过（避免偶发 IndexError）
            continue

        for sv_id in range(MAX_SV):
            x = sv_gt[idx, 6 * sv_id + 0] - VEH_LEN / 2
            y = sv_gt[idx, 6 * sv_id + 1] - VEH_WID / 2
            if np.isnan(x):
                continue
            rect = patches.Rectangle(
                (x, y), VEH_LEN, VEH_WID,
                linewidth=2, facecolor="none", edgecolor="cornflowerblue"
            )
            ax.add_patch(rect)

        # ---------- EV rectangle ----------
        ev_xt = sx_ls[t]
        ev_yt = sy_ls[t]
        ev_angle_t = fai_ls[t] if fai_ls is not None else 0.0
        rect_ev = patches.Rectangle(
            (ev_xt - VEH_LEN / 2, ev_yt - VEH_WID / 2),
            VEH_LEN, VEH_WID,
            linewidth=2, facecolor="none", edgecolor="orangered",
            angle=ev_angle_t
        )
        ax.add_patch(rect_ev)

        # ---------- optional: E2E predicted trajectory (spaghetti one-shot) ----------
        if pred_traj_glo_ls is not None and t < len(pred_traj_glo_ls):
            traj = pred_traj_glo_ls[t]
            if traj is not None:
                traj = np.asarray(traj)
                if traj.ndim == 2 and traj.shape[1] == 2:
                    ax.plot(traj[:, 0], traj[:, 1], "--", linewidth=1.5)  # 风格与混合框架 EV轨迹图一致（虚线）

        # ---------- labels / limits / title (match hybrid style) ----------
        ax.set_xlabel("X-axis", fontsize=20)
        ax.set_ylabel("Y-axis", fontsize=20)

        # 与混合框架一致：显示全部范围（基于执行轨迹首尾）
        ax.set_xlim(sx_ls[0] - 20, sx_ls[t_end - 1] + 20)
        ax.set_ylim(sy_ls[0] - 15, sy_ls[0] + 10)

        ax.set_title(f"Record Id: {rec_id}  Case Id: {case_id}  Time Id: {t}")

    plt.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.08, hspace=0.45)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{fname_prefix}_rec{rec_id+1}_case{case_id+1}.svg")
    plt.savefig(out_path)

    if show:
        plt.show()
    plt.close(fig)
    return 1
