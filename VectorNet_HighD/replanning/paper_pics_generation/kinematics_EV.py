import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys
sys.path.append("/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/VectorNet_HighD") 
from config_nw import *


def draw_kine_res_merged_e2e(
    sx_ls, vx_ls, sy_ls, vy_ls, fai_ls,
    svs_kin_ls,                 # 允许 None；若不为 None，需与混合框架一致结构：6个list，每个元素是(8,)的SV数组:contentReference[oaicite:2]{index=2}
    rec_id, case_id,
    sample_rate=0.1,
    t_sta=0,
    t_end=9999,
    plot_sv_in_v=True,          # 是否在 v-t 中画 SV1 和 Avg v（与你原版一致）:contentReference[oaicite:3]{index=3}
    tar_sv_idx=0,               # 你原版默认 SV1=idx0，但注释说有时要手动改:contentReference[oaicite:4]{index=4}
    set_default_ylims=True,     # 是否沿用你原版的手动 y 轴范围（v:20-33, heading:-2~6）:contentReference[oaicite:5]{index=5}
    save=True,
    show=False,
    save_dir=save_dir_rpl,
    fname_prefix="kin_res_e2e",
):
    

    """
    5张竖排 merged kinematics 图（对齐 res_kine_merged.py 的风格）:contentReference[oaicite:6]{index=6}
    figs: s-t, l-t, v-t(merged), a-t(merged), heading-t
    """

    if len(sx_ls) == 0:
        return 0

    # ---------- 1) downsample（与你原版一致：interv = sample_rate/0.1） ----------
    interv = int(sample_rate / 0.1)
    if interv <= 0:
        interv = 1

    kin_ls_avg = []
    for kin_ls in [sx_ls, sy_ls, vx_ls, vy_ls, fai_ls]:
        kin_ls_avg.append([np.mean(kin_ls[i:i+interv]) for i in range(0, len(sx_ls), interv)])

    # merged velocity and acceleration（与你原版一致的构造方式）:contentReference[oaicite:7]{index=7}
    v_merged_ls = [np.sqrt(kin_ls_avg[2][i]**2 + kin_ls_avg[3][i]**2) for i in range(len(kin_ls_avg[2]))]
    a_merged_ls = [(v_merged_ls[i] - v_merged_ls[i-1]) / 0.1 for i in range(1, len(v_merged_ls))]
    a_merged_ls.insert(0, (v_merged_ls[0] - np.sqrt(vx_ls[0]**2 + vy_ls[0]**2)) / 0.1)
    kin_ls_avg_merged = [kin_ls_avg[0], kin_ls_avg[1], v_merged_ls, a_merged_ls, kin_ls_avg[4]]

    # ---------- 2) SV downsample（允许 svs_kin_ls=None） ----------
    svs_kin_ls_avg = None
    if svs_kin_ls is not None and len(svs_kin_ls) > 0 and len(svs_kin_ls[0]) > 0:
        svs_kin_ls_avg = [[], [], [], [], [], []]
        for j in range(0, len(svs_kin_ls)):  # 6个指标：x,y,vx,vy,ax,ay
            for i in range(0, len(svs_kin_ls[0]), interv):
                svs_kin_ls_avg[j].append(np.mean(np.array(svs_kin_ls[j][i:i+interv]), axis=0))

    # ---------- 3) 截取窗口（与原版一致：steps 用 t_sta, t_end）:contentReference[oaicite:8]{index=8} ----------
    t_end_eff = min(len(kin_ls_avg_merged[0]), t_end)
    t_sta_eff = max(0, t_sta)
    if t_end_eff <= t_sta_eff + 1:
        return 0

    steps = [i * sample_rate for i in range(t_sta_eff, t_end_eff)]
    color_ev = "black"  # 与原版一致：EV黑色:contentReference[oaicite:9]{index=9}

    # ---------- 4) 画图：5张竖排 ----------
    num_figs = 5
    fig, axs = plt.subplots(num_figs, figsize=(7, 6))

    for i in range(num_figs):
        y = np.hstack(kin_ls_avg_merged[i])[t_sta_eff:t_end_eff]
        if i == 2:
            axs[i].plot(steps, y, linewidth=1.5, color=color_ev, label="Velocity profile of EV")
        else:
            axs[i].plot(steps, y, linewidth=1.5, color=color_ev)

    # ---------- 5) v-t 图：可选叠加 SV1 & 平均速度（沿用你原版）:contentReference[oaicite:10]{index=10} ----------
    if plot_sv_in_v and svs_kin_ls_avg is not None:
        # SV1 velocity profile
        vp_sv_x = np.array(svs_kin_ls_avg[2])[:, tar_sv_idx]
        vp_sv_y = np.array(svs_kin_ls_avg[3])[:, tar_sv_idx]
        vp_sv = np.sqrt(vp_sv_x**2 + vp_sv_y**2)
        axs[2].plot(steps, vp_sv[t_sta_eff:t_end_eff], linewidth=1.5, linestyle="dashed",
                    color=color_ev, label=f"Velocity profile of SV{tar_sv_idx+1}")

        # Average velocity（你原版是算所有SV的速度均值 + EV均值，取整体常数）:contentReference[oaicite:11]{index=11}
        # Average velocity（鲁棒版：跳过全 NaN 的 SV，不触发 Mean of empty slice）
        sv_vx = np.array(svs_kin_ls_avg[2])  # (T, nSV)
        sv_vy = np.array(svs_kin_ls_avg[3])  # (T, nSV)
        sv_speed = np.sqrt(sv_vx**2 + sv_vy**2)

        mean_per_sv = []
        nSV = sv_speed.shape[1] if sv_speed.ndim == 2 else 0
        for i in range(min(MAX_SV, nSV)):
            col = sv_speed[:, i]
            # 这辆 SV 在整个窗口内都缺失（全 NaN）=> 不参与平均
            if col.size == 0 or np.all(np.isnan(col)):
                continue
            mean_per_sv.append(np.nanmean(col))

        ev_mean = float(np.mean(kin_ls_avg_merged[2]))  # EV 的平均速度（你原逻辑里也加了它）

        # 如果所有 SV 都缺失：退化为 EV 平均速度（避免 nanmean([])）
        avg_v_sv = float(np.nanmean(mean_per_sv)) if len(mean_per_sv) > 0 else ev_mean
        # 保持你原来的风格：Average velocity 是一条常数水平线
        avg_v_sv_ls = [avg_v_sv for _ in range(len(v_merged_ls))]

        axs[2].plot(steps, np.array(avg_v_sv_ls)[t_sta_eff:t_end_eff], linewidth=1.5, linestyle="dotted",
                    color=color_ev, label="Average velocity")

    # ---------- 6) y 轴范围（保持 Hybrid 风格：bottom=20，但 top 必要时自动抬高，避免裁剪） ----------
    if set_default_ylims:
        # 收集 v-t 图里可能出现的所有速度曲线：EV、SV1、Average
        y_candidates = []

        # EV merged speed
        y_ev = np.array(kin_ls_avg_merged[2])[t_sta_eff:t_end_eff]
        y_candidates.append(y_ev)

        # SV1 speed（如果画了）
        if plot_sv_in_v and svs_kin_ls_avg is not None:
            vp_sv_x = np.array(svs_kin_ls_avg[2])[:, tar_sv_idx]
            vp_sv_y = np.array(svs_kin_ls_avg[3])[:, tar_sv_idx]
            vp_sv = np.sqrt(vp_sv_x**2 + vp_sv_y**2)[t_sta_eff:t_end_eff]
            y_candidates.append(vp_sv)

            # Average velocity（常数线）
            # 你这里如果已经算出了 avg_v_sv，就用它；否则就先跳过
            # y_candidates.append(np.array(avg_v_sv_ls)[t_sta_eff:t_end_eff])

        # 计算最大值（忽略 nan）
        y_all = np.hstack([np.ravel(y) for y in y_candidates if y is not None and len(y) > 0])
        y_all = y_all[np.isfinite(y_all)]

        v_top = 33
        if y_all.size > 0:
            v_top = max(33, float(np.nanmax(y_all)) + 1.0)  # +1 给一点 margin

        axs[2].set_ylim(bottom=20, top=v_top)   # ✅ bottom 保持一致，top 必要时扩展
        axs[4].set_ylim(bottom=-2, top=6)       # heading angle 保持原样


    # ---------- 7) 轴标签、刻度（保持原版风格）:contentReference[oaicite:13]{index=13} ----------
    for i in range(num_figs):
        axs[i].set_xlabel("Time(s)", fontsize=10)
        axs[i].xaxis.set_major_locator(MaxNLocator(nbins=int((len(vx_ls) * 0.1) / 1)))
        axs[i].set_xlim(t_sta_eff * 0.1, min(int(len(vx_ls)), t_end_eff) * 0.1)
        axs[i].tick_params(axis="x", labelsize=10)

    ylabels = ["s (m)", "l (m)", "Velocity\n(m/s)", "Acceleration\n(m/$s^2$)", "Heading Angle\n(degree)"]
    for i in range(num_figs):
        axs[i].set_ylabel(ylabels[i], fontsize=10)
        axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axs[i].tick_params(axis="y", labelsize=10)

    # legend（你原版只给速度图放 legend）:contentReference[oaicite:14]{index=14}
    axs[2].legend(loc="best", fontsize=8, ncol=3)

    plt.subplots_adjust(left=0.14, right=0.96, top=0.97, bottom=0.1, hspace=0.6)

    # ---------- 8) 保存/展示 ----------
    if save:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{fname_prefix}_rec{rec_id+1}_case{case_id+1}.svg")
        plt.savefig(out_path)

    if show:
        plt.show()

    plt.close(fig)
    return 1
