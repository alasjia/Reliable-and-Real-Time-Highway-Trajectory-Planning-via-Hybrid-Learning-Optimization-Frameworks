# ============================================================
# E2E multi-shot BEV (Hybrid res_shots.py v2 style clone)
# - No predicted trajectory line
# - Same: fre_bev=0.5s, bevs_per_sub=3, gradient fill rectangles
# - Same: colors, linewidth, xlim/ylim, time text placement, subplot spacing
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
# sys.path.append("E:/UTC_PHD/Compared1_VN_trajectory_prediction")  #记得改！！！  
sys.path.append("/home/chwei/Compared1_VN_trajectory_prediction") 
from config_nw import *

def _hex_to_rgb01(hex_color: str):
    """'#RRGGBB' -> (r,g,b) in [0,1]"""
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def _create_gradient(rgb01, num_steps: int):
    """
    Match Hybrid v2: generate light-to-original gradient colors (RGB tuples).
    """
    r, g, b = rgb01
    # light color: blend toward white
    light = np.array([1.0, 1.0, 1.0]) * 0.65 + np.array([r, g, b]) * 0.35
    base = np.array([r, g, b])
    if num_steps <= 1:
        return [tuple(base.tolist())]
    out = []
    for i in range(num_steps):
        a = i / (num_steps - 1)
        col = light * (1 - a) + base * a
        out.append(tuple(col.tolist()))
    return out


def draw_trajectory_multi_BEV_e2e(
    graph_data,
    sx_ls, sy_ls, fai_ls,
    sv_gt,                 # IMPORTANT: pass sv_gt[1:, :] from caller (you already did)
    line_num,
    rec_id, case_id,
    t_sta=0,
    t_end=None,
    subs_per_fig=4,        # one page 4 subplots (your request)
    fre_bev=0.5,           # Hybrid v2 default
    bevs_per_sub=3,        # Hybrid v2 default
    dt=0.1,                # your sampling period
    save_dir=SAVE_DIR4,
    fig_w=10,
    fig_h=8,
    x_pad=40,              # Hybrid v2: -40/+40
    y_pad=0.5,             # Hybrid v2: lane min/max +-0.5
    lane_lw=2.0,           # Hybrid v2: linewidth=2 for lanes
    rect_lw=1.5,           # Hybrid v2: linewidth=1.5 for vehicle rectangles
    time_text_dy=-2.5,     # Hybrid v2: ev_yt - 2.5
    out_ext="svg",         # "pdf"/"svg"
    verbose=False,
):
    """
    E2E shots BEV figure with EXACT style of Hybrid res_shots.py draw_trajectory_multi_BEV_v2.
    Each subplot overlays `bevs_per_sub` consecutive BEVs with gradient fill.
    Output: 1 page with `subs_per_fig` subplots (4 by default).

    NOTE:
    - This function does NOT draw predicted trajectory line (as you requested).
    - SV indexing follows Hybrid v2: sv_gt[F_HIS + t_raw, ...]
    """

    if t_end is None:
        t_end = len(sx_ls)
    t_end = min(t_end, len(sx_ls))

    if t_end <= t_sta + 1:
        if verbose:
            print("[E2E_v2style] t_end too small, skip.")
        return 0

    # --- lane markings (same source as your original E2E function) ---
    lane_markings_all = [i.item() for i in graph_data.lane_ys_nonorm]
    lane_markings = lane_markings_all[:line_num] if line_num is not None else lane_markings_all
    if len(lane_markings) == 0:
        lane_markings = [0.0]

    # --- color config (copy from Hybrid v2) ---
    color_ev = "#ED746A"
    colors_sv = ["#638DEE", "#66C999", "#F6C470", "#A7D0D6", "#DDB2D2"]

    rgb_ev = _hex_to_rgb01(color_ev)
    rgbs_sv = [_hex_to_rgb01(c) for c in colors_sv]

    grad_ev = _create_gradient(rgb_ev, bevs_per_sub)
    grads_sv = [_create_gradient(rgb, bevs_per_sub) for rgb in rgbs_sv]

    # --- convert fre_bev seconds to steps in your dt timeline (dt=0.1 => 0.5s = 5 steps) ---
    step_per_bev = int(round(fre_bev / dt))
    if step_per_bev <= 0:
        step_per_bev = 1

    # --- determine number of subplots from [t_sta, t_end) ---
    # Hybrid v2:
    # num_sub = int((t_end - t_sta) / (fre_bev/0.1 * bevs_per_sub)) + 1
    # Here fre_bev/0.1 == step_per_bev; so:
    span = (t_end - t_sta)
    denom = step_per_bev * bevs_per_sub
    num_sub = int(span / denom) + 1

    # we only draw one page with `subs_per_fig` subplots (4)
    num_sub_to_draw = min(subs_per_fig, num_sub)

    # --- figure ---
    fig, axs = plt.subplots(num_sub_to_draw, 1, figsize=(fig_w, fig_h), sharex=False)
    if num_sub_to_draw == 1:
        axs = [axs]

    # ---- draw each subplot ----
    for sub_idx in range(num_sub_to_draw):
        ax = axs[sub_idx]

        # lane
        for y in lane_markings:
            ax.axhline(y=y, color="gray", linestyle="--", linewidth=lane_lw)

        # time window in steps:
        # Hybrid v2:
        # t_raw_1 = t_raw = sub_idx*(step_per_bev)*bevs_per_sub + t_sta
        # and for each k: t_raw = sub_idx*(step_per_bev)*bevs_per_sub + k*(step_per_bev) + t_sta
        t_raw_1 = sub_idx * denom + t_sta
        t_raw_2 = t_raw_1 + denom  # end boundary for xlim calc (same spirit as v2)

        # overlay bevs_per_sub frames with gradient
        for k in range(1, bevs_per_sub + 1):
            t_raw = sub_idx * denom + k * step_per_bev + t_sta

            if t_raw < 0 or t_raw >= len(sx_ls):
                continue

            # SV rectangles (filled gradient + colored edge)
            idx_sv = F_HIS + t_raw
            if 0 <= idx_sv < sv_gt.shape[0]:
                for sv_id in range(MAX_SV):
                    x_c = sv_gt[idx_sv, 6 * sv_id + 0]
                    y_c = sv_gt[idx_sv, 6 * sv_id + 1]
                    if np.isnan(x_c) or np.isnan(y_c):
                        continue

                    x = x_c - VEH_LEN / 2
                    y = y_c - VEH_WID / 2

                    if k == 1:
                        face_col = "none"   # 第一个时间戳：空白
                    else:
                        # 从第二个时间戳开始才填充渐变色
                        face_col = grads_sv[sv_id % len(colors_sv)][k - 2]
                    rect = patches.Rectangle(
                        (x, y),
                        VEH_LEN,
                        VEH_WID,
                        linewidth=rect_lw,
                        edgecolor=colors_sv[sv_id % len(colors_sv)],
                        facecolor=face_col,
                    )

                    ax.add_patch(rect)

            # EV rectangle (filled gradient + colored edge)
            ev_x = sx_ls[t_raw]
            ev_y = sy_ls[t_raw]
            ev_angle = fai_ls[t_raw] if fai_ls is not None else 0.0

            if k == 1:
                face_col_ev = "none"
            else:
                face_col_ev = grad_ev[k - 2]

            rect_ev = patches.Rectangle(
                (ev_x - VEH_LEN / 2, ev_y - VEH_WID / 2),
                VEH_LEN,
                VEH_WID,
                linewidth=rect_lw,
                edgecolor=color_ev,
                facecolor=face_col_ev,
                angle=ev_angle,
            )
            
            ax.add_patch(rect_ev)

            # time text for EACH EV box (k=1..bevs_per_sub)
            t_sec = (t_raw) * dt
            # 为了避免 3 个文本完全重叠：让每个 k 的文本在 y 方向错开一点点
            # 你可以调这个 0.7（米）到更舒服的数
            dy_k = (k - 1) * (-0.7)
            ax.text(
                ev_x,
                ev_y + time_text_dy + dy_k,
                "t = %.1f s" % (t_sec),
                ha="center",
            )

        # x/y limits (match v2 spirit)
        t_raw_1_clip = np.clip(t_raw_1, 0, len(sx_ls) - 1)
        t_raw_2_clip = np.clip(t_raw_2, 0, len(sx_ls) - 1)

        ax.set_xlim([sx_ls[t_raw_1_clip] - x_pad, sx_ls[t_raw_2_clip] + x_pad])
        ax.set_ylim([lane_markings[0] - y_pad, lane_markings[-1] + y_pad])

        # keep clean (Hybrid v2: no title, no xlabel/ylabel, no legend)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.2)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir,
        f"shots_e2e_rec{rec_id+1}_case{case_id+1}.{out_ext}"
    )
    plt.savefig(out_path)
    plt.close(fig)

    if verbose:
        print(f"[E2E_v2style] saved: {out_path}")

    return 1
