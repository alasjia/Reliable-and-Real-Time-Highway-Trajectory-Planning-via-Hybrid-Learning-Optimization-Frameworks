# e2e_replanning_batch.py
# ------------------------------------------------------------
# Batch evaluation for E2E replanning success rate
# - Case success rate: #success_cases / #total_cases
# - Step success rate: #successful_steps / #total_steps
# Success definition aligns with Hybrid (replanning2.py):
#   done occurs either because reach end-of-GT (success)
#   or invalid / early termination (failure)
#   success iff t_ptr > gt_total_t at termination
# ------------------------------------------------------------



'''
当前这个成功率的计算没有意义：
因为Hybrid“成功”的定义是优化模型求解成功，而E2E的模型不存在无可行解问题，应该是从“是否避障成功"的角度去定义。
但是目前没有时间了，就先不做这个部分了
'''




from __future__ import annotations

import os
import re
import time
import argparse
import array
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import sys
sys.path.append("/home/lab/luyujia/projects_alasjia/Hybrid_Highway_Trajectory_Planning/VectorNet_HighD") 
from config_nw import *

# Reuse E2E env reset & step
# Make sure these two functions exist in your local online_rolling_upper.py
from online_rolling_upper import env_reset, step_e2e


# ============================================================
# Default experiment configuration (E2E batch replanning)
# ============================================================

DEFAULT_CFG = dict(
    # model
    ckpt          = os.path.join(read_dir_train, "traj_pred_VN_100ep_lrdf09.pth"),   #!!! 
    input_features= 15,
    pred_len      = 30,

    # device
    device        = "cuda:0",

    # evaluation range    #!!!
    rec_sta       = 54,       
    rec_end       = 55,
    case_sta      = 0,
    case_end      = 500,

    # runtime
    maxsteps      = 1000,
    verbose_every = 50,
)




def _list_valid_csv_files(read_dir_dp: str) -> List[str]:
    """List preprocessed csv files and filter out LibreOffice temp files etc."""
    files_ = os.listdir(read_dir_dp)
    pattern = r'^\.~lock.*#$'
    files = [f for f in files_ if not re.match(pattern, f)]
    files.sort()
    return files


def _load_recording_data(
    read_dir_hd: str,
    read_dir_dp: str,
    rec: int,
    files_sorted: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load recordingMeta (raw) and tracks_df (preprocessed) for a given recording id.
    This matches your online_rolling_upper.py demo logic.
    """
    rMetafile_name = f"{rec:02d}_recordingMeta.csv"
    recordingMeta = pd.read_csv(os.path.join(read_dir_hd, rMetafile_name))

    # pick the preprocessed csv ending with _{rec:02d}.csv
    matched = [f for f in files_sorted if f.endswith(f"_{int(rec):02d}.csv")]
    if len(matched) == 0:
        raise FileNotFoundError(f"No preprocessed csv found for rec={rec:02d} under {read_dir_dp}")
    csv_name = matched[0]
    tracks_df = pd.read_csv(os.path.join(read_dir_dp, csv_name), index_col=0)

    return recordingMeta, tracks_df


def _iter_cases(tracks_df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split a recording into cases by vehicle id (same as your demo)."""
    grouped = tracks_df.groupby("id")
    group_data = [g for _, g in grouped]
    return group_data


def _predict_pred_xy(model: torch.nn.Module, o, device: torch.device) -> np.ndarray:
    """
    Run VectorNet once and return pred_xy with shape (F_FUR, 2).
    Aligns with your online_rolling_upper.py:
        o_cpu = o.clone().to("cpu")
        DataLoader([o_cpu]) -> batch -> model(batch) -> out["pred"]
        reshape to (-1,2)
    """
    o_cpu = o.clone().to("cpu")  # VN may modify input
    o_dl = DataLoader([o_cpu], batch_size=1)
    batch = next(iter(o_dl)).to(device)

    with torch.no_grad():
        out = model(batch)

    y_hat = out["pred"].detach().cpu().numpy()
    if y_hat.ndim == 3:
        y_hat = y_hat[0]
    pred_xy = y_hat.reshape(-1, 2)
    return pred_xy


def execute_replan_e2e_batch(
    read_dir_hd: str,
    read_dir_dp: str,
    model: torch.nn.Module,
    device: torch.device,
    rec_sta: int,
    rec_end: int,
    case_sta: int,
    case_end: int,
    maxsteps_per_epoch: int = 1000,
    verbose_every: int = 50,
) -> Dict[str, Any]:
    """
    Batch replanning evaluation for E2E.
    Returns a dict with success stats and failure lists.
    """
    files_sorted = _list_valid_csv_files(read_dir_dp)

    # ---- stats (align with Hybrid replanning2.py) ----
    num_succ_cases, num_total_cases = 0, 0
    num_succ_steps, num_total_steps = 0, 0

    fail_cases_arr = array.array("i")         # store ep index (within recording)
    fail_steps_bycase_arr = array.array("i")  # remaining steps not finished (gt_total_t+1 - t_ptr)
    fail_rec_arr = array.array("i")           # store rec index for each failure (helpful!)

    # runtime stats (optional)
    ave_step_time_arr = array.array("f")
    all_step_time_arr = array.array("f")

    model.eval()

    for rec in range(rec_sta, rec_end):
        # load recording data
        try:
            recordingMeta, tracks_df = _load_recording_data(read_dir_hd, read_dir_dp, rec, files_sorted)
        except Exception as e:
            print(f"[WARN] Skip rec={rec:02d} due to loading error: {e}")
            continue

        group_data = _iter_cases(tracks_df)
        case_num = len(group_data)
        if case_num == 0:
            print(f"[WARN] rec={rec:02d} has 0 cases, skip.")
            continue

        ep_sta_ = max(0, case_sta)
        ep_end_ = min(case_end, case_num)
        if ep_sta_ >= ep_end_:
            print(f"[INFO] rec={rec:02d}: no cases in range [{case_sta},{case_end}), skip.")
            continue

        for ep in range(ep_sta_, ep_end_):
            num_total_cases += 1

            track_df = group_data[ep]

            # NOTE: in Hybrid you maintain a global cluster ptr; here we keep per-case ptr from env_reset/step
            cluster_global_ptr = 0

            # reset
            o, sv_gt, cluster_global_ptr, line_num = env_reset(
                track_df, recordingMeta, cluster_global_ptr=cluster_global_ptr
            )

            # success definition aligned with Hybrid:
            gt_total_t = sv_gt.shape[0] - F_FUR  # max t_ptr allowed before end condition triggers

            t_ptr = F_HIS
            step_times_this_case: List[float] = []
            done = False

            for _ in range(maxsteps_per_epoch):
                t_s = time.time()

                # ---- 1) model inference ----
                pred_xy = _predict_pred_xy(model, o, device)

                # ---- 2) environment step (E2E) ----
                o, done, t_ptr, cluster_global_ptr, exec_xy_g, exec_xy_l = step_e2e(
                    obs=o,
                    sv_gt=sv_gt,
                    cluster_global_ptr=cluster_global_ptr,
                    line_num=line_num,
                    t_ptr=t_ptr,
                    pred_xy=pred_xy,
                )

                dt_step = time.time() - t_s
                step_times_this_case.append(dt_step)
                all_step_time_arr.append(dt_step)

                # ---- 3) step success count ----
                # Align with Hybrid: only count steps that did NOT terminate
                if not done:
                    num_succ_steps += 1
                else:
                    break

            # ---- end of case handling ----
            # total steps possible in GT, align with Hybrid replanning2.py
            # (how many rolling decisions you *should* be able to make)
            num_total_steps += max(0, (gt_total_t + 1 - F_HIS))

            # average step time for this case (optional)
            if len(step_times_this_case) > 0:
                ave_step_time_arr.append(float(np.mean(step_times_this_case)))

            # case success check
            # success if termination was by reaching the end:
            # step_e2e sets done when (t_ptr > track_max_f - F_FUR)
            # which should match (t_ptr > gt_total_t) here
            if t_ptr > gt_total_t:
                num_succ_cases += 1
            else:
                fail_rec_arr.append(rec)
                fail_cases_arr.append(ep)
                fail_steps_bycase_arr.append(int((gt_total_t + 1) - t_ptr))

            # print progress
            if verbose_every > 0 and (num_total_cases % verbose_every == 0):
                case_sr = num_succ_cases / max(1, num_total_cases)
                step_sr = num_succ_steps / max(1, num_total_steps)
                print(
                    f"[E2E-BATCH] cases {num_total_cases} | "
                    f"case_SR={case_sr:.4f} step_SR={step_sr:.4f} | "
                    f"latest rec={rec:02d} ep={ep} t_ptr={t_ptr} gt_total_t={gt_total_t}"
                )

    # finalize
    case_success_rate = num_succ_cases / max(1, num_total_cases)
    step_success_rate = num_succ_steps / max(1, num_total_steps)

    ret = {
        "num_succ_cases": int(num_succ_cases),
        "num_total_cases": int(num_total_cases),
        "case_success_rate": float(case_success_rate),
        "num_succ_steps": int(num_succ_steps),
        "num_total_steps": int(num_total_steps),
        "step_success_rate": float(step_success_rate),
        "fail_rec_arr": fail_rec_arr,
        "fail_cases_arr": fail_cases_arr,
        "fail_steps_bycase_arr": fail_steps_bycase_arr,
        "ave_step_time_arr": ave_step_time_arr,
        "all_step_time_arr": all_step_time_arr,
    }
    return ret


def build_model_and_load_ckpt(
    device: torch.device,
    ckpt_path: str,
    input_features: int,
    pred_len: int,
) -> torch.nn.Module:
    """
    Align with online_rolling_upper.py:
        from train_VN import VectorNet
        model = VectorNet(input_features, pred_len, device, with_aux=False).to(device)
        model.load_state_dict(torch.load(ckpt_path))
    """
    from train_VN import VectorNet

    model = VectorNet(input_features, pred_len, device, with_aux=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args(default_cfg: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser()

    for k, v in default_cfg.items():
        arg_type = type(v)
        if v is None:
            p.add_argument(f"--{k}", default=None)
        else:
            p.add_argument(f"--{k}", type=arg_type, default=v)

    return p.parse_args()


def main():
    args = parse_args(DEFAULT_CFG)

    # device
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    model = build_model_and_load_ckpt(
        device=device,
        ckpt_path=args.ckpt,
        input_features=args.input_features,
        pred_len=args.pred_len,
    )

    # run batch
    t0 = time.time()
    res = execute_replan_e2e_batch(
        read_dir_hd=read_dir_hd,
        read_dir_dp=read_dir_dp,
        model=model,
        device=device,
        rec_sta=args.rec_sta,
        rec_end=args.rec_end,
        case_sta=args.case_sta,
        case_end=args.case_end,
        maxsteps_per_epoch=args.maxsteps,
        verbose_every=args.verbose_every,
    )
    t1 = time.time()

    # print summary
    print("\n========== E2E Batch Replanning Summary ==========")
    print(f"recordings: [{args.rec_sta}, {args.rec_end})  cases: [{args.case_sta}, {args.case_end})")
    print(f"case_success_rate = {res['case_success_rate']:.6f}   ({res['num_succ_cases']}/{res['num_total_cases']})")
    print(f"step_success_rate = {res['step_success_rate']:.6f}   ({res['num_succ_steps']}/{res['num_total_steps']})")
    print(f"total_time = {t1 - t0:.2f} s")
    # ---- runtime stats ----
    if len(res["ave_step_time_arr"]) > 0:
        print(f"avg_step_time (mean over cases) = {np.mean(res['ave_step_time_arr']):.6f} s")
    if len(res["all_step_time_arr"]) > 0:
        print(f"avg_step_time (mean over all steps) = {np.mean(res['all_step_time_arr']):.6f} s")

    # optional: show worst failures quickly
    if len(res["fail_cases_arr"]) > 0:
        print(f"\n#fail_cases = {len(res['fail_cases_arr'])}")
        # print first 10
        for i in range(min(10, len(res["fail_cases_arr"]))):
            print(
                f"fail[{i}] rec={int(res['fail_rec_arr'][i]):02d} "
                f"ep={int(res['fail_cases_arr'][i])} "
                f"remain_steps={int(res['fail_steps_bycase_arr'][i])}"
            )


if __name__ == "__main__":
    main()
