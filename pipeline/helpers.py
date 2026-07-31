"""
Shared helper functions for the RLVR checkpoint sweep.

run_inspect_eval uses subprocess.run rather than Colab's shell magic, so
this module works as a plain script as well as inside a notebook kernel.
"""

import glob
import json
import os
import random
import shlex
import shutil
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

import config


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    rng = random.Random(42)
    means = [
        np.mean([arr[rng.randrange(len(arr))] for _ in range(len(arr))])
        for _ in range(n_boot)
    ]
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(means, alpha * 100)),
        float(np.percentile(means, (1 - alpha) * 100)),
    )


def wilson_ci(successes, total, ci=0.95):
    from scipy import stats

    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def is_ea_complete(checkpoint):
    if not os.path.exists(config.EA_RESULTS_PATH):
        return False
    df = pd.read_csv(config.EA_RESULTS_PATH)
    return checkpoint in df["checkpoint"].values


def is_ic_complete(checkpoint):
    if not os.path.exists(config.IC_RESULTS_PATH):
        return False
    df = pd.read_csv(config.IC_RESULTS_PATH)
    return checkpoint in df["checkpoint"].values


def is_math_complete(checkpoint):
    if not os.path.exists(config.MATH_RESULTS_PATH):
        return False
    df = pd.read_csv(config.MATH_RESULTS_PATH)
    return checkpoint in df["checkpoint"].values


def check_disk_space(min_free_gb=20):
    """
    Raise before starting a measurement that would likely fail partway
    through from running out of local disk space, rather than losing an
    in-progress run to a mid-generation crash.
    """
    free_gb = shutil.disk_usage("/").free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f}GB free disk space remaining (need at least "
            f"{min_free_gb}GB). Restart the runtime for a fresh disk, then "
            f"resume the sweep; already-completed checkpoints are safe and "
            f"will not be redone."
        )


LOGS_DIR = os.path.abspath("logs")


RAW_LOG_ARCHIVE_DIR = os.environ.get("RLVR_LOG_ARCHIVE_DIR")


def archive_log(log_path):
    """
    Copy a log file to RLVR_LOG_ARCHIVE_DIR if that environment variable
    is set. Returns the archive path, or None if archiving is disabled.
    Safe to call unconditionally; does nothing when the variable is unset.
    """
    if not RAW_LOG_ARCHIVE_DIR:
        return None
    os.makedirs(RAW_LOG_ARCHIVE_DIR, exist_ok=True)
    dest = os.path.join(RAW_LOG_ARCHIVE_DIR, os.path.basename(log_path))
    shutil.copy2(log_path, dest)
    return dest


def get_latest_log():
    logs = sorted(glob.glob(os.path.join(LOGS_DIR, "*.eval")))
    if not logs:
        raise FileNotFoundError(f"No .eval log files found in {LOGS_DIR}")
    return logs[-1]


def run_inspect_eval(command):
    """
    Run an `inspect eval ...` command and raise on a nonzero exit code,
    so a failed run stops the sweep loudly instead of falling through to
    get_latest_log() and picking up a stale log from a previous checkpoint.

    The command is parsed with shlex before being run, so it is not
    interpreted by a live shell. This avoids the argument-splitting and
    special-character risks of shell=True while still accepting a single
    command string built with normal quoting.
    """
    args = shlex.split(command)
    result = subprocess.run(args)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n{command}"
        )


def process_ea_log(log_path, checkpoint):
    log = read_eval_log(log_path)
    accuracy_vals, vea_vals, judge_failed = [], [], 0

    for sample in log.samples or []:
        for _, score_val in (sample.scores or {}).items():
            if hasattr(score_val, "value") and score_val.value is not None:
                accuracy_vals.append(float(score_val.value))
                if hasattr(score_val, "metadata") and score_val.metadata:
                    vea_vals.append(1 if score_val.metadata.get("vea_present", False) else 0)
                    if score_val.metadata.get("judge_failed", False):
                        judge_failed += 1
                else:
                    vea_vals.append(0)
                break

    n = len(accuracy_vals)
    acc_ci = bootstrap_ci(accuracy_vals)
    vea_ci = bootstrap_ci(vea_vals)

    return {
        "checkpoint": checkpoint,
        "ea_accuracy": round(float(np.mean(accuracy_vals)) if accuracy_vals else 0.0, 4),
        "ea_ci_low": round(acc_ci[0], 4),
        "ea_ci_high": round(acc_ci[1], 4),
        "ea_vea_rate": round(float(np.mean(vea_vals)) if vea_vals else 0.0, 4),
        "ea_vea_ci_low": round(vea_ci[0], 4),
        "ea_vea_ci_high": round(vea_ci[1], 4),
        "ea_judge_failed": judge_failed,
        "n_samples": n,
        "timestamp": datetime.now().isoformat(),
    }


def process_ic_log(log_path, checkpoint):
    log = read_eval_log(log_path)
    convergence_by_cat, all_convergence, judge_failed = {}, [], 0
    unique_ids = set()

    for sample in log.samples or []:
        unique_ids.add(getattr(sample, "id", None))
        for _, score_val in (sample.scores or {}).items():
            if hasattr(score_val, "metadata") and score_val.metadata:
                cat = score_val.metadata.get("category", "unknown")
                conv = score_val.metadata.get("convergence", False)
                failed = score_val.metadata.get("judge_failed", False)
                if failed:
                    judge_failed += 1
                else:
                    all_convergence.append(1 if conv else 0)
                    convergence_by_cat.setdefault(cat, []).append(1 if conv else 0)
                break

    n = len(all_convergence)
    ci = wilson_ci(sum(all_convergence), n)
    cat_rates = {cat: round(float(np.mean(vals)), 4) for cat, vals in convergence_by_cat.items()}

    if judge_failed > 5:
        print(f"warning: {judge_failed} IC judge calls failed for {checkpoint}")

    return {
        "checkpoint": checkpoint,
        "convergence_rate": round(float(np.mean(all_convergence)) if all_convergence else 0.0, 4),
        "convergence_ci_low": round(ci[0], 4),
        "convergence_ci_high": round(ci[1], 4),
        "convergence_by_category": json.dumps(cat_rates),
        "ic_judge_failed": judge_failed,
        "n_observations": n,
        "n_unique_tasks": len(unique_ids),
        "n_rollouts": 3,
        "timestamp": datetime.now().isoformat(),
    }


def process_math_log(log_path, checkpoint):
    log = read_eval_log(log_path)
    accuracy_vals, verify_failures = [], 0

    for sample in log.samples or []:
        for _, score_val in (sample.scores or {}).items():
            if hasattr(score_val, "value") and score_val.value is not None:
                accuracy_vals.append(float(score_val.value))
                if hasattr(score_val, "metadata") and score_val.metadata:
                    if score_val.metadata.get("verify_failed", False):
                        verify_failures += 1
                break

    n = len(accuracy_vals)
    ci = bootstrap_ci(accuracy_vals)

    return {
        "checkpoint": checkpoint,
        "accuracy": round(float(np.mean(accuracy_vals)) if accuracy_vals else 0.0, 4),
        "accuracy_ci_low": round(ci[0], 4),
        "accuracy_ci_high": round(ci[1], 4),
        "math_verify_failures": verify_failures,
        "n_problems": n,
        "timestamp": datetime.now().isoformat(),
    }


def append_result(csv_path, row):
    """
    Append one row to a result CSV, refusing to duplicate a checkpoint.
    Writes to a temp file first, then replaces the real file with
    os.replace, which is atomic on POSIX filesystems. A crash mid write
    can then only ever leave the original file untouched or the new file
    fully written; it cannot leave the file truncated or corrupted.
    """
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        if row["checkpoint"] in df_existing["checkpoint"].values:
            print(f"{row['checkpoint']} already in {os.path.basename(csv_path)}, skipping")
            return
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    tmp_path = csv_path + ".tmp"
    df_combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)


def init_result_csvs():
    if not os.path.exists(config.EA_RESULTS_PATH):
        pd.DataFrame(columns=[
            "checkpoint", "ea_accuracy", "ea_ci_low", "ea_ci_high",
            "ea_vea_rate", "ea_vea_ci_low", "ea_vea_ci_high",
            "ea_judge_failed", "n_samples", "timestamp",
        ]).to_csv(config.EA_RESULTS_PATH, index=False)

    if not os.path.exists(config.IC_RESULTS_PATH):
        pd.DataFrame(columns=[
            "checkpoint", "convergence_rate", "convergence_ci_low", "convergence_ci_high",
            "convergence_by_category", "ic_judge_failed",
            "n_observations", "n_unique_tasks", "n_rollouts", "timestamp",
        ]).to_csv(config.IC_RESULTS_PATH, index=False)

    if not os.path.exists(config.MATH_RESULTS_PATH):
        pd.DataFrame(columns=[
            "checkpoint", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
            "math_verify_failures", "n_problems", "timestamp",
        ]).to_csv(config.MATH_RESULTS_PATH, index=False)
