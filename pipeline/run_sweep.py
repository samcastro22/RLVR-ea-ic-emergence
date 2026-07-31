"""
Runs EA, IC, and MATH evaluation across every checkpoint listed in
config.CHECKPOINTS_PATH.

Requires OPENAI_API_KEY set in the environment for the EA and IC judge
calls, and HF_TOKEN set if the model needs authentication to download.

Resume-safe: a checkpoint's measurement is only run if it is missing from
that measurement's result CSV, so this can be rerun after any interruption
without redoing completed work.
"""

import csv
import glob
import json
import os
import random
import tempfile
import time

import pandas as pd
from datasets import load_dataset

import config
import helpers


def load_checkpoints():
    checkpoints_df = pd.read_csv(config.CHECKPOINTS_PATH)
    checkpoints = checkpoints_df.iloc[:, 0].dropna().tolist()
    return [c for c in checkpoints if str(c).startswith("step_")]


def load_math_problems():
    math_ds = load_dataset("allenai/RLVR-MATH", split="train")
    rng = random.Random(config.MATH_SAMPLE_SEED)
    indices = list(range(len(math_ds)))
    rng.shuffle(indices)
    return list(math_ds.select(indices[: config.MATH_SAMPLE_SIZE]))


def run_checkpoint(checkpoint, math_problems):
    print(f"\n{'=' * 60}\nCheckpoint: {checkpoint}\n{'=' * 60}")
    start = time.time()

    ea_done = helpers.is_ea_complete(checkpoint)
    ic_done = helpers.is_ic_complete(checkpoint)
    math_done = helpers.is_math_complete(checkpoint)

    if not ea_done:
        helpers.check_disk_space()
        print("Running EA (976 samples)")
        helpers.run_inspect_eval(
            f"inspect eval tasks/ea_task.py "
            f"--model {config.MODEL_ID} "
            f"-M revision={checkpoint} "
            f'-T ea_path="{config.EA_DATA_PATH}"'
        )
        ea_log = helpers.get_latest_log()
        ea_result = helpers.process_ea_log(ea_log, checkpoint)
        helpers.append_result(config.EA_RESULTS_PATH, ea_result)
        helpers.archive_log(ea_log)
        os.remove(ea_log)
        print(f"EA accuracy={ea_result['ea_accuracy']:.3f} vea_rate={ea_result['ea_vea_rate']:.3f}")
    else:
        print("EA already complete, skipping")

    if not ic_done:
        helpers.check_disk_space()
        print("Running IC (76 tasks, 3 rollouts each)")
        helpers.run_inspect_eval(
            f"inspect eval tasks/ic_task.py "
            f"--model {config.MODEL_ID} "
            f"-M revision={checkpoint} "
            f'-T ic_folder="{config.IC_DATA_PATH}"'
        )
        ic_log = helpers.get_latest_log()
        ic_result = helpers.process_ic_log(ic_log, checkpoint)
        helpers.append_result(config.IC_RESULTS_PATH, ic_result)
        helpers.archive_log(ic_log)
        os.remove(ic_log)
        print(f"IC convergence_rate={ic_result['convergence_rate']:.3f}")
    else:
        print("IC already complete, skipping")

    if not math_done:
        helpers.check_disk_space()
        print(f"Running MATH ({config.MATH_SAMPLE_SIZE} problems)")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(math_problems, f)
            math_tmp = f.name
        try:
            helpers.run_inspect_eval(
                f"inspect eval tasks/math_task.py "
                f"--model {config.MODEL_ID} "
                f"-M revision={checkpoint} "
                f'-T problems_path="{math_tmp}"'
            )
            math_log = helpers.get_latest_log()
            math_result = helpers.process_math_log(math_log, checkpoint)
            helpers.append_result(config.MATH_RESULTS_PATH, math_result)
            helpers.archive_log(math_log)
            os.remove(math_log)
        finally:
            os.remove(math_tmp)
        print(f"MATH accuracy={math_result['accuracy']:.3f}")
    else:
        print("MATH already complete, skipping")

    print(f"Checkpoint done in {(time.time() - start) / 60:.1f} min")


def validate_openai_key():
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    try:
        openai.OpenAI(api_key=api_key).models.list()
    except Exception as e:
        raise RuntimeError(
            f"Could not verify the OpenAI API key before starting the sweep: {e}\n"
            f"Check that OPENAI_API_KEY is set to a valid, active key."
        )


def validate_setup():
    problems = []

    if not os.path.exists(config.CHECKPOINTS_PATH):
        problems.append(f"Checkpoints file not found: {config.CHECKPOINTS_PATH}")
    else:
        try:
            df = pd.read_csv(config.CHECKPOINTS_PATH)
            step_values = [c for c in df.iloc[:, 0].dropna().tolist() if str(c).startswith("step_")]
            if not step_values:
                problems.append(
                    f"{config.CHECKPOINTS_PATH} contains no values starting with "
                    f"'step_' in its first column"
                )
        except Exception as e:
            problems.append(f"Could not read {config.CHECKPOINTS_PATH}: {e}")

    if not os.path.exists(config.EA_DATA_PATH):
        problems.append(f"EA dataset not found: {config.EA_DATA_PATH}")
    else:
        try:
            with open(config.EA_DATA_PATH) as f:
                ea_data = json.load(f)
            if not isinstance(ea_data, list) or len(ea_data) == 0:
                problems.append(f"{config.EA_DATA_PATH} does not contain a non-empty list of transcripts")
        except Exception as e:
            problems.append(f"Could not read {config.EA_DATA_PATH}: {e}")

    if not os.path.exists(config.IC_DATA_PATH):
        problems.append(f"IC data folder not found: {config.IC_DATA_PATH}")
    else:
        csv_files = [f for f in os.listdir(config.IC_DATA_PATH) if f.endswith(".csv")]
        if not csv_files:
            problems.append(f"IC data folder exists but contains no CSV files: {config.IC_DATA_PATH}")
        else:
            has_rows = False
            for fname in csv_files:
                with open(os.path.join(config.IC_DATA_PATH, fname)) as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    has_rows = True
                    break
            if not has_rows:
                problems.append(f"IC data folder contains CSV files but none have any scenario rows: {config.IC_DATA_PATH}")

    if problems:
        raise FileNotFoundError(
            "Data setup is incomplete, refusing to start the sweep:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


def main():
    validate_openai_key()
    validate_setup()
    os.makedirs(helpers.LOGS_DIR, exist_ok=True)
    helpers.init_result_csvs()

    stale_logs = glob.glob(os.path.join(helpers.LOGS_DIR, "*.eval"))
    if stale_logs:
        print(f"Removing {len(stale_logs)} leftover .eval log(s) from a previous run")
        for path in stale_logs:
            os.remove(path)

    checkpoints = load_checkpoints()
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {config.CHECKPOINTS_PATH} after loading")
    print(f"Checkpoints to evaluate: {checkpoints}")

    needs_math = any(not helpers.is_math_complete(c) for c in checkpoints)
    math_problems = load_math_problems() if needs_math else None
    if needs_math:
        print(f"Sampled {len(math_problems)} MATH problems (seed={config.MATH_SAMPLE_SEED})")
    else:
        print("MATH already complete for every checkpoint, skipping dataset load")

    sweep_start = time.time()
    completed = []

    for checkpoint in checkpoints:
        if helpers.is_ea_complete(checkpoint) and helpers.is_ic_complete(checkpoint) and helpers.is_math_complete(checkpoint):
            print(f"\n{checkpoint} already complete, skipping")
            continue
        run_checkpoint(checkpoint, math_problems)
        completed.append(checkpoint)

    total_hours = (time.time() - sweep_start) / 3600
    print(f"\nSweep finished in {total_hours:.1f}h. Newly completed: {len(completed)}")


if __name__ == "__main__":
    main()
