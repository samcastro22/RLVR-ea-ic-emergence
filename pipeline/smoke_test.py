"""
Runs a small number of samples through each of EA, IC, and MATH on a
single checkpoint, to check that the environment, model, and API keys
are all working before committing to the full multi-hour sweep.

Also reports scored versus total counts for each measurement, since a
model or environment problem can cause many samples to error out
silently while inspect eval still exits with a success status.

Usage:
    python smoke_test.py [checkpoint_name]

If no checkpoint is given, uses the first one listed in checkpoints.csv.
"""

import json
import os
import sys
import time

from inspect_ai.log import read_eval_log

import config
import helpers
import run_sweep


def probe(label, command, expected_count):
    print(f"\nRunning {label} probe...")
    start = time.time()
    helpers.run_inspect_eval(command)
    elapsed = time.time() - start

    log_path = helpers.get_latest_log()
    log = read_eval_log(log_path)
    total = len(log.samples or [])
    scored = sum(1 for s in (log.samples or []) if s.scores)

    print(f"{label}: {scored}/{total} scored in {elapsed:.1f}s")
    if total < expected_count:
        print(f"  warning: expected around {expected_count} samples, got {total}")
    if scored < total:
        print(f"  warning: {total - scored} sample(s) did not score, check the log for errors")

    rate = elapsed / total if total else float("nan")
    os.remove(log_path)
    return rate


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    if checkpoint is None:
        checkpoints = run_sweep.load_checkpoints()
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {config.CHECKPOINTS_PATH}")
        checkpoint = checkpoints[0]

    print(f"Smoke testing against checkpoint: {checkpoint}")

    ea_rate = probe(
        "EA",
        f"inspect eval tasks/ea_task.py --model {config.MODEL_ID} -M revision={checkpoint} "
        f"--limit 5 -T ea_path=\"{config.EA_DATA_PATH}\"",
        expected_count=5,
    )

    ic_rate = probe(
        "IC",
        f"inspect eval tasks/ic_task.py --model {config.MODEL_ID} -M revision={checkpoint} "
        f"--limit 3 -T ic_folder=\"{config.IC_DATA_PATH}\"",
        expected_count=9,
    )

    math_problems = run_sweep.load_math_problems()[:5]
    math_tmp = "/tmp/smoke_test_math_problems.json"
    with open(math_tmp, "w") as f:
        json.dump(math_problems, f)
    math_rate = probe(
        "MATH",
        f"inspect eval tasks/math_task.py --model {config.MODEL_ID} -M revision={checkpoint} "
        f'-T problems_path="{math_tmp}"',
        expected_count=5,
    )
    os.remove(math_tmp)

    print(f"\nMeasured rates from this smoke test:")
    print(f"  EA:   {ea_rate:.1f}s/sample")
    print(f"  IC:   {ic_rate:.1f}s/generation")
    print(f"  MATH: {math_rate:.1f}s/problem")
    print("\nThese rates come from only 5 to 9 samples per measurement and can vary")
    print("substantially between runs depending on how much reasoning the model")
    print("does on any given sample. They are not extrapolated into a full-sweep")
    print("time estimate here, since a projection built on this little data would")
    print("look more precise and trustworthy than it actually is. Run a larger")
    print("batch, or watch the first real checkpoint of the full sweep complete,")
    print("for a more reliable sense of total duration.")


if __name__ == "__main__":
    main()
