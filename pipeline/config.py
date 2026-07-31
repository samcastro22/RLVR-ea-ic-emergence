"""
Shared configuration for the RLVR EA/IC/MATH evaluation pipeline.

Set RLVR_BASE_PATH to point at a directory containing a data/ subfolder
with ea_dataset.json, instrumental_eval/, and checkpoints.csv, and a
results/ subfolder where the three output CSVs will be written.
"""

import os

BASE_PATH = os.environ.get("RLVR_BASE_PATH", "./RLVR_Project")

DATA_PATH = f"{BASE_PATH}/data"
RESULTS_PATH = f"{BASE_PATH}/results"

EA_DATA_PATH = f"{DATA_PATH}/ea_dataset.json"
IC_DATA_PATH = f"{DATA_PATH}/instrumental_eval"
CHECKPOINTS_PATH = f"{DATA_PATH}/checkpoints.csv"

EA_RESULTS_PATH = f"{RESULTS_PATH}/ea_results.csv"
IC_RESULTS_PATH = f"{RESULTS_PATH}/ic_results.csv"
MATH_RESULTS_PATH = f"{RESULTS_PATH}/math_results.csv"

MODEL_ID = "vllm/allenai/Olmo-3-7B-Think"

# 200 problems sampled once with a fixed seed, reused for every checkpoint
# and every rerun, so the MATH comparison stays fair across checkpoints.
MATH_SAMPLE_SIZE = 200
MATH_SAMPLE_SEED = 42

# EA's rubric (see tasks/ea_task.py) is sourced from Aranguri and Bloom's
# verbalized eval awareness paper, originally validated with GPT-5-mini.
# IC's judge prompt is separately sourced from InstrumentalEval (He et al.)
# and was not validated against any specific judge model in that paper.
# Both tasks currently use GPT-4.1-mini as a practical default rather than
# a matched choice; see README for the comparison evidence behind this.

os.makedirs(RESULTS_PATH, exist_ok=True)
