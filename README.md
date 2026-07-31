# RLVR EA/IC/MATH Pipeline

Tracks Evaluation Awareness (EA), Instrumental Convergence (IC), and MATH
capability across RLVR training checkpoints of OLMo 3 7B Think, to study
whether verbalized evaluation awareness precedes instrumental convergence
behavior as RLVR training progresses.

## Repository layout

This README describes the whole project. The repository is organized as:

- `pipeline/` - code: config, helpers, the sweep entry point, tasks/
- `data/` — EA transcripts, IC scenarios, the checkpoint list
- `results/` — the three result CSVs from the completed sweep
- `notebooks/` — exploration, data retrieval, and analysis notebooks

`pipeline/` and `data/` together are everything needed to run the sweep.
`results/` and `notebooks/` are the project's output and analysis, not
inputs the pipeline reads.

## Sources

- EA classification and dataset: Needham et al., "LLMs Often Know They Are
  Being Evaluated."
- EA verbalized eval awareness rubric: Aranguri and Bloom, "Verbalized Eval
  Awareness Inflates Measured Safety," a joint publication between Goodfire
  and UK AISI.
- IC benchmark and adversarial system prompt: He et al., "Evaluating the
  Paperclip Maximizer" (InstrumentalEval).
- MATH problems: allenai/RLVR-MATH, part of the Tulu 3 release.
- Checkpoint selection and original data retrieval: see
  [`notebooks/01_data_retrieval_and_exploration.ipynb`](https://github.com/samcastro22/RLVR-ea-ic-emergence/tree/main/notebooks).
  OLMo 3 7B Think has 55 released checkpoints; this project uses every
  fourth one, 14 total, for complete temporal coverage at a tractable
  compute cost. The data already committed under `data/` is the output of
  that notebook; it does not need to be rerun to use the pipeline.

## Getting only what you need

A full `git clone` downloads everything, including `notebooks/` and
`results/`, and is the simplest option. To download only the code and
the data required to actually run the sweep:

```bash
git clone --no-checkout https://github.com/samcastro22/RLVR-ea-ic-emergence.git
cd RLVR-ea-ic-emergence
git sparse-checkout init --cone
git sparse-checkout set pipeline data
git checkout main
```


## Getting the EA dataset

`data/ea_dataset.json` is not included in this repository. Fetch it
directly from the original source:

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
import zipfile, os

path = snapshot_download(repo_id='jjpn2/eval_awareness', repo_type='dataset')
zip_path = [f for f in os.listdir(path) if f.endswith('.zip')][0]
with zipfile.ZipFile(os.path.join(path, zip_path)) as z:
    z.extractall('data', pwd=b'isthisreallythepassword')
"
```

See `notebooks/01_data_retrieval_and_exploration.ipynb` for the full
original exploration this was drawn from.


## Setup (Colab or any machine with Python and a GPU)

Install the base requirements, then vLLM separately in the exact order
below. This order matters: vLLM 0.20.0 and later default to a CUDA 13.0
build, which can silently conflict with an environment already using an
earlier CUDA version.

```bash
cd pipeline
pip install -r requirements.txt
pip install --force-reinstall --no-deps vllm==0.19.0
pip install vllm==0.19.0
```

Copy `.env.example` to `.env` and fill in real values, or export the same
variables directly. `RLVR_BASE_PATH` should point at the repository root,
one level up from `pipeline/`, since that is where `data/` and `results/`
live:

```bash
export RLVR_BASE_PATH=..
export OPENAI_API_KEY=...
export HF_TOKEN=...   # only needed if the model repository requires auth
```

## Running

Check the setup first on a small number of samples:

```bash
python smoke_test.py
```

Then run the full sweep:

```bash
python run_sweep.py
```

Resume-safe: rerunning after any interruption picks up exactly where it
left off. A checkpoint's measurement is only run if it is missing from
that measurement's own result CSV, so a partial failure in one
measurement does not force the other two to be redone.

## Running outside Colab (Docker)

Colab does not support running Docker containers itself; this path is
for a local workstation or a cloud GPU instance with the NVIDIA Container
Toolkit installed instead.

```bash
cd pipeline
docker build -t rlvr-pipeline .
docker run --gpus all \
  -e OPENAI_API_KEY=... \
  -e HF_TOKEN=... \
  -e RLVR_BASE_PATH=/workspace \
  -v /path/to/RLVR-ea-ic-emergence:/workspace \
  rlvr-pipeline
```

The Dockerfile uses the same install sequence as the manual setup above,
built on `nvidia/cuda:12.8.0-runtime-ubuntu22.04`. The base requirements
install and the vLLM force-reinstall step were both confirmed to succeed
in testing; the final full dependency resolution was not run to
completion end to end in a live container due to a disk space limit in
the environment used to check this Dockerfile, separate from the
pipeline itself. The install sequence is identical to the one already
confirmed working directly in Colab, so this is not expected to behave
differently, but has not been verified with a full `docker build` on a
real machine.

## Optional: archiving raw logs

By default, raw per-sample model completions are discarded after each
checkpoint's aggregate metrics are computed. Setting
`RLVR_LOG_ARCHIVE_DIR` to a folder path causes each log to be copied
there before the local temporary copy is deleted. A full 14 checkpoint
sweep produces roughly 250MB of logs in total, so this is cheap to
enable. The original 14 checkpoint results in `results/` were produced
without this enabled.

## Known limitations

- The EA and IC judge model (GPT-4.1-mini) differs from the model used to
  validate the original verbalized eval awareness rubric (GPT-5-mini).
  This was a practical default rather than a matched choice. This
  substitution is not expected to weaken the results: an independent study comparing the two on
  a comparable verbalization-detection task found GPT-4.1-mini achieved
  higher agreement with human annotators (90% vs 84% accuracy)
- The subject model runs at temperature 0.6 rather than temperature 0,
  matching OLMo's own model card rather than the deterministic settings
  used in the IC and EA source papers. This is a deliberate choice: it
  reflects how the model is actually meant to run, and the source
  rubric's own methodology (sampling many rollouts per question)
  depends on real sampling variance existing rather than eliminating
  it. Confidence intervals in the result CSVs reflect this variance.
- MATH problems are presented without the few-shot examples that
  RLVR-MATH's own format includes, since this pipeline does not use chat
  template formatting. See the comment in `pipeline/tasks/math_task.py`
  for the specific failure mode this avoids.
- OLMo 3 7B Think has 55 released training checkpoints. This project
  evaluates 14 of them, every fourth one, for temporal coverage at a
  tractable compute cost. It reproduces this specific study rather than
  serving as a tool for evaluating an arbitrary set of checkpoints
  without editing `data/checkpoints.csv`.
