# RLVR EA/IC/MATH Pipeline

Tracks Evaluation Awareness (EA), Instrumental Convergence (IC), and MATH
capability across RLVR training checkpoints of OLMo 3 7B Think, to study
whether verbalized evaluation awareness precedes instrumental convergence
behavior as RLVR training progresses.

## Sources

- EA classification and dataset: Needham et al., "LLMs Often Know They Are
  Being Evaluated."
- EA verbalized eval awareness rubric: Aranguri and Bloom, "Verbalized Eval
  Awareness Inflates Measured Safety," a joint publication between Goodfire
  and UK AISI.
- IC benchmark and adversarial system prompt: He et al., "Evaluating the
  Paperclip Maximizer" (InstrumentalEval).
- MATH problems: allenai/RLVR-MATH, part of the Tulu 3 release.

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths and constants shared across the pipeline |
| `helpers.py` | Statistics, log processing, and resume logic |
| `tasks/ea_task.py` | EA measurement: classify eval versus deployment, detect verbalized awareness |
| `tasks/ic_task.py` | IC measurement: adversarial system prompt plus judge |
| `tasks/math_task.py` | MATH measurement: symbolic scoring with math-verify |
| `run_sweep.py` | Entry point, runs all three measurements across every checkpoint |
| `smoke_test.py` | Runs a small number of samples per measurement to check the setup before committing to the full sweep |

## Setup

Install the base requirements, then vLLM separately in the exact order
described in `requirements.txt`. This order matters: vLLM 0.20.0 and
later default to a CUDA 13.0 build, which can silently conflict with an
environment already using an earlier CUDA version.

```bash
pip install -r requirements.txt
pip install --force-reinstall --no-deps vllm==0.19.0
pip install vllm==0.19.0
```

Copy `.env.example` to `.env` and fill in real values, or export the same
variables directly:

```bash
export RLVR_BASE_PATH=/path/to/RLVR_Project
export OPENAI_API_KEY=...
export HF_TOKEN=...   # only needed if the model repository requires auth
```

Expected data layout under `$RLVR_BASE_PATH/data/`:

- `ea_dataset.json`: EA transcripts
- `instrumental_eval/`: six category CSVs from InstrumentalEval
- `checkpoints.csv`: one column of checkpoint names, for example `step_0025`

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

## Known limitations

- The EA and IC judge model (GPT-4.1-mini) differs from the model used to
  validate the original verbalized eval awareness rubric (GPT-5-mini).
  This was a practical default rather than a matched choice.
- The subject model runs at temperature 0.6 rather than temperature 0,
  matching OLMo's own model card rather than the deterministic settings
  used in the IC and EA source papers. Confidence intervals in the
  result CSVs reflect the resulting sampling variance.
- MATH problems are presented without the few-shot examples that
  RLVR-MATH's own format includes, since this pipeline does not use chat
  template formatting. See the comment in `tasks/math_task.py` for the specific
  failure mode this avoids.
- Raw per-sample model completions are discarded after each checkpoint's
  aggregate metrics are computed, unless `RLVR_LOG_ARCHIVE_DIR` is set
  (see `.env.example`). This is off by default since it was not enabled
  during the original 14-checkpoint run this pipeline produced, so those
  results do not have retained raw logs; enabling it going forward costs
  little (roughly 250MB total for a full sweep) and allows retrospective
  calibration analysis or pulling qualitative examples that this
  pipeline does not otherwise support.
