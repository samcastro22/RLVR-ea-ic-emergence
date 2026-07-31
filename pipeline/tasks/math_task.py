"""
MATH Task: Capability control using allenai/RLVR-MATH dataset.
"""

import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy
from inspect_ai.solver import generate, TaskState

MATH_PROMPT_TEMPLATE = """Solve the following math problem. Show your reasoning, \
then provide your final answer.

Problem: {problem}

After working through the problem, state your final answer clearly."""

# --- Dataset builder ---

def build_math_samples(problems: list[dict]) -> list[Sample]:
    samples = []
    for i, problem in enumerate(problems):
        problem_text = ""
        if "messages" in problem:
            for msg in problem["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                problem_text = part.get("text", "")
                                break
                    else:
                        problem_text = str(content)
                    break
        elif "problem" in problem:
            problem_text = problem["problem"]

        # RLVR-MATH bundles several few-shot "Question:...Answer:..." examples
        # before the real question (designed for chat-template formatting,
        # which this pipeline doesn't use). Extract just the real question -
        # the last segment after the final "Question:" marker.
        if "Question:" in problem_text:
            problem_text = problem_text.split("Question:")[-1].strip()

        answer = problem.get("answer", problem.get("solution", problem.get("ground_truth", "")))
        if isinstance(answer, list):
            answer = str(answer[0]) if answer else ""

        samples.append(Sample(
            id=f"math_{i}",
            input=MATH_PROMPT_TEMPLATE.format(problem=problem_text[:2000]),
            target=str(answer),
            metadata={"problem_index": i}
        ))

    return samples

# --- Scorer ---

@scorer(metrics=[accuracy()])
def math_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion if state.output else ""

        if "<think>" in completion and "</think>" in completion:
            thinking = completion.split("<think>")[1].split("</think>")[0].strip()
            answer_text = completion.split("</think>")[1].strip()
        else:
            thinking = None
            answer_text = completion.strip()

        ground_truth = target.text.strip()
        correct = False
        verify_failed = False
        extracted_answer = None

        # math-verify scans the FULL text and finds \boxed{} itself: hand it
        # the whole post-</think> block, not a pre-truncated last line.
        try:
            from math_verify import verify, parse
            parsed_answer = parse(answer_text)
            parsed_gt = parse(ground_truth)
            if parsed_answer is not None and parsed_gt is not None:
                correct = verify(parsed_gt, parsed_answer)
                extracted_answer = str(parsed_answer)
            else:
                verify_failed = True
        except Exception:
            verify_failed = True

        if verify_failed:
            answer_lines = [l.strip() for l in answer_text.strip().split("\n") if l.strip()]
            extracted_answer = answer_lines[-1] if answer_lines else answer_text.strip()
            correct = extracted_answer.strip() == ground_truth.strip()

        return Score(
            value=1.0 if correct else 0.0,
            answer=(extracted_answer or "")[:200],
            explanation=thinking[:300] if thinking else "No thinking trace",
            metadata={
                "has_thinking": thinking is not None,
                "thinking_length": len(thinking) if thinking else 0,
                "ground_truth": ground_truth,
                "extracted_answer": (extracted_answer or "")[:200],
                "verify_failed": verify_failed,
            }
        )

    return score

# --- Task ---

@task
def math_task(problems_path: str) -> Task:
    with open(problems_path, "r") as f:
        problems = json.load(f)
    samples = build_math_samples(problems)

    return Task(
        dataset=samples,
        solver=[generate()],
        scorer=math_scorer(),
        config=GenerateConfig(
            temperature=0.6,
            top_p=0.95,
            max_tokens=32768,
        ),
        fail_on_error=False,
    )