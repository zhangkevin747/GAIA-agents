"""
Quick test: run GAIA questions with all tools, track timing, token costs, and correctness.
Usage:
  python test_run.py --start 0 --count 10          # first 10 from API (20 questions)
  python test_run.py --source dataset --count 165   # all 165 from HuggingFace dataset
  python test_run.py --source dataset --level 1     # only Level 1 questions
"""

import os
import json
import time
import re
import sys
import signal
import argparse
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Token tracking via litellm callbacks
import litellm

token_log = {"prompt_tokens": 0, "completion_tokens": 0}

def _track_tokens(kwargs, completion_response, start_time, end_time):
    usage = getattr(completion_response, "usage", None)
    if usage:
        token_log["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
        token_log["completion_tokens"] += getattr(usage, "completion_tokens", 0)

litellm.success_callback = [_track_tokens]

from agents import Agent
from tool import get_tools
from model import get_model
from smolagents.memory import ActionStep

# All tool function names the agent can call inside code blocks.
# CodeAgent wraps everything in python_interpreter, so we parse these from code.
TOOL_NAMES = {
    "web_search", "wikipedia_search", "visit_webpage",
    "vision_tool", "youtube_frames_to_images", "ask_youtube_video",
    "read_text_file", "file_from_url", "transcribe_youtube",
    "audio_to_text", "extract_text_via_ocr", "summarize_csv_data",
    "summarize_excel_data", "final_answer",
}
_TOOL_CALL_RE = re.compile(r'\b(' + '|'.join(TOOL_NAMES) + r')\s*\(')

# --- Config ---
API_URL = "https://agents-course-unit4-scoring.hf.space"
MODEL_ID = "openai/gpt-4o-mini"

# GPT-4o-mini pricing (per 1M tokens)
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60


def load_gaia_dataset():
    """Load full GAIA validation dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset(
        "gaia-benchmark/GAIA", "2023_all", split="validation",
    )
    return ds


def load_ground_truth():
    """Load GAIA validation ground truth answers, keyed by task_id."""
    ds = load_gaia_dataset()
    return {row["task_id"]: row["Final answer"] for row in ds}


def load_questions_from_dataset(level=None):
    """Load questions from HuggingFace dataset (all 165, not just API's 20).

    Returns list of dicts matching the API format: task_id, question, Level, file_name.
    """
    ds = load_gaia_dataset()
    questions = []
    for row in ds:
        if level is not None and row["Level"] != level:
            continue
        questions.append({
            "task_id": row["task_id"],
            "question": row["Question"],
            "Level": row["Level"],
            "file_name": row.get("file_name", "") or "",
        })
    return questions


def normalize_answer(text):
    """Normalize an answer string for comparison.

    - Strip outer whitespace, quotes, markdown bold
    - Normalize comma-separated lists: 'a,b, c' -> 'a, b, c'
    - Case-insensitive
    """
    text = text.strip().strip("'\"").strip()
    # Remove markdown bold markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Normalize comma spacing: any comma followed by optional spaces -> ", "
    text = re.sub(r'\s*,\s*', ', ', text)
    return text.lower()


def check_answer(submitted, expected):
    """GAIA exact-match scoring with normalization."""
    return normalize_answer(submitted) == normalize_answer(expected)


def fetch_questions():
    resp = requests.get(f"{API_URL}/questions", timeout=15)
    resp.raise_for_status()
    return resp.json()


def extract_steps(agent):
    """Extract structured step info from the agent's memory after a run."""
    steps = []
    for step in agent.agent.memory.steps:
        if not isinstance(step, ActionStep):
            continue
        tools_in_code = []
        if step.code_action:
            tools_in_code = list(dict.fromkeys(_TOOL_CALL_RE.findall(step.code_action)))
        info = {
            "step": step.step_number,
            "code": step.code_action,
            "tools_called": tools_in_code,
            "observations": step.observations,
            "is_final": step.is_final_answer,
        }
        steps.append(info)
    return steps


def print_step_detail(steps):
    """Pretty-print the step-by-step tool usage."""
    for s in steps:
        prefix = "[FINAL] " if s["is_final"] else ""
        print(f"\n  {prefix}Step {s['step']}:")

        if s["tools_called"]:
            print(f"    Tools: {', '.join(s['tools_called'])}")

        if s["code"]:
            code_lines = s["code"].strip().split("\n")
            code_preview = "\n    ".join(code_lines[:10])
            if len(code_lines) > 10:
                code_preview += f"\n    ... ({len(code_lines) - 10} more lines)"
            print(f"    Code:\n    {code_preview}")

        if s["observations"]:
            obs = s["observations"].strip()
            if len(obs) > 500:
                obs = obs[:500] + f"... [{len(obs)} chars total]"
            print(f"    Observation: {obs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index into questions list")
    parser.add_argument("--count", type=int, default=5, help="Number of questions to run")
    parser.add_argument("--source", choices=["api", "dataset"], default="api",
                        help="Question source: 'api' (20 questions) or 'dataset' (all 165)")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None,
                        help="Filter by GAIA difficulty level (dataset source only)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-question timeout in seconds (default 300 = 5 min)")
    args = parser.parse_args()

    # Load questions from chosen source
    if args.source == "dataset":
        print("Loading questions from HuggingFace GAIA dataset...")
        questions = load_questions_from_dataset(level=args.level)
        level_str = f" (Level {args.level})" if args.level else " (all levels)"
        print(f"Loaded {len(questions)} questions{level_str}.")
    else:
        questions = fetch_questions()
        print(f"Fetched {len(questions)} questions from API.")

    subset = questions[args.start : args.start + args.count]
    n_questions = len(subset)

    print(f"\n{'='*70}")
    print(f"  GAIA Test Run — {n_questions} questions [{args.start}:{args.start + n_questions}]")
    print(f"  Source: {args.source} | Model: {MODEL_ID}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    # Load ground truth from HuggingFace dataset
    print("Loading GAIA ground truth...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground truth answers.\n")

    model = get_model("LiteLLMModel", MODEL_ID)
    tools = get_tools()
    agent = Agent(model=model, tools=tools)

    # Prepare incremental output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/test_run_{timestamp}.jsonl"
    os.makedirs("results", exist_ok=True)

    results = []
    question_timeout = args.timeout

    # Timeout handler using SIGALRM
    class QuestionTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise QuestionTimeout(f"Question exceeded {question_timeout}s timeout")

    for i, q in enumerate(subset):
        task_id = q["task_id"]
        question = q["question"] if "question" in q else q.get("Question", "")
        file_name = q.get("file_name", "")
        expected = ground_truth.get(task_id, None)

        print(f"\n{'='*70}")
        print(f"  Question {i+1}/{n_questions}  (Level {q.get('Level', '?')})")
        print(f"{'='*70}")
        print(f"  Task ID:  {task_id}")
        print(f"  Question: {question}")
        if file_name:
            print(f"  File:     {file_name}")
        if expected is not None:
            print(f"  Expected: {expected}")
        print(f"{'-'*70}")

        # Reset per-question token counter
        q_prompt_before = token_log["prompt_tokens"]
        q_comp_before = token_log["completion_tokens"]

        start = time.time()
        try:
            # Set per-question timeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(question_timeout)

            answer = agent.answer_question(question, file_name if file_name else None)

            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)
        except QuestionTimeout:
            answer = f"TIMEOUT: exceeded {question_timeout}s"
            signal.alarm(0)
            print(f"\n  *** TIMEOUT after {question_timeout}s ***")
        except Exception as e:
            answer = f"ERROR: {e}"
            signal.alarm(0)
        elapsed = time.time() - start

        q_prompt = token_log["prompt_tokens"] - q_prompt_before
        q_comp = token_log["completion_tokens"] - q_comp_before
        q_cost = (q_prompt * INPUT_COST_PER_M + q_comp * OUTPUT_COST_PER_M) / 1_000_000

        # Extract and display step details
        steps = extract_steps(agent)
        print(f"\n  --- Agent steps ({len(steps)} total) ---")
        print_step_detail(steps)

        # Score
        correct = check_answer(answer, expected) if expected is not None else None
        score_str = "CORRECT" if correct else ("WRONG" if correct is False else "N/A")

        print(f"\n{'-'*70}")
        print(f"  FINAL ANSWER: {answer}")
        if expected is not None:
            print(f"  EXPECTED:     {expected}")
            print(f"  SCORE:        {score_str}")
        print(f"  Time: {elapsed:.1f}s | Tokens: {q_prompt} in + {q_comp} out | Cost: ${q_cost:.4f}")
        sys.stdout.flush()

        result = {
            "task_id": task_id,
            "question": question,
            "level": q.get("Level", ""),
            "file_name": file_name,
            "answer": answer,
            "expected": expected,
            "correct": correct,
            "time_seconds": round(elapsed, 1),
            "prompt_tokens": q_prompt,
            "completion_tokens": q_comp,
            "cost_usd": round(q_cost, 6),
            "num_steps": len(steps),
            "tools_used": list(dict.fromkeys(
                t for s in steps for t in s["tools_called"] if t != "final_answer"
            )),
        }
        results.append(result)

        # Save incrementally after each question
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Summary
    total_time = sum(r["time_seconds"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_comp = sum(r["completion_tokens"] for r in results)
    total_cost = sum(r["cost_usd"] for r in results)
    scored = [r for r in results if r["correct"] is not None]
    n_correct = sum(1 for r in scored if r["correct"])

    print(f"\n\n{'='*70}")
    print(f"  SUMMARY ({n_questions} questions)")
    print(f"{'='*70}")
    if scored:
        print(f"  Accuracy:       {n_correct}/{len(scored)} ({100*n_correct/len(scored):.0f}%)")
    print(f"  Total time:       {total_time:.1f}s ({total_time/n_questions:.1f}s avg)")
    print(f"  Total tokens:     {total_prompt} prompt + {total_comp} completion")
    print(f"  Total cost:       ${total_cost:.4f} (${total_cost/n_questions:.4f} avg/question)")
    print(f"  Projected 165 Qs: ${total_cost/n_questions * 165:.2f}")
    print()

    # Per-question summary table
    print(f"  {'#':<3} {'Cor':<4} {'Answer':<30} {'Expected':<20} {'Tools Used':<30} {'Time':>6}")
    print(f"  {'-'*3} {'-'*4} {'-'*30} {'-'*20} {'-'*30} {'-'*6}")
    for i, r in enumerate(results):
        mark = "Y" if r["correct"] else ("N" if r["correct"] is False else "?")
        ans = r["answer"][:27] + "..." if len(r["answer"]) > 30 else r["answer"]
        exp = (r["expected"] or "")[:17] + "..." if len(r["expected"] or "") > 20 else (r["expected"] or "")
        tools_str = ", ".join(r["tools_used"]) if r["tools_used"] else "(none)"
        if len(tools_str) > 30:
            tools_str = tools_str[:27] + "..."
        print(f"  {i+1:<3} {mark:<4} {ans:<30} {exp:<20} {tools_str:<30} {r['time_seconds']:>5.1f}s")

    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
