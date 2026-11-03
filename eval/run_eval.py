"""Evaluate the agent on the 15-question set (FR-9 evidence).

Run from the project folder, with your .env filled in:

    python eval/run_eval.py --scores-only     # no LLM calls: shows retrieval scores
    python eval/run_eval.py                   # full run, writes eval/results.md

--pause (default 15 seconds) waits between questions so Groq's per-minute
token limit is not hit.
"""

import argparse
import json
import re
import sys
import unicodedata
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docqa.agents.graph import run_agent  # noqa: E402
from docqa.config import load_settings  # noqa: E402
from docqa.errors import DocQAError  # noqa: E402
from docqa.llm import build_llm  # noqa: E402
from docqa.rag import NOT_FOUND_MESSAGE  # noqa: E402
from docqa.workspace import Workspace  # noqa: E402

SAMPLE_FILES = ["company_policy.txt", "handbook.pdf", "sales.csv", "staff.xlsx"]


def check_answer(item: dict, answer: str) -> bool:
    """A fact question passes when every expected string appears in the answer
    (lowercased, commas and [n] citation markers removed). A refusal question
    passes only when the system declined."""
    if item["expect_refusal"]:
        return answer == NOT_FOUND_MESSAGE or "doesn't seem to be about" in answer
    # NFKC turns models' no-break spaces into plain spaces; then collapse runs of whitespace.
    text = unicodedata.normalize("NFKC", answer)
    cleaned = re.sub(r"\s+", " ", re.sub(r"\[\d+\]", "", text).replace(",", "")).lower()
    squeezed = cleaned.replace(" ", "")  # "9 am" and "9am" are the same fact
    return all(
        expected.lower() in cleaned or expected.lower().replace(" ", "") in squeezed
        for expected in item["expect_all"]
    )


def build_workspace(samples_dir: Path, embedding_function=None) -> Workspace:
    workspace = Workspace(embedding_function=embedding_function)
    for name in SAMPLE_FILES:
        workspace.add_file(name, (Path(samples_dir) / name).read_bytes())
    return workspace


def score_questions(items: list[dict], workspace: Workspace) -> list[tuple[str, float]]:
    """Best retrieval score per question, with no LLM calls."""
    scores = []
    for item in items:
        hits = workspace.store.query(item["question"], k=1)
        scores.append((item["question"], hits[0].score if hits else 0.0))
    return scores


def run_eval(items, workspace, llm, pause: float = 15.0, sleep=time.sleep) -> list[dict]:
    results = []
    for index, item in enumerate(items):
        if index:
            sleep(pause)
        expected = "refusal" if item["expect_refusal"] else ", ".join(item["expect_all"])
        try:
            outcome = run_agent(item["question"], workspace, llm)
            answer, calls = outcome.answer, outcome.llm_calls
        except DocQAError as error:
            answer, calls = f"ERROR: {error}", 0
        results.append(
            {
                "question": item["question"],
                "expected": expected,
                "answer": answer,
                "llm_calls": calls,
                "passed": not answer.startswith("ERROR:") and check_answer(item, answer),
            }
        )
    return results


def format_results(results: list[dict]) -> str:
    passed = sum(1 for r in results if r["passed"])
    calls = sum(r["llm_calls"] for r in results)
    lines = [
        "# Evaluation results",
        "",
        f"Passed {passed}/{len(results)} ({100 * passed // max(len(results), 1)}%). "
        f"Total LLM calls: {calls}.",
        "",
        "| # | Question | Expected | Result | Answer |",
        "| --- | --- | --- | --- | --- |",
    ]
    for number, r in enumerate(results, start=1):
        answer = r["answer"].replace("|", "/").replace("\n", " ")
        verdict = "PASS" if r["passed"] else "FAIL"
        lines.append(f"| {number} | {r['question']} | {r['expected']} | {verdict} | {answer} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument("--pause", type=float, default=15.0)
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--out", type=Path, default=Path("eval/results.md"))
    args = parser.parse_args(argv)

    if not all((args.samples / name).exists() for name in SAMPLE_FILES):
        from samples.make_samples import make_samples

        make_samples(args.samples)
    items = json.loads((Path(__file__).parent / "eval_set.json").read_text())
    workspace = build_workspace(args.samples)

    if args.scores_only:
        scores = score_questions(items, workspace)
        for item, (question, score) in zip(items, scores):
            kind = "refusal " if item["expect_refusal"] else "answer  "
            print(f"{score:5.2f}  {kind} {question}")
        answerable = [s for it, (_, s) in zip(items, scores) if not it["expect_refusal"]]
        refusals = [s for it, (_, s) in zip(items, scores) if it["expect_refusal"]]
        print(f"\nlowest answerable score: {min(answerable):.2f}")
        print(f"highest refusal score:   {max(refusals):.2f}")
        print("Pick MIN_SCORE between those two if they do not overlap; otherwise keep 0.25.")
        return

    results = run_eval(items, workspace, build_llm(load_settings()), pause=args.pause)
    report = format_results(results)
    args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
