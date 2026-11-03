import json
from pathlib import Path

import pytest
from conftest import FakeLLM

from docqa.ingestion import parse_file
from docqa.rag import NOT_FOUND_MESSAGE
from eval.run_eval import (
    build_workspace,
    check_answer,
    format_results,
    run_eval,
    score_questions,
)
from samples.make_samples import make_samples

ROOT = Path(__file__).parent.parent
EVAL_SET = json.loads((ROOT / "eval" / "eval_set.json").read_text())


@pytest.fixture
def samples_dir(tmp_path):
    return make_samples(tmp_path)


def test_eval_set_is_well_formed():
    assert len(EVAL_SET) == 15
    assert sum(1 for item in EVAL_SET if item["expect_refusal"]) >= 3
    for item in EVAL_SET:
        assert item["question"].strip()
        if not item["expect_refusal"]:
            assert item["expect_all"]


def test_samples_contain_the_facts_the_eval_set_expects(samples_dir):
    def parse(name):
        return parse_file(name, (samples_dir / name).read_bytes())

    assert "30 days" in parse("company_policy.txt").sections[0].text
    handbook = parse("handbook.pdf")
    assert "90 days" in handbook.sections[0].text
    assert "24 days" in handbook.sections[1].text and "14 days" in handbook.sections[1].text
    sales = parse("sales.csv").tables["csv"]
    assert len(sales) == 8 and sales["amount"].sum() == 5250
    assert sales.loc[sales["region"] == "West", "amount"].sum() == 2400
    staff = parse("staff.xlsx").tables
    engineering = staff["Staff"][staff["Staff"]["department"] == "Engineering"]
    assert engineering["salary"].sum() == 170000
    assert (staff["Staff"]["department"] == "Sales").sum() == 2
    assert staff["Offices"].set_index("city").loc["Pune", "seats"] == 120


def test_check_answer_rules():
    fact = {"question": "q", "expect_all": ["5250"], "expect_refusal": False}
    refusal = {"question": "q", "expect_all": [], "expect_refusal": True}
    assert check_answer(fact, "The total is 5,250 [1].")
    assert not check_answer(fact, "The total is 5,000 [1].")
    assert check_answer(refusal, NOT_FOUND_MESSAGE)
    assert check_answer(refusal, "This question doesn't seem to be about your uploaded documents.")
    assert not check_answer(refusal, "It is 5250.")


def test_run_eval_scores_each_answer(samples_dir, offline_embedding):
    workspace = build_workspace(samples_dir, embedding_function=offline_embedding)
    items = [
        {"question": "What is the refund window?", "expect_all": ["30 days"], "expect_refusal": False},
        {"question": "What is the capital of France?", "expect_all": [], "expect_refusal": True},
    ]
    llm = FakeLLM(
        ['{"route": "lookup"}', "Refunds take 30 days [1].", '{"supported": true}',
         '{"route": "unrelated"}']
    )
    pauses = []
    results = run_eval(items, workspace, llm, pause=7, sleep=pauses.append)
    assert [r["passed"] for r in results] == [True, True]
    assert [r["llm_calls"] for r in results] == [3, 1]
    assert pauses == [7]  # pause between questions, not after the last one
    report = format_results(results)
    assert "2/2" in report and "What is the refund window?" in report


def test_score_questions_reports_best_score_per_question(samples_dir, offline_embedding):
    workspace = build_workspace(samples_dir, embedding_function=offline_embedding)
    scores = score_questions(EVAL_SET[:2], workspace)
    assert [question for question, _ in scores] == [item["question"] for item in EVAL_SET[:2]]
    assert all(0.0 <= score <= 1.0 for _, score in scores)
