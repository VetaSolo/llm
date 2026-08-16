"""CLI for the LLM text pipeline. No web server — run from the terminal."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from llm_client import LLMError, RetryableAPIError, complete, scripted
from pipeline import Outcome, run_pipeline
from schemas import Category, ClassifyResult, ExtractResult, Sentiment, StructuredParseError, parse_model
from utils import MAX_ANSWER_CHARS, clip_text

EXAMPLES_DIR = Path(__file__).parent / "examples"
OUTPUTS_DIR = Path(__file__).parent / "outputs"

DEMO_SCENARIOS = [
    "01_laptop_overheating.txt",
    "02_app_feedback.txt",
    "03_rag_question.txt",
    "04_billing_complaint.txt",
    "06_sales_trial.txt",
]

EXPECTED_CATEGORY = {
    "01_laptop_overheating.txt": Category.support,
    "02_app_feedback.txt": Category.feedback,
    "03_rag_question.txt": Category.general_question,
    "04_billing_complaint.txt": Category.complaint,
    "05_messy_note.txt": Category.general_question,
    "06_sales_trial.txt": Category.sales,
    "07_login_support.txt": Category.support,
    "08_delivery_complaint.txt": Category.complaint,
    "09_onboarding_feedback.txt": Category.feedback,
    "10_sales_pricing.txt": Category.sales,
}

DEMO_ERRORS = [
    ("broken_json", "{summary: nope, this is not json"),
    ("missing_fields", '{"summary": "Short summary.", "category": "support"}'),
    (
        "wrong_types",
        json.dumps(
            {
                "summary": "Short summary.",
                "category": "support",
                "sentiment": "neutral",
                "intent": "ask_help",
                "key_points": "one, two, three",
            }
        ),
    ),
    (
        "bad_enum",
        json.dumps(
            {
                "summary": "Short summary.",
                "category": "hello",
                "sentiment": "neutral",
                "intent": "ask_help",
                "key_points": ["a", "b", "c"],
            }
        ),
    ),
]


def setup_logging() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(OUTPUTS_DIR / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.WARNING, handlers=[stream, file_handler], force=True)
    logging.getLogger("pipeline").setLevel(logging.INFO)
    logging.getLogger("llm").setLevel(logging.INFO)


def print_outcome(outcome: Outcome) -> None:
    print("=" * 72)
    print(f"INPUT: {outcome.source}")
    print("-" * 72)
    print(outcome.input_text.strip())
    print("-" * 72)

    for i, event in enumerate(outcome.steps, start=1):
        print(f"STEP {i} {event.name:11} [{event.status}]  uses {event.uses}")
        print(f"         {event.detail}")

    if not outcome.ok or outcome.result is None:
        print()
        print("STATUS: failed (item skipped, batch continues)")
        print(f"ERROR: {outcome.error}")
        return

    result = outcome.result
    if result.degraded:
        used = ", ".join(result.fallback_used) or "—"
        print(f"STATUS: degraded  fallback={used}")
    else:
        print("STATUS: ok")

    expected = EXPECTED_CATEGORY.get(outcome.source)
    if expected is not None:
        match = "MATCH" if result.category is expected else f"MISMATCH (expected {expected.value})"
        print(f"CHECK   category {result.category.value} [{match}]")
    if result.sentiment is Sentiment.negative:
        print("GUARD   negative sentiment")
    if result.truncated:
        print(f"GUARD   answer clipped to {MAX_ANSWER_CHARS} chars")
    print()
    print("SUMMARY")
    print(result.summary)
    print()
    print("KEY POINTS")
    for i, point in enumerate(result.key_points, start=1):
        print(f"  {i}. {point}")
    print()
    suffix = ""
    if result.revised:
        suffix = "  (revised after self-check)"
    elif result.degraded:
        suffix = "  (fallback)"
    print("FINAL ANSWER" + suffix)
    print(result.final_answer)
    print()


def run_demo_errors() -> int:
    print("Schema guards (no API calls)")
    failed_as_expected = 0
    for name, raw in DEMO_ERRORS:
        print("=" * 72)
        print(f"CASE: {name}")
        try:
            parse_model(raw, ClassifyResult)
            print("UNEXPECTED: this payload was accepted")
        except StructuredParseError as exc:
            failed_as_expected += 1
            print("STATUS: rejected by schema")
            print(f"ERROR: {exc}")
    print("=" * 72)
    print(f"{failed_as_expected}/{len(DEMO_ERRORS)} invalid payloads were caught")
    return 0 if failed_as_expected == len(DEMO_ERRORS) else 1


def run_demo_failures() -> int:
    """Three required negative scenarios + length guard. No live API."""
    os.environ["LLM_MAX_RETRIES"] = "3"
    os.environ["LLM_RETRY_BASE"] = "0.05"
    report: list[dict] = []
    passed = 0

    print("=" * 72)
    print("FAIL 1: model returned text instead of JSON")
    try:
        parse_model("Sure! Here's a helpful summary without any JSON.", ExtractResult)
        detail = "UNCAUGHT — parser accepted plain text"
        ok = False
    except StructuredParseError as exc:
        detail = str(exc)
        ok = True
    print("STATUS:", "caught" if ok else "FAILED")
    print("ERROR:", detail)
    report.append({"case": "plain_text", "ok": ok, "detail": detail})
    passed += int(ok)

    print("=" * 72)
    print("FAIL 2: JSON parsed but keys are missing")
    try:
        parse_model('{"summary": "Laptop is hot."}', ExtractResult)
        detail = "UNCAUGHT — missing key_points accepted"
        ok = False
    except StructuredParseError as exc:
        detail = str(exc)
        ok = True
    print("STATUS:", "caught" if ok else "FAILED")
    print("ERROR:", detail)
    report.append({"case": "missing_keys", "ok": ok, "detail": detail})
    passed += int(ok)

    print("=" * 72)
    print("FAIL 3: API temporarily unavailable (3 retryable errors)")
    try:
        with scripted(
            [
                RetryableAPIError("simulated TLS/connection drop"),
                RetryableAPIError("simulated TLS/connection drop"),
                RetryableAPIError("simulated TLS/connection drop"),
            ]
        ):
            complete("ping")
        detail = "UNCAUGHT — complete() succeeded"
        ok = False
    except LLMError as exc:
        detail = str(exc)
        ok = "after 3 attempts" in str(exc)
    print("STATUS:", "caught after retries" if ok else "FAILED")
    print("ERROR:", detail)
    report.append({"case": "api_unavailable", "ok": ok, "detail": detail})
    passed += int(ok)

    print("=" * 72)
    print("FAIL 3b: retry then success")
    with scripted(
        [
            RetryableAPIError("one blip"),
            '{"summary": "ok", "key_points": ["a", "b", "c"]}',
        ]
    ):
        recovered = complete("ping")
    ok = "ok" in recovered
    print("STATUS:", "recovered on attempt 2" if ok else "FAILED")
    report.append({"case": "retry_then_success", "ok": ok, "detail": recovered[:80]})
    passed += int(ok)

    print("=" * 72)
    print("GUARD: too-long answer is clipped")
    clipped, truncated = clip_text("word " * 500, MAX_ANSWER_CHARS)
    ok = truncated and len(clipped) <= MAX_ANSWER_CHARS
    print("STATUS:", "clipped" if ok else "FAILED", f"len={len(clipped)}")
    report.append({"case": "too_long", "ok": ok, "detail": f"len={len(clipped)}"})
    passed += int(ok)

    print("=" * 72)
    print("USABLE: full pipeline with broken JSON still returns a reply")
    sample = "С карты списали оплату дважды 15 марта. Верните деньги."
    with scripted(["this is not json", "still not json"]):
        outcome = run_pipeline(sample, source="demo_plain_text")
    print_outcome(outcome)
    ok = bool(outcome.ok and outcome.result and outcome.degraded)
    report.append(
        {
            "case": "pipeline_fallback",
            "ok": ok,
            "detail": {
                "ok": outcome.ok,
                "degraded": outcome.degraded,
                "fallback_used": outcome.fallback_used,
                "final_answer": outcome.result.final_answer if outcome.result else None,
            },
        }
    )
    passed += int(ok)

    out = save_json(report, "failure_demo.json")
    print("=" * 72)
    print(f"Failure demos: {passed}/{len(report)} passed")
    print(f"Saved {out}")
    return 0 if passed == len(report) else 1


def run_demo_routing() -> int:
    path = EXAMPLES_DIR / "04_billing_complaint.txt"
    text = path.read_text(encoding="utf-8")
    print("Routing demo: same text, different routes")
    records = []
    for category in (Category.complaint, Category.sales, Category.support):
        outcome = run_pipeline(text, source=path.name, force_category=category)
        print_outcome(outcome)
        records.append(
            {
                "forced_route": category.value,
                "ok": outcome.ok,
                "degraded": outcome.degraded,
                "final_answer": outcome.result.final_answer if outcome.result else None,
                "error": outcome.error,
            }
        )
    out = save_json(records, "routing_demo.json")
    print(f"Saved routing demo to {out}")
    return 0 if all(item["ok"] for item in records) else 1


def load_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.text:
        return [("cli", args.text)]
    if args.file:
        path = Path(args.file)
        return [(path.name, path.read_text(encoding="utf-8"))]
    if args.demo:
        files = [EXAMPLES_DIR / name for name in DEMO_SCENARIOS]
    else:
        files = sorted(EXAMPLES_DIR.glob("*.txt"))
    missing = [path for path in files if not path.exists()]
    if missing:
        raise SystemExit(f"Example file not found: {missing[0]}")
    if not files:
        raise SystemExit(f"No example files found in {EXAMPLES_DIR}")
    return [(path.name, path.read_text(encoding="utf-8")) for path in files]


def list_examples() -> int:
    print("Demo scenarios (python main.py --demo):")
    for name in DEMO_SCENARIOS:
        expected = EXPECTED_CATEGORY.get(name)
        label = expected.value if expected else "?"
        print(f"  {name:32}  → {label}")
    print()
    print("All example inputs:")
    for path in sorted(EXAMPLES_DIR.glob("*.txt")):
        print(f"  {path.name}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM text pipeline: extract → classify → route → answer → self-check.",
        epilog=(
            "Examples:\n"
            "  python main.py --demo\n"
            "  python main.py --text \"The app is great but I need dark mode\"\n"
            "  python main.py --file examples/04_billing_complaint.txt\n"
            "  python main.py --demo-routing\n"
            "  python main.py --demo-failures\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--text", help="Process a single text from the command line")
    parser.add_argument("--file", help="Process a single text file")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the 5 showcase scenarios (support, feedback, question, complaint, sales)",
    )
    parser.add_argument("--list", action="store_true", help="List example input files")
    parser.add_argument(
        "--force-category",
        choices=[item.value for item in Category],
        help="Ignore the classifier and generate with this route",
    )
    parser.add_argument("--demo-errors", action="store_true", help="Schema validation demos (no API)")
    parser.add_argument("--demo-routing", action="store_true", help="Same complaint, three different routes")
    parser.add_argument(
        "--demo-failures",
        action="store_true",
        help="Negative scenarios: plain text, missing keys, API outage (no live API)",
    )
    return parser.parse_args()


def save_json(payload: list[dict] | dict, filename: str) -> Path:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / filename
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.demo_errors:
        raise SystemExit(run_demo_errors())
    if args.demo_failures:
        raise SystemExit(run_demo_failures())
    if args.demo_routing:
        raise SystemExit(run_demo_routing())
    if args.list:
        raise SystemExit(list_examples())

    force = Category(args.force_category) if args.force_category else None
    items = load_inputs(args)
    records: list[dict] = []
    failures = 0
    degraded = 0
    matches = 0
    expected_total = 0
    revised = 0

    for name, text in items:
        try:
            outcome = run_pipeline(text, source=name, force_category=force)
        except Exception as exc:
            logging.getLogger("pipeline").exception("item %s crashed", name)
            outcome = Outcome(name, text, False, None, str(exc), force, [], True, [])
        print_outcome(outcome)
        record: dict = {
            "source": name,
            "ok": outcome.ok,
            "degraded": outcome.degraded,
            "input": text,
        }
        if outcome.ok and outcome.result is not None:
            record.update(outcome.result.model_dump(mode="json"))
            revised += int(outcome.result.revised)
            degraded += int(outcome.result.degraded)
            expected = EXPECTED_CATEGORY.get(name)
            if expected is not None:
                expected_total += 1
                hit = outcome.result.category is expected
                record["expected_category"] = expected.value
                record["category_match"] = hit
                matches += int(hit)
        else:
            failures += 1
            record["error"] = outcome.error
            record["steps"] = [event.model_dump() for event in outcome.steps]
        records.append(record)

    out_path = save_json(records, "results.json")
    print("=" * 72)
    print(f"Saved {len(records)} result(s) to {out_path}")
    print(
        f"Accepted: {len(records) - failures}   Failed: {failures}   "
        f"Degraded: {degraded}   Revised: {revised}"
    )
    if expected_total:
        print(f"Category accuracy: {matches}/{expected_total}")
    print(f"Log file: {OUTPUTS_DIR / 'pipeline.log'}")
    if failures == len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
