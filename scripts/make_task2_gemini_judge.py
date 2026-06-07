"""Judge Task 2 explanations with Google Gemini models.

This script samples Task 2 submission rows, builds strict JSON judge prompts, and
optionally calls the Gemini API. It writes outputs compatible with
`scripts/aggregate_task2_judge_scores.py`.

Important:
- Do NOT hard-code your API key in this file.
- Set GEMINI_API_KEY or GOOGLE_API_KEY in your shell before running API mode.

PowerShell example:
    $env:GEMINI_API_KEY="..."

Prompt-only mode:
    python scripts/make_task2_gemini_judge.py \
      --task2-jsonl hf_submission/cata_multitask_final/submission_task2.jsonl \
      --sample-size 300 \
      --seed 42 \
      --output-dir paper_assets/task2_judge_gemini \
      --model gemini-3.1-flash-lite \
      --prompts-only

API mode:
    python scripts/make_task2_gemini_judge.py \
      --task2-jsonl hf_submission/cata_multitask_final/submission_task2.jsonl \
      --sample-size 300 \
      --seed 42 \
      --output-dir paper_assets/task2_judge_gemini \
      --model gemini-3.1-flash-lite

Then aggregate:
    python scripts/aggregate_task2_judge_scores.py \
      --judge-results paper_assets/task2_judge_gemini/task2_gemini_judge_results.jsonl \
      --output-dir paper_assets/tables \
      --judge-name Gemini-3.1-Flash-Lite
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CRITERIA = ["correctness", "faithfulness", "relevance", "clarity", "completeness"]

SYSTEM_PROMPT = """You are an expert medical visual question answering evaluator.
You evaluate explanation quality for GI endoscopy VQA outputs.
Return strict JSON only. Do not include markdown or extra text.
""".strip()

RUBRIC = """Score each criterion from 1 to 5:
1 = poor, 2 = weak, 3 = acceptable, 4 = good, 5 = excellent.

Criteria:
- correctness: the answer and explanation are consistent with the question and available reference information.
- faithfulness: the explanation supports the predicted answer without inventing unsupported claims.
- relevance: the explanation focuses on visual/question evidence that matters for the answer.
- clarity: the explanation is readable, concise, and clinically understandable.
- completeness: the explanation covers the key evidence needed to justify the answer.

Return JSON with exactly these keys:
{
  "correctness": <integer 1-5>,
  "faithfulness": <integer 1-5>,
  "relevance": <integer 1-5>,
  "clarity": <integer 1-5>,
  "completeness": <integer 1-5>,
  "rationale": "short explanation"
}
""".strip()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["__line_number"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def normalize_text(value: Any, max_chars: int = 1800) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def visual_items_summary(row: Dict[str, Any]) -> str:
    items = row.get("visual_explanation") or []
    if not isinstance(items, list):
        return "No visual_explanation list provided."

    parts: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = normalize_text(item.get("type", ""), 80)
        data = normalize_text(item.get("data", ""), 180)
        desc = normalize_text(item.get("description", ""), 320)
        parts.append(f"- type={item_type}; path={data}; description={desc}")
    return "\n".join(parts) if parts else "No visual explanation entries."


def build_prompt(row: Dict[str, Any]) -> str:
    reference = row.get("ground_truth") or row.get("reference") or row.get("gt") or ""
    return f"""{RUBRIC}

Evaluate this Task 2 explanation.

val_id: {normalize_text(row.get('val_id', row.get('__line_number')))}
img_id: {normalize_text(row.get('img_id', ''))}
question: {normalize_text(row.get('question', ''))}
reference_answer_if_available: {normalize_text(reference)}
predicted_answer: {normalize_text(row.get('answer', ''))}
confidence_score: {normalize_text(row.get('confidence_score', ''))}
textual_explanation: {normalize_text(row.get('textual_explanation', ''), 2400)}
visual_explanation_summary:
{visual_items_summary(row)}

Remember: return strict JSON only.
""".strip()


def sample_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[i] for i in indices]


def write_prompts(rows: List[Dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "task2_gemini_judge_prompts.jsonl"
    csv_path = output_dir / "task2_gemini_judge_prompts.csv"

    with prompts_path.open("w", encoding="utf-8") as f_jsonl, csv_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["sample_id", "val_id", "img_id", "question", "answer", "prompt"],
        )
        writer.writeheader()
        for sample_id, row in enumerate(rows):
            record = {
                "sample_id": sample_id,
                "val_id": str(row.get("val_id", row.get("__line_number", sample_id))),
                "img_id": str(row.get("img_id", "")),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "prompt": build_prompt(row),
            }
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow(record)

    print(f"Saved prompts: {prompts_path}")
    print(f"Saved prompts CSV: {csv_path}")
    return prompts_path


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_score_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in CRITERIA:
        value = obj.get(key)
        try:
            score = int(round(float(value)))
        except Exception:
            score = 1
        out[key] = max(1, min(5, score))
    out["rationale"] = str(obj.get("rationale", "")).strip()
    return out


def load_done_sample_ids(results_path: Path) -> set[int]:
    done: set[int] = set()
    if not results_path.exists():
        return done
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if str(row.get("status", "ok")) == "ok":
                    done.add(int(row["sample_id"]))
            except Exception:
                continue
    return done


def parse_retry_after_seconds(message: str) -> Optional[float]:
    compact = re.search(
        r"(?:try again in|retry in|retry after)\s+(?:(?P<hours>[0-9]+)h)?(?:(?P<minutes>[0-9]+)m)?(?:(?P<seconds>[0-9]+(?:\.[0-9]+)?)s)?",
        message,
        flags=re.IGNORECASE,
    )
    if compact:
        hours = float(compact.group("hours") or 0)
        minutes = float(compact.group("minutes") or 0)
        seconds = float(compact.group("seconds") or 0)
        total = hours * 3600 + minutes * 60 + seconds
        if total > 0:
            return total

    patterns = [
        r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)s",
        r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)\s*seconds",
        r"retry after\s+([0-9]+(?:\.[0-9]+)?)\s*seconds",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def make_gemini_client() -> Any:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set. Set it in your shell; do not put it in code.")

    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("Missing dependency: install with `pip install google-genai`.") from exc

    return genai.Client(api_key=api_key)


def call_gemini(client: Any, model: str, prompt: str, max_tokens: int) -> str:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )

    text = getattr(response, "text", None)
    if text:
        return str(text)

    # Fallback for SDK response variants.
    try:
        return response.candidates[0].content.parts[0].text or ""
    except Exception:
        return str(response)


def run_gemini_api(
    prompts_path: Path,
    output_dir: Path,
    model: str,
    max_tokens: int,
    retry: int,
    sleep_seconds: float,
    resume: bool,
    max_wait_seconds: float,
) -> Path:
    client = make_gemini_client()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "task2_gemini_judge_results.jsonl"
    done = load_done_sample_ids(results_path) if resume else set()
    if done:
        print(f"Resume mode: skipping {len(done)} already judged samples.")

    mode = "a" if resume else "w"
    with prompts_path.open("r", encoding="utf-8") as f_in, results_path.open(mode, encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            sample_id = int(record["sample_id"])
            if sample_id in done:
                continue

            last_error = ""
            response_text = ""
            status = "error"
            parsed = {key: 1 for key in CRITERIA}
            parsed["rationale"] = "api_error"

            for attempt in range(1, retry + 1):
                try:
                    response_text = call_gemini(
                        client=client,
                        model=model,
                        prompt=record["prompt"],
                        max_tokens=max_tokens,
                    )
                    parsed = validate_score_record(extract_json_object(response_text))
                    status = "ok"
                    last_error = ""
                    break
                except Exception as exc:
                    last_error = str(exc)
                    retry_after = parse_retry_after_seconds(last_error)
                    if retry_after is not None and retry_after > max_wait_seconds:
                        print(
                            f"sample={sample_id} hit a long rate limit wait ({retry_after:.1f}s). "
                            "Stopping cleanly without writing a failed row. Re-run later to resume."
                        )
                        return results_path
                    wait = retry_after + 1.0 if retry_after is not None else max(sleep_seconds, 1.0) * attempt
                    print(f"sample={sample_id} attempt={attempt}/{retry} error={last_error}; sleep={wait:.1f}s")
                    time.sleep(wait)

            if status != "ok":
                print(f"sample={sample_id} failed after {retry} attempts. Stopping without writing a failed row.")
                return results_path

            out = {
                **{k: record.get(k) for k in ("sample_id", "val_id", "img_id", "question", "answer")},
                **parsed,
                "status": status,
                "error": last_error,
                "judge_model": model,
                "raw_response": response_text,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            f_out.flush()
            print(f"sample={sample_id} status={status} scores=" + ",".join(str(out[k]) for k in CRITERIA))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print(f"Saved judge results: {results_path}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run Gemini judge for Task 2 explanations.")
    parser.add_argument("--task2-jsonl", required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="paper_assets/task2_judge_gemini")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--retry", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-wait-seconds", type=float, default=300.0)
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument("--prompts-only", action="store_true", default=False)
    args = parser.parse_args()

    task2_path = Path(args.task2_jsonl)
    if not task2_path.is_absolute():
        task2_path = PROJECT_ROOT / task2_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    rows = load_jsonl(task2_path)
    sampled = sample_rows(rows, args.sample_size, args.seed)
    print(f"Loaded {len(rows)} Task 2 rows; sampled {len(sampled)} rows.")
    prompts_path = write_prompts(sampled, output_dir)

    if args.prompts_only:
        print("Prompt-only mode complete. Remove --prompts-only to call the Gemini API.")
        return

    run_gemini_api(
        prompts_path=prompts_path,
        output_dir=output_dir,
        model=args.model,
        max_tokens=args.max_tokens,
        retry=args.retry,
        sleep_seconds=args.sleep_seconds,
        resume=not args.no_resume,
        max_wait_seconds=args.max_wait_seconds,
    )


if __name__ == "__main__":
    main()
