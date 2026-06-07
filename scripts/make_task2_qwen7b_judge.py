"""Judge Task 2 explanations with local Qwen Instruct models.

This is a generic Qwen judge script for Qwen 7B/14B/32B/72B-class models.
It samples Task 2 JSONL rows, builds strict JSON judge prompts, and optionally
runs a local Transformers model. Outputs are compatible with
`scripts/aggregate_task2_judge_scores.py`.

Recommended H100 examples:

    # One H100 80GB: use a quantized 72B checkpoint when possible.
    python scripts/make_task2_qwen7b_judge.py \
      --task2-jsonl hf_submission/cata_multitask_final/submission_task2.jsonl \
      --sample-size 300 \
      --seed 42 \
      --output-dir paper_assets/task2_judge_qwen72b_awq \
      --run-local \
      --model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
      --torch-dtype bfloat16 \
      --attn-implementation flash_attention_2 \
      --max-new-tokens 384

    # One H100 80GB: full checkpoint with bitsandbytes 4-bit.
    python scripts/make_task2_qwen7b_judge.py \
      --task2-jsonl hf_submission/cata_multitask_final/submission_task2.jsonl \
      --sample-size 300 \
      --seed 42 \
      --output-dir paper_assets/task2_judge_qwen72b_4bit \
      --run-local \
      --model-name Qwen/Qwen2.5-72B-Instruct \
      --load-in-4bit \
      --torch-dtype bfloat16 \
      --attn-implementation flash_attention_2 \
      --max-new-tokens 384

    # Multi-H100: full BF16 if memory is sufficient.
    python scripts/make_task2_qwen7b_judge.py \
      --task2-jsonl hf_submission/cata_multitask_final/submission_task2.jsonl \
      --sample-size 300 \
      --seed 42 \
      --output-dir paper_assets/task2_judge_qwen72b_bf16 \
      --run-local \
      --model-name Qwen/Qwen2.5-72B-Instruct \
      --torch-dtype bfloat16 \
      --max-new-tokens 384

Then aggregate:

    python scripts/aggregate_task2_judge_scores.py \
      --judge-results paper_assets/task2_judge_qwen72b_awq/task2_qwen_judge_results.jsonl \
      --output-dir paper_assets/tables \
      --judge-name Qwen2.5-72B-Instruct-AWQ
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
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


def write_prompts(rows: List[Dict[str, Any]], output_dir: Path, output_prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{output_prefix}_prompts.jsonl"
    csv_path = output_dir / f"{output_prefix}_prompts.csv"

    with jsonl_path.open("w", encoding="utf-8") as f_jsonl, csv_path.open("w", encoding="utf-8", newline="") as f_csv:
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

    print(f"Saved prompts: {jsonl_path}")
    print(f"Saved prompts CSV: {csv_path}")
    return jsonl_path


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


def dtype_from_name(torch_module: Any, dtype_name: str) -> Any:
    dtype_map = {
        "auto": "auto",
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    return dtype_map.get(dtype_name, "auto")


def parse_max_memory(max_memory: str) -> Optional[Dict[str, str]]:
    if not max_memory:
        return None
    parsed: Dict[str, str] = {}
    for item in max_memory.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --max-memory item: {item!r}. Expected e.g. cuda:0=76GiB,cpu=256GiB")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed or None


def get_input_device(model: Any, torch_module: Any) -> Any:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for device in model.hf_device_map.values():
            device_text = str(device)
            if device_text.startswith("cuda"):
                return torch_module.device(device_text)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def build_model_kwargs(args: argparse.Namespace, torch_module: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": True,
    }

    if args.local_files_only:
        kwargs["local_files_only"] = True

    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    max_memory = parse_max_memory(args.max_memory)
    if max_memory:
        kwargs["max_memory"] = max_memory

    dtype = dtype_from_name(torch_module, args.torch_dtype)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = "auto"

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one of --load-in-4bit or --load-in-8bit.")

    if args.load_in_4bit or args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError("Missing bitsandbytes support. Install `bitsandbytes` and a recent `transformers`.") from exc

        compute_dtype = torch_module.bfloat16 if args.torch_dtype in {"auto", "bfloat16"} else dtype
        if args.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    return kwargs


def run_local_qwen(
    prompts_path: Path,
    output_dir: Path,
    output_prefix: str,
    args: argparse.Namespace,
) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = build_model_kwargs(args, torch)
    print(f"Loading judge model: {args.model_name}")
    print(f"Model kwargs: {json.dumps({k: str(v) for k, v in model_kwargs.items() if k != 'quantization_config'}, ensure_ascii=False)}")
    if "quantization_config" in model_kwargs:
        print(f"Quantization: {'4-bit' if args.load_in_4bit else '8-bit'} bitsandbytes")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    input_device = get_input_device(model, torch)
    print(f"Input device: {input_device}")
    if hasattr(model, "hf_device_map"):
        print(f"Device map: {model.hf_device_map}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{output_prefix}_results.jsonl"
    done = load_done_sample_ids(results_path) if not args.no_resume else set()
    if done:
        print(f"Resume mode: skipping {len(done)} already judged samples.")

    mode = "a" if not args.no_resume else "w"
    with prompts_path.open("r", encoding="utf-8") as f_in, results_path.open(mode, encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            sample_id = int(record["sample_id"])
            if sample_id in done:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record["prompt"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(input_device)

            response = ""
            status = "parse_error"
            error = ""
            parsed = {key: 1 for key in CRITERIA}
            parsed["rationale"] = "parse_error"

            for attempt in range(1, args.parse_retries + 1):
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = generated[:, inputs.input_ids.shape[1] :]
                response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
                try:
                    parsed = validate_score_record(extract_json_object(response))
                    status = "ok"
                    error = ""
                    break
                except Exception as exc:
                    error = str(exc)
                    print(f"sample={sample_id} parse_attempt={attempt}/{args.parse_retries} error={error}")

            out = {
                **{k: record.get(k) for k in ("sample_id", "val_id", "img_id", "question", "answer")},
                **parsed,
                "status": status,
                "error": error,
                "judge_model": args.model_name,
                "raw_response": response,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            f_out.flush()
            print(f"sample={sample_id} status={status} scores=" + ",".join(str(out[k]) for k in CRITERIA))

    print(f"Saved judge results: {results_path}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run a local Qwen judge for Task 2 explanations.")
    parser.add_argument("--task2-jsonl", required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="paper_assets/task2_judge_qwen")
    parser.add_argument("--output-prefix", default="task2_qwen_judge")
    parser.add_argument("--run-local", action="store_true", default=False)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory", default="", help="Optional, e.g. cuda:0=76GiB,cpu=256GiB")
    parser.add_argument("--attn-implementation", default="", help="Optional: flash_attention_2, sdpa, or eager")
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--load-in-8bit", action="store_true", default=False)
    parser.add_argument("--bnb-4bit-quant-type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--no-resume", action="store_true", default=False)
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
    prompts_path = write_prompts(sampled, output_dir, args.output_prefix)

    if args.run_local:
        run_local_qwen(
            prompts_path=prompts_path,
            output_dir=output_dir,
            output_prefix=args.output_prefix,
            args=args,
        )
    else:
        print("Prompt-only mode complete. Re-run with --run-local to judge with a local Qwen model.")


if __name__ == "__main__":
    main()
