import argparse
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_JOBS = ("original", "aug1", "aug3", "aug10")


JOB_CONFIGS = {
    "original": {
        "jsonl": "data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl",
        "images_dir": "data/raw/Kvasir-VQA-x1/images",
        "output_dir": "data/processed/structural_features/train_original",
        "manifest_csv": "data/processed/structural_features/train_original_manifest.csv",
    },
    "aug1": {
        "jsonl": "data/processed/Kvasir-VQA-x1/Kvasir-VQA-x1-train-aug1.jsonl",
        "images_dir": "data/processed/Kvasir-VQA-x1/images_aug10",
        "output_dir": "data/processed/structural_features/train_aug1",
        "manifest_csv": "data/processed/structural_features/train_aug1_manifest.csv",
    },
    "aug3": {
        "jsonl": "data/processed/Kvasir-VQA-x1/Kvasir-VQA-x1-train-aug3.jsonl",
        "images_dir": "data/processed/Kvasir-VQA-x1/images_aug10",
        "output_dir": "data/processed/structural_features/train_aug3",
        "manifest_csv": "data/processed/structural_features/train_aug3_manifest.csv",
    },
    "aug10": {
        "jsonl": "data/processed/Kvasir-VQA-x1/Kvasir-VQA-x1-train-aug10.jsonl",
        "images_dir": "data/processed/Kvasir-VQA-x1/images_aug10",
        "output_dir": "data/processed/structural_features/train_aug10",
        "manifest_csv": "data/processed/structural_features/train_aug10_manifest.csv",
    },
}


def parse_jobs(value):
    if not value:
        return list(DEFAULT_JOBS)

    jobs = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in JOB_CONFIGS:
            valid = ", ".join(JOB_CONFIGS.keys())
            raise argparse.ArgumentTypeError(f"Job không hợp lệ: {item}. Hợp lệ: {valid}")
        jobs.append(item)
    return jobs


def ensure_inputs(job_name, config):
    missing = []
    for key in ("jsonl", "images_dir"):
        path = Path(config[key])
        if not path.exists():
            missing.append(f"{key}={path}")
    if missing:
        raise FileNotFoundError(f"Job {job_name} thiếu input: {', '.join(missing)}")


def run_job(job_name, config, no_compress=True, overwrite=False, max_samples=None):
    ensure_inputs(job_name, config)

    cmd = [
        sys.executable,
        "-m",
        "scripts.precompute_structural_features",
        "--jsonl",
        config["jsonl"],
        "--images-dir",
        config["images_dir"],
        "--output-dir",
        config["output_dir"],
        "--manifest-csv",
        config["manifest_csv"],
    ]

    if no_compress:
        cmd.append("--no-compress")
    if overwrite:
        cmd.append("--overwrite")
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    print("\n======================================")
    print(f"🚀 RUN STRUCTURAL PRECOMPUTE: {job_name}")
    print("======================================")
    print("Command:")
    print(" ".join(cmd))
    print("--------------------------------------")

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"Job {job_name} failed với exit code {result.returncode}")

    print("--------------------------------------")
    print(f"✅ Job {job_name} done in {elapsed / 60:.2f} min")
    print("======================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Chạy lần lượt precompute structural features cho original/aug1/aug3/aug10."
    )
    parser.add_argument(
        "--jobs",
        type=parse_jobs,
        default=list(DEFAULT_JOBS),
        help="Danh sách job, ví dụ: original,aug1,aug3,aug10",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Bật nén .npz. Mặc định tắt nén để chạy nhanh hơn.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè cache đã tồn tại.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn mẫu mỗi job để test nhanh.",
    )
    args = parser.parse_args()

    print("\n======================================")
    print("🧬 STRUCTURAL PRECOMPUTE BATCH RUNNER")
    print("======================================")
    print(f"Jobs:       {', '.join(args.jobs)}")
    print(f"Compress:   {args.compress}")
    print(f"Overwrite:  {args.overwrite}")
    print(f"Max samples: {args.max_samples}")
    print("======================================")

    for job_name in args.jobs:
        run_job(
            job_name=job_name,
            config=JOB_CONFIGS[job_name],
            no_compress=not args.compress,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
        )

    print("\n✅ Tất cả jobs hoàn tất.")


if __name__ == "__main__":
    main()
