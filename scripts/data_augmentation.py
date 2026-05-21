import argparse
import json
import shutil
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm


SAFE_AUGMENTATIONS_PER_SAMPLE = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_safe_endoscopy_transform(image_size=None):
    """
    Albumentations pipeline an toàn cho ảnh nội soi.

    Ràng buộc chính:
    - Shift/Scale <= 5%
    - Rotate <= 10 độ
    - Hue <= 0.02
    - Brightness/Contrast <= 0.10

    Không dùng flip mặc định vì ảnh VQA có câu hỏi vị trí trái/phải/trên/dưới.
    Không dùng crop mạnh vì có thể làm mất lesion/instrument/text.
    """
    transforms = [
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-10, 10),
            shear=(-2, 2),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.85,
        ),
        A.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.08,
            hue=0.02,
            p=0.65,
        ),
        A.GaussianBlur(
            blur_limit=(3, 3),
            sigma_limit=(0.1, 0.6),
            p=0.12,
        ),
        A.GaussNoise(
            std_range=(0.01, 0.03),
            mean_range=(0.0, 0.0),
            per_channel=False,
            p=0.12,
        ),
    ]

    if image_size is not None:
        transforms.append(
            A.Resize(
                height=image_size[0],
                width=image_size[1],
                interpolation=cv2.INTER_AREA,
                p=1.0,
            )
        )

    return A.Compose(transforms)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL lỗi ở dòng {line_number}: {exc}") from exc


def resolve_image_path(image_ref, images_dir):
    """
    Ưu tiên ảnh trong images_dir theo basename để tránh phụ thuộc absolute path cũ.
    Nếu không có, fallback sang đường dẫn trong JSONL.
    """
    images_dir = Path(images_dir)
    image_ref_path = Path(image_ref)

    candidate = images_dir / image_ref_path.name
    if candidate.exists():
        return candidate

    if image_ref_path.exists():
        return image_ref_path

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_ref_path.stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Không tìm thấy ảnh '{image_ref}'. Đã kiểm tra images_dir='{images_dir}'."
    )


def make_augmented_name(original_path, aug_idx, output_ext=None):
    output_ext = output_ext or original_path.suffix.lower()
    if output_ext not in IMAGE_EXTENSIONS:
        output_ext = ".jpg"
    return f"{original_path.stem}_aug{aug_idx:02d}{output_ext}"


def update_record_image_path(record, new_image_path):
    updated = dict(record)
    updated["images"] = [str(Path(new_image_path))]
    updated["augmentation"] = {
        "source_image": Path(record["images"][0]).name if record.get("images") else None,
        "policy": "safe_endoscopy_affine_colorjitter",
    }
    return updated


def augment_jsonl_dataset(
    original_jsonl,
    images_dir,
    augmented_jsonl,
    aug_images_dir,
    num_augments=SAFE_AUGMENTATIONS_PER_SAMPLE,
    image_size=None,
    output_ext=".jpg",
    copy_original=False,
):
    original_jsonl = Path(original_jsonl)
    images_dir = Path(images_dir)
    augmented_jsonl = Path(augmented_jsonl)
    aug_images_dir = Path(aug_images_dir)

    if not original_jsonl.exists():
        raise FileNotFoundError(f"Không tìm thấy original_jsonl: {original_jsonl}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy images_dir: {images_dir}")

    augmented_jsonl.parent.mkdir(parents=True, exist_ok=True)
    aug_images_dir.mkdir(parents=True, exist_ok=True)

    transform = build_safe_endoscopy_transform(image_size=image_size)
    records = list(read_jsonl(original_jsonl))

    written_rows = 0
    skipped_rows = 0

    with augmented_jsonl.open("w", encoding="utf-8", newline="\n") as writer:
        for _, record in tqdm(records, desc="Safe endoscopy augmentation"):
            try:
                if "images" not in record or not record["images"]:
                    raise ValueError("Record không có field images hợp lệ.")

                image_path = resolve_image_path(record["images"][0], images_dir)
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError(f"cv2 không đọc được ảnh: {image_path}")

                if copy_original:
                    original_out = aug_images_dir / image_path.name
                    if not original_out.exists():
                        shutil.copy2(image_path, original_out)
                    original_record = update_record_image_path(record, original_out)
                    original_record["augmentation"]["type"] = "original_copy"
                    writer.write(json.dumps(original_record, ensure_ascii=False) + "\n")
                    written_rows += 1

                for aug_idx in range(1, num_augments + 1):
                    augmented = transform(image=image_bgr)["image"]
                    out_name = make_augmented_name(image_path, aug_idx, output_ext=output_ext)
                    out_path = aug_images_dir / out_name

                    success = cv2.imwrite(str(out_path), augmented)
                    if not success:
                        raise IOError(f"Không thể ghi ảnh augment: {out_path}")

                    aug_record = update_record_image_path(record, out_path)
                    aug_record["augmentation"].update(
                        {
                            "type": "offline_albumentations",
                            "index": aug_idx,
                            "num_augments_per_sample": num_augments,
                            "constraints": {
                                "scale": "<=5%",
                                "shift": "<=5%",
                                "rotate": "<=10deg",
                                "hue": "<=0.02",
                                "brightness": "<=0.10",
                                "contrast": "<=0.10",
                            },
                        }
                    )
                    writer.write(json.dumps(aug_record, ensure_ascii=False) + "\n")
                    written_rows += 1

            except Exception as exc:
                skipped_rows += 1
                print(f"⚠️ Bỏ qua record do lỗi: {exc}")

    print("\n======================================")
    print("✅ OFFLINE DATA AUGMENTATION HOÀN TẤT")
    print("======================================")
    print(f"JSONL gốc:             {original_jsonl}")
    print(f"Thư mục ảnh gốc:       {images_dir}")
    print(f"JSONL augmented:       {augmented_jsonl}")
    print(f"Thư mục ảnh augmented: {aug_images_dir}")
    print(f"Số record gốc:         {len(records)}")
    print(f"Số record ghi ra:      {written_rows}")
    print(f"Số record lỗi/bỏ qua:  {skipped_rows}")
    print("======================================\n")


def parse_image_size(value):
    if value is None:
        return None
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("image_size phải có dạng HxW, ví dụ 224x224")
    h, w = value.lower().split("x", 1)
    return int(h), int(w)


def main():
    parser = argparse.ArgumentParser(
        description="Offline safe data augmentation cho Kvasir-VQA-x1 bằng Albumentations."
    )
    parser.add_argument("--original-jsonl", required=True, help="Đường dẫn JSONL gốc.")
    parser.add_argument("--images-dir", required=True, help="Thư mục chứa ảnh gốc.")
    parser.add_argument("--augmented-jsonl", required=True, help="Đường dẫn JSONL đầu ra.")
    parser.add_argument("--aug-images-dir", required=True, help="Thư mục ảnh augmented đầu ra.")
    parser.add_argument("--num-augments", type=int, default=SAFE_AUGMENTATIONS_PER_SAMPLE)
    parser.add_argument("--image-size", type=parse_image_size, default=None, help="Tuỳ chọn resize HxW, ví dụ 224x224.")
    parser.add_argument("--output-ext", default=".jpg", choices=sorted(IMAGE_EXTENSIONS))
    parser.add_argument("--copy-original", action="store_true", help="Ghi thêm bản copy ảnh gốc vào output.")
    args = parser.parse_args()

    augment_jsonl_dataset(
        original_jsonl=args.original_jsonl,
        images_dir=args.images_dir,
        augmented_jsonl=args.augmented_jsonl,
        aug_images_dir=args.aug_images_dir,
        num_augments=args.num_augments,
        image_size=args.image_size,
        output_ext=args.output_ext,
        copy_original=args.copy_original,
    )


if __name__ == "__main__":
    main()
