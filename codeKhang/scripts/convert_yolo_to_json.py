"""
Script chuyển đổi YOLO dataset → JSON annotation format cho MultimodalDataset.

Chức năng:
1. Đọc YOLO labels (bbox format)
2. Gọi Gemini 2.5 Flash API sinh text prompt đa dạng (có cache + fallback)
3. Random assign text prompt cho từng ảnh dựa trên class labels
4. Xuất ra file JSON theo format MultimodalDataset

Chạy:
    python scripts/convert_yolo_to_json.py \
        --data-yaml datasets/Dataset.v12-yolov11/data.yaml \
        --output-dir data/ \
        --gemini-api-key YOUR_KEY
"""

import json
import os
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import yaml

from gemini_prompt_generator import GeminiPromptGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

SYMPTOM_TEMPLATES = {
    "Dom-la": [
        "chấm tròn li ti có quầng màu vàng bao quanh",
        "nhiều đốm tròn xuất hiện dày đặc trên lá già",
        "đốm tròn li ti viền vàng",
        "các chấm tròn li ti và quầng vàng bao xung quanh"
    ],
    
    "Ri-sat": [
        "đốm li ti nâu sẫm giống màu rỉ sắt",
        "vệt màu nâu sẫm như rỉ sét trên mặt lá",
        "chấm li ti nâu sẫm lan rộng được bao quanh bởi quầng vàng",
        "các đốm li ti nâu sẫm giống rỉ sắt"
    ],
    
    "Nhen-do": [
        "chấm nhỏ li ti màu trắng bạc",
        "chấm nhỏ li ti màu vàng nhạt rải rác",
        "mảng lá cứng thô chuyển sang màu nâu đồng",
        "bề mặt lá có chấm trắng bạc và ngả màu nâu đồng thô ráp"
    ],
    
    "Than-thu": [
        "vết chấm nhỏ màu nâu nhạt hoặc xám bạc",
        "đốm lớn bất thường, trung tâm khô bạc màu",
        "vết bệnh khô bạc màu với viền ngoài nâu đậm ",
        "lá bị cháy khô từng mảng lớn có viền sẫm"
    ]
}


# =============================================================================
# YOLO Label Parser
# =============================================================================

def parse_yolo_label(label_path: Path) -> List[Dict]:
    """
    Parse YOLO label file (bbox format only).
    
    Format: class_id x_center y_center width height (normalized 0-1)
    
    Args:
        label_path: Path to .txt label file.
        
    Returns:
        List of annotation dicts with bbox [x_center, y_center, w, h] normalized.
    """
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                logger.warning(f"Skipping malformed line in {label_path}: {line[:80]}")
                continue

            class_id = int(parts[0])
            x_center, y_center, w, h = [float(x) for x in parts[1:]]

            annotations.append({
                "bbox": [
                    round(x_center, 6),
                    round(y_center, 6),
                    round(w, 6),
                    round(h, 6),
                ],
                "class_id": class_id,
            })

    return annotations


# =============================================================================
# Text Prompt Assignment
# =============================================================================

def assign_text_prompt(
    class_ids: List[int],
    class_names: List[str],
    generator: GeminiPromptGenerator,
) -> str:
    # Nếu ảnh hoàn toàn không có nhãn nào (rỗng)
    if not class_ids:
        return ""

    unique_ids = sorted(list(set(class_ids)))
    unique_names = [class_names[cid] for cid in unique_ids if cid < len(class_names)]

    # Lọc bỏ nhãn "La-khoe" nếu có bệnh khác đi kèm
    disease_names = [n for n in unique_names if n != "La-khoe"]

    # 1. Nếu chỉ có La-khoe
    if len(disease_names) == 0:
        if "La-khoe" in unique_names:
            return generator.get_random_prompt("La-khoe")
        return ""

    # 2. Nếu có đúng 1 loại bệnh (Single-disease) -> Dùng cơ chế Queue cũ
    if len(disease_names) == 1:
        return generator.get_random_prompt(disease_names[0])

    # 3. Nếu đa bệnh (Multi-disease) -> Dùng Template nối chuỗi
    phrases = []
    for cls_name in disease_names:
        templates = SYMPTOM_TEMPLATES.get(cls_name, [f"bị {cls_name}"])
        phrases.append(random.choice(templates))
        
    return f"Lá xuất hiện {' và '.join(phrases)}."



# =============================================================================
# Convert Split
# =============================================================================

def convert_split(
    images_dir: Path,
    labels_dir: Path,
    class_names: List[str],
    output_path: Path,
    generator: GeminiPromptGenerator,
) -> int:
    """
    Convert một split (train/valid/test) từ YOLO format sang JSON.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    result = {"images": []}

    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    for idx, img_path in enumerate(image_files, start=1):
        label_path = labels_dir / (img_path.stem + ".txt")
        annotations = parse_yolo_label(label_path)

        # Add class_name
        for ann in annotations:
            cid = ann["class_id"]
            ann["class_name"] = class_names[cid] if cid < len(class_names) else f"class_{cid}"

        # Assign diverse text prompt from Gemini cache
        class_ids_in_image = [ann["class_id"] for ann in annotations]
        text_prompt = assign_text_prompt(class_ids_in_image, class_names, generator)

        result["images"].append({
            "id": idx,
            "file_name": img_path.name,
            "text_prompt": text_prompt,
            "annotations": annotations,
        })

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return len(image_files)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO dataset to MultimodalDataset JSON format (with Gemini text prompts)"
    )
    parser.add_argument(
        "--data-yaml", type=str,
        default="datasets/data.yaml",
        help="Path to YOLO data.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/",
        help="Output directory for JSON annotation files",
    )
    parser.add_argument(
        "--gemini-api-key", type=str, default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--prompt-count", type=int, default=250,
        help="Number of text prompts to generate per class",
    )
    parser.add_argument(
        "--cache-path", type=str, default="data/train_prompt_cache.json",
        help="Path to prompt cache file for train set",
    )
    parser.add_argument(
        "--val-cache-path", type=str, default="data/val_prompt_cache.json",
        help="Path to prompt cache file for validation/test sets",
    )
    args = parser.parse_args()

    # --- Step 1: Generate text prompts with Gemini ---
    logger.info("=" * 50)
    logger.info("Step 1: Generating text prompts...")
    
    train_generator = GeminiPromptGenerator(
        api_key=args.gemini_api_key,
        cache_path=args.cache_path,
    )
    # Tắt gọi Gemini API vì đã dùng cách ghép template
    # train_generator.generate_all_classes(count_per_class=args.prompt_count)
    
    val_generator = GeminiPromptGenerator(
        api_key=args.gemini_api_key,
        cache_path=args.val_cache_path,
    )
    # Không gọi generate_all_classes cho val_generator để giữ nguyên file cache cũ (150 câu) mà không tự động sinh thêm.

    # --- Step 2: Convert YOLO dataset ---
    logger.info("=" * 50)
    logger.info("Step 2: Converting YOLO dataset...")

    data_yaml_path = Path(args.data_yaml)
    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    class_names = data_config["names"]
    dataset_root = data_yaml_path.parent
    output_dir = Path(args.output_dir)

    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Classes ({len(class_names)}): {class_names}")

    splits = {
        "train": ("train", "train_annotations.json"),
        "valid": ("valid", "val_annotations.json"),
        "test": ("test", "test_annotations.json"),
    }

    for split_name, (split_folder, output_filename) in splits.items():
        images_dir = dataset_root / split_folder / "images"
        labels_dir = dataset_root / split_folder / "labels"

        if not images_dir.exists():
            logger.warning(f"Skip {split_name}: {images_dir} not found")
            continue

        output_path = output_dir / output_filename
        current_generator = train_generator if split_name == "train" else val_generator
        count = convert_split(images_dir, labels_dir, class_names, output_path, current_generator)
        logger.info(f"✓ {split_name}: {count} images → {output_path}")

    # --- Summary ---
    logger.info("=" * 50)
    logger.info("Conversion complete!")
    
    sample_path = output_dir / "train_annotations.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Text prompt examples:")
        for img in data["images"][:8]:
            classes = [a["class_name"] for a in img["annotations"]]
            logger.info(f"  {img['file_name']} → '{img['text_prompt']}' (classes: {classes})")


if __name__ == "__main__":
    main()
