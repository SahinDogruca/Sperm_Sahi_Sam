"""
Prepare YOLO Segmentation Dataset
==================================
- datasets/train/images -> OutputDataset/train/images
- datasets/val/images   -> OutputDataset/val/images
- datasets/test/images  -> OutputDataset/test/images
- OksidatifStress/json_files/<name>.json -> YOLO segment label -> OutputDataset/<split>/labels/<name>.txt

Label mapping (sorted alphabetically for consistency):
  BH  -> 0
  DEG -> 1
  MH  -> 2
  NH  -> 3
  SH  -> 4
"""

import os
import json
import shutil
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
JSON_DIR    = BASE_DIR / "OksidatifStress" / "json_files"
OUTPUT_DIR  = BASE_DIR / "OutputDataset"

SPLITS = ["train", "val", "test"]

# ──────────────────────────────────────────────
# Build global class list from ALL json files
# (so indices are consistent across splits)
# ──────────────────────────────────────────────
def collect_all_labels(json_dir: Path) -> dict[str, int]:
    # Fixed class order as requested: DEG=0, NH=1, SH=2, MH=3, BH=4
    CLASS_ORDER = ["DEG", "NH", "SH", "MH", "BH"]
    label2id = {lbl: idx for idx, lbl in enumerate(CLASS_ORDER)}
    return label2id


def convert_json_to_yolo(json_path: Path, label2id: dict[str, int]) -> str | None:
    """
    Convert a labelme-style polygon JSON to YOLO segmentation format.
    Returns the text content or None if no valid shapes found.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] Could not read {json_path.name}: {e}")
        return None

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    if not img_w or not img_h:
        print(f"  [WARN] Missing image dimensions in {json_path.name}")
        return None

    lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        label = shape.get("label", "")
        if label not in label2id:
            print(f"  [WARN] Unknown label '{label}' in {json_path.name}, skipping")
            continue
        cls_id = label2id[label]
        points = shape.get("points", [])
        if len(points) < 3:
            continue  # degenerate polygon
        # Normalise coordinates
        coords = []
        for x, y in points:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))

    return "\n".join(lines) if lines else None


def main():
    # Collect label mapping
    print("Collecting class labels from all JSON files ...")
    label2id = collect_all_labels(JSON_DIR)
    print(f"  Found {len(label2id)} classes: {label2id}\n")

    stats = {split: {"images": 0, "labels": 0, "missing_json": 0} for split in SPLITS}

    for split in SPLITS:
        src_img_dir = DATASET_DIR / split / "images"
        dst_img_dir = OUTPUT_DIR / split / "images"
        dst_lbl_dir = OUTPUT_DIR / split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"[SKIP] {src_img_dir} does not exist — skipping split '{split}'")
            continue

        img_files = [
            f for f in src_img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ]

        print(f"[{split.upper()}]  {len(img_files)} images found in {src_img_dir}")

        for img_path in sorted(img_files):
            stem = img_path.stem
            json_path = JSON_DIR / f"{stem}.json"

            # ── Copy image ──────────────────────────────────
            dst_img = dst_img_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            stats[split]["images"] += 1

            # ── Convert annotation ──────────────────────────
            if not json_path.exists():
                print(f"  [WARN] No JSON for {img_path.name}")
                stats[split]["missing_json"] += 1
                # Write empty label file so YOLO doesn't complain
                (dst_lbl_dir / f"{stem}.txt").write_text("")
                continue

            yolo_content = convert_json_to_yolo(json_path, label2id)
            if yolo_content is None:
                yolo_content = ""
            (dst_lbl_dir / f"{stem}.txt").write_text(yolo_content, encoding="utf-8")
            stats[split]["labels"] += 1

        print(f"  -> images copied : {stats[split]['images']}")
        print(f"  -> labels written: {stats[split]['labels']}")
        if stats[split]["missing_json"]:
            print(f"  -> missing JSONs : {stats[split]['missing_json']}")
        print()

    # ── Write data.yaml ─────────────────────────────────────
    yaml_lines = [
        f"path: {OUTPUT_DIR.resolve()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        f"nc: {len(label2id)}",
        "names: [" + ", ".join(f"'{k}'" for k in sorted(label2id, key=lambda x: label2id[x])) + "]",
    ]
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"[OK] data.yaml written -> {yaml_path}")

    # ── Summary ─────────────────────────────────────────────
    print("\n=== Summary ===")
    for split in SPLITS:
        s = stats[split]
        print(f"  {split:5s}: {s['images']} images, {s['labels']} labels")
    print(f"\nClass mapping:")
    for lbl, idx in sorted(label2id.items(), key=lambda x: x[1]):
        print(f"  {idx}: {lbl}")
    print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
