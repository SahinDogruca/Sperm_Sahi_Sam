"""
Normal YOLO Dataset Preparation (No Slicing)
==============================================
- /kaggle/working/dataset/train/images -> /kaggle/working/YOLO_Normal_Dataset/train/images
- /kaggle/working/dataset/val/images   -> /kaggle/working/YOLO_Normal_Dataset/val/images
- /kaggle/working/dataset/test/images  -> /kaggle/working/YOLO_Normal_Dataset/test/images
- /kaggle/working/OksidatifStress/json_files/<name>.json -> YOLO segment label -> YOLO_Normal_Dataset/<split>/labels/<name>.txt

Label mapping (sorted alphabetically for consistency):
  DEG -> 0
  NH  -> 1
  SH  -> 2
  MH  -> 3
  BH  -> 4
"""

import os
import json
import shutil
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
WORKING     = Path("/kaggle/working")
DATASET_DIR = WORKING / "dataset"
JSON_DIR    = WORKING / "OksidatifStress" / "json_files"
OUTPUT_DIR  = WORKING / "YOLO_Normal_Dataset"

SPLITS = ["train", "val", "test"]

# Fixed class order as requested: DEG=0, NH=1, SH=2, MH=3, BH=4
CLASS_ORDER = ["DEG", "NH", "SH", "MH", "BH"]
LABEL2ID = {lbl: idx for idx, lbl in enumerate(CLASS_ORDER)}


def convert_json_to_yolo(json_path: Path) -> str | None:
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
        if label not in LABEL2ID:
            continue
        
        cls_id = LABEL2ID[label]
        points = shape.get("points", [])
        if len(points) < 3:
            continue  # degenerate polygon
        
        # Normalise coordinates
        coords = []
        for x, y in points:
            coords.append(f"{max(0.0, min(1.0, x / img_w)):.6f}")
            coords.append(f"{max(0.0, min(1.0, y / img_h)):.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))

    return "\n".join(lines) if lines else None


def main():
    print("=" * 60)
    print("Normal YOLO Dataset Preparation (No Slicing)")
    print(f"JSON kaynak: {JSON_DIR}")
    print(f"Dataset    : {DATASET_DIR}")
    print(f"Çıktı      : {OUTPUT_DIR}")
    print("=" * 60)

    stats = {split: {"images": 0, "labels": 0, "missing_json": 0} for split in SPLITS}

    for split in SPLITS:
        src_img_dir = DATASET_DIR / split / "images"
        dst_img_dir = OUTPUT_DIR / split / "images"
        dst_lbl_dir = OUTPUT_DIR / split / "labels"

        # Temiz başla
        if dst_img_dir.exists():
            shutil.rmtree(dst_img_dir)
        if dst_lbl_dir.exists():
            shutil.rmtree(dst_lbl_dir)
            
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"[SKIP] {src_img_dir} does not exist — skipping split '{split}'")
            continue

        img_files = [
            f for f in src_img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ]

        print(f"\n[{split.upper()}]  {len(img_files)} images found in {src_img_dir}")

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

            yolo_content = convert_json_to_yolo(json_path)
            if yolo_content is None:
                yolo_content = ""
            (dst_lbl_dir / f"{stem}.txt").write_text(yolo_content, encoding="utf-8")
            stats[split]["labels"] += 1

        print(f"  -> images copied : {stats[split]['images']}")
        print(f"  -> labels written: {stats[split]['labels']}")
        if stats[split]["missing_json"]:
            print(f"  -> missing JSONs : {stats[split]['missing_json']}")

    # ── Write data.yaml ─────────────────────────────────────
    yaml_lines = [
        f"path: {OUTPUT_DIR}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        f"nc: {len(LABEL2ID)}",
        "names: [" + ", ".join(f"'{k}'" for k in CLASS_ORDER) + "]",
    ]
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"\n[OK] data.yaml written -> {yaml_path}")

    # ── Summary ─────────────────────────────────────────────
    print("\n=== Summary ===")
    for split in SPLITS:
        s = stats[split]
        print(f"  {split:5s}: {s['images']} images, {s['labels']} labels")
    print(f"\nClass mapping: {LABEL2ID}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nSonraki adım: python oversample_deg.py --dataset_dir {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
