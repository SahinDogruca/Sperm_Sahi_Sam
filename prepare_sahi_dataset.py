"""
SAHI-style Sliced Dataset Preparation for YOLO Segmentation
=============================================================
1280×720 görüntüleri overlap'li 640×640 patch'lere böler.
Polygon annotation'ları her patch'e göre kırpılır ve normalize edilir.

Küçük objelerin (özellikle DEG ~20×20px) efektif boyutunu 2x artırır.

Ön koşul: setup_kaggle_data.py çalıştırılmış olmalı.

Kullanım:
  python prepare_sahi_dataset.py

Çıktı:
  /kaggle/working/YOLO_Sahi_Dataset/
    ├── train/images/   train/labels/
    ├── val/images/     val/labels/
    ├── test/images/    test/labels/
    └── data.yaml

Sınıf sıralaması:
  DEG=0, NH=1, SH=2, MH=3, BH=4
"""

import json
import shutil
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict

# ──────────────────────────────────────────────
# Kaggle Paths (/kaggle/working altındaki kopyalar)
# ──────────────────────────────────────────────
WORKING     = Path("/kaggle/working")
DATASET_DIR = WORKING / "dataset"                          # train/val/test split'leri
JSON_DIR    = WORKING / "OksidatifStress" / "json_files"   # SAM-refined veya orijinal
OUTPUT_DIR  = WORKING / "YOLO_Sahi_Dataset"

SPLITS = ["train", "val", "test"]

# SAHI slice parametreleri
SLICE_W = 640
SLICE_H = 640
OVERLAP_W = 0.3   # %30 yatay overlap
OVERLAP_H = 0.3   # %30 dikey overlap

# Minimum polygon alan oranı (patch alanına göre)
# Bu değerin altındaki kırpılmış polygon'lar atılır
MIN_AREA_RATIO = 0.05  # patch alanının %5'inden küçükse atla

# Sınıf sıralaması
CLASS_ORDER = ["DEG", "NH", "SH", "MH", "BH"]
LABEL2ID = {lbl: idx for idx, lbl in enumerate(CLASS_ORDER)}


def compute_slice_positions(img_w: int, img_h: int,
                             slice_w: int, slice_h: int,
                             overlap_w: float, overlap_h: float) -> list[tuple[int, int, int, int]]:
    """
    Görüntü boyutuna göre slice pozisyonlarını hesapla.
    Returns: [(x1, y1, x2, y2), ...]
    """
    step_x = int(slice_w * (1 - overlap_w))
    step_y = int(slice_h * (1 - overlap_h))

    positions = []
    y = 0
    while y < img_h:
        y2 = min(y + slice_h, img_h)
        # Eğer kalan alan slice_h'den küçükse, sondan başla
        if y2 - y < slice_h and img_h >= slice_h:
            y = img_h - slice_h
            y2 = img_h

        x = 0
        while x < img_w:
            x2 = min(x + slice_w, img_w)
            if x2 - x < slice_w and img_w >= slice_w:
                x = img_w - slice_w
                x2 = img_w

            positions.append((x, y, x2, y2))

            if x2 >= img_w:
                break
            x += step_x

        if y2 >= img_h:
            break
        y += step_y

    # Tekrarları kaldır
    return list(dict.fromkeys(positions))


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace formula ile polygon alanı hesapla."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def clip_polygon_to_box(points: list[list[float]],
                         x1: float, y1: float,
                         x2: float, y2: float) -> list[list[float]] | None:
    """
    Sutherland-Hodgman algoritması ile polygon'u dikdörtgene kırp.
    Returns: Kırpılmış polygon noktaları veya None
    """
    def inside(p, edge, is_left):
        if edge == "left":
            return p[0] >= x1
        elif edge == "right":
            return p[0] <= x2
        elif edge == "top":
            return p[1] >= y1
        elif edge == "bottom":
            return p[1] <= y2

    def intersection(p1, p2, edge):
        x_a, y_a = p1
        x_b, y_b = p2
        dx = x_b - x_a
        dy = y_b - y_a

        if edge == "left":
            if abs(dx) < 1e-10:
                return [x1, y_a]
            t = (x1 - x_a) / dx
            return [x1, y_a + t * dy]
        elif edge == "right":
            if abs(dx) < 1e-10:
                return [x2, y_a]
            t = (x2 - x_a) / dx
            return [x2, y_a + t * dy]
        elif edge == "top":
            if abs(dy) < 1e-10:
                return [x_a, y1]
            t = (y1 - y_a) / dy
            return [x_a + t * dx, y1]
        elif edge == "bottom":
            if abs(dy) < 1e-10:
                return [x_a, y2]
            t = (y2 - y_a) / dy
            return [x_a + t * dx, y2]

    output = list(points)
    for edge in ["left", "right", "top", "bottom"]:
        if len(output) == 0:
            return None
        input_list = output
        output = []
        for i in range(len(input_list)):
            current = input_list[i]
            prev = input_list[i - 1]

            curr_inside = inside(current, edge, True)
            prev_inside = inside(prev, edge, True)

            if curr_inside:
                if not prev_inside:
                    output.append(intersection(prev, current, edge))
                output.append(current)
            elif prev_inside:
                output.append(intersection(prev, current, edge))

    if len(output) < 3:
        return None
    return output


def process_annotations_for_slice(json_path: Path,
                                    x1: int, y1: int, x2: int, y2: int,
                                    slice_w: int, slice_h: int) -> str:
    """
    JSON annotation'ları belirli bir slice'a göre kırp ve YOLO formatına çevir.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    patch_area = slice_w * slice_h
    lines = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        label = shape.get("label", "")
        if label not in LABEL2ID:
            continue

        points = shape.get("points", [])
        if len(points) < 3:
            continue

        # Polygon'u slice box'a kırp
        clipped = clip_polygon_to_box(points, x1, y1, x2, y2)
        if clipped is None or len(clipped) < 3:
            continue

        # Kırpılmış polygon alanını kontrol et
        clipped_area = polygon_area(clipped)
        original_area = polygon_area(points)

        # Çok küçük kırpıntıları atla
        if clipped_area < patch_area * MIN_AREA_RATIO * 0.01:
            continue

        # Orijinal alanın çok azı kaldıysa atla (kenar artefaktı)
        if original_area > 0 and clipped_area / original_area < 0.15:
            continue

        # Koordinatları slice'a göre normalize et
        cls_id = LABEL2ID[label]
        coords = []
        for px, py in clipped:
            nx = max(0.0, min(1.0, (px - x1) / slice_w))
            ny = max(0.0, min(1.0, (py - y1) / slice_h))
            coords.append(f"{nx:.6f}")
            coords.append(f"{ny:.6f}")

        lines.append(f"{cls_id} " + " ".join(coords))

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("SAHI-Style Sliced Dataset Preparation")
    print(f"Slice: {SLICE_W}×{SLICE_H}, Overlap: {OVERLAP_W:.0%}×{OVERLAP_H:.0%}")
    print(f"JSON kaynak: {JSON_DIR}")
    print(f"Dataset    : {DATASET_DIR}")
    print(f"Çıktı      : {OUTPUT_DIR}")
    print("=" * 60)

    global_stats = {split: {"images": 0, "patches": 0, "labels_with_obj": 0}
                    for split in SPLITS}
    global_class_counts = {split: Counter() for split in SPLITS}

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
            print(f"[SKIP] {src_img_dir} does not exist")
            continue

        img_files = sorted([
            f for f in src_img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ])

        print(f"\n[{split.upper()}] {len(img_files)} görüntü işleniyor...")

        for img_path in img_files:
            stem = img_path.stem
            json_path = JSON_DIR / f"{stem}.json"

            # Görüntü boyutunu oku
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"  [WARN] Cannot open {img_path.name}: {e}")
                continue

            global_stats[split]["images"] += 1

            # Slice pozisyonlarını hesapla
            positions = compute_slice_positions(
                img_w, img_h, SLICE_W, SLICE_H, OVERLAP_W, OVERLAP_H
            )

            for patch_idx, (x1, y1, x2, y2) in enumerate(positions):
                patch_name = f"{stem}_p{patch_idx}"
                pw = x2 - x1
                ph = y2 - y1

                # Patch'i kes ve kaydet
                with Image.open(img_path) as img:
                    patch = img.crop((x1, y1, x2, y2))
                    patch_path = dst_img_dir / f"{patch_name}.jpg"
                    patch.save(patch_path, "JPEG", quality=95)

                global_stats[split]["patches"] += 1

                # Annotation'ları kırp
                if json_path.exists():
                    label_content = process_annotations_for_slice(
                        json_path, x1, y1, x2, y2, pw, ph
                    )
                else:
                    label_content = ""

                # Label dosyasını yaz
                label_path = dst_lbl_dir / f"{patch_name}.txt"
                label_path.write_text(label_content, encoding="utf-8")

                if label_content.strip():
                    global_stats[split]["labels_with_obj"] += 1
                    # Sınıf sayımı
                    for line in label_content.strip().split("\n"):
                        cls_id = int(line.split()[0])
                        cls_name = CLASS_ORDER[cls_id]
                        global_class_counts[split][cls_name] += 1

        s = global_stats[split]
        print(f"  → {s['images']} görüntü → {s['patches']} patch")
        print(f"  → {s['labels_with_obj']} patch'te obje var")
        print(f"  → Sınıf dağılımı: {dict(global_class_counts[split])}")

    # ── data.yaml yaz ──────────────────────────────────
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
    print(f"\n[OK] data.yaml → {yaml_path}")

    # ── Özet ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    for split in SPLITS:
        s = global_stats[split]
        print(f"  {split:5s}: {s['images']} img → {s['patches']} patches "
              f"({s['labels_with_obj']} with objects)")
    print(f"\nSınıf sıralaması: {LABEL2ID}")
    print(f"Çıktı: {OUTPUT_DIR}")
    print(f"\nSonraki adım: python oversample_deg.py")


if __name__ == "__main__":
    main()
