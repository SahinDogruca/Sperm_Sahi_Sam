"""
SAM (Segment Anything) ile Mask Refinement — Kaggle-Ready
==========================================================
SAM1 (segment-anything) kullanarak kaba polygon annotation'ları
piksel-düzeyinde hassas mask'lara dönüştürür.

Bu script doğrudan Kaggle'da çalışacak şekilde tasarlanmıştır.

Kaggle Notebook'ta çalıştırma:
  !pip install segment-anything opencv-python-headless
  !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  !python refine_masks_sam.py

Alternatif (daha hızlı, daha az VRAM):
  SAM ViT-B: sam_vit_b_01ec64.pth (375 MB)
  SAM ViT-L: sam_vit_l_0b3195.pth (1.2 GB)
  SAM ViT-H: sam_vit_h_4b8939.pth (2.4 GB) — en iyi kalite
"""

import json
import os
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import torch

# ──────────────────────────────────────────────
# KAGGLE PATHS
# ──────────────────────────────────────────────
# Kaggle:
JSON_DIR        = Path("/kaggle/input/datasets/sahindogruca/oksidatifstress/OksidatifStress/json_files")
IMAGE_DIR       = Path("/kaggle/input/datasets/sahindogruca/oksidatifstress/OksidatifStress/images")
OUTPUT_JSON_DIR = Path("/kaggle/working/json_files_refined")
SAM_CHECKPOINT  = "/kaggle/working/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE  = "vit_h"   # vit_b, vit_l, vit_h

# Yerel test:
# JSON_DIR        = Path("OksidatifStress/json_files")
# IMAGE_DIR       = Path("OksidatifStress/images")
# OUTPUT_JSON_DIR = Path("OksidatifStress/json_files_refined")
# SAM_CHECKPOINT  = "sam_vit_h_4b8939.pth"

# Sınıflar
CLASS_ORDER = ["DEG", "NH", "SH", "MH", "BH"]

# Douglas-Peucker toleransı: düşük = daha fazla nokta = daha hassas
# 1.0: ~50-80 nokta, 2.0: ~20-40 nokta, 0.5: ~100-150 nokta
POLYGON_TOLERANCE = 1.0

# Bbox genişletme oranı
BBOX_EXPAND = 0.3


def polygon_centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def polygon_bbox_expanded(points, img_w, img_h, expand=BBOX_EXPAND):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    ex, ey = w * expand, h * expand
    return np.array([
        max(0, x1 - ex), max(0, y1 - ey),
        min(img_w, x2 + ex), min(img_h, y2 + ey)
    ])


def mask_to_polygon(mask, tolerance=POLYGON_TOLERANCE):
    """Binary mask → polygon noktaları."""
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, tolerance, True)

    if len(approx) < 3:
        return None
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def iou_polygon_mask(polygon_pts, mask, img_w, img_h):
    """Orijinal polygon ile SAM mask arasındaki IoU'yu hesapla (kalite kontrolü)."""
    # Polygon'u mask'a çevir
    pts_np = np.array(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
    poly_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts_np], 1)

    sam_mask = mask.astype(np.uint8)

    intersection = np.logical_and(poly_mask, sam_mask).sum()
    union = np.logical_or(poly_mask, sam_mask).sum()

    if union == 0:
        return 0.0
    return intersection / union


def main():
    print("=" * 70)
    print("SAM Mask Refinement Pipeline (Kaggle-Ready)")
    print("=" * 70)

    # ── SAM yükle ──
    print("\n[1/4] SAM modeli yükleniyor...")
    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f"  ✓ SAM yüklendi: {SAM_MODEL_TYPE}")

    # ── Çıktı dizini ──
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    # ── JSON dosyalarını işle ──
    json_files = sorted(JSON_DIR.glob("*.json"))
    print(f"\n[2/4] {len(json_files)} JSON dosyası işlenecek...")

    stats = defaultdict(int)
    point_counts_before = []
    point_counts_after = []
    iou_scores = []

    for idx, jf in enumerate(json_files):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except:
            continue

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            continue

        # Görüntüyü bul ve yükle
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = IMAGE_DIR / (jf.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            stats["missing_img"] += 1
            # Orijinal JSON'u kopyala
            (OUTPUT_JSON_DIR / jf.name).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            stats["read_fail"] += 1
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # Her shape'i refine et
        refined_shapes = []

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                refined_shapes.append(shape)
                continue

            label = shape.get("label", "")
            points = shape.get("points", [])

            if len(points) < 3 or label not in CLASS_ORDER:
                refined_shapes.append(shape)
                continue

            point_counts_before.append(len(points))

            # SAM prompt hazırla
            cx, cy = polygon_centroid(points)
            bbox = polygon_bbox_expanded(points, img_w, img_h)

            try:
                # Point + Box prompt
                masks, scores, logits = predictor.predict(
                    point_coords=np.array([[cx, cy]]),
                    point_labels=np.array([1]),
                    box=bbox,
                    multimask_output=True,
                )

                # En iyi mask'ı seç
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                best_score = float(scores[best_idx])

                # Mask → polygon
                new_points = mask_to_polygon(best_mask, tolerance=POLYGON_TOLERANCE)

                if new_points and len(new_points) >= 4:
                    # Kalite kontrolü: IoU hesapla
                    iou = iou_polygon_mask(points, best_mask, img_w, img_h)
                    iou_scores.append(iou)

                    # IoU çok düşükse SAM yanlış objeyi segmente etmiş olabilir
                    if iou < 0.15:
                        # SAM tamamen farklı bir şey bulmuş → orijinali koru
                        refined_shapes.append(shape)
                        point_counts_after.append(len(points))
                        stats["low_iou_skip"] += 1
                        continue

                    # Başarılı refinement
                    new_shape = shape.copy()
                    new_shape["points"] = new_points
                    new_shape["flags"] = shape.get("flags", {}).copy()
                    new_shape["flags"]["sam_refined"] = True
                    new_shape["flags"]["sam_score"] = best_score
                    new_shape["flags"]["sam_iou"] = round(iou, 3)
                    new_shape["flags"]["pts_before"] = len(points)
                    new_shape["flags"]["pts_after"] = len(new_points)
                    refined_shapes.append(new_shape)
                    point_counts_after.append(len(new_points))
                    stats["refined"] += 1
                else:
                    refined_shapes.append(shape)
                    point_counts_after.append(len(points))
                    stats["mask_fail"] += 1

            except Exception as e:
                refined_shapes.append(shape)
                point_counts_after.append(len(points))
                stats["error"] += 1

        # Kaydet
        data["shapes"] = refined_shapes
        (OUTPUT_JSON_DIR / jf.name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stats["processed"] += 1

        if (idx + 1) % 25 == 0 or idx == len(json_files) - 1:
            print(f"  [{idx+1}/{len(json_files)}] refined={stats['refined']} "
                  f"kept={stats['mask_fail']+stats['low_iou_skip']}")

    # ── Rapor ──
    print(f"\n[3/4] İstatistikler:")
    print(f"  İşlenen dosya   : {stats['processed']}")
    print(f"  Refined polygon  : {stats['refined']}")
    print(f"  Mask fail        : {stats['mask_fail']}")
    print(f"  Low IoU skip     : {stats['low_iou_skip']}")
    print(f"  Error            : {stats['error']}")
    print(f"  Missing image    : {stats['missing_img']}")

    if point_counts_before:
        print(f"\n  Polygon nokta sayısı değişimi:")
        print(f"    ÖNCE  : avg={sum(point_counts_before)/len(point_counts_before):.1f}  "
              f"min={min(point_counts_before)}  max={max(point_counts_before)}")
        print(f"    SONRA : avg={sum(point_counts_after)/len(point_counts_after):.1f}  "
              f"min={min(point_counts_after)}  max={max(point_counts_after)}")

    if iou_scores:
        print(f"\n  Orijinal↔SAM IoU:")
        print(f"    avg={sum(iou_scores)/len(iou_scores):.3f}  "
              f"min={min(iou_scores):.3f}  max={max(iou_scores):.3f}")

    print(f"\n[4/4] Çıktı: {OUTPUT_JSON_DIR.resolve()}")
    print(f"\n{'='*70}")
    print("SONRAKİ ADIMLAR:")
    print("  1. json_files_refined/ → json_files/ olarak kullan")
    print("     (prepare_sahi_dataset.py'de JSON_DIR'ı güncelle)")
    print("  2. python prepare_sahi_dataset.py")
    print("  3. python oversample_deg.py")
    print("  4. python train_sahi_optimized.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
