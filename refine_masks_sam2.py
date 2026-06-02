"""
SAM2 ile Mask Refinement Pipeline
==================================
Mevcut kaba polygon annotation'ları (3-8 nokta) SAM2 kullanarak
piksel-düzeyinde hassas mask'lara dönüştürür.

Strateji:
  1. Mevcut polygon'un centroid + bbox'unu hesapla
  2. SAM2'ye point prompt (centroid) + box prompt (bbox) ver
  3. SAM2'nin ürettiği hassas mask'ı polygon'a çevir (~50-100 nokta)
  4. Sınıf bilgisini koru, sadece polygon kalitesini artır

Bu script Kaggle'da GPU ile çalıştırılmalıdır (SAM2 GPU gerektirir).

Kullanım (Kaggle):
  pip install segment-anything-2 opencv-python-headless
  python refine_masks_sam2.py

Neden mAP@50-95 artacak:
  - mAP@50: IoU≥0.50 → kaba polygon zaten yakalıyor (mevcut ~0.70)
  - mAP@75: IoU≥0.75 → 5 noktalı üçgen mask asla 0.75 IoU'ya ulaşamaz!
  - mAP@50-95: Tüm IoU eşiklerinin ortalaması → mask kalitesi kritik
  - 5 noktalı kaba polygon ile gerçek sperm başı arasındaki IoU max ~0.6-0.7
  - SAM ile 50+ noktalı mask → IoU 0.85-0.95'e çıkar
"""

import json
import os
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────
# KAGGLE PATHS — kendi ortamınıza göre güncelleyin
# ──────────────────────────────────────────────
# Yerel geliştirme (Mac):
BASE_DIR  = Path(__file__).parent
JSON_DIR  = BASE_DIR / "OksidatifStress" / "json_files"
IMAGE_DIR = BASE_DIR / "OksidatifStress" / "images"
OUTPUT_JSON_DIR = BASE_DIR / "OksidatifStress" / "json_files_refined"

# Kaggle'da:
# BASE_DIR  = Path("/kaggle/working")
# JSON_DIR  = Path("/kaggle/input/your-dataset/OksidatifStress/json_files")
# IMAGE_DIR = Path("/kaggle/input/your-dataset/OksidatifStress/images")
# OUTPUT_JSON_DIR = Path("/kaggle/working/json_files_refined")

# SAM2 model
SAM2_CHECKPOINT = "facebook/sam2.1-hiera-large"  # veya "facebook/sam2.1-hiera-base-plus"

# Sınıf sıralaması
CLASS_ORDER = ["DEG", "NH", "SH", "MH", "BH"]

# Minimum polygon çevresi (piksel) — bundan küçük annotation'ları atla
MIN_PERIMETER = 10

# Mask'tan polygon'a çevirirken tolerans (piksel)
# Daha düşük = daha fazla nokta = daha hassas
POLYGON_TOLERANCE = 1.5


def polygon_centroid(points):
    """Polygon centroid hesapla."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def polygon_bbox(points):
    """Polygon bounding box [x1, y1, x2, y2]."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def polygon_bbox_expanded(points, img_w, img_h, expand_ratio=0.3):
    """Bounding box'ı genişlet (SAM'e daha iyi context verir)."""
    x1, y1, x2, y2 = polygon_bbox(points)
    w = x2 - x1
    h = y2 - y1
    ex = w * expand_ratio
    ey = h * expand_ratio
    return [
        max(0, x1 - ex),
        max(0, y1 - ey),
        min(img_w, x2 + ex),
        min(img_h, y2 + ey),
    ]


def mask_to_polygon(mask, tolerance=POLYGON_TOLERANCE):
    """
    Binary mask'ı polygon noktalarına çevir.
    cv2.findContours + approxPolyDP kullanır.
    """
    import cv2

    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # En büyük konturu al
    largest = max(contours, key=cv2.contourArea)

    # Douglas-Peucker ile sadeleştir (ama orijinalden çok daha fazla nokta kalacak)
    approx = cv2.approxPolyDP(largest, tolerance, True)

    if len(approx) < 3:
        return None

    points = [[float(p[0][0]), float(p[0][1])] for p in approx]
    return points


def refine_with_sam2(sam_predictor, image, shapes, img_w, img_h):
    """
    Bir görüntüdeki tüm shape'leri SAM2 ile refine et.
    
    Args:
        sam_predictor: SAM2 predictor (set_image yapılmış)
        image: numpy array (H, W, 3)
        shapes: labelme shapes listesi
        img_w, img_h: görüntü boyutları
    
    Returns:
        Refined shapes listesi
    """
    refined_shapes = []

    for shape in shapes:
        if shape.get("shape_type") != "polygon":
            refined_shapes.append(shape)
            continue

        label = shape.get("label", "")
        points = shape.get("points", [])

        if len(points) < 3 or label not in CLASS_ORDER:
            refined_shapes.append(shape)
            continue

        # Centroid ve expanded bbox hesapla
        centroid = polygon_centroid(points)
        bbox = polygon_bbox_expanded(points, img_w, img_h, expand_ratio=0.3)

        # SAM2'ye prompt ver
        try:
            # Point prompt (centroid) + Box prompt (expanded bbox)
            input_point = np.array([[centroid[0], centroid[1]]])
            input_label = np.array([1])  # 1 = foreground
            input_box = np.array(bbox)

            masks, scores, _ = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True,
            )

            # En yüksek skorlu mask'ı seç
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]

            # Mask'ı polygon'a çevir
            new_points = mask_to_polygon(best_mask, tolerance=POLYGON_TOLERANCE)

            if new_points and len(new_points) >= 3:
                # SAM mask geçerli → kullan
                refined_shape = shape.copy()
                refined_shape["points"] = new_points
                refined_shape["flags"] = shape.get("flags", {})
                refined_shape["flags"]["sam_refined"] = True
                refined_shape["flags"]["sam_score"] = float(best_score)
                refined_shape["flags"]["original_points"] = len(points)
                refined_shape["flags"]["refined_points"] = len(new_points)
                refined_shapes.append(refined_shape)
            else:
                # SAM başarısız → orijinali koru
                refined_shapes.append(shape)

        except Exception as e:
            print(f"    [WARN] SAM failed for {label}: {e}")
            refined_shapes.append(shape)

    return refined_shapes


def main():
    import cv2

    print("=" * 70)
    print("SAM2 Mask Refinement Pipeline")
    print("=" * 70)

    # ── SAM2 yükle ──
    print("\n[1/3] SAM2 modeli yükleniyor...")
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor.from_pretrained(SAM2_CHECKPOINT)
        print(f"  ✓ SAM2 yüklendi: {SAM2_CHECKPOINT}")
    except ImportError:
        print("  ✗ sam2 modülü bulunamadı!")
        print("  Kurulum: pip install sam2")
        print("\n  Alternatif olarak segment-anything kullanabilirsiniz:")
        print("  pip install segment-anything")
        print("  Bu durumda refine_masks_sam1.py scriptini kullanın.")
        return

    # ── Output dizini ──
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    # ── JSON dosyalarını işle ──
    json_files = sorted(JSON_DIR.glob("*.json"))
    print(f"\n[2/3] {len(json_files)} JSON dosyası işlenecek...")

    stats = defaultdict(int)
    point_before = []
    point_after = []

    for idx, jf in enumerate(json_files):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] {jf.name}: {e}")
            continue

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            continue

        # Görüntüyü yükle
        img_name = data.get("imagePath", jf.stem + ".jpg")
        img_path = IMAGE_DIR / img_name
        if not img_path.exists():
            # Alternatif yolları dene
            for ext in [".jpg", ".jpeg", ".png"]:
                alt = IMAGE_DIR / (jf.stem + ext)
                if alt.exists():
                    img_path = alt
                    break

        if not img_path.exists():
            stats["missing_img"] += 1
            # Orijinal JSON'u kopyala
            (OUTPUT_JSON_DIR / jf.name).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            continue

        # Görüntüyü oku
        image = cv2.imread(str(img_path))
        if image is None:
            stats["read_fail"] += 1
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # SAM2'ye görüntüyü set et
        with predictor.set_image(image_rgb) as _:
            pass
        predictor.set_image(image_rgb)

        # Shape'leri refine et
        original_shapes = data.get("shapes", [])
        for s in original_shapes:
            if s.get("shape_type") == "polygon":
                point_before.append(len(s.get("points", [])))

        refined_shapes = refine_with_sam2(
            predictor, image_rgb, original_shapes, img_w, img_h
        )

        for s in refined_shapes:
            if s.get("shape_type") == "polygon":
                point_after.append(len(s.get("points", [])))
                if s.get("flags", {}).get("sam_refined"):
                    stats["refined"] += 1
                else:
                    stats["kept_original"] += 1

        # Refined JSON'u kaydet
        data["shapes"] = refined_shapes
        out_path = OUTPUT_JSON_DIR / jf.name
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stats["processed"] += 1

        if (idx + 1) % 20 == 0 or idx == len(json_files) - 1:
            print(f"  [{idx+1}/{len(json_files)}] {jf.stem} "
                  f"({stats['refined']} refined, {stats['kept_original']} kept)")

    # ── İstatistikler ──
    print(f"\n[3/3] Sonuçlar:")
    print(f"  İşlenen dosya: {stats['processed']}")
    print(f"  Refined shape: {stats['refined']}")
    print(f"  Orijinal kalan: {stats['kept_original']}")
    if stats['missing_img']:
        print(f"  Eksik görüntü: {stats['missing_img']}")

    if point_before and point_after:
        print(f"\n  Polygon nokta sayısı:")
        print(f"    Önce:  avg={sum(point_before)/len(point_before):.1f}  "
              f"min={min(point_before)}  max={max(point_before)}")
        print(f"    Sonra: avg={sum(point_after)/len(point_after):.1f}  "
              f"min={min(point_after)}  max={max(point_after)}")

    print(f"\n  Çıktı: {OUTPUT_JSON_DIR.resolve()}")
    print(f"\n  Sonraki adım:")
    print(f"    1. json_files_refined/ klasörünü json_files/ ile değiştir")
    print(f"    2. prepare_sahi_dataset.py çalıştır")
    print(f"    3. oversample_deg.py çalıştır")
    print(f"    4. train_sahi_optimized.py ile eğit")


if __name__ == "__main__":
    main()
