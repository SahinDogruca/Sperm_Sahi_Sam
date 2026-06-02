"""
DEG Oversampling
==================================================
DEG sınıfı içeren patch'leri 2x çoğaltarak sınıf dengesizliğini azaltır.
SAHI dataset üzerinde çalışır (prepare_sahi_dataset.py çıktısı).

Ön koşul: prepare_sahi_dataset.py çalıştırılmış olmalı.

Kullanım:
  python oversample_deg.py
"""

import shutil
from pathlib import Path
from collections import Counter

# ──────────────────────────────────────────────
# Kaggle Paths
# ──────────────────────────────────────────────
DATASET_DIR = Path("/kaggle/working/YOLO_Sahi_Dataset")
REPEAT      = 2   # DEG içeren patch'leri kaç kere tekrarla (2 = 2 ek kopya)

# Sınıf sıralaması
CLASS_ORDER = ["DEG", "NH", "SH", "MH", "BH"]
DEG_CLASS_ID = 0   # DEG sınıfının YOLO ID'si


def main():
    train_img_dir = DATASET_DIR / "train" / "images"
    train_lbl_dir = DATASET_DIR / "train" / "labels"

    if not train_img_dir.exists():
        print(f"[HATA] {train_img_dir} bulunamadı!")
        print("Önce prepare_sahi_dataset.py çalıştırın.")
        return

    # DEG içeren label dosyalarını bul
    deg_labels = []
    total_labels = 0
    class_counts_before = Counter()

    for lbl_path in sorted(train_lbl_dir.glob("*.txt")):
        content = lbl_path.read_text(encoding="utf-8").strip()
        if not content:
            total_labels += 1
            continue

        total_labels += 1
        has_deg = False
        for line in content.split("\n"):
            parts = line.strip().split()
            if parts:
                cls_id = int(parts[0])
                class_counts_before[CLASS_ORDER[cls_id]] += 1
                if cls_id == DEG_CLASS_ID:
                    has_deg = True

        if has_deg:
            deg_labels.append(lbl_path)

    print("=" * 60)
    print("DEG Oversampling")
    print("=" * 60)
    print(f"\nToplam train patch: {total_labels}")
    print(f"DEG içeren patch: {len(deg_labels)}")
    print(f"Tekrar sayısı: {REPEAT}x")
    print(f"\nÖnceki sınıf dağılımı:")
    total = sum(class_counts_before.values())
    for cls_name in CLASS_ORDER:
        cnt = class_counts_before[cls_name]
        print(f"  {cls_name:5s}: {cnt:5d} ({cnt/total*100:5.1f}%)")

    # DEG içeren patch'leri kopyala
    added_patches = 0
    class_counts_added = Counter()

    for lbl_path in deg_labels:
        stem = lbl_path.stem
        img_path = train_img_dir / f"{stem}.jpg"

        if not img_path.exists():
            # PNG dene
            img_path = train_img_dir / f"{stem}.png"
            if not img_path.exists():
                print(f"  [WARN] Görüntü bulunamadı: {stem}")
                continue

        content = lbl_path.read_text(encoding="utf-8").strip()

        for rep in range(REPEAT):
            new_stem = f"{stem}_degR{rep}"
            new_img = train_img_dir / f"{new_stem}{img_path.suffix}"
            new_lbl = train_lbl_dir / f"{new_stem}.txt"

            shutil.copy2(img_path, new_img)
            new_lbl.write_text(content, encoding="utf-8")
            added_patches += 1

            # Eklenen sınıf sayımı
            for line in content.split("\n"):
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts_added[CLASS_ORDER[cls_id]] += 1

    # Sonuç
    print(f"\nEklenen patch: {added_patches}")
    print(f"\nSonraki sınıf dağılımı:")
    total_after = total + sum(class_counts_added.values())
    for cls_name in CLASS_ORDER:
        cnt = class_counts_before[cls_name] + class_counts_added[cls_name]
        print(f"  {cls_name:5s}: {cnt:5d} ({cnt/total_after*100:5.1f}%)")

    print(f"\nToplam train patch (sonrası): {total_labels + added_patches}")
    print("[OK] DEG oversampling tamamlandı.")
    print(f"\nSonraki adım: python train_sahi_optimized.py")


if __name__ == "__main__":
    main()
