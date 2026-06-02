"""
Kaggle Data Setup — OksidatifStress + SAM Refined JSON Birleştirme
===================================================================
1. /kaggle/input/.../OksidatifStress → /kaggle/working/OksidatifStress kopyalar
2. SAM refined JSON'lar varsa → json_files'ı bunlarla değiştirir
3. dataset split'leri (train/val/test) → /kaggle/working/dataset kopyalar

Bu script tüm pipeline'ın ilk adımıdır.

Akış:
  setup_kaggle_data.py  →  prepare_sahi_dataset.py  →  oversample_deg.py
       →  train_sahi_optimized.py  →  evaluate_tta.py
"""

import shutil
from pathlib import Path

# ──────────────────────────────────────────────
# Kaynak (Kaggle input — read-only)
# ──────────────────────────────────────────────
INPUT_BASE = Path("/kaggle/input/datasets/sahindogruca")
INPUT_OKSIDATIF = INPUT_BASE / "oksidatifstress" / "OksidatifStress"
INPUT_DATASET   = INPUT_BASE / "dataset" / "dataset"

# ──────────────────────────────────────────────
# Hedef (Kaggle working — read-write)
# ──────────────────────────────────────────────
WORKING = Path("/kaggle/working")
WORKING_OKSIDATIF = WORKING / "OksidatifStress"
WORKING_DATASET   = WORKING / "dataset"

# SAM refined JSON çıktısı (refine_masks_sam.py tarafından üretilir)
SAM_REFINED_DIR = WORKING / "json_files_refined"


def main():
    print("=" * 60)
    print("Kaggle Data Setup")
    print("=" * 60)

    # ── 1. OksidatifStress klasörünü kopyala ──
    print(f"\n[1/3] OksidatifStress kopyalanıyor...")
    print(f"  Kaynak: {INPUT_OKSIDATIF}")
    print(f"  Hedef : {WORKING_OKSIDATIF}")

    if WORKING_OKSIDATIF.exists():
        shutil.rmtree(WORKING_OKSIDATIF)

    shutil.copytree(INPUT_OKSIDATIF, WORKING_OKSIDATIF)

    # Dosya sayısını kontrol et
    json_count = len(list((WORKING_OKSIDATIF / "json_files").glob("*.json")))
    img_count = len(list((WORKING_OKSIDATIF / "images").glob("*")))
    print(f"  ✓ {json_count} JSON + {img_count} görüntü kopyalandı")

    # ── 2. SAM refined JSON'ları birleştir ──
    print(f"\n[2/3] SAM Refined JSON kontrolü...")

    if SAM_REFINED_DIR.exists():
        refined_files = list(SAM_REFINED_DIR.glob("*.json"))
        if refined_files:
            print(f"  ✓ {len(refined_files)} refined JSON bulundu → birleştiriliyor")

            target_json_dir = WORKING_OKSIDATIF / "json_files"

            # Orijinal JSON'ların yedeğini al
            backup_dir = WORKING_OKSIDATIF / "json_files_original"
            if not backup_dir.exists():
                shutil.copytree(target_json_dir, backup_dir)
                print(f"  ✓ Orijinal JSON'lar yedeklendi → {backup_dir.name}/")

            # Refined JSON'ları kopyala (orijinallerin üzerine yaz)
            replaced = 0
            for rf in refined_files:
                target = target_json_dir / rf.name
                shutil.copy2(rf, target)
                replaced += 1

            print(f"  ✓ {replaced} JSON refined versiyonla değiştirildi")
        else:
            print("  ⚠ json_files_refined/ klasörü boş → orijinal JSON'lar kullanılacak")
    else:
        print("  ⚠ json_files_refined/ bulunamadı → orijinal JSON'lar kullanılacak")
        print("    (SAM refinement atlandıysa bu normaldir)")

    # ── 3. Dataset split'lerini kopyala ──
    print(f"\n[3/3] Dataset split'leri kopyalanıyor...")
    print(f"  Kaynak: {INPUT_DATASET}")
    print(f"  Hedef : {WORKING_DATASET}")

    if WORKING_DATASET.exists():
        shutil.rmtree(WORKING_DATASET)

    shutil.copytree(INPUT_DATASET, WORKING_DATASET)

    for split in ["train", "val", "test"]:
        split_dir = WORKING_DATASET / split / "images"
        if split_dir.exists():
            count = len(list(split_dir.iterdir()))
            print(f"  ✓ {split:5s}: {count} görüntü")
        else:
            print(f"  ⚠ {split:5s}: bulunamadı")

    # ── Özet ──
    print(f"\n{'='*60}")
    print("SETUP TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  OksidatifStress : {WORKING_OKSIDATIF}")
    print(f"    └─ json_files : {'SAM-refined ✓' if SAM_REFINED_DIR.exists() and list(SAM_REFINED_DIR.glob('*.json')) else 'orijinal'}")
    print(f"  Dataset splits  : {WORKING_DATASET}")
    print(f"\n  Sonraki adım: python prepare_sahi_dataset.py")


if __name__ == "__main__":
    main()
