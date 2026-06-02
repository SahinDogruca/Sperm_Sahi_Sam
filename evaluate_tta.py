"""
TTA (Test-Time Augmentation) ile Değerlendirme — Val + Test
=============================================================
Eğitilmiş modeli TTA aktif/kapalı olarak hem val hem test setinde
değerlendirir ve karşılaştırmalı tablo çıkarır.

Ön koşul: train_sahi_optimized.py ile eğitim tamamlanmış olmalı.

Kullanım:
  python evaluate_tta.py

Otomatik olarak:
  - best.pt weights'ini bulur
  - Val split: Normal + TTA
  - Test split: Normal + TTA
  - 4 sonucu karşılaştırmalı tablo ile gösterir
"""

import numpy as np

# NumPy 2.0+ uyumluluk yaması
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from ultralytics import YOLO
from pathlib import Path

# ──────────────────────────────────────────────
# Kaggle Paths
# ──────────────────────────────────────────────
DATA_YAML = "/kaggle/working/YOLO_Sahi_Dataset/data.yaml"
WEIGHTS   = "/kaggle/working/final_v2/sahi_optimized/weights/best.pt"
PROJECT   = "/kaggle/working/final_v2"

IMGSZ  = 640
BATCH  = 8
DEVICE = "0"
CONF   = 0.001
IOU    = 0.6


def safe_get(results, key, default=0.0):
    """results_dict'ten güvenli değer çek."""
    try:
        return float(results.results_dict.get(key, default))
    except Exception:
        return default


def evaluate_split(model, split, augment, tag):
    """Tek bir split'i değerlendir."""
    print(f"\n{'─'*50}")
    print(f"  [{tag}] split={split}, TTA={'AÇIK' if augment else 'KAPALI'}")
    print(f"{'─'*50}")

    results = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        split=split,
        conf=CONF,
        iou=IOU,
        augment=augment,
        plots=True,
        verbose=True,
        project=PROJECT,
        name=f"eval_{split}_{'tta' if augment else 'normal'}",
        exist_ok=True,
    )

    return results


def main():
    print("=" * 70)
    print("Model Değerlendirme — Val + Test × Normal + TTA")
    print("=" * 70)

    # Weights kontrolü
    if not Path(WEIGHTS).exists():
        print(f"\n[HATA] Weights bulunamadı: {WEIGHTS}")
        print("Önce train_sahi_optimized.py çalıştırın.")

        # Alternatif weights ara
        alt = Path(PROJECT).glob("**/best.pt")
        for p in alt:
            print(f"  Bulunan alternatif: {p}")
        return

    print(f"\n  Weights : {WEIGHTS}")
    print(f"  Data    : {DATA_YAML}")
    print(f"  ImgSz   : {IMGSZ}")

    model = YOLO(WEIGHTS)

    # ── 4 değerlendirme çalıştır ──
    results = {}

    # 1. Val — Normal
    results["val_normal"] = evaluate_split(model, "val", augment=False, tag="1/4")

    # 2. Val — TTA
    results["val_tta"] = evaluate_split(model, "val", augment=True, tag="2/4")

    # 3. Test — Normal
    results["test_normal"] = evaluate_split(model, "test", augment=False, tag="3/4")

    # 4. Test — TTA
    results["test_tta"] = evaluate_split(model, "test", augment=True, tag="4/4")

    # ── Karşılaştırma Tablosu ──
    metrics = [
        ("mAP50(B)",    "metrics/mAP50(B)"),
        ("mAP50-95(B)", "metrics/mAP50-95(B)"),
        ("mAP50(M)",    "metrics/mAP50(M)"),
        ("mAP50-95(M)", "metrics/mAP50-95(M)"),
        ("Precision(B)", "metrics/precision(B)"),
        ("Recall(B)",    "metrics/recall(B)"),
    ]

    print("\n" + "=" * 90)
    print("SONUÇ KARŞILAŞTIRMASI")
    print("=" * 90)

    header = f"{'Metric':<16s} │ {'Val':>8s} {'Val+TTA':>8s} {'Δ':>7s} │ {'Test':>8s} {'Test+TTA':>9s} {'Δ':>7s}"
    print(header)
    print("─" * 90)

    for name, key in metrics:
        vn = safe_get(results["val_normal"], key)
        vt = safe_get(results["val_tta"], key)
        tn = safe_get(results["test_normal"], key)
        tt = safe_get(results["test_tta"], key)

        vd = vt - vn
        td = tt - tn

        vs = f"+{vd:.4f}" if vd > 0 else f"{vd:.4f}"
        ts = f"+{td:.4f}" if td > 0 else f"{td:.4f}"

        print(f"{name:<16s} │ {vn:>8.4f} {vt:>8.4f} {vs:>7s} │ {tn:>8.4f} {tt:>9.4f} {ts:>7s}")

    print("─" * 90)

    # Per-class sonuçlar (varsa)
    print("\n" + "=" * 70)
    print("PER-CLASS DETAY (Val + TTA — en iyi sonuç)")
    print("=" * 70)

    try:
        best_result = results["val_tta"]
        # Per-class mAP erişimi YOLO versiyonuna göre değişebilir
        if hasattr(best_result, 'box'):
            class_names = best_result.names if hasattr(best_result, 'names') else {}
            print("  (Per-class detaylar plots/ klasöründeki confusion matrix'te mevcuttur)")
    except Exception:
        pass

    print(f"\n{'='*70}")
    print("DEĞERLENDİRME TAMAMLANDI")
    print(f"{'='*70}")
    print(f"  Plots: {PROJECT}/eval_*/")
    print(f"  Best weights: {WEIGHTS}")


if __name__ == "__main__":
    main()
