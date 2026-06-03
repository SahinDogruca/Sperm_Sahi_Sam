"""
Geliştirilmiş YOLOv12m-seg Eğitim Scripti
==========================================
SAHI sliced dataset + DEG oversampling sonrası kullanılır.

Ön koşul: prepare_sahi_dataset.py + oversample_deg.py çalıştırılmış olmalı.

Değişiklikler vs orijinal train_best.py:
  1. box=7.5, cls=2.5, dfl=2.0  → mAP@50-95 ve DEG için optimize
  2. mixup=0.15 eklendi
  3. copy_paste=0.5 (artırıldı)
  4. scale=0.5 (artırıldı — zoom-in küçük objeleri büyütür)
  5. warmup_epochs=5
  6. epochs=400, patience=60
  7. close_mosaic=15
  8. perspective=0.0003 eklendi

Kullanım:
  python train_sahi_optimized.py
"""

import json
from pathlib import Path
import numpy as np

# NumPy 2.0+ uyumluluk yaması (np.trapz kaldırıldı, yerine np.trapezoid geldi)
# Ultralytics içindeki metrics hesaplamasında hata vermemesi için:
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from ultralytics import YOLO

# ──────────────────────────────────────────────
# Kaggle Paths
# ──────────────────────────────────────────────
DATA_YAML = "/kaggle/working/YOLO_Sahi_Dataset/data.yaml"
MODEL_PT = "yolov12m-seg.pt"
PROJECT = "/kaggle/working/final_v2"

# Opsiyonel: Optuna best params varsa yükle
BEST_PARAMS_PATH = "/kaggle/working/tune/best_params.json"
USE_TUNED = False  # True yaparsanız Optuna sonuçlarını birleştirir

# ──────────────────────────────────────────────
# Parametreler
# ──────────────────────────────────────────────
# Varsayılan (optimize edilmiş) parametreler
params = {
    "lr0": 0.0008,
    "lrf": 0.01,
    "warmup_epochs": 5,
    "box": 7.5,  # ↑ Kesin lokalizasyon → mAP@50-95 artışı
    "cls": 2.0,  # ↑ Sınıflandırma odağı → DEG recall artışı
    "dfl": 2.0,  # ↑ Distribution focal loss → bbox kalitesi
    "copy_paste": 0.3,  # ↑ Küçük objeleri çoğalt
    "degrees": 10.0,
    "scale": 0.2,
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "erasing": 0.3,
    "close_mosaic": 100,
    "mixup": 0.15,  # YENİ: Görüntü blend → regularization
    "perspective": 0.0003,  # YENİ: Hafif perspektif → robustness
}

# Optuna sonuçlarını birleştir (opsiyonel)
if USE_TUNED and Path(BEST_PARAMS_PATH).exists():
    with open(BEST_PARAMS_PATH) as f:
        info = json.load(f)
    tuned = info["params"]
    print(f"Optuna Trial #{info['trial']} — tune mAP50-95(M): {info['value']:.4f}")

    # Tuned parametreleri güncelle, ama box/cls/dfl override'larımızı koru
    for key in [
        "lr0",
        "lrf",
        "warmup_epochs",
        "copy_paste",
        "degrees",
        "scale",
        "hsv_s",
        "hsv_v",
        "erasing",
        "close_mosaic",
    ]:
        if key in tuned:
            params[key] = tuned[key]
            print(f"  Tuned: {key} = {tuned[key]}")

    # box, cls, dfl için: tune değeri varsa ortalamasını al
    for key in ["box", "cls", "dfl"]:
        if key in tuned:
            # Tuned ve optimize edilmiş arasında ortalama
            params[key] = (params[key] + tuned[key]) / 2
            print(f"  Blended: {key} = {params[key]:.3f}")

print("\nKullanılan parametreler:")
for k, v in params.items():
    print(f"  {k:20s} = {v}")

# ──────────────────────────────────────────────
# Model eğitimi
# ──────────────────────────────────────────────
model = YOLO(MODEL_PT)

model.train(
    data=DATA_YAML,
    epochs=400,
    imgsz=640,  # SAHI patch'ler 640×640
    batch=32,  # SAHI ile patch sayısı arttığı için batch artırılabilir
    device=[0, 1],
    workers=4,
    cache="ram",
    pretrained=True,
    optimizer="AdamW",
    cos_lr=True,
    amp=True,
    single_cls=False,
    patience=60,  # ↑ Daha sabırlı early stopping
    save=True,
    save_period=50,
    val=True,
    plots=True,
    verbose=True,
    # ── Optimize edilmiş parametreler ──
    lr0=params["lr0"],
    lrf=params["lrf"],
    warmup_epochs=params["warmup_epochs"],
    box=params["box"],
    cls=params["cls"],
    dfl=params["dfl"],
    copy_paste=params["copy_paste"],
    degrees=params["degrees"],
    scale=params["scale"],
    hsv_h=params["hsv_h"],
    hsv_s=params["hsv_s"],
    hsv_v=params["hsv_v"],
    erasing=params["erasing"],
    close_mosaic=params["close_mosaic"],
    mixup=params["mixup"],
    perspective=params["perspective"],
    # ── Sabit augmentation ──
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    # ── Output ──
    project=PROJECT,
    name="sahi_optimized",
    exist_ok=True,
)

print("\n" + "=" * 60)
print("Eğitim tamamlandı!")
print(f"Sonuçlar: {PROJECT}/sahi_optimized/")
print(f"\nSonraki adım: python evaluate_tta.py")
print("=" * 60)
