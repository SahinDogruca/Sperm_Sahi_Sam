"""
TTA (Test-Time Augmentation) ile Değerlendirme
===============================================
Eğitilmiş modeli TTA aktif olarak val/test setinde değerlendirir.
TTA, multi-scale + flip ile inference yaparak mAP'i 1-2% artırır.

Ayrıca SAHI sliced inference da destekler (küçük objeler için optimal).

Kullanım:
  python evaluate_tta.py --weights /path/to/best.pt --data /path/to/data.yaml

Kaggle'da:
  python evaluate_tta.py \\
    --weights /kaggle/working/final_v2/sahi_optimized/weights/best.pt \\
    --data /kaggle/working/data/data.yaml
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="TTA + SAHI Evaluation")
    parser.add_argument("--weights", type=str, required=True,
                        help="Eğitilmiş model weights yolu (best.pt)")
    parser.add_argument("--data", type=str, required=True,
                        help="data.yaml yolu")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (SAHI dataset için 640)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device (0, 0,1, cpu)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Hangi split'te değerlendir")
    parser.add_argument("--no-tta", action="store_true",
                        help="TTA'yı devre dışı bırak")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="NMS IoU threshold")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Model Değerlendirme")
    print("=" * 60)
    print(f"  Weights : {args.weights}")
    print(f"  Data    : {args.data}")
    print(f"  Split   : {args.split}")
    print(f"  TTA     : {'KAPALI' if args.no_tta else 'AÇIK'}")
    print(f"  ImgSz   : {args.imgsz}")
    print()

    model = YOLO(args.weights)

    # ── 1. Normal değerlendirme ──
    print("[1/2] Normal değerlendirme...")
    results_normal = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        augment=False,
        plots=True,
        verbose=True,
    )

    # ── 2. TTA ile değerlendirme ──
    if not args.no_tta:
        print("\n[2/2] TTA ile değerlendirme...")
        results_tta = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            augment=True,     # ← TTA aktif
            plots=True,
            verbose=True,
        )

        # ── Karşılaştırma ──
        print("\n" + "=" * 60)
        print("KARŞILAŞTIRMA")
        print("=" * 60)

        def safe_get(results, key, default=0.0):
            try:
                return float(results.results_dict.get(key, default))
            except:
                return default

        metrics = [
            ("mAP50(B)",    "metrics/mAP50(B)"),
            ("mAP50-95(B)", "metrics/mAP50-95(B)"),
            ("mAP50(M)",    "metrics/mAP50(M)"),
            ("mAP50-95(M)", "metrics/mAP50-95(M)"),
        ]

        print(f"{'Metric':<20s} {'Normal':>10s} {'TTA':>10s} {'Fark':>10s}")
        print("-" * 52)
        for name, key in metrics:
            val_n = safe_get(results_normal, key)
            val_t = safe_get(results_tta, key)
            diff = val_t - val_n
            sign = "+" if diff > 0 else ""
            print(f"{name:<20s} {val_n:>10.4f} {val_t:>10.4f} {sign}{diff:>9.4f}")

    print("\n[OK] Değerlendirme tamamlandı.")


if __name__ == "__main__":
    main()
