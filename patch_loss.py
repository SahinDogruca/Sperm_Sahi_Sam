"""
YOLOv12 Custom Segmentation Loss Patcher
=========================================
Kaggle ortamında ultralytics/utils/loss.py dosyasını bularak
orijinal v8SegmentationLoss sınıfını özel Combo Loss ile değiştirir.

Kullanım (Kaggle hücresinde):
  !python patch_loss.py
"""

import sys
import re
from pathlib import Path

# YOLOv12'nin Kaggle'daki konumu (Klonlandığı yer)
LOSS_FILE = Path("/kaggle/working/yolov12/ultralytics/utils/loss.py")

# Eğer standart ultralytics paketi kurulduysa buraya düşer:
if not LOSS_FILE.exists():
    import ultralytics
    LOSS_FILE = Path(ultralytics.__file__).parent / "utils" / "loss.py"

if not LOSS_FILE.exists():
    print("❌ HATA: loss.py dosyası bulunamadı!")
    sys.exit(1)

print(f"🎯 loss.py bulundu: {LOSS_FILE}")

# Orijinal dosyayı oku
content = LOSS_FILE.read_text(encoding="utf-8")

# Eğer zaten yamalandıysa uyar, ama yine de GÜNCELLE
if "CUSTOM SPERM MORPHOLOGY COMBO LOSS" in content:
    print("⚠️ Sistem daha önce yamalanmış. Eski yama yeni sürümle OVERRIDE ediliyor (Değiştiriliyor)...")

# Değiştirilecek Özel Sınıf Kodu
CUSTOM_CLASS = '''class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses with Custom SpermSeg Combo Loss."""

    def __init__(self, model):
        super().__init__(model)
        self.overlap = model.args.overlap_mask
        print("\n" + "="*60)
        print("🚀 YOLOV12 ÖZEL LOSS AKTİF: SpermSeg Combo Loss devrede!")
        print("   - Boundary (Sınır) Loss (Kenar hassasiyeti x3)")
        print("   - Focal Tversky Loss (Zor hücreler odaklı)")
        print("="*60 + "\n")

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        import torch
        import torch.nn.functional as F
        
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ segment dataset incorrectly formatted.") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss + Özel Focal Loss Enjeksiyonu (gamma=1.5)
        # Sınıflandırma dengesizliğini çözmek için argüman yerine doğrudan matematiğe eklendi
        bce_cls = self.bce(pred_scores, target_scores.to(dtype))
        probs_cls = pred_scores.sigmoid()
        focal_weight_cls = torch.abs(target_scores - probs_cls) ** 1.5
        loss[2] = (bce_cls * focal_weight_cls).sum() / target_scores_sum

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box  # seg gain (uses box gain multiplier in YOLO architecture)
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        🎯 CUSTOM SPERM MORPHOLOGY COMBO LOSS 🎯
        BCE + Boundary Loss + Focal Tversky Loss
        """
        import torch
        import torch.nn.functional as F
        from ultralytics.utils.ops import crop_mask

        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)

        # 1. STANDART BCE LOSS
        bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        cropped_bce = crop_mask(bce_loss, xyxy)

        # Olasılıklar ve Hedefleri kırpma
        probs = torch.sigmoid(pred_mask)
        cropped_probs = crop_mask(probs, xyxy)
        cropped_gt = crop_mask(gt_mask, xyxy)

        # 2. BOUNDARY (SINIR) LOSS EKLENTİSİ
        gt_unsqueeze = cropped_gt.unsqueeze(0)
        gt_pool = F.max_pool2d(gt_unsqueeze, kernel_size=3, stride=1, padding=1)
        boundary_mask = (gt_pool - gt_unsqueeze) > 0
        boundary_mask = boundary_mask.squeeze(0)

        # Kenar piksellerinde BCE kaybını 3 katına çıkar
        boundary_bce = cropped_bce * (1 + 2.0 * boundary_mask.float())
        bce_loss_per_obj = boundary_bce.mean(dim=(1, 2)) / area

        # 3. FOCAL TVERSKY LOSS
        TP = (cropped_probs * cropped_gt).sum(dim=(1, 2))
        FP = (cropped_probs * (1 - cropped_gt)).sum(dim=(1, 2))
        FN = ((1 - cropped_probs) * cropped_gt).sum(dim=(1, 2))

        alpha = 0.7   # False Negative cezası
        beta = 0.3    # False Positive cezası
        gamma = 2.0   # Focal gamma
        smooth = 1e-6

        tversky = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
        focal_tversky_loss = (1 - tversky) ** gamma

        # 4. KOMBİNASYON (%50 Boundary BCE + %50 Focal Tversky)
        total_loss_per_obj = 0.5 * bce_loss_per_obj + 0.5 * focal_tversky_loss

        return total_loss_per_obj.sum()

    def calculate_segmentation_loss(
        self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, overlap
    ):
        import torch
        from ultralytics.utils.ops import xyxy2xywh
        
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()

        return loss / fg_mask.sum()
'''

# Regex ile orijinal sınıfı bul ve değiştir
# class v8SegmentationLoss ile başlar, bir sonraki class tanımına veya dosya sonuna kadar eşleşir
pattern = re.compile(r"^class v8SegmentationLoss\(v8DetectionLoss\):.*?(?=^class |\Z)", re.MULTILINE | re.DOTALL)

if not pattern.search(content):
    print("❌ HATA: v8SegmentationLoss sınıfı dosyada bulunamadı!")
    sys.exit(1)

new_content = pattern.sub(CUSTOM_CLASS + "\n", content)

# Dosyayı kaydet
LOSS_FILE.write_text(new_content, encoding="utf-8")

print("✅ BAŞARILI: YOLOv12 kayıp fonksiyonu (loss.py) başarıyla güncellendi!")
print("   - Boundary (Sınır) Loss eklendi.")
print("   - Focal Tversky Loss eklendi.")
print("   Artık eğitimi başlatabilirsiniz.")
