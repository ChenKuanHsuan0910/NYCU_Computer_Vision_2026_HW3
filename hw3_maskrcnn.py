"""
NYCU Visual Recognition using Deep Learning 2026 Spring - HW3
Instance Segmentation with Mask R-CNN (Compliant Version)

Compliance:
  - ImageNet pretrained backbone (ResNet-50) only
  - NO COCO pretrained detection/segmentation weights
  - Detection and mask heads trained from scratch on HW3 dataset
  - Trainable parameters < 200M
  - No external data
"""

import argparse
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tifffile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import nms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Guard: never allow COCO pretrained weights to slip in
# ---------------------------------------------------------------------------
try:
    from torchvision.models.detection.mask_rcnn import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
    )
    _FORBIDDEN = {
        MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
    }
except Exception:
    _FORBIDDEN = set()


def _check_no_coco_weights(w):
    if w in _FORBIDDEN or w == "DEFAULT":
        raise ValueError(
            "[COMPLIANCE ERROR] COCO pretrained detection weights are forbidden!"
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_tif_as_rgb_tensor(path: str) -> torch.Tensor:
    """Load a .tif image and return a float32 [3, H, W] tensor in [0, 1]."""
    img = tifffile.imread(path)  # numpy array

    # Handle different channel configurations
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.shape[2] >= 3:
            img = img[:, :, :3]
        # if shape is (C, H, W)
        elif img.shape[0] in (1, 3, 4):
            img = img.transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = np.concatenate([img, img, img], axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # Normalize to [0, 1]
    img = img.astype(np.float32)
    max_val = img.max()
    if max_val > 1.0:
        # Infer bit depth
        if max_val > 255.0:
            img = img / 65535.0
        else:
            img = img / 255.0

    # [H, W, 3] -> [3, H, W]
    tensor = torch.from_numpy(img.transpose(2, 0, 1))
    return tensor


def load_mask_tif(path: str) -> np.ndarray:
    """Load a single-channel mask tif. Returns [H, W] uint32 array."""
    if not os.path.exists(path):
        return None
    m = tifffile.imread(path)
    if m.ndim == 3:
        m = m[:, :, 0] if m.shape[2] < m.shape[0] else m[0]
    return m.astype(np.int32)


def compute_bbox_from_mask(mask_2d: np.ndarray):
    """mask_2d: binary [H, W]. Returns [x1, y1, x2, y2] or None."""
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax + 1), float(rmax + 1)]


def apply_augmentation(image: torch.Tensor, masks: list, boxes: list, labels: list):
    """
    Synchronised augmentation for image [3,H,W], list of binary masks [H,W], boxes, labels.
    Returns augmented (image, masks, boxes, labels).
    """
    # Random scale jitter: resize image + masks between 0.75x ~ 1.25x
    if random.random() > 0.3:
        scale = random.uniform(0.75, 1.25)
        H0, W0 = image.shape[1], image.shape[2]
        new_H = max(64, int(H0 * scale))
        new_W = max(64, int(W0 * scale))
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False
        ).squeeze(0)
        new_masks = []
        new_boxes = []
        for m, b in zip(masks, boxes):
            m_t = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
            m_r = torch.nn.functional.interpolate(m_t, size=(new_H, new_W), mode='nearest').squeeze().numpy().astype(np.uint8)
            new_masks.append(m_r)
            rb = compute_bbox_from_mask(m_r)
            if rb is None:
                rb = [b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale]
            new_boxes.append(rb)
        masks = new_masks
        boxes = new_boxes

    # Horizontal flip
    if random.random() > 0.5:
        image = torch.flip(image, dims=[2])
        masks = [np.fliplr(m) for m in masks]
        W = image.shape[2]
        boxes = [[W - b[2], b[1], W - b[0], b[3]] for b in boxes]

    # Vertical flip
    if random.random() > 0.5:
        image = torch.flip(image, dims=[1])
        masks = [np.flipud(m) for m in masks]
        H = image.shape[1]
        boxes = [[b[0], H - b[3], b[2], H - b[1]] for b in boxes]

    # Random 90-degree rotation
    k = random.choice([0, 1, 2, 3])
    if k > 0:
        image = torch.rot90(image, k=k, dims=[1, 2])
        rot_masks = []
        rot_boxes = []
        H, W = image.shape[1], image.shape[2]
        for m, b in zip(masks, boxes):
            rm = np.rot90(m, k=k)
            rot_masks.append(rm)
            rb = compute_bbox_from_mask(rm)
            if rb is None:
                rb = [0.0, 0.0, 1.0, 1.0]
            rot_boxes.append(rb)
        masks = rot_masks
        boxes = rot_boxes

    # Stronger color jitter (image only)
    cj = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08)
    image = cj(image)

    # Random Gaussian noise
    if random.random() > 0.5:
        noise = torch.randn_like(image) * 0.02
        image = torch.clamp(image + noise, 0.0, 1.0)

    # Random grayscale (helps model be robust to staining variations)
    if random.random() > 0.85:
        gray = image.mean(dim=0, keepdim=True).expand_as(image)
        image = gray

    return image, masks, boxes, labels


class HW3CellDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train",
                 val_ratio: float = 0.2, seed: int = 42,
                 train_all: bool = False, use_aug: bool = True,
                 min_area: int = 5):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.use_aug = use_aug and (split == "train")
        self.min_area = min_area

        train_dir = Path(data_root) / "train"
        all_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])

        if train_all or split == "all":
            self.samples = all_dirs
        else:
            set_seed(seed)
            indices = list(range(len(all_dirs)))
            random.shuffle(indices)
            n_val = max(1, int(len(all_dirs) * val_ratio))
            if split == "val":
                self.samples = [all_dirs[i] for i in indices[:n_val]]
            else:
                self.samples = [all_dirs[i] for i in indices[n_val:]]
            # restore seed state isn't critical here

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        image_path = str(sample_dir / "image.tif")
        image = load_tif_as_rgb_tensor(image_path)  # [3, H, W]
        H, W = image.shape[1], image.shape[2]

        all_masks = []
        all_labels = []
        all_boxes = []

        for cls_idx in range(1, 5):
            mask_path = str(sample_dir / f"class{cls_idx}.tif")
            mask = load_mask_tif(mask_path)
            if mask is None:
                continue
            unique_vals = np.unique(mask)
            unique_vals = unique_vals[unique_vals != 0]
            for val in unique_vals:
                bin_mask = (mask == val).astype(np.uint8)
                area = bin_mask.sum()
                if area < self.min_area:
                    continue
                bbox = compute_bbox_from_mask(bin_mask)
                if bbox is None:
                    continue
                all_masks.append(bin_mask)
                all_labels.append(cls_idx)
                all_boxes.append(bbox)

        # Augment
        if self.use_aug and len(all_masks) > 0:
            image, all_masks, all_boxes, all_labels = apply_augmentation(
                image, all_masks, all_boxes, all_labels
            )
            # Re-derive boxes after augmentation to be safe
            new_boxes = []
            new_masks = []
            new_labels = []
            for m, b, l in zip(all_masks, all_boxes, all_labels):
                m_cont = np.ascontiguousarray(m)
                area = m_cont.sum()
                if area < self.min_area:
                    continue
                bbox = compute_bbox_from_mask(m_cont)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                new_masks.append(m_cont)
                new_boxes.append(bbox)
                new_labels.append(l)
            all_masks = new_masks
            all_boxes = new_boxes
            all_labels = new_labels

        if len(all_masks) == 0:
            # Dummy target to avoid crash
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
            return image, target

        boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
        labels_t = torch.tensor(all_labels, dtype=torch.int64)
        masks_t = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(m)) for m in all_masks], dim=0
        ).to(torch.uint8)
        areas = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": torch.zeros((len(all_labels),), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 5,
                image_min_size: int = 800,
                image_max_size: int = 1333,
                multiscale: bool = False) -> nn.Module:
    """
    Build a compliant Mask R-CNN model.
    - ImageNet pretrained backbone only
    - NO COCO weights
    - Box and mask heads trained from scratch
    """
    # Safety check
    _check_no_coco_weights(None)  # just verify guard works

    print("[COMPLIANCE] Using ImageNet-pretrained backbone only.")
    print("[COMPLIANCE] Detection and mask heads are trained from scratch.")
    print("[COMPLIANCE] No COCO pretrained detection/segmentation weights are loaded.")

    # Multi-scale training: randomly pick from list during training forward pass
    if multiscale:
        min_size = (640, 672, 704, 736, 768, 800, 832)
        print("[INFO] Multi-scale training enabled:", min_size)
    else:
        min_size = image_min_size

    model = maskrcnn_resnet50_fpn_v2(
        weights=None,  # NO COCO weights
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2,  # ImageNet only
        num_classes=num_classes,
        min_size=min_size,
        max_size=image_max_size,
        # RPN tuning for small/dense objects
        rpn_pre_nms_top_n_train=4000,
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        box_detections_per_img=512,
        box_score_thresh=0.03,
        box_nms_thresh=0.45,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
    )

    # Replace anchor generator with smaller anchors tuned for small cells
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=aspect_ratios
    )

    # Replace box predictor (fresh init)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor with larger hidden layer for better mask quality
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512  # increased from 256 for better mask quality
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[COMPLIANCE] Trainable parameters: {total_params:,}")
    assert total_params < 200_000_000, (
        f"[COMPLIANCE ERROR] Trainable params {total_params} >= 200M!"
    )
    print(f"[COMPLIANCE] Parameter count OK (< 200M).")

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, scaler, epoch, use_amp):
    model.train()
    total_loss = 0.0
    loss_keys = [
        "loss_classifier", "loss_box_reg", "loss_mask",
        "loss_objectness", "loss_rpn_box_reg"
    ]
    accum = {k: 0.0 for k in loss_keys}
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        for k in loss_keys:
            accum[k] += loss_dict.get(k, torch.tensor(0.0)).item()
        n += 1
        pbar.set_postfix(loss=f"{losses.item():.4f}")

    avg = total_loss / max(n, 1)
    parts = " | ".join(f"{k}={accum[k]/max(n,1):.4f}" for k in loss_keys)
    print(f"  [Epoch {epoch}] total_loss={avg:.4f} | {parts}")
    return avg


@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp):
    """Compute average loss on validation set using model in train mode."""
    model.train()  # need train mode to get losses
    total_loss = 0.0
    n = 0
    for images, targets in tqdm(loader, desc="Val loss", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
        total_loss += losses.item()
        n += 1
    return total_loss / max(n, 1)


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, args):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }, path)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Datasets
    train_dataset = HW3CellDataset(
        args.data_root,
        split="all" if args.train_all else "train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_all=args.train_all,
        use_aug=args.use_aug,
        min_area=args.min_area,
    )
    print(f"Train samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if not args.train_all:
        val_dataset = HW3CellDataset(
            args.data_root,
            split="val",
            val_ratio=args.val_ratio,
            seed=args.seed,
            use_aug=False,
            min_area=args.min_area,
        )
        print(f"Val samples: {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Model
    model = build_model(
        num_classes=5,
        image_min_size=args.image_min_size,
        image_max_size=args.image_max_size,
        multiscale=args.multiscale,
    )
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )

    # LR scheduler: linear warmup for first 5 epochs, then cosine annealing
    warmup_epochs = min(5, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, scaler, epoch, args.amp
        )
        scheduler.step()

        # Save latest
        save_checkpoint(
            os.path.join(args.output_dir, "latest.pt"),
            model, optimizer, epoch, best_val_loss, args,
        )

        # Validation
        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, device, args.amp)
            print(f"  [Epoch {epoch}] val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    os.path.join(args.output_dir, "best.pt"),
                    model, optimizer, epoch, best_val_loss, args,
                )
                print(f"  -> Saved best.pt (val_loss={best_val_loss:.4f})")

    # If train_all, save final
    if args.train_all:
        save_checkpoint(
            os.path.join(args.output_dir, "best.pt"),
            model, optimizer, args.epochs, best_val_loss, args,
        )
        print("Saved final best.pt (train_all mode)")

    print("Training complete.")
    print(f"Checkpoints saved to: {args.output_dir}")


# ---------------------------------------------------------------------------
# Inference & Submission
# ---------------------------------------------------------------------------

def mask_to_coco_rle(mask_2d: np.ndarray, height: int, width: int) -> dict:
    """Encode a binary mask as COCO RLE. mask_2d: [H, W] binary."""
    from pycocotools import mask as coco_mask

    if mask_2d.shape[0] != height or mask_2d.shape[1] != width:
        import cv2
        mask_2d = cv2.resize(
            mask_2d.astype(np.uint8), (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    fort = np.asfortranarray(mask_2d.astype(np.uint8))
    rle = coco_mask.encode(fort)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    rle["size"] = [height, width]
    return rle


def run_tta_inference(model, image_tensor, device, score_thresh):
    """Run inference with 4x TTA (orig, h-flip, v-flip, both) and merge results."""
    from torchvision.ops import nms as tv_nms

    H_orig, W_orig = image_tensor.shape[1], image_tensor.shape[2]
    results = []

    # 4 augmentation variants: (flip_h, flip_v)
    for flip_h, flip_v in [(False, False), (True, False), (False, True), (True, True)]:
        img = image_tensor.clone()
        if flip_h:
            img = torch.flip(img, dims=[2])
        if flip_v:
            img = torch.flip(img, dims=[1])

        with torch.no_grad():
            out = model([img.to(device)])[0]

        boxes = out["boxes"].cpu()
        scores = out["scores"].cpu()
        labels = out["labels"].cpu()
        masks = out["masks"].cpu()

        # Reverse flip on boxes and masks
        if flip_h:
            boxes_f = boxes.clone()
            boxes_f[:, 0] = W_orig - boxes[:, 2]
            boxes_f[:, 2] = W_orig - boxes[:, 0]
            boxes = boxes_f
            masks = torch.flip(masks, dims=[3])
        if flip_v:
            boxes_f = boxes.clone()
            boxes_f[:, 1] = H_orig - boxes[:, 3]
            boxes_f[:, 3] = H_orig - boxes[:, 1]
            boxes = boxes_f
            masks = torch.flip(masks, dims=[2])

        keep = scores >= score_thresh
        results.append({
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
            "masks": masks[keep],
        })

    # Merge all TTA results
    all_boxes = torch.cat([r["boxes"] for r in results], dim=0)
    all_scores = torch.cat([r["scores"] for r in results], dim=0)
    all_labels = torch.cat([r["labels"] for r in results], dim=0)
    all_masks = torch.cat([r["masks"] for r in results], dim=0)

    if all_boxes.shape[0] == 0:
        return {"boxes": all_boxes, "scores": all_scores,
                "labels": all_labels, "masks": all_masks}

    # Per-class NMS to de-duplicate
    final_idx = []
    for cls in all_labels.unique():
        cls_mask_idx = (all_labels == cls).nonzero(as_tuple=True)[0]
        keep = tv_nms(all_boxes[cls_mask_idx], all_scores[cls_mask_idx], iou_threshold=0.4)
        final_idx.append(cls_mask_idx[keep])
    if final_idx:
        final_idx = torch.cat(final_idx)
    else:
        final_idx = torch.tensor([], dtype=torch.long)

    return {
        "boxes": all_boxes[final_idx],
        "scores": all_scores[final_idx],
        "labels": all_labels[final_idx],
        "masks": all_masks[final_idx],
    }


def infer(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load name -> id mapping
    mapping_path = os.path.join(args.data_root, "test_image_name_to_ids.json")
    with open(mapping_path, "r") as f:
        mapping_raw = json.load(f)
    # Support both list of dicts and plain dict formats
    if isinstance(mapping_raw, list):
        name_to_id = {entry["file_name"]: entry["id"] for entry in mapping_raw}
        # Also map without extension just in case
        name_to_id.update({
            Path(entry["file_name"]).stem: entry["id"] for entry in mapping_raw
        })
    else:
        name_to_id = mapping_raw

    # Build model and load checkpoint
    model = build_model(
        num_classes=5,
        image_min_size=args.image_min_size,
        image_max_size=args.image_max_size,
        multiscale=False,  # no multi-scale needed at inference
    )

    ckpt_path = args.checkpoint
    if not ckpt_path:
        # Try to find best.pt
        ckpt_path = os.path.join(args.output_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.output_dir, "latest.pt")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Collect test images
    test_dir = Path(args.data_root) / "test_release"
    test_files = sorted(test_dir.glob("*.tif"))
    print(f"Found {len(test_files)} test images")

    results = []

    for img_path in tqdm(test_files, desc="Inference"):
        fname = img_path.stem  # filename without extension
        image_id = name_to_id.get(fname) or name_to_id.get(str(img_path.name))
        if image_id is None:
            # Try matching with extension
            image_id = name_to_id.get(img_path.name)
        if image_id is None:
            print(f"  [WARN] Cannot find image_id for {fname}, skipping.")
            continue

        img_tensor = load_tif_as_rgb_tensor(str(img_path))
        H, W = img_tensor.shape[1], img_tensor.shape[2]

        if args.use_tta:
            out = run_tta_inference(model, img_tensor, device, args.score_thresh)
        else:
            with torch.no_grad():
                out = model([img_tensor.to(device)])[0]

        boxes = out["boxes"].cpu().numpy()       # [N, 4] xyxy
        scores = out["scores"].cpu().numpy()     # [N]
        labels = out["labels"].cpu().numpy()     # [N]
        masks = out["masks"].cpu().numpy()       # [N, 1, H', W']

        for i in range(len(scores)):
            score = float(scores[i])
            if score < args.score_thresh:
                continue

            cat_id = int(labels[i])
            if cat_id == 0 or cat_id > 4:
                continue

            # bbox: xyxy -> xywh
            x1, y1, x2, y2 = boxes[i]
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            bbox_xywh = [float(x1), float(y1), bw, bh]

            # Mask: threshold at 0.5, resize to original if needed
            mask_prob = masks[i, 0]  # [H', W']
            mask_bin = (mask_prob >= 0.5).astype(np.uint8)

            if mask_bin.sum() == 0:
                continue

            rle = mask_to_coco_rle(mask_bin, H, W)

            results.append({
                "image_id": int(image_id),
                "bbox": bbox_xywh,
                "score": score,
                "category_id": cat_id,
                "segmentation": rle,
            })

    print(f"Total predictions: {len(results)}")

    # Save test-results.json
    json_path = os.path.join(args.output_dir, "test-results.json")
    with open(json_path, "w") as f:
        json.dump(results, f)
    print(f"Saved: {json_path}")

    # Create submission.zip
    zip_path = os.path.join(args.output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="test-results.json")
    print(f"Saved: {zip_path}")

    # Validate
    validate_submission_format(json_path, zip_path)

    # Final checklist
    print("\n========== FINAL CHECKLIST ==========")
    print(f"[1] test-results.json exists: {os.path.exists(json_path)}")
    print(f"[2] submission.zip exists: {os.path.exists(zip_path)}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    print(f"[3] zip contains exactly one file 'test-results.json': "
          f"{names == ['test-results.json']}")
    cats = [r["category_id"] for r in results]
    print(f"[4] all category_id in 1~4: {all(1 <= c <= 4 for c in cats)}")
    counts_ok = all(isinstance(r["segmentation"]["counts"], str) for r in results)
    print(f"[5] all RLE counts are strings: {counts_ok}")
    print("[6] No data/checkpoint/output included in code submission (manual step).")
    print("======================================\n")

    return zip_path


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_submission_format(json_path: str, zip_path: str):
    print("\n--- Validating submission format ---")
    with open(json_path) as f:
        data = json.load(f)
    assert isinstance(data, list), "test-results.json must be a list"

    required_keys = {"image_id", "bbox", "score", "category_id", "segmentation"}
    for i, r in enumerate(data):
        missing = required_keys - set(r.keys())
        assert not missing, f"Entry {i} missing keys: {missing}"
        assert isinstance(r["image_id"], int), f"Entry {i}: image_id must be int"
        assert 1 <= r["category_id"] <= 4, f"Entry {i}: category_id={r['category_id']} out of range"
        assert len(r["bbox"]) == 4, f"Entry {i}: bbox must have 4 elements"
        assert isinstance(r["score"], float), f"Entry {i}: score must be float"
        seg = r["segmentation"]
        assert isinstance(seg, dict), f"Entry {i}: segmentation must be dict"
        assert "size" in seg and "counts" in seg, f"Entry {i}: segmentation missing size/counts"
        assert len(seg["size"]) == 2, f"Entry {i}: segmentation size must be [H, W]"
        assert isinstance(seg["counts"], str), f"Entry {i}: counts must be string"

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    assert names == ["test-results.json"], \
        f"zip must contain exactly ['test-results.json'], got {names}"

    print(f"  Validation PASSED. {len(data)} predictions, zip OK.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HW3 Mask R-CNN Instance Segmentation")

    # Paths
    p.add_argument("--data_root", default="data/hw3-data-release")
    p.add_argument("--output_dir", default="outputs/run")
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    p.add_argument("--checkpoint", default="", help="Checkpoint for inference")

    # Mode
    p.add_argument("--mode", choices=["train", "infer", "both"], default="both")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--train_all", action="store_true", default=False)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")

    # Dataset
    p.add_argument("--min_area", type=int, default=5)
    p.add_argument("--use_aug", action="store_true", default=True)
    p.add_argument("--no_aug", dest="use_aug", action="store_false")

    # Model
    p.add_argument("--image_min_size", type=int, default=800)
    p.add_argument("--image_max_size", type=int, default=1333)
    p.add_argument("--multiscale", action="store_true", default=False)

    # Inference
    p.add_argument("--score_thresh", type=float, default=0.05)
    p.add_argument("--use_tta", action="store_true", default=True)
    p.add_argument("--no_tta", dest="use_tta", action="store_false")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("HW3 Mask R-CNN | NYCU VR 2026 Spring")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")

    if args.mode in ("train", "both"):
        train(args)

    if args.mode in ("infer", "both"):
        # After training, use best.pt for inference
        if not args.checkpoint:
            best = os.path.join(args.output_dir, "best.pt")
            latest = os.path.join(args.output_dir, "latest.pt")
            args.checkpoint = best if os.path.exists(best) else latest
        zip_path = infer(args)
        print(f"\nSubmission ready: {zip_path}")
        print("Upload submission.zip to CodaBench to get your AP50 score.")
