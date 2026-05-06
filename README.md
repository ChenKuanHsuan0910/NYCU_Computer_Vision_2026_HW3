# NYCU Computer Vision 2026 — HW3 Instance Segmentation

- **Student ID:** 313552026
- **Name:** 陳冠瑄

---

## Introduction

This project implements instance segmentation using **Mask R-CNN** (ResNet-50 + FPN v2 backbone) for the NYCU Visual Recognition HW3 task. The goal is to detect and segment individual cell instances across 4 cell classes from fluorescence microscopy TIFF images.

Key features of this implementation:
- ImageNet-pretrained ResNet-50 backbone (no COCO weights, fully compliant)
- All detection and mask heads trained from scratch
- Multi-scale training (640–832px) + strong data augmentation
- Test-Time Augmentation (2× horizontal flip) during inference
- ~45.9M trainable parameters (< 200M limit)

**Final AP50 score: 0.5061**

---

## Environment Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (tested on NVIDIA H200)
- Conda environment

### Install Dependencies

```bash
conda create -n hw3_env python=3.9
conda activate hw3_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tifffile imagecodecs pycocotools tqdm opencv-python-headless
```

Or use the provided environment file:

```bash
conda env create -f environment.yml
conda activate Visual_Recognition
```

### Dataset Structure

```
hw3-data-release/
├── train/
│   ├── <uuid>/
│   │   ├── image.tif
│   │   ├── class1.tif
│   │   ├── class2.tif
│   │   ├── class3.tif
│   │   └── class4.tif
│   └── ...
├── test_release/
│   └── *.tif
└── test_image_name_to_ids.json
```

---

## Usage

### Training

Train on all 209 samples with multi-scale augmentation for 200 epochs:

```bash
python hw3_maskrcnn.py \
    --data_root ./hw3-data-release \
    --output_dir ./outputs/my_run \
    --mode train_all \
    --epochs 200 \
    --batch_size 2 \
    --lr 2e-4 \
    --optimizer adamw \
    --amp \
    --use_aug \
    --multiscale \
    --seed 42
```

Or use the provided SLURM script:

```bash
sbatch slurm_hw3_train.sh train_all
```

### Inference

Run inference with 2× TTA on the test set:

```bash
python hw3_maskrcnn.py \
    --data_root ./hw3-data-release \
    --output_dir ./outputs/my_run \
    --checkpoint ./outputs/my_run/best.pt \
    --mode infer \
    --image_min_size 800 \
    --image_max_size 1333 \
    --score_thresh 0.03 \
    --use_tta \
    --seed 42
```

Or use the provided SLURM inference script:

```bash
sbatch slurm_hw3_infer.sh
```

The output `submission.zip` will be located in `--output_dir`.

### SLURM Notes

- Partition: `h200q`
- Memory: `96G` required for inference with TTA (due to large TIFF images)
- Training memory: `64G` is sufficient

---

## Performance Snapshot

![Leaderboard](images/leaderboard.png)

| Setting | AP50 |
|---|---|
| Baseline (150 epochs, 2× TTA) | 0.4468 |
| 200 epochs, multi-scale, strong aug (no TTA) | 0.4451 |
| **200 epochs, multi-scale, strong aug + 2× TTA** | **0.5061** |


