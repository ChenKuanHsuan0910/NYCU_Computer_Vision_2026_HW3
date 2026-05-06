#!/bin/bash
#SBATCH -J vr_hw3_infer
#SBATCH -o /home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw3/logs/slurm_%j.out
#SBATCH -e /home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw3/logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gpu:1
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --partition=h200q

set -euo pipefail

module purge
module load anaconda
module load slurm
module load nvidia-hpc
module load nvhpc-hpcx-cuda12

eval "$(conda shell.bash hook)"
conda deactivate || true
conda activate Visual_Recognition

export PYTHONNOUSERSITE=1

PROJECT_DIR="/home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw3"
DATA_ROOT="${PROJECT_DIR}/hw3-data-release"
CKPT="${PROJECT_DIR}/outputs/run_52277_20260429_112429/best.pt"
OUTPUT_DIR="${PROJECT_DIR}/outputs/run_52277_20260429_112429"
SCRIPT="${PROJECT_DIR}/hw3_maskrcnn.py"

echo "======================================================"
echo " Inference only (resume from completed training)"
echo " CKPT      : ${CKPT}"
echo " OUTPUT    : ${OUTPUT_DIR}"
echo " GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "======================================================"

pip install --quiet tifffile imagecodecs pycocotools tqdm "opencv-python-headless" 2>/dev/null || \
pip install --quiet tifffile imagecodecs pycocotools tqdm opencv-python 2>/dev/null

python "${SCRIPT}" \
    --data_root      "${DATA_ROOT}" \
    --output_dir     "${OUTPUT_DIR}" \
    --checkpoint     "${CKPT}" \
    --mode           infer \
    --image_min_size 800 \
    --image_max_size 1333 \
    --score_thresh   0.03 \
    --use_tta \
    --seed           42

echo ""
echo "======================================================"
echo " submission.zip: ${OUTPUT_DIR}/submission.zip"
echo " Upload to CodaBench!"
echo "======================================================"
