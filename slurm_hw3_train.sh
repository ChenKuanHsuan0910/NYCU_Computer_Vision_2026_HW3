#!/bin/bash
# =============================================================================
# NYCU Visual Recognition 2026 Spring — HW3
# Mask R-CNN Instance Segmentation (Compliant: ImageNet backbone only)
#
# Usage:
#   sbatch slurm_train_hw3.sh smoke      # quick end-to-end test (2 epochs)
#   sbatch slurm_train_hw3.sh full       # train/val split, 100 epochs
#   sbatch slurm_train_hw3.sh train_all  # all 209 images, 120 epochs (final)
#
# MODE can also be set via the variable below instead of CLI arg.
# =============================================================================

#SBATCH -J vr_hw3_maskrcnn
#SBATCH -o /home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw3/logs/slurm_%j.out
#SBATCH -e /home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw3/logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=h200q

# =============================================================================
# MODE: smoke | full | train_all
#   smoke     - 2 epochs, end-to-end sanity check
#   full      - 100 epochs, train/val split, best by val loss
#   train_all - 120 epochs, all 209 images, final submission
# =============================================================================
MODE=${1:-train_all}   # accept CLI arg or default to train_all

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load anaconda
module load slurm
module load nvidia-hpc
module load nvhpc-hpcx-cuda12
module list

eval "$(conda shell.bash hook)"
conda deactivate || true
conda activate Visual_Recognition

export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# Paths  (adjust PROJECT_DIR to your actual HW3 folder on the cluster)
# ---------------------------------------------------------------------------
PROJECT_DIR="/home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw3"
DATA_ROOT="${PROJECT_DIR}/hw3-data-release"
OUTPUT_BASE="${PROJECT_DIR}/outputs"
RUN_DIR="${OUTPUT_BASE}/run_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_DIR}/logs"
SCRIPT="${PROJECT_DIR}/hw3_maskrcnn.py"

mkdir -p "${OUTPUT_BASE}"
mkdir -p "${RUN_DIR}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

echo "======================================================"
echo " PROJECT_DIR : ${PROJECT_DIR}"
echo " DATA_ROOT   : ${DATA_ROOT}"
echo " RUN_DIR     : ${RUN_DIR}"
echo " MODE        : ${MODE}"
echo " Job ID      : ${SLURM_JOB_ID}"
echo " Node        : $(hostname)"
echo " GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "======================================================"

# ---------------------------------------------------------------------------
# Verify data exists
# ---------------------------------------------------------------------------
if [ ! -d "${DATA_ROOT}/train" ]; then
    echo "[ERROR] Train directory not found: ${DATA_ROOT}/train"
    exit 1
fi
if [ ! -d "${DATA_ROOT}/test_release" ]; then
    echo "[ERROR] Test directory not found: ${DATA_ROOT}/test_release"
    exit 1
fi
if [ ! -f "${DATA_ROOT}/test_image_name_to_ids.json" ]; then
    echo "[ERROR] Not found: ${DATA_ROOT}/test_image_name_to_ids.json"
    exit 1
fi
TRAIN_COUNT=$(ls -d "${DATA_ROOT}/train"/*/ 2>/dev/null | wc -l)
TEST_COUNT=$(ls "${DATA_ROOT}/test_release"/*.tif 2>/dev/null | wc -l)
echo "[INFO] Data check passed. train=${TRAIN_COUNT}  test=${TEST_COUNT}"

# ---------------------------------------------------------------------------
# Install / verify Python dependencies
# ---------------------------------------------------------------------------
echo "[INFO] Checking Python dependencies..."
pip install --quiet tifffile imagecodecs pycocotools tqdm "opencv-python-headless" 2>/dev/null || \
pip install --quiet tifffile imagecodecs pycocotools tqdm opencv-python 2>/dev/null
python -c "
import torch, torchvision, tifffile, pycocotools, numpy, cv2, tqdm
print('[INFO] All packages OK')
print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'  torchvision={torchvision.__version__}')
"

# ---------------------------------------------------------------------------
# Hyper-parameters per MODE
# ---------------------------------------------------------------------------
case "${MODE}" in

# -----------------------------------------------------------------------
smoke)
    echo "[INFO] MODE=smoke: 2-epoch end-to-end test"
    EPOCHS=2
    BATCH_SIZE=2
    LR=2e-4
    WEIGHT_DECAY=1e-4
    NUM_WORKERS=4
    IMG_MIN=800
    IMG_MAX=1333
    SCORE_THRESH=0.05
    TRAIN_ALL_FLAG=""
    ;;

# -----------------------------------------------------------------------
full)
    echo "[INFO] MODE=full: 100 epochs, train/val split"
    EPOCHS=100
    BATCH_SIZE=2
    LR=2e-4
    WEIGHT_DECAY=1e-4
    NUM_WORKERS=8
    IMG_MIN=800
    IMG_MAX=1333
    SCORE_THRESH=0.05
    TRAIN_ALL_FLAG=""
    ;;

# -----------------------------------------------------------------------
train_all)
    echo "[INFO] MODE=train_all: 200 epochs, all 209 images, multi-scale"
    EPOCHS=200
    BATCH_SIZE=2
    LR=2e-4
    WEIGHT_DECAY=1e-4
    NUM_WORKERS=8
    IMG_MIN=800
    IMG_MAX=1333
    SCORE_THRESH=0.03
    TRAIN_ALL_FLAG="--train_all --multiscale"
    ;;

*)
    echo "[ERROR] Unknown MODE=${MODE}. Use: smoke | full | train_all"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Run hw3_maskrcnn.py
# ---------------------------------------------------------------------------
echo ""
echo "[INFO] Starting hw3_maskrcnn.py ..."

python "${SCRIPT}" \
    --data_root       "${DATA_ROOT}" \
    --output_dir      "${RUN_DIR}" \
    --mode            both \
    --epochs          "${EPOCHS}" \
    --batch_size      "${BATCH_SIZE}" \
    --lr              "${LR}" \
    --weight_decay    "${WEIGHT_DECAY}" \
    --num_workers     "${NUM_WORKERS}" \
    --seed            42 \
    --amp \
    --val_ratio       0.2 \
    --image_min_size  "${IMG_MIN}" \
    --image_max_size  "${IMG_MAX}" \
    --score_thresh    "${SCORE_THRESH}" \
    --min_area        5 \
    --use_aug \
    --use_tta \
    --optimizer       adamw \
    ${TRAIN_ALL_FLAG}

echo ""
echo "[INFO] hw3_maskrcnn.py finished."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
BEST_CKPT="${RUN_DIR}/best.pt"
LATEST_CKPT="${RUN_DIR}/latest.pt"
JSON_PATH="${RUN_DIR}/test-results.json"
ZIP_PATH="${RUN_DIR}/submission.zip"

echo ""
echo "======================================================"
echo " Run Summary"
echo "======================================================"
echo " Checkpoint (best):   ${BEST_CKPT}"
echo " Checkpoint (latest): ${LATEST_CKPT}"
echo " test-results.json:   ${JSON_PATH}"
echo " submission.zip:      ${ZIP_PATH}"

if [ -f "${BEST_CKPT}" ]; then
    echo "[OK] best.pt found"
else
    echo "[WARN] best.pt not found"
fi

if [ -f "${JSON_PATH}" ]; then
    PRED_COUNT=$(python3 -c "import json; d=json.load(open('${JSON_PATH}')); print(len(d))")
    echo "[OK] test-results.json found | predictions: ${PRED_COUNT}"
else
    echo "[WARN] test-results.json not found"
fi

if [ -f "${ZIP_PATH}" ]; then
    echo "[OK] submission.zip found"
    python3 -c "
import zipfile
with zipfile.ZipFile('${ZIP_PATH}', 'r') as zf:
    for name in zf.namelist():
        info = zf.getinfo(name)
        print(f'  {name}  ({info.file_size} bytes)')
"
else
    echo "[WARN] submission.zip not found"
fi

echo ""
echo "======================================================"
echo " DONE. Upload ${ZIP_PATH} to CodaBench."
echo "======================================================"
