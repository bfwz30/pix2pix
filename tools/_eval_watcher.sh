#!/usr/bin/env bash
set -euo pipefail
GPU_ID="$1"
RUNNAME="$2"
SAVE_EVERY="$3"
VAL_MASKS_DIR="$4"
EVAL_LOG="$5"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ★ 新增：激活 conda 环境（和训练用同一个）
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate liutao || true
fi

LAST_DONE=0
echo "[`date`] 评估观察者启动（GPU${GPU_ID}）" | tee -a "$EVAL_LOG"

while true; do
  CKPT_DIR="checkpoints/${RUNNAME}"
  if [ -d "$CKPT_DIR" ]; then
    LATEST=$(ls "$CKPT_DIR" | grep -E '^[0-9]+_net_G\.pth$' | sed -E 's/_net_G\.pth//g' | sort -n | tail -1 || echo 0)
    if [[ "$LATEST" =~ ^[0-9]+$ ]] && [ "$LATEST" -gt "$LAST_DONE" ]; then
      START=$(( (LAST_DONE/SAVE_EVERY + 1) * SAVE_EVERY ))
      [ "$START" -eq 0 ] && START=$SAVE_EVERY
      for ep in $(seq $START $SAVE_EVERY $LATEST); do
        echo "[`date`] 评估 epoch ${ep}" | tee -a "$EVAL_LOG"

        # ★★ 关键修正：加 --phase val，并保持与你训练一致的 preprocess/dataset_mode
        CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py \
          --dataroot datasets/rhino_aligned \
          --checkpoints_dir checkpoints \
          --name "${RUNNAME}" \
          --model pix2pix --direction AtoB \
          --dataset_mode aligned --preprocess none \
          --phase val --num_test -1 \
          --epoch ${ep} \
          --results_dir results/ep${ep} >> "$EVAL_LOG" 2>&1 || true

        # 计算 L1/SSIM/LPIPS（含 Masked）
        python tools/evaluate.py \
          --results_root results \
          --epoch ${ep} \
          --mask_dir "${VAL_MASKS_DIR}" \
          --out_csv metrics_val.csv >> "$EVAL_LOG" 2>&1 || true

        echo "[`date`] 完成评估 epoch ${ep}" | tee -a "$EVAL_LOG"
        LAST_DONE=$ep
      done
    fi
  fi
  sleep 60
done
