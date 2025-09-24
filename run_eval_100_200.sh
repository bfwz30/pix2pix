#!/usr/bin/env bash
set -euo pipefail

# ===== 可按需修改 =====
GPU=2                                         # 选一张空闲卡（如 2）
RUN=rhino_pix2pix_roi                         # checkpoints 子目录名
DATASET=datasets/rhino_aligned                # 数据根
VAL_MASKS=${DATASET}/val_masks                # 验证集鼻部 mask
RESULTS=results                               # 结果输出根目录
CSV=metrics_val.csv                           # 指标汇总 CSV
EPOCH_BEGIN=100
EPOCH_END=200
EPOCH_STEP=5
# ===========================================

# 避免显存碎片导致的 OOM（可按需调大/调小数值）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# 0) 清理旧评估
rm -f "${CSV}"
for ep in $(seq ${EPOCH_BEGIN} ${EPOCH_STEP} ${EPOCH_END}); do
  [ -d "${RESULTS}/ep${ep}" ] && rm -rf "${RESULTS}/ep${ep}"
done

# 1) 逐 epoch 生成 fake_B（一定要用 --phase val 读取 val/）
for ep in $(seq ${EPOCH_BEGIN} ${EPOCH_STEP} ${EPOCH_END}); do
  echo "=== [gen] epoch ${ep} ==="
  CUDA_VISIBLE_DEVICES=${GPU} python test.py \
    --dataroot "${DATASET}" \
    --checkpoints_dir checkpoints \
    --name "${RUN}" \
    --model pix2pix --direction AtoB \
    --dataset_mode aligned --preprocess none \
    --phase val \
    --epoch ${ep} --num_test -1 \
    --results_dir "${RESULTS}/ep${ep}"
done

# 2) 逐 epoch 计算 L1/SSIM/LPIPS + Masked 版本，并写入 CSV
for ep in $(seq ${EPOCH_BEGIN} ${EPOCH_STEP} ${EPOCH_END}); do
  echo "=== [eval] epoch ${ep} ==="
  if [ ! -d "${RESULTS}/ep${ep}/images" ]; then
    echo "[WARN] missing ${RESULTS}/ep${ep}/images，跳过 ${ep}"
    continue
  fi
  python tools/evaluate.py \
    --results_root "${RESULTS}" \
    --epoch ${ep} \
    --mask_dir "${VAL_MASKS}" \
    --out_csv "${CSV}"
done

echo "完成，指标汇总在 ${CSV} ："
tail -n 10 "${CSV}"
