# #!/usr/bin/env bash
# set -euo pipefail

# ############ 配置区 ############
# GPU_ID=2
# PROJECT_DIR="$HOME/pix2pix"                 # 项目根目录
# ENV_NAME="liutao"                           # ← 你的 conda 环境名
# RUNNAME="150epoch_mask"                 # checkpoints 子目录名
# DATASET="datasets/rhino_aligned"            # 数据集根
# MASKS_DIR="$DATASET/train_masks"            # 训练 mask 目录
# VAL_MASKS_DIR="$DATASET/val_masks"          # 验证 mask 目录
# SAVE_EVERY=5                                # 每隔多少 epoch 存一次权重/评一次
# MEM_FREE=13000                               # 认为“空闲”的显存阈值 MiB
# LAMBDA_L1=70
# LAMBDA_ROI=6                               
# LAMBDA_BG=2                                 # ← 新增：非 ROI 保真
# ################################

# cd "$PROJECT_DIR"

# # ---- 激活环境（有就激活，无则跳过）----
# if command -v conda >/dev/null 2>&1; then
#   # 让非交互 shell 也能用 conda
#   source "$(conda info --base)/etc/profile.d/conda.sh" || true
#   if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
#     conda activate "${ENV_NAME}" || true
#   fi
# fi

# mkdir -p logs checkpoints results tools

# STAMP="$(date +%F_%H%M%S)"
# TRAIN_LOG="logs/train_${STAMP}_gpu${GPU_ID}.log"
# EVAL_LOG="logs/eval_${STAMP}_gpu${GPU_ID}.log"

# # ---- 把当前 Python/torch 环境写进日志，便于确认 ----
# {
#   echo "[`date`] which python: $(which python)"
#   python -V
#   python - <<'PY'
# import sys
# try:
#     import torch
#     print("torch", torch.__version__, "python", sys.version.split()[0], "cuda?", torch.cuda.is_available())
# except Exception as e:
#     print("torch import error:", e)
# PY
# } | tee -a "$TRAIN_LOG"

# # 让 python 日志不缓存，tail 能实时看到
# export PYTHONUNBUFFERED=1

# echo "[`date`] 监控 GPU ${GPU_ID}，阈值 ${MEM_FREE} MiB；项目: ${PROJECT_DIR}" | tee -a "$TRAIN_LOG"

# # ---- 等卡空闲 ----
# while true; do
#   USED=$(nvidia-smi --id=${GPU_ID} --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 999999)
#   if [[ "$USED" =~ ^[0-9]+$ ]] && [ "$USED" -lt "$MEM_FREE" ]; then
#     echo "[`date`] GPU${GPU_ID} 空闲，显存 ${USED} MiB，开始训练…" | tee -a "$TRAIN_LOG"
#     break
#   fi
#   echo "[`date`] GPU${GPU_ID} 显存 ${USED} MiB，等待中…" | tee -a "$TRAIN_LOG"
#   sleep 300
# done

# # ---- 启动训练（nohup 后台）----
# CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python train.py \
#   --dataroot "${DATASET}" \
#   --checkpoints_dir checkpoints \
#   --name "${RUNNAME}" \
#   --model pix2pix --direction AtoB \
#   --dataset_mode aligned --preprocess none \
#   --netG unet_512 --netD n_layers --n_layers_D 4 \
#   --norm instance --gan_mode lsgan --no_dropout \
#   --batch_size 1 \
#   --lr 0.0002 \
#   --n_epochs 150 --n_epochs_decay 0 \
#   --lambda_L1 ${LAMBDA_L1} \
#   --masks_dir "${MASKS_DIR}" \
#   --lambda_roi ${LAMBDA_ROI} \
#   --lambda_bg ${LAMBDA_BG} \
#   --num_threads 0 \
#   --use_mask_condition \            
#   --save_epoch_freq ${SAVE_EVERY} \
#   > "${TRAIN_LOG}" 2>&1 &



# TRAIN_PID=$!
# echo "[`date`] 训练已启动：PID=${TRAIN_PID}，日志：${TRAIN_LOG}"



cd ~/pix2pix

RUNNAME="150epoch_mask"
DATASET="datasets/rhino_aligned"
MASKS_DIR="$DATASET/train_masks"
STAMP="$(date +%F_%H%M%S)"
TRAIN_LOG="logs/train_${STAMP}_gpu2.log"

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py \
  --dataroot "${DATASET}" \
  --checkpoints_dir checkpoints \
  --name "${RUNNAME}" \
  --model pix2pix --direction AtoB \
  --dataset_mode aligned --preprocess none \
  --netG unet_512 --netD n_layers --n_layers_D 4 \
  --norm instance --gan_mode lsgan --no_dropout \
  --batch_size 1 \
  --lr 0.0002 \
  --n_epochs 150 --n_epochs_decay 0 \
  --lambda_L1 70 \
  --masks_dir "${MASKS_DIR}" \
  --lambda_roi 6 \
  --lambda_bg 2 \
  --use_mask_condition \
  --num_threads 0 \
  --save_epoch_freq 5 \
  > "${TRAIN_LOG}" 2>&1 &

echo $! > logs/last_train_pid.txt
echo "PID=$(cat logs/last_train_pid.txt)  LOG=${TRAIN_LOG}"
sleep 2
pgrep -fa "python train.py" || echo "no train.py"
tail -n 50 "${TRAIN_LOG}"