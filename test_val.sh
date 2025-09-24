#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=~/pix2pix
RUNNAME=150epoch
DATASET=datasets/rhino_aligned
RESULTS_DIR=$PROJECT_DIR/results/${RUNNAME}_val

mkdir -p "$RESULTS_DIR"

for E in {105..150..5}; do
  echo "===> Testing epoch $E"
  CUDA_VISIBLE_DEVICES=2 python test.py \
    --dataroot "$DATASET" \
    --checkpoints_dir "$PROJECT_DIR/checkpoints" \
    --name "$RUNNAME" \
    --model pix2pix --direction AtoB \
    --dataset_mode aligned --preprocess none \
    --netG unet_512 \
    --norm instance --no_dropout \
    --phase val \
    --epoch $E \
    --num_test 999999 \
    --masks_dir "$DATASET/val_masks" \
    --results_dir "$RESULTS_DIR/epoch_${E}"
done