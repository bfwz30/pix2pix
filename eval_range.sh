for ep in $(seq 100 5 200); do
  echo "=== Evaluating epoch $ep ==="

  CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataroot datasets/rhino_aligned \
    --checkpoints_dir checkpoints \
    --name rhino_pix2pix_roi \
    --model pix2pix --direction AtoB \
    --dataset_mode aligned --preprocess none \
    --epoch $ep --num_test -1 \
    --results_dir results/ep${ep}

  python tools/evaluate.py \
    --results_root results \
    --epoch ${ep} \
    --mask_dir datasets/rhino_aligned/val_masks \
    --out_csv metrics_val.csv
done
