# split_dataset.py  （放在仓库根目录）
import os, random, shutil
from pathlib import Path

# 数据集名称（可改）
DS_NAME = os.environ.get("DS_NAME", "rhino_aligned")

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "datasets_src"
DST  = ROOT / "datasets" / DS_NAME

(TRAIN := DST / "train").mkdir(parents=True, exist_ok=True)
(VAL   := DST / "val").mkdir(parents=True, exist_ok=True)

# 收集图片（按需加其他后缀）
ex = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
images = [p for p in SRC.iterdir() if p.suffix in ex]
random.shuffle(images)

n = len(images)
train_n = int(0.8 * n)   # 80% 训练，20% 验证
train_files = images[:train_n]
val_files   = images[train_n:]

for p in train_files:
    shutil.copy2(p, TRAIN / p.name)

for p in val_files:
    shutil.copy2(p, VAL / p.name)

print(f"Total: {n} | Train: {len(train_files)} | Val: {len(val_files)}")
print("Train samples:", sorted(os.listdir(TRAIN))[:5])
print("Val samples  :", sorted(os.listdir(VAL))[:5])