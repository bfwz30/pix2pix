# preprocess_aligned.py
# 将 datasets/rhino_aligned/{train,val} 下的 1200x1600 拼接图
# 转换成合法的 aligned 尺寸：
#   - "resize" 模式: 缩放到 1024x1536 (-> A,B: 512x1536)
#   - "pad" 模式: 补边到 1280x1664 (-> A,B: 640x1664)

from pathlib import Path
from PIL import Image

MODE = "resize"  # 改成 "pad" 可切换方案B

def process_dir(d: Path):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for p in sorted(d.iterdir()):
        if p.suffix not in exts:
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size  # 原图应是 (1200,1600)

        if MODE == "resize":
            # 直接缩放到 1024x1536
            im = im.resize((1024, 1536), Image.BICUBIC)

        elif MODE == "pad":
            # 等比保持原尺寸，居中补边到 1280x1664
            target_w, target_h = 1280, 1664
            canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            x = (target_w - w) // 2
            y = (target_h - h) // 2
            canvas.paste(im, (x, y))
            im = canvas

        im.save(p, quality=95)
    print(f"Processed {d} with mode={MODE}")

if __name__ == "__main__":
    root = Path("datasets/rhino_aligned")
    for sub in ["train", "val"]:
        d = root / sub
        if d.exists():
            process_dir(d)
    print("Done! Aligned images converted.")