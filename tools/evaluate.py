# tools/evaluate.py
import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def try_import_lpips():
    """Try import LPIPS, return a callable (or None if unavailable)."""
    try:
        import lpips  # type: ignore
        net = lpips.LPIPS(net='vgg')  # downloads VGG weights at first run (CPU 也可以)
        net = net.to('cpu')
        def _lpips(a_rgb01: np.ndarray, b_rgb01: np.ndarray) -> float:
            # a/b: HxWx3, 0..1
            import torch
            ta = torch.from_numpy(a_rgb01.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1  # to [-1,1]
            tb = torch.from_numpy(b_rgb01.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1
            with torch.no_grad():
                v = net(ta, tb).item()
            return float(v)
        return _lpips
    except Exception:
        return None


def load_img_rgb(path: Path) -> np.ndarray:
    """Load image as float32 RGB in [0,1], shape HxWx3."""
    im = Image.open(path).convert('RGB')
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def load_mask_gray01(mask_dir: Path, stem: str) -> Optional[np.ndarray]:
    """Load mask by stem from mask_dir; return float32 [0,1] 2D array or None."""
    for ext in ('.png', '.jpg', '.jpeg'):
        p = mask_dir / f'{stem}{ext}'
        if p.exists():
            m = Image.open(p).convert('L')
            arr = np.asarray(m, dtype=np.float32) / 255.0
            return arr
    return None


def masked_l1(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    """L1 over masked region. a/b: HxWxC in [0,1]; m: HxW in [0,1]."""
    diff = np.abs(a - b)  # HxWxC
    w = np.clip(m, 0.0, 1.0)
    w_sum = w.sum()
    if w_sum < 1e-6:
        return np.nan
    return float((diff * w[..., None]).sum() / (w_sum * a.shape[2]))


def masked_ssim(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    """
    SSIM over masked region (近似做法：对整图 SSIM，再按 mask 作像素平均；
    更严格可分块/裁剪，这里采用近似实现，足够比较不同 epoch。)
    """
    try:
        s, s_map = ssim(a, b, channel_axis=2, full=True, data_range=1.0)
        w = np.clip(m, 0.0, 1.0)
        w_sum = w.sum()
        if w_sum < 1e-6:
            return np.nan
        return float((s_map * w).sum() / w_sum)
    except Exception:
        return np.nan


def ensure_images_dir(results_root: Path, epoch: Optional[int], images_dir_arg: Optional[Path]) -> Path:
    """
    Resolve images directory:
    - if images_dir_arg provided: use it
    - else: search results_root/ep{epoch}/**/images (first match)
    """
    if images_dir_arg:
        if not images_dir_arg.is_dir():
            raise SystemExit(f"[ERR] images_dir 不存在: {images_dir_arg}")
        return images_dir_arg

    if epoch is None:
        raise SystemExit("[ERR] 未提供 --images_dir，且缺少 --epoch 用于推断目录")

    ep_dir = results_root / f'ep{epoch}'
    if not ep_dir.is_dir():
        raise SystemExit(f"[ERR] {ep_dir} 不是有效目录")

    # 在 ep{epoch} 下递归找 images/
    candidates = sorted(ep_dir.glob('**/images'))
    if not candidates:
        raise SystemExit(f"[ERR] 未在 {ep_dir} 下找到 images/ 目录")
    return candidates[0]


def pair_paths(imgdir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Collect (fake_B, real_B, stem) pairs from images dir.
    stem: 原始文件名（不含后缀、不含 _fake_B / _real_B 部分）
    """
    fakes = sorted([p for p in imgdir.glob('*_fake_B.*') if p.is_file()])
    pairs = []
    for fp in fakes:
        stem = fp.name
        for suffix in ['_fake_B', '_real_A', '_real_B']:
            stem = stem.replace(suffix, '')
        stem = Path(stem).stem  # 保险再去后缀

        # 找 real_B（必须同一目录）
        real = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = imgdir / f'{stem}_real_B{ext}'
            if cand.exists():
                real = cand
                break
        if real is None:
            print(f"[WARN] 无 real_B 对应: {fp.name}", flush=True)
            continue

        pairs.append((fp, real, stem))
    return pairs


def compute_metrics(fake: np.ndarray, real: np.ndarray,
                    mask: Optional[np.ndarray],
                    lpips_fn) -> dict:
    """
    Compute global & masked metrics for a pair (fake, real).
    """
    out = {}

    # ---- Global L1/SSIM/LPIPS
    l1 = float(np.abs(fake - real).mean())
    try:
        s, _ = ssim(fake, real, channel_axis=2, full=True, data_range=1.0)
    except Exception:
        s = np.nan
    out.update({"L1": l1, "SSIM": float(s)})

    if lpips_fn is not None:
        try:
            out["LPIPS"] = float(lpips_fn(fake, real))
        except Exception:
            out["LPIPS"] = np.nan
    else:
        out["LPIPS"] = np.nan

    # ---- Masked
    if mask is not None:
        out["Masked_L1"] = masked_l1(fake, real, mask)
        out["Masked_SSIM"] = masked_ssim(fake, real, mask)
        if lpips_fn is not None:
            try:
                # 仅对 mask 区域做 LPIPS：用 mask 将未覆盖部分置为 real，近似局部对比
                mm = mask[..., None].clip(0.0, 1.0)
                fake_m = fake * mm + real * (1 - mm)
                real_m = real
                out["Masked_LPIPS"] = float(lpips_fn(fake_m, real_m))
            except Exception:
                out["Masked_LPIPS"] = np.nan
        else:
            out["Masked_LPIPS"] = np.nan
    else:
        out.update({"Masked_L1": np.nan, "Masked_SSIM": np.nan, "Masked_LPIPS": np.nan})

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results",
                    help="包含 epXXX 子目录的根目录（或忽略，用 --images_dir）")
    ap.add_argument("--epoch", type=int, default=None,
                    help="评估的 epoch（或目录名 epXXX）")
    ap.add_argument("--mask_dir", default="datasets/rhino_aligned/val_masks",
                    help="mask 目录，文件名需与 real_B/fake_B 的“主干名”一致")
    ap.add_argument("--out_csv", default="metrics_val.csv",
                    help="把结果追加写入到这个 CSV")
    ap.add_argument("--images_dir", type=str, default="",
                    help="（可选）直接指定 val_X/images 目录，优先于 results_root/epoch 推断")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    images_dir = Path(args.images_dir) if args.images_dir else None
    imgdir = ensure_images_dir(results_root, args.epoch, images_dir)

    pairs = pair_paths(imgdir)
    if not pairs:
        raise SystemExit(f"[ERR] 没有可评估的样本（找不到 *_fake_B.*）")

    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    lpips_fn = try_import_lpips()
    if lpips_fn is None:
        print("[WARN] 未安装 lpips，LPIPS 与 Masked_LPIPS 将为 NaN（pip install lpips）", flush=True)

    rows = []
    for fake_p, real_p, stem in pairs:
        fake = load_img_rgb(fake_p)
        real = load_img_rgb(real_p)
        m = load_mask_gray01(mask_dir, stem) if (mask_dir and mask_dir.is_dir()) else None

        metrics = compute_metrics(fake, real, m, lpips_fn)
        rows.append(metrics)

    # 按列求平均
    def colmean(key: str) -> float:
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else np.nan

    summary = {
        "epoch": args.epoch if args.epoch is not None else imgdir.parent.name,  # 记录个标识
        "N": len(rows),
        "L1": colmean("L1"),
        "SSIM": colmean("SSIM"),
        "LPIPS": colmean("LPIPS"),
        "Masked_L1": colmean("Masked_L1"),
        "Masked_SSIM": colmean("Masked_SSIM"),
        "Masked_LPIPS": colmean("Masked_LPIPS"),
    }

    out_csv = Path(args.out_csv)
    new_file = not out_csv.exists()
    with out_csv.open('a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "N", "L1", "SSIM", "LPIPS",
                        "Masked_L1", "Masked_SSIM", "Masked_LPIPS"]
        )
        if new_file:
            writer.writeheader()
        writer.writerow(summary)

    print(f"[OK] 写入 {out_csv} ：{summary}", flush=True)


if __name__ == "__main__":
    main()
