import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    """Paired dataset (A|B in one image). Also loads a per-sample mask aligned to B."""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        assert self.opt.load_size >= self.opt.crop_size
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

        # 这些开关可在命令行给（都有默认）
        self.use_mask_condition = getattr(self.opt, "use_mask_condition", False)  # 是否把 mask 拼到 A 作为条件
        self.mask_hard         = getattr(self.opt, "mask_hard", True)             # 是否把 mask 二值化
        self.mask_thresh       = getattr(self.opt, "mask_thresh", 0.5)            # 二值阈值

    def __len__(self):
        return len(self.AB_paths)

    # -------------------- 关键：按与 B 完全一致的参数变换 mask --------------------
    def _apply_mask_transform(self, mask_img: Image.Image, params):
        """
        对 mask 应用与 B 完全一致的几何变换；插值一律用 NEAREST，避免灰边。
        输出 [1,H,W] 的 float32，范围 0~1（可选二值化）。
        """
        preprocess = getattr(self.opt, "preprocess", "resize_and_crop")
        load_size  = getattr(self.opt, "load_size", None)
        crop_size  = getattr(self.opt, "crop_size", None)

        # 1) resize / scale_width / crop（与 B 同步）
        if preprocess == "resize_and_crop":
            assert load_size is not None and crop_size is not None
            mask_img = mask_img.resize((load_size, load_size), Image.NEAREST)
            x, y = params["crop_pos"]
            mask_img = mask_img.crop((x, y, x + crop_size, y + crop_size))

        elif preprocess == "scale_width_and_crop":
            assert load_size is not None and crop_size is not None
            ow, oh = mask_img.size
            if ow != load_size:
                new_h = int(round(oh * load_size / ow))
                mask_img = mask_img.resize((load_size, new_h), Image.NEAREST)
            x, y = params["crop_pos"]
            mask_img = mask_img.crop((x, y, x + crop_size, y + crop_size))

        elif preprocess == "resize":
            assert load_size is not None
            mask_img = mask_img.resize((load_size, load_size), Image.NEAREST)

        elif preprocess == "crop":
            assert crop_size is not None
            x, y = params["crop_pos"]
            mask_img = mask_img.crop((x, y, x + crop_size, y + crop_size))

        elif preprocess == "none":
            pass  # 不做几何变化

        # 2) flip（与 B 同步）
        if (not self.opt.no_flip) and params.get("flip", False):
            mask_img = TF.hflip(mask_img)

        # 3) ToTensor → float32 的 0~1
        m = torch.from_numpy(np.array(mask_img, dtype=np.uint8))  # [H,W], 0..255
        m = m.float().div(255.0)

        # 4) （可选）二值化
        if self.mask_hard:
            m = (m >= self.mask_thresh).float()

        # [1,H,W]
        return m.unsqueeze(0)

    # -------------------------------------------------------------------------

    def __getitem__(self, index):
        """Return a dict with A, B, mask_B (and optionally A_cond)."""
        # 读取 A|B 拼接图
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")

        # 拆分 A、B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # 统一参数，保证 A、B、mask 变换一致
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)  # [C,H,W], float
        B = B_transform(B)  # [C,H,W], float

        # 读取并对齐 mask（与 B 对齐）
        mask = None
        if getattr(self.opt, "masks_dir", ""):
            stem = os.path.splitext(os.path.basename(AB_path))[0]
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                mp = os.path.join(self.opt.masks_dir, stem + ext)
                if os.path.exists(mp):
                    mimg = Image.open(mp).convert("L")
                    mask = self._apply_mask_transform(mimg, transform_params)  # [1,H,W], 0~1
                    break

        # 若没有 mask，给全 0（尺寸与 B 一致）
        if mask is None:
            mask = torch.zeros_like(B[:1, :, :])  # [1,H,W]

        out = {
            "A": A,
            "B": B,
            "A_paths": AB_path,
            "B_paths": AB_path,
            "mask_B": mask,  # [1,H,W], float32, 0~1（可能是二值）
        }

        # 可选：直接返回拼好的条件输入；若你在 model 里拼，这块可不要
        if self.use_mask_condition:
            out["A_cond"] = torch.cat([A, mask], dim=0)  # [C+1,H,W]

        return out