import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """pix2pix with optional mask-conditioned inputs and ROI-aware losses."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # 保持 pix2pix 原始默认
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")

        # ⚠️ 不要在这里再注册 use_mask_condition / lambda_roi / lambda_bg / masks_dir
        # 它们已经在 BaseOptions 里统一定义了，避免 argparse 冲突

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument(
                "--lambda_L1", type=float, default=100.0,
                help="weight for full-image L1 loss"
            )

        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.device = opt.device

        # 要显示/保存的名字
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:
            self.model_names = ["G"]

        # ===== 构建网络：当 use_mask_condition 时，G/D 的输入通道数 +1 =====
        in_nc_G = opt.input_nc + (1 if opt.use_mask_condition else 0)
        in_nc_D = (opt.input_nc + (1 if opt.use_mask_condition else 0)) + opt.output_nc

        self.netG = networks.define_G(
            in_nc_G, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain
        )

        if self.isTrain:
            self.netD = networks.define_D(
                in_nc_D, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain
            )

            # 损失 & 优化器
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers += [self.optimizer_G, self.optimizer_D]

    # ------------------------- data I/O -------------------------

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)  # [N,C,H,W]
        self.real_B = input["B" if AtoB else "A"].to(self.device)  # [N,C,H,W]
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

        # mask（来自 Dataset 的 [N,1,H,W]、0~1）
        self.mask_B = input.get("mask_B", None)
        if self.mask_B is not None:
            self.mask_B = self.mask_B.to(self.device)
            # 保险：确保 float、范围正确
            if not self.mask_B.dtype.is_floating_point:
                self.mask_B = self.mask_B.float()
            mmax = self.mask_B.max().detach()
            if mmax > 1.5:  # 万一是 0/255
                self.mask_B = self.mask_B / 255.0
        else:
            # 没有 mask 时用全 0；这样开启 use_mask_condition 也不会报错
            self.mask_B = torch.zeros(self.real_B.size(0), 1, self.real_B.size(2), self.real_B.size(3), device=self.device)

        # 条件输入：是否把 mask 拼进 A
        if self.opt.use_mask_condition:
            self.cond_A = torch.cat([self.real_A, self.mask_B], dim=1)  # [N,C+1,H,W]
        else:
            self.cond_A = self.real_A

                # —— 到这里 cond_A 已经就绪 —— #

        # 仅打印一次，确认通道数与 mask 范围
        if not hasattr(self, "_logged_shapes"):
            aC = self.real_A.shape[1]
            cC = self.cond_A.shape[1]
            mmn = float(self.mask_B.min().item()) if self.mask_B is not None else -1
            mmx = float(self.mask_B.max().item()) if self.mask_B is not None else -1
            print(f"[DEBUG] real_A {tuple(self.real_A.shape)}, cond_A {tuple(self.cond_A.shape)}, "
                  f"mask min/max {mmn:.3f}/{mmx:.3f}, use_mask={self.opt.use_mask_condition}",
                  flush=True)
            self._logged_shapes = True

    # ------------------------- forward / backward -------------------------

    def forward(self):
        """G(A or [A,mask]) -> fake_B"""
        self.fake_B = self.netG(self.cond_A)

    # 判别器
    def backward_D(self):
        """D 条件输入也用 cond_A"""
        # Fake（detach 阻断回到 G）
        fake_AB = torch.cat((self.cond_A, self.fake_B), dim=1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.cond_A, self.real_B), dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # 合并
        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()

    # 生成器
    def backward_G(self):
        """GAN + 全图 L1 + ROI 面积归一 L1（可选） + 非ROI保真（可选）"""
        # 1) GAN（不 detach）
        fake_AB = torch.cat((self.cond_A, self.fake_B), dim=1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # 2) 全图 L1
        l1_global = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # 3) ROI/背景的面积归一 L1
        l1_roi = torch.tensor(0.0, device=self.device)
        l1_bg = torch.tensor(0.0, device=self.device)

        # 条件：给了 mask 且权重大于 0
        if (self.mask_B is not None) and (getattr(self.opt, "lambda_roi", 0.0) > 0.0):
            # 建议硬边界；如果你想保留软边界，把下一行注释掉
            m = (self.mask_B >= 0.5).float()  # [N,1,H,W]
            diff = torch.abs(self.fake_B - self.real_B)  # [N,C,H,W]
            C = self.fake_B.shape[1]
            eps = 1e-6

            # ROI：只用 ROI 像素数 × C 做分母（逐样本归一再 batch 平均）
            roi_pix = m.sum(dim=[1, 2, 3])  # [N]
            roi_sum = (diff * m).sum(dim=[1, 2, 3])
            roi_mean = roi_sum / (roi_pix * C + eps)
            l1_roi = roi_mean.mean() * self.opt.lambda_roi

            # 非 ROI：可选
            if getattr(self.opt, "lambda_bg", 0.0) > 0.0:
                bg = 1.0 - m
                bg_pix = bg.sum(dim=[1, 2, 3])
                bg_sum = (diff * bg).sum(dim=[1, 2, 3])
                bg_mean = bg_sum / (bg_pix * C + eps)
                l1_bg = bg_mean.mean() * self.opt.lambda_bg

        # 聚合
        self.loss_G_L1 = l1_global + l1_roi + l1_bg
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    # 优化流程
    def optimize_parameters(self):
        self.forward()
        # 先更 D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # 再更 G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()