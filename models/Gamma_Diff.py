import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from models.decom import CTDN
import os
import cv2
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as compare_psnr,
    structural_similarity as compare_ssim
)
import lpips
import kornia
from pytorch_wavelets import DWTForward, DWTInverse

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)
        if self.args.mode == 'training':
            self.decom = self.load_stage1(CTDN(), 'ckpt/stage1')
        else:
            self.decom = CTDN()

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    @staticmethod
    def load_stage1(model, model_dir):
        checkpoint = utils.logging.load_checkpoint(os.path.join(model_dir, 'stage1_weight.pth.tar'), 'cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    def compute_base_gamma(self, img_rgb):
        if img_rgb.dim() == 3:
            img_rgb = img_rgb.unsqueeze(0)

        img_float = img_rgb / 255.0
        B, C, H, W = img_float.shape
        scales = [20, 50, 150, 250]
        illuminations = []

        for s in scales:
            ksize = int(4 * s + 1)
            ksize = min(ksize, H if H < W else W)
            if ksize % 2 == 0:
                ksize -= 1
            blur = kornia.filters.gaussian_blur2d(img_float, (ksize, ksize), (s, s))
            illuminations.append(blur)

        fused_illum = torch.mean(torch.stack(illuminations, dim=0), dim=0)

        gamma_base = 3.50
        illum_weight = 1.0
        base_gamma = torch.pow(gamma_base + fused_illum * illum_weight, 2 * fused_illum - 1)

        return img_float, base_gamma

    def partition_gamma(self, img_float, base_gamma, num_regions):
        B, C, H, W = img_float.shape
        if num_regions >= H * W:
            gamma_partition = base_gamma
        else:
            grid_size = max(1, int(num_regions ** 0.5))
            ph, pw = max(1, H // grid_size), max(1, W // grid_size)
            pooled = F.avg_pool2d(base_gamma, kernel_size=(ph, pw), stride=(ph, pw))
            gamma_partition = F.interpolate(pooled, size=(H, W), mode='nearest')
        enhanced = torch.clamp(img_float.pow(gamma_partition) * 255.0, 0, 255)

        B = enhanced.shape[0]
        sigma_color = torch.full((B,), 45.0, device=enhanced.device, dtype=enhanced.dtype)
        sigma_space = torch.full((B, 2), 45.0, device=enhanced.device, dtype=enhanced.dtype)

        enhanced = kornia.filters.bilateral_blur(enhanced, (3, 3), sigma_color=sigma_color, sigma_space=sigma_space)
        return enhanced

    def single_scale_retinex(self, img, sigma: float):
        B, C, H, W = img.shape
        ksize = int(2 * round(3 * sigma) + 1)
        ksize = min(ksize, min(H, W) - 1)
        if ksize % 2 == 0:
            ksize -= 1
        ksize = max(3, ksize)

        sigma_tensor = torch.full((B, 2), sigma, device=img.device, dtype=img.dtype)

        blur = kornia.filters.gaussian_blur2d(img, (ksize, ksize), sigma_tensor)

        return torch.log10(img + 1e-6) - torch.log10(blur + 1e-6)

    def multi_scale_retinex(self, img, sigma_list=[15, 80, 250]):
        out = torch.zeros_like(img)
        for sigma in sigma_list:
            out += self.single_scale_retinex(img, sigma)
        return out / len(sigma_list)

    def color_restoration(self, img, alpha, beta):
        img_sum = img.sum(dim=1, keepdim=True)
        return beta * (torch.log10(alpha * img + 1e-6) - torch.log10(img_sum + 1e-6))

    def msrcr(self, img, sigma_list=[15, 80, 250], G=4.0, b=25, alpha=125, beta=46):
        img = img.float() + 1.0
        retinex = self.multi_scale_retinex(img, sigma_list)
        color = self.color_restoration(img, alpha, beta)
        out = G * (retinex * color + b)
        out = torch.clamp(out, 0, 255)
        return out

    def wavelet_fusion(self, img1, img2, weight1=0.64, weight2=0.36):
        dwt = DWTForward(J=1, wave='haar').to(self.device)
        idwt = DWTInverse(wave='haar').to(self.device)
        """
        img1,img2: [B,3,H,W]
        """
        fused = []
        for c in range(3):
            Yl1, Yh1 = dwt(img1[:, c:c + 1])
            Yl2, Yh2 = dwt(img2[:, c:c + 1])
            Yl = weight1 * Yl1 + weight2 * Yl2
            Yh = [weight1 * h1 + weight2 * h2 for h1, h2 in zip(Yh1, Yh2)]
            f_c = idwt((Yl, Yh))
            fused.append(f_c)
        fused = torch.cat(fused, dim=1)
        return fused.clamp(0, 255)

    def color_denoise(self, img):
        yuv = kornia.color.rgb_to_yuv(img / 255.0)
        u = kornia.filters.gaussian_blur2d(yuv[:, 1:2], (5, 5), (1, 1))
        v = kornia.filters.gaussian_blur2d(yuv[:, 2:3], (5, 5), (1, 1))
        yuv = torch.cat([yuv[:, 0:1], u, v], dim=1)
        return (kornia.color.yuv_to_rgb(yuv) * 255.0).clamp(0, 255)

    def enhance_with_gamma(self, img_tensor, num_regions):
        img_rgb = img_tensor * 255.0
        img_float, base_gamma = self.compute_base_gamma(img_rgb)
        gamma_img = self.partition_gamma(img_float, base_gamma, num_regions)
        msrcr_img = self.msrcr(img_rgb)
        fused_img = self.wavelet_fusion(self.color_denoise(gamma_img), msrcr_img)
        final_img = torch.clamp(fused_img, 0, 255)

        B = final_img.shape[0]  # batch size
        sigma_color = torch.full((B,), 45.0, device=final_img.device, dtype=final_img.dtype)
        sigma_space = torch.full((B, 2), 45.0, device=final_img.device, dtype=final_img.dtype)

        final_img = kornia.filters.bilateral_blur(final_img, (3, 3), sigma_color=sigma_color, sigma_space=sigma_space)
        return (final_img / 255.0).clamp(0, 1)

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)


            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]


    def forward(self, inputs):
        data_dict = {}
        b = self.betas.to(inputs.device)

        if self.training:
            N = self.args.N
            lowlight_input = inputs[:, :3, :, :]
            low_fea = self.decom(lowlight_input, pred_fea=None)

            low_fea_norm = utils.data_transform(low_fea)

            t = torch.randint(low=0, high=self.num_timesteps, size=(lowlight_input.shape[0] // 2 + 1,)).to(
                self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:lowlight_input.shape[0]].to(inputs.device)

            max_regions = lowlight_input.shape[2] * lowlight_input.shape[3]
            pixel_img = []
            for idx in range(lowlight_input.shape[0]):
                pixel_gamma_tensor = self.enhance_with_gamma(lowlight_input[idx].unsqueeze(0), num_regions=max_regions)
                pixel_img.append(pixel_gamma_tensor)
            pixel_img = torch.cat(pixel_img, dim=0)
            pixel_img_fea = self.decom(pixel_img, pred_fea=None)

            seg_len = self.num_timesteps // N
            num_regions_list = [2 ** i for i in range(N)]
            enhanced_list = []
            for idx in range(lowlight_input.shape[0]):
                region_id = min(t[idx] // seg_len, N - 1)
                num_regions = num_regions_list[region_id]
                enhanced_tensor = self.enhance_with_gamma(lowlight_input[idx].unsqueeze(0), num_regions)
                enhanced_list.append(enhanced_tensor)
            enhanced_label = torch.cat(enhanced_list, dim=0)
            enhanced_label_fea = self.decom(enhanced_label, pred_fea=None)


            enhanced_label_fea_norm = utils.data_transform(enhanced_label_fea)


            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

            e = torch.randn_like(low_fea_norm)

            x = enhanced_label_fea_norm * a.sqrt() + e * (1.0 - a).sqrt()

            noise_output = self.Unet(torch.cat([low_fea_norm, x], dim=1), t.float())

            pred_fea = self.sample_training(low_fea_norm, b)
            pred_fea = utils.inverse_data_transform(pred_fea)

            data_dict["noise_output"] = noise_output
            data_dict["e"] = e
            data_dict["pred_fea"] = pred_fea
            data_dict["reference_fea"] = pixel_img_fea

        else:
            lowlight_input = inputs[:, :3, :, :]
            low_fea = self.decom(lowlight_input, pred_fea=None)
            low_fea_norm = utils.data_transform(low_fea)

            pred_fea = self.sample_training(low_fea_norm, b)
            pred_fea = utils.inverse_data_transform(pred_fea)
            pred_x = self.decom(inputs[:, :3, :, :], pred_fea=pred_fea)["pred_img"]

            data_dict["pred_x"] = pred_x

        return data_dict

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.view(-1, 1, 1, 1)


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.generator = Net(args, config)
        self.generator.to(self.device)

        self.discriminator = Discriminator(input_channels=3)
        self.discriminator.to(self.device)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.generator)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.BCELoss()



        self.optimizer_G = utils.optimize.get_optimizer(self.config, self.generator.parameters())
        self.optimizer_D = utils.optimize.get_optimizer(self.config, self.discriminator.parameters())


        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)

        self.generator.load_state_dict(checkpoint['state_dict'], strict=True)
        if 'discriminator' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator'])

        if ema:
            self.ema_helper.ema(self.generator)
        print(f"=> loaded checkpoint {load_path} (epoch {self.start_epoch}, step {self.step})")

    def train(self, DATASET, local_rank=0, world_size=1):
        cudnn.benchmark = True

        train_loader, val_loader = DATASET.get_loaders(world_size, local_rank)

        if local_rank == 0:
            _, non_dist_val_loader = DATASET.get_loaders(1, 0)

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for name, param in self.generator.named_parameters():
            if "decom" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            if local_rank == 0:
                print(f'[Epoch {epoch + 1}/{self.config.training.n_epochs}]')

            data_start = time.time()
            data_time = 0


            for i, (x, y) in enumerate(train_loader):
                batch_size = x.size(0)
                valid = torch.ones((batch_size, 1, 1, 1), requires_grad=False).to(self.device)
                fake = torch.zeros((batch_size, 1, 1, 1), requires_grad=False).to(self.device)

                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x = x.to(self.device, non_blocking=True)

                real_imgs = x[:, 3:, :, :]

                self.discriminator.train()
                self.optimizer_D.zero_grad()

                with torch.no_grad():
                    output = self.generator(x)
                    if "pred_x" in output:
                        gen_imgs = output["pred_x"]
                    else:
                        gen_imgs = self.generator.decom(x[:, :3, :, :],
                                                        pred_fea=output["pred_fea"])["pred_img"]

                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)

                loss_D = (real_loss + fake_loss) / 2
                loss_D.backward()
                self.optimizer_D.step()

                self.generator.train()
                self.optimizer_G.zero_grad()

                output = self.generator(x)

                if "pred_x" in output:
                    gen_imgs = output["pred_x"]
                else:
                    gen_imgs = self.generator.decom(x[:, :3, :, :],
                                                    pred_fea=output["pred_fea"])["pred_img"]

                noise_loss, scc_loss = self.noise_estimation_loss(output)

                validity = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid)

                lambda_adv= self.args.lambda_adv
                loss_G = noise_loss + scc_loss + lambda_adv * g_loss

                loss_G.backward()
                self.optimizer_G.step()
                self.ema_helper.update(self.generator)

                self.step += 1
                data_time += time.time() - data_start

                if local_rank == 0 and self.step % 10 == 0:
                    print(f"[Step {self.step}] noise_loss:{noise_loss.item():.5f} "
                          f"scc_loss:{scc_loss.item():.5f} "
                          f"G_loss:{g_loss.item():.5f} "
                          f"D_loss:{loss_D.item():.5f} "
                          f"time:{data_time / (i + 1):.5f}")

                data_start = time.time()

                if local_rank == 0 and self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.generator.eval()
                    self.sample_validation_patches(non_dist_val_loader, self.step)

                    checkpoint_filename = os.path.join(self.config.data.ckpt_dir, f'model_step_{self.step}')

                    utils.logging.save_checkpoint({
                        'step': self.step,
                        'epoch': epoch + 1,
                        'state_dict': self.generator.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'optimizer_G': self.optimizer_G.state_dict(),
                        'optimizer_D': self.optimizer_D.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=checkpoint_filename)

    def noise_estimation_loss(self, output):
        pred_fea, reference_fea = output["pred_fea"], output["reference_fea"]

        noise_output, e = output["noise_output"], output["e"]

        noise_loss = self.l2_loss(noise_output, e)

        scc_loss = self.args.lambda_scc * self.l1_loss(pred_fea, reference_fea)

        return noise_loss, scc_loss

    def sample_validation_patches(self, val_loader, step):
        if self.args.local_rank != 0:
            return

        image_folder = os.path.join(self.args.image_folder,
                                    self.config.data.type + str(self.config.data.patch_size))
        os.makedirs(os.path.join(image_folder, str(step)), exist_ok=True)

        self.generator.eval()


        with torch.no_grad():
            print(f'[Validation] Performing validation at step: {step}')
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape

                img_h_64 = int(64 * np.ceil(img_h / 64.0))
                img_w_64 = int(64 * np.ceil(img_w / 64.0))
                x = F.pad(x, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect')

                pred_x = self.generator(x.to(self.device))["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]

                file_name_with_extension = os.path.basename(y[0])
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), f'{file_name_with_extension}'))
        print("success!")

        def calculate_image_metrics(img_path1, img_path2,
                                    show_images=False,
                                    interpolation=cv2.INTER_AREA,
                                    device=torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'):
            try:
                img1 = cv2.imread(img_path1)
                img2 = cv2.imread(img_path2)
                if img1 is None or img2 is None:
                    raise ValueError("Error reading images")

                height, width = img1.shape[:2]
                img2_resized = cv2.resize(img2, (width, height), interpolation=interpolation)

                img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2_rgb = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB)

                img1_float = img1_rgb.astype(np.float32) / 255.0
                img2_float = img2_rgb.astype(np.float32) / 255.0

                psnr = compare_psnr(img1_float, img2_float)

                ssim = compare_ssim(img1_float, img2_float,
                                    multichannel=True,
                                    channel_axis=2,
                                    data_range=1.0)

                metrics = {'PSNR': psnr, 'SSIM': ssim}

                if lpips is not None and torch is not None:
                    loss_fn = lpips.LPIPS(net='alex', version='0.1').to(device)

                    def prepare_lpips(img):
                        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                        img_t = (img_t * 2 - 1).to(device)
                        return img_t

                    img1_lpips = prepare_lpips(img1_float)
                    img2_lpips = prepare_lpips(img2_float)

                    with torch.no_grad():
                        lpips_value = loss_fn(img1_lpips, img2_lpips).item()
                    metrics['LPIPS'] = lpips_value
                else:
                    metrics['LPIPS'] = None

                if show_images:
                    info_text = f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}"
                    if metrics['LPIPS'] is not None:
                        info_text += f" | LPIPS: {metrics['LPIPS']:.3f}"

                    vis_img = np.hstack([img1, img2_resized])
                    cv2.putText(vis_img, info_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('Comparison', vis_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                return metrics

            except Exception as e:
                print(f"Error: {str(e)}")
                return None

        def calculate_folder_metrics(gt_folder, test_folder, show_images=False, interpolation=cv2.INTER_AREA,
                                     device='cuda' if torch.cuda.is_available() else 'cpu'):
            compute_lpips = lpips is not None and torch is not None

            psnr_list = []
            ssim_list = []
            lpips_list = []

            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            test_files = [f for f in os.listdir(test_folder)
                          if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(test_folder, f))]

            total_pairs = 0
            for filename in test_files:
                gt_path = os.path.join(gt_folder, filename)
                test_path = os.path.join(test_folder, filename)

                if not os.path.exists(gt_path):
                    print(f"Warning：Pass {filename}，No GT")
                    continue

                metrics = calculate_image_metrics(gt_path, test_path, show_images=show_images,
                                                  interpolation=interpolation, device=device)
                if not metrics:
                    continue

                psnr_list.append(metrics['PSNR'])
                ssim_list.append(metrics['SSIM'])
                if compute_lpips:
                    if metrics['LPIPS'] is not None:
                        lpips_list.append(metrics['LPIPS'])
                    else:
                        print(f"Warning：{filename} LPIPS failed")

                total_pairs += 1

            avg_metrics = {
                'PSNR': np.mean(psnr_list) if psnr_list else 0,
                'SSIM': np.mean(ssim_list) if ssim_list else 0,
                'LPIPS': np.mean(lpips_list) if compute_lpips and lpips_list else None
            }

            print(f"\n Sucess: {total_pairs} pairs")
            return avg_metrics

        print("Start")
        generated_folder = os.path.join(image_folder, str(step))
        gt_folder = self.args.eval_path
        avg_metrics = calculate_folder_metrics(
            gt_folder=gt_folder,
            test_folder=generated_folder,
            show_images=False
        )

        result_txt = os.path.join(image_folder, 'validation_results.txt')
        with open(result_txt, 'a') as f:
            f.write(f"\n====== Step {step} ======\n")
            f.write(f"PSNR: {avg_metrics['PSNR']:.2f} dB\n")
            f.write(f"SSIM: {avg_metrics['SSIM']:.4f}\n")
            if avg_metrics['LPIPS'] is not None:
                f.write(f"LPIPS: {avg_metrics['LPIPS']:.4f}\n")
            f.write("\n")

        print(f"Results: (Step {step}):")
        print(f"PSNR: {avg_metrics['PSNR']:.2f} dB")
        print(f"SSIM: {avg_metrics['SSIM']:.4f}")
        if avg_metrics['LPIPS'] is not None:
            print(f"LPIPS: {avg_metrics['LPIPS']:.4f}")