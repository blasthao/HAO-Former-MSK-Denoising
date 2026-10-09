import os
import time
import argparse
import math
from glob import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn
from torch.amp import autocast, GradScaler

# ==========================================
# 引入强大的 Transformer 网络
# ==========================================
from networks import Restormer

# ==========================================
# 0. 基础图像差异与质控评价引擎 (完全对齐最新 measure_5cv.py)
# ==========================================
def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def compute_PSNR(img1, img2, data_range=1.0):
    mse_ = compute_MSE(img1, img2)
    if mse_ < 1e-10:
        return 80.0
    return 10 * torch.log10((data_range ** 2) / mse_).item()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

def compute_SSIM(img1, img2, data_range=1.0, window_size=11, channel=1):
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1, 1, shape_, shape_)
        img2 = img2.view(1, 1, shape_, shape_)
    window = create_window(window_size, channel).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.relu(F.conv2d(img1 * img1, window, padding=window_size//2) - mu1_sq)
    sigma2_sq = F.relu(F.conv2d(img2 * img2, window, padding=window_size//2) - mu2_sq)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def get_intra_tissue_noise(vol_tensor, tissue_thresh):
    """
    组织内部同质图块底噪估算法 (ITHP 金标准)。
    寻找在组织内部方差最小的 5% 图块作为评价本底热噪声的金标准标尺。
    """
    stds = []
    for z in range(vol_tensor.shape[0]):
        slice_t = vol_tensor[z]
        if slice_t.shape[0] < 8 or slice_t.shape[1] < 8:
            continue
        patches = slice_t.unfold(0, 8, 8).unfold(1, 8, 8).reshape(-1, 64)
        mins = patches.min(dim=1).values
        valid_patches = patches[mins > tissue_thresh]
        if len(valid_patches) > 0:
            stds.append(valid_patches.std(dim=1))
    if len(stds) == 0:
        return torch.tensor(1.0)
    all_stds = torch.cat(stds)
    noise_std = torch.quantile(all_stds, 0.05)
    return torch.clamp(noise_std, min=1.0)

def clean_outliers_iqr(data):
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    clean_data = data[(data >= lower) & (data <= upper)]
    if len(clean_data) == 0:
        clean_data = data
    return np.mean(clean_data), np.std(clean_data)

def plot_and_stats(df, save_dir):
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    print("\n==== 🔬 深度切片级统计检验结果 (Validation Final) ====")
    print(f"[*] 参与统计的有效切片数量 (N) = {len(df)}")
    
    paired_metrics = ['AutoSNR', 'AutoCNR']
    titles = ['Absolute Quality\n(Auto-SNR)', 'Tissue Contrast\n(Auto-CNR)']
    
    for i, metric in enumerate(paired_metrics):
        ax = axes[i]
        con_data = df[f'Input_{metric}'].values
        pred_data = df[f'Output_{metric}'].values
        
        t_stat, p_val = stats.ttest_rel(con_data, pred_data)
        
        m1, s1 = clean_outliers_iqr(con_data)
        m2, s2 = clean_outliers_iqr(pred_data)
        means = [m1, m2]
        stds = [s1, s2]
        
        ax.bar(['Baseline\n(Input)', 'Transformer\n(Output)'], means, yerr=stds, capsize=8, 
               color=['#4C72B0', '#55A868'], alpha=0.85, edgecolor='black', linewidth=1.2, width=0.6)
        ax.set_ylabel('Score', fontsize=12)
        
        if p_val < 0.001: sig = '***'
        elif p_val < 0.01: sig = '**'
        elif p_val < 0.05: sig = '*'
        else: sig = 'ns'
        
        print(f"[{metric:18s}] Input={m1:.2f} ± {s1:.2f} -> Output={m2:.2f} ± {s2:.2f} | P-value: {p_val:.4e} | 显著性: {sig}")
        
        y_max = max(means[0] + stds[0], means[1] + stds[1])
        y_range = max(y_max * 0.15, 1.0)
        line_y = y_max + y_range * 0.3
        
        ax.plot([0, 0, 1, 1], [y_max + y_range * 0.1, line_y, line_y, y_max + y_range * 0.1], lw=1.5, c='k')
        ax.text(0.5, line_y + y_range * 0.05, sig, ha='center', va='bottom', color='k', fontsize=16, fontweight='bold')
        
        ax.set_title(titles[i], fontsize=14, fontweight='bold', pad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0, top=line_y + y_range * 0.4)

    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    psnr_mean, psnr_std = clean_outliers_iqr(df['Diff_PSNR'].values)
    rna_mean, rna_std = clean_outliers_iqr(df['Removed_Noise_Pct'].values)
    ssim_mean, ssim_std = clean_outliers_iqr(df['Diff_SSIM'].values)
    
    x_pos = [0, 1, 2]
    x_labels = ['Diff-PSNR\n(dB)', 'Removed Noise\n(%)', 'Diff-SSIM\n(Score)']
    
    ax3.bar(0, psnr_mean, yerr=psnr_std, capsize=8, color='#DD8452', alpha=0.85, edgecolor='black', linewidth=1.2, width=0.6)
    ax3.bar(1, rna_mean, yerr=rna_std, capsize=8, color='#8172B3', alpha=0.85, edgecolor='black', linewidth=1.2, width=0.6)
    ax3_twin.bar(2, ssim_mean, yerr=ssim_std, capsize=8, color='#64B5CD', alpha=0.85, edgecolor='black', linewidth=1.2, width=0.6)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, fontsize=12)
    ax3.set_ylabel('dB  /  Percentage (%)', fontsize=12)
    ax3_twin.set_ylabel('SSIM Score (0-1)', fontsize=12)
    
    ax3.text(0, psnr_mean + psnr_std + (psnr_mean * 0.05), f"{psnr_mean:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax3.text(1, rna_mean + rna_std + (psnr_mean * 0.05), f"{rna_mean:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax3_twin.text(2, ssim_mean + ssim_std + 0.02, f"{ssim_mean:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax3.set_title('Global Difference Metrics\n(Residual values)', fontsize=14, fontweight='bold', pad=15)
    ax3.spines['top'].set_visible(False)
    ax3_twin.spines['top'].set_visible(False)
    ax3.set_ylim(bottom=0)
    ax3_twin.set_ylim(bottom=0, top=min(1.2, ssim_mean + ssim_std + 0.2))
    
    plt.tight_layout()
    img_path = os.path.join(save_dir, 'statistical_slice_metrics_comparison.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[*] 📊 验证集高清学术融合柱状图已保存至: {img_path}")

# ==========================================
# 1. 物理动态范围无损锚定与高频微结构补偿 (完全对齐 infer_5cv.py)
# ==========================================
def process_3d_tensor_gpu_infer(tensor_3d):
    flat_tensor = tensor_3d.view(-1)
    valid_flat = flat_tensor[flat_tensor > 0]
    if len(valid_flat) == 0:
        valid_flat = flat_tensor
        
    raw_min = tensor_3d.min()
    raw_max = tensor_3d.max()
    
    p1 = torch.quantile(valid_flat, 0.005)
    p99 = torch.quantile(valid_flat, 0.992)
    
    tensor_clipped = torch.clamp(tensor_3d, min=p1, max=p99)
    tensor_norm = (tensor_clipped - p1) / (p99 - p1 + 1e-8)
    return tensor_norm, p1, p99, raw_min, raw_max

def inverse_process_and_refine(pred_tensor, p1, p99, raw_tensor_3d, raw_min, raw_max, strength=0.72, detail_boost=0.25):
    pred_restored = pred_tensor * (p99 - p1 + 1e-8) + p1
    pred_restored = torch.clamp(pred_restored, min=0.0)
    
    flat_raw = raw_tensor_3d.view(-1)
    valid_raw = flat_raw[flat_raw > 0]
    if len(valid_raw) == 0:
        return pred_restored
        
    smoothed_raw = F.avg_pool2d(
        raw_tensor_3d.unsqueeze(1), 
        kernel_size=3, stride=1, padding=1, count_include_pad=False
    ).squeeze(1)
    
    tissue_thresh = torch.quantile(valid_raw, 0.15)
    tissue_mask = smoothed_raw > tissue_thresh
    
    if tissue_mask.sum() > 100:
        raw_mean = raw_tensor_3d[tissue_mask].mean()
        pred_mean = pred_restored[tissue_mask].mean()
        
        gain = raw_mean / (pred_mean + 1e-8)
        gain = torch.clamp(gain, min=0.5, max=2.0)
        
        pred_restored[tissue_mask] = pred_restored[tissue_mask] * gain

    high_pass_detail = raw_tensor_3d - smoothed_raw

    refined_output = strength * pred_restored + (1.0 - strength) * raw_tensor_3d
    refined_output[tissue_mask] = refined_output[tissue_mask] + detail_boost * high_pass_detail[tissue_mask]

    high_mask = smoothed_raw > p99
    if high_mask.sum() > 0:
        refined_output[high_mask] = raw_tensor_3d[high_mask]

    bg_thresh = torch.quantile(valid_raw, 0.05)
    bg_mask = smoothed_raw <= bg_thresh
    refined_output[bg_mask] = torch.min(refined_output[bg_mask], raw_tensor_3d[bg_mask])
    
    refined_output = torch.clamp(refined_output, min=raw_min, max=raw_max)
    return refined_output

# ==========================================
# 2. 核心锐化引擎：SSIM + L1 + 索贝尔梯度损失
# ==========================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.relu(F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq)
        sigma2_sq = F.relu(F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq)
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

class SobelGradLoss(nn.Module):
    def __init__(self):
        super(SobelGradLoss, self).__init__()
        kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)

    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.weight_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.weight_y, padding=1)
        target_grad_x = F.conv2d(target, self.weight_x, padding=1)
        target_grad_y = F.conv2d(target, self.weight_y, padding=1)
        
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

class HybridLoss(nn.Module):
    def __init__(self, alpha_ssim=0.4, alpha_l1=0.4, alpha_grad=0.2):
        super(HybridLoss, self).__init__()
        self.alpha_ssim = alpha_ssim
        self.alpha_l1 = alpha_l1
        self.alpha_grad = alpha_grad
        self.ssim = SSIMLoss()
        self.l1 = nn.L1Loss() 
        self.grad = SobelGradLoss()

    def forward(self, pred, target):
        return (self.alpha_ssim * self.ssim(pred, target) + 
                self.alpha_l1 * self.l1(pred, target) + 
                self.alpha_grad * self.grad(pred, target))

# ==========================================
# 3. 数据加载部分 (直接读取 preprocess 根目录，无损分割患者)
# ==========================================
class ct_dataset(Dataset):
    def __init__(self, data_dir, patient_subset=None, patch_n=None, patch_size=None, is_train=True):
        input_path = sorted(glob(os.path.join(data_dir, '*_input.npy')))
        target_path = sorted(glob(os.path.join(data_dir, '*_target.npy')))
        
        if patient_subset is not None:
            self.input_ = [f for f in input_path if os.path.basename(f).split('_')[0] in patient_subset]
            self.target_ = [f for f in target_path if os.path.basename(f).split('_')[0] in patient_subset]
        else:
            self.input_ = input_path
            self.target_ = target_path
            
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img = np.load(self.input_[idx])
        target_img = np.load(self.target_[idx])

        if self.is_train and self.patch_size:
            input_patches, target_patches = get_patch(input_img, target_img, self.patch_n, self.patch_size)
            return (input_patches, target_patches)
        else:
            # 验证集返回原始切片与文件路径，以便输出 CSV 统计表
            return (input_img, target_img, self.input_[idx])

def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    patch_input_imgs, patch_target_imgs = [], []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    
    for _ in range(patch_n):
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        patch_input_imgs.append(full_input_img[top:top+new_h, left:left+new_w])
        patch_target_imgs.append(full_target_img[top:top+new_h, left:left+new_w])
        
    return np.array(patch_input_imgs), np.array(patch_target_imgs)

# ==========================================
# 4. 核心训练与金标准验证引擎 (Solver)
# ==========================================
class Solver(object):
    def __init__(self, args, train_loader, val_loader, fold_idx="0"):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold_idx = fold_idx 

        self.device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.trunc_min, self.trunc_max = args.trunc_min, args.trunc_max
        self.norm_range_max = args.norm_range_max

        self.save_path = args.save_path
        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.result_fig = args.result_fig
        self.patch_size = args.patch_size
        
        self.patience = args.patience
        self.early_stop_counter = 0

        self.model = Restormer().to(self.device)
        self.criterion = HybridLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), args.lr)
        self.scaler = GradScaler(device='cuda', enabled=self.device.type == 'cuda')

        self.history = {'train_loss': [], 'train_psnr': [], 'val_loss': [], 'val_psnr': []}
        
        # 构建专属 Fold 目录系统：save/fold_0/ ...
        self.fold_dir = os.path.join(self.save_path, f"fold_{self.fold_idx}")
        os.makedirs(self.fold_dir, exist_ok=True)
        if self.result_fig:
            os.makedirs(os.path.join(self.fold_dir, 'fig'), exist_ok=True)

        log_file_path = os.path.join(self.fold_dir, 'train_log.txt')
        self.log_file = open(log_file_path, 'a', encoding='utf-8')
        
    def log(self, message):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def save_model(self, epoch, is_best=False):
        if is_best:
            f = os.path.join(self.fold_dir, 'Transformer_best.ckpt')
        else:
            f = os.path.join(self.fold_dir, f'Transformer_epoch_{epoch}.ckpt')
        torch.save(self.model.state_dict(), f)

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, epoch, psnr, ssim):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Input', fontsize=30)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title(f'Transformer Result\nPSNR: {psnr:.2f} | SSIM: {ssim:.4f}', fontsize=25)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Target', fontsize=30)

        f.savefig(os.path.join(self.fold_dir, 'fig', f'val_result_epoch_{epoch}.png'))
        plt.close()

    def plot_training_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(14, 6))
        plt.rcParams.update({'font.size': 12})
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss (Hybrid)', color='#4C72B0', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss (MSE)', color='#DD8452', linewidth=2)
        plt.title('Training & Validation Loss', fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_psnr'], label='Train PSNR', color='#4C72B0', linewidth=2)
        plt.plot(epochs, self.history['val_psnr'], label='Validation PSNR', color='#55A868', linewidth=2)
        plt.title('Training & Validation PSNR', fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(self.fold_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.log(f"\n[*] 📊 训练曲线图已保存至: {plot_path}")

    def train(self):
        best_val_psnr = 0.0 
        total_iters = 0
        
        self.log(f"==========================================")
        self.log(f"[*] HAO-Former 核心训练引擎启动 | Fold: {self.fold_idx} | 设备: {str(self.device).upper()}")
        self.log(f"==========================================\n")

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_train_loss = []
            epoch_train_psnr_mse = [] 
            start_time = time.time()

            for iter_, (x, y) in enumerate(self.train_loader):
                total_iters += 1
                x, y = x.float().to(self.device), y.float().to(self.device)

                if self.patch_size: 
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)
                else:
                    x = x.unsqueeze(1)
                    y = y.unsqueeze(1)

                self.optimizer.zero_grad()
                
                with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                    pred = self.model(x)
                    
                pred_f32 = pred.float()
                y_f32 = y.float()
                
                loss = self.criterion(pred_f32, y_f32)
                pure_mse = nn.MSELoss()(pred_f32.detach(), y_f32).item()
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_train_loss.append(loss.item())
                epoch_train_psnr_mse.append(pure_mse)

                if total_iters % self.print_iters == 0:
                    current_train_psnr = 10 * np.log10(1.0 / (pure_mse + 1e-8))
                    self.log(f"[Fold {self.fold_idx}] EPOCH [{epoch}/{self.num_epochs}], ITER [{iter_+1}/{len(self.train_loader)}] "
                             f"| Loss(Hybrid): {loss.item():.5f} | Batch Train PSNR: {current_train_psnr:.2f} | Time: {time.time() - start_time:.1f}s")

            avg_train_loss = np.mean(epoch_train_loss)
            avg_train_psnr = 10 * np.log10(1.0 / (np.mean(epoch_train_psnr_mse) + 1e-8))
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_psnr'].append(avg_train_psnr)

            self.model.eval()
            epoch_val_loss, epoch_val_psnr, epoch_val_ssim = [], [], []
            
            with torch.no_grad():
                for i, val_data in enumerate(self.val_loader):
                    x_tensor = val_data[0].unsqueeze(1).float().to(self.device)
                    y_tensor = val_data[1].unsqueeze(1).float().to(self.device)

                    with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                        pred_tensor = self.model(x_tensor)
                        
                    pred_tensor_f32 = pred_tensor.float()
                    y_tensor_f32 = y_tensor.float()
                    v_loss = nn.MSELoss()(pred_tensor_f32, y_tensor_f32).item()
                    
                    epoch_val_loss.append(v_loss)

                    pred_psnr = compute_PSNR(pred_tensor_f32, y_tensor_f32, data_range=self.norm_range_max)
                    pred_ssim = compute_SSIM(pred_tensor_f32, y_tensor_f32, data_range=self.norm_range_max)
                    
                    epoch_val_psnr.append(pred_psnr)
                    epoch_val_ssim.append(pred_ssim)

                    if i == 0 and self.result_fig:
                        x_view = self.trunc(x_tensor.squeeze().cpu().detach())
                        y_view = self.trunc(y_tensor.squeeze().cpu().detach())
                        pred_view = self.trunc(pred_tensor_f32.squeeze().cpu().detach())
                        self.save_fig(x_view, y_view, pred_view, epoch, pred_psnr, pred_ssim)

            avg_val_loss = np.mean(epoch_val_loss)
            avg_val_psnr = np.mean(epoch_val_psnr)
            avg_val_ssim = np.mean(epoch_val_ssim)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_psnr'].append(avg_val_psnr)
            
            self.log(f"====> EPOCH {epoch} 总结 | Train Loss(Hybrid): {avg_train_loss:.5f} | Train PSNR: {avg_train_psnr:.2f}\n"
                     f"                   | Val Loss(MSE): {avg_val_loss:.5f} | Val PSNR: {avg_val_psnr:.2f} | Val SSIM: {avg_val_ssim:.4f}")

            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                self.early_stop_counter = 0 
                self.log(f"[*] 🚀 发现更佳视觉质量模型! PSNR 上升至: {best_val_psnr:.2f}，执行保存。")
                self.save_model(epoch, is_best=True)
            else:
                self.early_stop_counter += 1
                self.log(f"[*] ⚠️ 模型未见明显提升 (早停计数: {self.early_stop_counter}/{self.patience})")
                
            if self.early_stop_counter >= self.patience:
                self.log(f"\n[!] 🛑 触发早停机制：验证集 PSNR 连续 {self.patience} 个 Epoch 未提升。训练提前结束！")
                break

        self.plot_training_curves()

    def run_final_evaluation(self, eval_loader):
        """
        训练完成后，全自动加载 Best 权重，利用微结构补偿推理与金标准 ITHP 评价引擎进行全面客观体检
        """
        self.log(f"\n[*] 🌟 开始执行全自动独立验证 (Auto Evaluation & ITHP Metrics) ...")
        out_dir = os.path.join(self.fold_dir, 'validation')
        os.makedirs(out_dir, exist_ok=True)
        
        # 加载最佳模型权重
        best_model_path = os.path.join(self.fold_dir, 'Transformer_best.ckpt')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
            self.log(f"[*] 成功加载最佳权重用于体验验证: {best_model_path}")
        else:
            self.log("[!] 警告: 未找到最佳权重，将使用当前 Epoch 权重进行评估。")
            
        self.model.eval()
        
        all_in_slices = []
        all_pred_slices = []
        all_fnames = []

        with torch.no_grad():
            for val_data in eval_loader:
                x_tensor = val_data[0].float().to(self.device)  # [1, H, W]
                fpath = val_data[2][0]
                fname = os.path.basename(fpath).replace('_input.npy', '')
                
                # 1. 动态 Padding 适配 U-Net 跳跃连接 (16 的倍数)
                _, H_orig, W_orig = x_tensor.shape
                pad_h = (16 - H_orig % 16) % 16
                pad_w = (16 - W_orig % 16) % 16

                x_model_input = x_tensor.unsqueeze(1) # [1, 1, H, W]
                if pad_h > 0 or pad_w > 0:
                    x_model_input = F.pad(x_model_input, (0, pad_w, 0, pad_h), mode='reflect')

                with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                    pred = self.model(x_model_input)

                if pad_h > 0 or pad_w > 0:
                    pred = pred[:, :, :H_orig, :W_orig]

                # 2. 对齐 infer.py：分位数锚定与高频解剖微结构自适应补偿还原
                p1 = torch.quantile(x_tensor, 0.005)
                p99 = torch.quantile(x_tensor, 0.992)
                raw_min = x_tensor.min()
                raw_max = x_tensor.max()

                pred_restored = inverse_process_and_refine(
                    pred.squeeze(1), p1, p99, x_tensor, raw_min, raw_max,
                    strength=self.args.strength, detail_boost=self.args.detail_boost
                )

                all_in_slices.append(x_tensor.squeeze(0).cpu())
                all_pred_slices.append(pred_restored.squeeze(0).cpu())
                all_fnames.append(fname)

        if len(all_in_slices) > 0:
            val_vol_in = torch.stack(all_in_slices).to(self.device)
            val_vol_out = torch.stack(all_pred_slices).to(self.device)

            # 3. 对齐 measure_5cv.py：采用组织内部同质图块法 (ITHP) 提取真实底噪
            flat_in = val_vol_in.view(-1)
            valid_in = flat_in[flat_in > 1e-3]
            if len(valid_in) == 0:
                valid_in = flat_in

            tissue_thresh_3d = torch.quantile(valid_in, 0.45)
            vol_sigma_in = get_intra_tissue_noise(val_vol_in, tissue_thresh_3d)
            vol_sigma_out = get_intra_tissue_noise(val_vol_out, tissue_thresh_3d)

            p1_norm = torch.quantile(valid_in, 0.005)
            p99_norm = torch.quantile(valid_in, 0.995)

            results = []
            for z in range(val_vol_in.shape[0]):
                in_slice = val_vol_in[z]
                out_slice = val_vol_out[z]

                t_mask = in_slice > tissue_thresh_3d
                if t_mask.sum() < 200:
                    continue

                tissue_in = in_slice[t_mask]
                tissue_out = out_slice[t_mask]

                sig_in = torch.mean(tissue_in).item()
                sig_out = torch.mean(tissue_out).item()

                snr_in = sig_in / vol_sigma_in.item()
                snr_out = sig_out / vol_sigma_out.item()

                high_thresh = torch.quantile(tissue_in, 0.80)
                low_thresh = torch.quantile(tissue_in, 0.20)

                high_mask = in_slice >= high_thresh
                low_mask = (in_slice <= low_thresh) & t_mask

                if high_mask.sum() > 50 and low_mask.sum() > 50:
                    contrast_in = abs(torch.mean(in_slice[high_mask]).item() - torch.mean(in_slice[low_mask]).item())
                    contrast_out = abs(torch.mean(out_slice[high_mask]).item() - torch.mean(out_slice[low_mask]).item())
                else:
                    contrast_in = sig_in * 0.5
                    contrast_out = sig_out * 0.5

                cnr_in = contrast_in / vol_sigma_in.item()
                cnr_out = contrast_out / vol_sigma_out.item()

                in_slice_norm = (torch.clamp(in_slice, min=p1_norm, max=p99_norm) - p1_norm) / (p99_norm - p1_norm + 1e-8)
                out_slice_norm = (torch.clamp(out_slice, min=p1_norm, max=p99_norm) - p1_norm) / (p99_norm - p1_norm + 1e-8)

                in_slice_4d = in_slice_norm.unsqueeze(0).unsqueeze(0)
                out_slice_4d = out_slice_norm.unsqueeze(0).unsqueeze(0)

                psnr_val = compute_PSNR(out_slice_4d, in_slice_4d, data_range=1.0)
                ssim_val = compute_SSIM(out_slice_4d, in_slice_4d, data_range=1.0)
                rna_val = (10 ** (-psnr_val / 20.0)) * 100

                slice_data = {
                    'Volume_Slice': all_fnames[z],
                    'Input_AutoSNR': snr_in,
                    'Output_AutoSNR': snr_out,
                    'Input_AutoCNR': cnr_in,
                    'Output_AutoCNR': cnr_out,
                    'Diff_PSNR': psnr_val,
                    'Removed_Noise_Pct': rna_val,
                    'Diff_SSIM': ssim_val
                }
                results.append(slice_data)

            if len(results) > 0:
                df = pd.DataFrame(results)
                csv_path = os.path.join(out_dir, 'slice_level_evaluation_metrics.csv')
                df.to_csv(csv_path, index=False)
                self.log(f"[*] 📝 验证集客观指标 CSV 已保存至: {csv_path}")

                try:
                    plot_and_stats(df, out_dir)
                    self.log(f"[*] 📊 验证集对比柱状图已绘制并保存至: {out_dir}")
                except Exception as e:
                    self.log(f"[!] 绘制统计图时发生错误: {e}")

        self.log_file.close()

# ==========================================
# 5. Main 控制流 (患者级五折交叉防泄露 + 全量数据训练)
# ==========================================
def main(args):
    cudnn.benchmark = True
    os.makedirs(args.save_path, exist_ok=True)

    data_dir = args.saved_path
    all_files = glob(os.path.join(data_dir, '*_input.npy'))
    if len(all_files) == 0:
        print(f"[!] 错误：在 {data_dir} 中未找到预处理后的 _input.npy 数据，请检查路径。")
        return

    # 【无泄露核心】：通过 "_" 切分提取患者ID，确保划分单位是患者，而非切片
    patients = sorted(list(set([os.path.basename(f).split('_')[0] for f in all_files])))
    print(f"\n[*] 成功加载数据集，共识别到 {len(patients)} 位独立患者。")

    np.random.seed(args.seed)
    np.random.shuffle(patients)

    # ---------------------------------------------------------
    # 阶段一：进行 5 折患者级交叉验证 (fold_0 到 fold_4)
    # ---------------------------------------------------------
    if args.run_cv:
        print("\n" + "="*50)
        print(f"  执行患者级 {args.n_folds} 折交叉验证模式 (Patient-level CV Mode)")
        print("="*50)
        
        patient_folds = np.array_split(patients, args.n_folds)

        for fold in range(args.n_folds):
            print(f"\n>>>> 启动 Fold {fold} / {args.n_folds - 1} 训练 <<<<")
            
            fold_val_patients = patient_folds[fold].tolist()
            fold_train_patients = [p for p in patients if p not in fold_val_patients]

            train_dataset = ct_dataset(data_dir, patient_subset=fold_train_patients, 
                                       patch_n=args.patch_n, patch_size=args.patch_size, is_train=True)
            val_dataset = ct_dataset(data_dir, patient_subset=fold_val_patients, 
                                     patch_size=None, is_train=False)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                      num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True)

            solver = Solver(args, train_loader, val_loader, fold_idx=str(fold))
            solver.train()
            
            # 训练结束后，全自动执行本折独立验证集体检
            solver.run_final_evaluation(val_loader)

    # ---------------------------------------------------------
    # 阶段二：进行全量数据训练 (fold_all)
    # ---------------------------------------------------------
    if args.run_fold_all:
        print("\n" + "="*50)
        print(f"  执行全量数据训练模式 (Fold_All: 100% Training Data)")
        print("="*50)
        
        # 抽取 10% 兼作训练时的监控防过拟合
        val_subset = np.random.choice(patients, max(1, int(len(patients)*0.1)), replace=False).tolist()

        train_dataset = ct_dataset(data_dir, patient_subset=patients, 
                                   patch_n=args.patch_n, patch_size=args.patch_size, is_train=True)
        val_dataset = ct_dataset(data_dir, patient_subset=val_subset, 
                                 patch_size=None, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=True)

        solver = Solver(args, train_loader, val_loader, fold_idx="all")
        solver.train()
        
        # 由于是 fold_all，我们构建一个包含 100% 数据的体检 Loader
        all_eval_dataset = ct_dataset(data_dir, patient_subset=patients, patch_size=None, is_train=False)
        all_eval_loader = DataLoader(all_eval_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        solver.run_final_evaluation(all_eval_loader)

    if not args.run_cv and not args.run_fold_all:
        print("[!] 警告：您既没有开启 --run_cv 也没有开启 --run_fold_all，训练脚本未执行任何操作。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_path', type=str, default='./preprocess/', help="存放.npy的文件夹")
    parser.add_argument('--save_path', type=str, default='./save/', help="模型保存路径")
    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--device', type=str, default='', help="留空则自动识别 GPU")
    parser.add_argument('--num_workers', type=int, default=4)

    # 开关
    parser.add_argument('--run_cv', action='store_true', help="加入此标签以开启五折交叉验证")
    parser.add_argument('--run_fold_all', action='store_true', help="加入此标签以开启全量数据训练")
    
    parser.add_argument('--n_folds', type=int, default=5, help="交叉验证折数")
    parser.add_argument('--seed', type=int, default=42, help="K-Fold划分的随机种子")

    parser.add_argument('--norm_range_min', type=float, default=0.0)
    parser.add_argument('--norm_range_max', type=float, default=1.0)
    parser.add_argument('--trunc_min', type=float, default=0.0)
    parser.add_argument('--trunc_max', type=float, default=1.0)

    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--print_iters', type=int, default=50)
    parser.add_argument('--patience', type=int, default=20, help="多少个Epoch模型不提升则触发早停停止训练")

    # 对齐高频微结构补偿参数
    parser.add_argument('--strength', type=float, default=0.72, help="降噪强度融合系数 (0.70~0.75 兼顾信噪比与骨小梁真实感)")
    parser.add_argument('--detail_boost', type=float, default=0.25, help="高频微结构补偿系数 (0.20~0.35 强化骨小梁/软骨边缘)")

    args = parser.parse_args()
    main(args)