import os
import argparse
from glob import glob
import torch
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import exp
import torch.nn.functional as F

# ==========================================
# 1. 基础图像差异评价函数
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

# ==========================================
# 2. 独家金标准：组织内部同质图块底噪估算法 (Intra-Tissue Homogeneous Patch)
# ==========================================
def get_intra_tissue_noise(vol_tensor, tissue_thresh):
    """
    完全抛弃不可靠的体外空气背景。
    将 3D 解剖组织切分为 8x8 的网格图块，寻找在组织内部方差最小的 5% 图块
    （对应质地极其均匀的深层肌肉或脂肪垫），作为评价本底热噪声的金标准标尺。
    """
    stds = []
    
    for z in range(vol_tensor.shape[0]):
        slice_t = vol_tensor[z]
        if slice_t.shape[0] < 8 or slice_t.shape[1] < 8:
            continue
            
        # [N, 64] 获取所有 8x8 的像素图块
        patches = slice_t.unfold(0, 8, 8).unfold(1, 8, 8).reshape(-1, 64)
        
        # 获取每个图块内的最低亮度值
        mins = patches.min(dim=1).values
        
        # 仅保留完全身处于解剖组织内部的图块 (最小亮度都大于组织阈值)
        valid_patches = patches[mins > tissue_thresh]
        if len(valid_patches) > 0:
            stds.append(valid_patches.std(dim=1))
            
    if len(stds) == 0:
        return torch.tensor(1.0)
        
    all_stds = torch.cat(stds)
    
    # 提取这些纯净组织图块中方差最小的 5% 群体，即代表了剔除了解剖纹理后的纯粹物理热噪声
    noise_std = torch.quantile(all_stds, 0.05)
    
    # 物理下限保护：分母绝对不能小于 1，防止极端除零崩塌
    return torch.clamp(noise_std, min=1.0)

# ==========================================
# 3. 统计绘图引擎 (1x3 学术排版)
# ==========================================
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
    
    print("\n==== 🔬 深度切片级统计检验结果 ====")
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
    print(f"\n[*] 📊 高清学术 1x3 统计图已更新至: {img_path}")

# ==========================================
# 4. 主评测流程 (组织内部热噪声提取版)
# ==========================================
def main(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n==========================================")
    print(f"[*] 医学影像客观质控中心启动 (同质解剖图块底噪金标准版)")
    print(f"[*] 设备: {str(device).upper()}")
    print(f"==========================================\n")

    input_files = sorted(glob(os.path.join(args.input_dir, '*.nii.gz')))
    if len(input_files) == 0:
        print(f"[!] 错误: 未在 {args.input_dir} 找到 NIfTI 文件。")
        return

    results = []

    with torch.no_grad():
        for input_path in input_files:
            file_name = os.path.basename(input_path)
            output_path = os.path.join(args.output_dir, file_name)
            
            if not os.path.exists(output_path):
                print(f"[警告] 找不到对应的输出文件，跳过: {file_name}")
                continue
                
            print(f"Evaluating: {file_name}")

            in_arr = sitk.GetArrayFromImage(sitk.ReadImage(input_path))
            out_arr = sitk.GetArrayFromImage(sitk.ReadImage(output_path))
            original_shape = in_arr.shape 

            in_vol_raw = torch.from_numpy(in_arr).float().to(device)
            out_vol_raw = torch.from_numpy(out_arr).float().to(device)

            # ----------------------------------------------------
            # 1. 提取 3D 体素级别的基本有效信号与组织阈值
            # ----------------------------------------------------
            flat_in = in_vol_raw.view(-1)
            valid_in = flat_in[flat_in > 1.0]
            if len(valid_in) == 0:
                valid_in = flat_in

            # 定义组织信号的最低物理阈值 (选取有效信号的前 15% 避开空气过渡带)
            tissue_thresh_3d = torch.quantile(valid_in, 0.15)
            
            # 2. 采用 Intra-Tissue 算法提取全 3D 极为稳健的组织内部真实底噪
            vol_sigma_in = get_intra_tissue_noise(in_vol_raw, tissue_thresh_3d)
            vol_sigma_out = get_intra_tissue_noise(out_vol_raw, tissue_thresh_3d)

            # 为无参考指标确立 0-1 缩放映射边界
            p1 = torch.quantile(valid_in, 0.005)
            p99 = torch.quantile(valid_in, 0.995)

            for z in range(original_shape[0]):
                in_slice = in_vol_raw[z]
                out_slice = out_vol_raw[z]

                # ----------------------------------------------------
                # 3. 提取有效解剖切片，自动过滤无聊的纯空气空层
                # ----------------------------------------------------
                t_mask = in_slice > tissue_thresh_3d
                if t_mask.sum() < 500:
                    continue  

                # 提取纯粹组织均值 (Signal)
                tissue_in = in_slice[t_mask]
                tissue_out = out_slice[t_mask]
                
                sig_in = torch.mean(tissue_in).item()
                sig_out = torch.mean(tissue_out).item()

                # SNR = Signal / Noise
                snr_in = sig_in / vol_sigma_in.item()
                snr_out = sig_out / vol_sigma_out.item()

                # ----------------------------------------------------
                # 4. 提取图像解剖高低对比信号 (CNR)
                # ----------------------------------------------------
                high_thresh = torch.quantile(tissue_in, 0.75)
                low_thresh = torch.quantile(tissue_in, 0.25)

                high_mask = in_slice >= high_thresh
                low_mask = (in_slice <= low_thresh) & t_mask

                if high_mask.sum() > 50 and low_mask.sum() > 50:
                    # CNR = |Signal1 - Signal2| / Noise
                    contrast_in = abs(torch.mean(in_slice[high_mask]).item() - torch.mean(in_slice[low_mask]).item())
                    contrast_out = abs(torch.mean(out_slice[high_mask]).item() - torch.mean(out_slice[low_mask]).item())
                else:
                    contrast_in = sig_in * 0.5
                    contrast_out = sig_out * 0.5

                cnr_in = contrast_in / vol_sigma_in.item()
                cnr_out = contrast_out / vol_sigma_out.item()

                # ----------------------------------------------------
                # 5. 计算 PSNR 与 SSIM (统一缩放至完全等比的 [0, 1] 空间)
                # ----------------------------------------------------
                in_slice_norm = (torch.clamp(in_slice, min=p1, max=p99) - p1) / (p99 - p1 + 1e-8)
                out_slice_norm = (torch.clamp(out_slice, min=p1, max=p99) - p1) / (p99 - p1 + 1e-8)

                in_slice_4d = in_slice_norm.unsqueeze(0).unsqueeze(0)
                out_slice_4d = out_slice_norm.unsqueeze(0).unsqueeze(0)

                psnr_val = compute_PSNR(out_slice_4d, in_slice_4d, data_range=1.0)
                ssim_val = compute_SSIM(out_slice_4d, in_slice_4d, data_range=1.0)
                noise_removed_pct = (10 ** (-psnr_val / 20.0)) * 100

                slice_data = {
                    'Volume': file_name.replace('.nii.gz', ''),
                    'Slice_Idx': z,
                    'Input_AutoSNR': snr_in,
                    'Output_AutoSNR': snr_out,
                    'Input_AutoCNR': cnr_in,
                    'Output_AutoCNR': cnr_out,
                    'Diff_PSNR': psnr_val,
                    'Removed_Noise_Pct': noise_removed_pct,
                    'Diff_SSIM': ssim_val
                }
                results.append(slice_data)

    if len(results) > 0:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, 'slice_level_evaluation_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[*] 📝 逐切片评估数据明细已更新保存至: {csv_path}")
        plot_and_stats(df, args.output_dir)
    else:
        print("[!] 未提取到有效切片，请检查输入图像路径及组织信号强度。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Medical Image Metrics Evaluation Script")
    parser.add_argument('--input_dir', type=str, default='./test/input/', help="推理前原图路径")
    parser.add_argument('--output_dir', type=str, default='./test/output/', help="推理后降噪图路径")
    parser.add_argument('--device', type=str, default='', help="留空则自动识别 GPU")
    args = parser.parse_args()
    main(args)