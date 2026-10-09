import os
import argparse
from glob import glob
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import gc

from networks import Restormer

# ==========================================
# 1. 物理动态范围无损锚定与高频微结构补偿
# ==========================================
def process_3d_tensor_gpu_infer(tensor_3d):
    """
    稳健分位数归一化，并记录全图物理真实极值
    """
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
    """
    1. 基础物理逆变换
    2. 原图物理均值乘性对齐 (Multiplicative Gain，根除暗斑缺口)
    3. 【核心新增】：高频解剖微结构自适应补偿 (High-Frequency Detail Booster)
       定向强化骨小梁、软骨间隙边缘与肌束纹理
    4. 降噪强度加权融合 (默认 0.72，保留 28% 原生结构质感)
    5. 超高亮信号无损补全 (杜绝窗位腰斩)
    6. 背景空气底噪单向门控
    """
    # 1. 基础逆归一化
    pred_restored = pred_tensor * (p99 - p1 + 1e-8) + p1
    pred_restored = torch.clamp(pred_restored, min=0.0)
    
    flat_raw = raw_tensor_3d.view(-1)
    valid_raw = flat_raw[flat_raw > 0]
    if len(valid_raw) == 0:
        return pred_restored
        
    # 采用 3x3 均值滤波提取低频形态基底
    smoothed_raw = F.avg_pool2d(
        raw_tensor_3d.unsqueeze(1), 
        kernel_size=3, stride=1, padding=1, count_include_pad=False
    ).squeeze(1)
    
    # 提取解剖组织掩膜 (包容暗信号肌腱)
    tissue_thresh = torch.quantile(valid_raw, 0.15)
    tissue_mask = smoothed_raw > tissue_thresh
    
    # 2. 组织区域物理亮度乘性对齐 (Multiplicative Gain)
    if tissue_mask.sum() > 100:
        raw_mean = raw_tensor_3d[tissue_mask].mean()
        pred_mean = pred_restored[tissue_mask].mean()
        
        gain = raw_mean / (pred_mean + 1e-8)
        gain = torch.clamp(gain, min=0.5, max=2.0)
        
        pred_restored[tissue_mask] = pred_restored[tissue_mask] * gain

    # 3. 提取骨小梁、软骨间隙与肌束的高频纹理残差 (High-Pass Detail)
    high_pass_detail = raw_tensor_3d - smoothed_raw

    # 4. 降噪强度加权融合 + 组织内高频解剖细节定向注入
    refined_output = strength * pred_restored + (1.0 - strength) * raw_tensor_3d
    refined_output[tissue_mask] = refined_output[tissue_mask] + detail_boost * high_pass_detail[tissue_mask]

    # 5. 超高亮信号无损补全 (使用平滑掩膜，拒绝孤立白噪点带入)
    high_mask = smoothed_raw > p99
    if high_mask.sum() > 0:
        refined_output[high_mask] = raw_tensor_3d[high_mask]

    # 6. 背景空气底噪单向门控 (阈值 0.05 绝对保护低信号肌腱)
    bg_thresh = torch.quantile(valid_raw, 0.05)
    bg_mask = smoothed_raw <= bg_thresh
    refined_output[bg_mask] = torch.min(refined_output[bg_mask], raw_tensor_3d[bg_mask])
    
    # 7. 强制钳制在原图真实物理区间 [raw_min, raw_max]
    refined_output = torch.clamp(refined_output, min=raw_min, max=raw_max)

    return refined_output

def load_model_fresh(model_path, device):
    model = Restormer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

# ==========================================
# 2. 5CV 主推理流程
# ==========================================
def main(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n==========================================")
    print(f"[*] 5CV 全视野推理引擎启动 (骨小梁与微结构高保真强化版)")
    print(f"[*] 设备: {str(device).upper()} | 融合强度: {args.strength} | 细节补偿: {args.detail_boost}")
    print(f"==========================================\n")

    for fold in range(5):
        print(f"\n>>>> 🚀 开始处理 Fold {fold} <<<<")
        fold_name = f"fold_{fold}"
        input_dir = os.path.join(args.base_input, fold_name)
        output_dir = os.path.join(args.base_output, fold_name)
        
        if not os.path.exists(input_dir):
            print(f"[警告] 找不到输入文件夹 {input_dir}，跳过此折。")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        input_files = sorted(glob(os.path.join(input_dir, '*.nii.gz')))
        if len(input_files) == 0:
            print(f"在 {input_dir} 中未找到 .nii.gz 文件。")
            continue

        loaded_models = {}

        with torch.no_grad():
            for file_path in input_files:
                file_name = os.path.basename(file_path)
                file_name_upper = file_name.upper()
                
                if 'PD' in file_name_upper:
                    seq_type = 'PD'
                elif 'T1' in file_name_upper:
                    seq_type = 'T1'
                else:
                    print(f"[跳过] {file_name} 未检测到 PD 或 T1 标识。")
                    continue
                
                model_path = os.path.join(args.weight_dir, f"{seq_type}_{fold_name}_Transformer_best.ckpt")
                
                if not os.path.exists(model_path):
                    print(f"[错误] 缺少权重文件: {model_path}，跳过该文件。")
                    continue
                
                if model_path not in loaded_models:
                    print(f"   [+] 加载模型权重: {os.path.basename(model_path)}")
                    loaded_models[model_path] = load_model_fresh(model_path, device)
                
                model = loaded_models[model_path]

                print(f"   -> 推理中: {file_name}")

                con_img = sitk.ReadImage(file_path)
                con_arr = sitk.GetArrayFromImage(con_img)
                original_shape = con_arr.shape 

                con_tensor_3d = torch.from_numpy(con_arr).float().to(device)
                x_model_input, p1, p99, raw_min, raw_max = process_3d_tensor_gpu_infer(con_tensor_3d)
                x_model_input = x_model_input.unsqueeze(1)

                _, _, H_orig, W_orig = x_model_input.shape
                pad_h = (16 - H_orig % 16) % 16
                pad_w = (16 - W_orig % 16) % 16

                if pad_h > 0 or pad_w > 0:
                    x_model_input = F.pad(x_model_input, (0, pad_w, 0, pad_h), mode='reflect')

                pred_tensors = []
                batch_size = args.batch_size 
                
                for b in range(0, original_shape[0], batch_size):
                    batch_x = x_model_input[b : b+batch_size]
                    batch_pred = model(batch_x)
                    pred_tensors.append(batch_pred)
                
                pred_tensor = torch.cat(pred_tensors, dim=0)

                if pad_h > 0 or pad_w > 0:
                    pred_tensor = pred_tensor[:, :, :H_orig, :W_orig]

                pred_restored = inverse_process_and_refine(
                    pred_tensor.squeeze(1), p1, p99, con_tensor_3d, raw_min, raw_max, 
                    strength=args.strength, detail_boost=args.detail_boost
                )
                pred_vol = pred_restored.cpu().numpy()

                # 格式类型还原与 Gzip 高效压缩存储
                pred_itk = sitk.GetImageFromArray(pred_vol)
                pred_itk.CopyInformation(con_img) 
                pred_itk = sitk.Cast(pred_itk, con_img.GetPixelID())
                sitk.WriteImage(pred_itk, os.path.join(output_dir, file_name), useCompression=True)
                
                del con_tensor_3d, x_model_input, pred_tensors, pred_tensor, pred_restored
                torch.cuda.empty_cache() 
                gc.collect()             

    print(f"\n[*] 🎉 5CV 推理全部完成！输出文件已恢复轻量无损压缩。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Fold Transformer Inference Script (Micro-Structure High-Fidelity)")
    parser.add_argument('--base_input', type=str, default='./test/input/', help="存放 fold_x 输入图像的根目录")
    parser.add_argument('--base_output', type=str, default='./test/output/', help="存放 fold_x 推理结果的根目录")
    parser.add_argument('--weight_dir', type=str, default='./save/', help="存放模型 ckpt 权重的目录")
    parser.add_argument('--batch_size', type=int, default=8, help="推理时的批次大小")
    parser.add_argument('--strength', type=float, default=0.72, help="降噪强度融合系数 (0.70~0.75 兼顾信噪比与骨小梁真实感)")
    parser.add_argument('--detail_boost', type=float, default=0.25, help="高频微结构补偿系数 (0.20~0.35 强化骨小梁/软骨边缘)")
    parser.add_argument('--device', type=str, default='', help="留空则自动识别 GPU")
    args = parser.parse_args()
    main(args)