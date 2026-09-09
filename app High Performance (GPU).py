# -*- coding: utf-8 -*-
"""
程序名称：HAO-Former MSK Denoising (Streamlit 网页云端发布版 + PACS传输功能)
功能描述：肌骨磁共振通用智能降噪与增强诊断平台。
          集成：多DICOM ZIP解析、NIfTI解析、Transformer 核心推理、多重指标自动计算。
          优化：推理端对齐骨小梁高频补偿(防模糊缺口)，评价端对齐最新金标准 ITHP 引擎。
"""

import os
import sys
import time
import shutil
import tempfile
import zipfile
import numpy as np
import SimpleITK as sitk
import pandas as pd
import torch
import torch.nn.functional as F
from math import exp
import gc
import streamlit as st
import warnings
import logging

# 全局静音 SimpleITK/GDCM 的底层 C++ 扫描警告
sitk.ProcessObject.SetGlobalWarningDisplay(False)
warnings.filterwarnings("ignore")
for logger_name in logging.root.manager.loggerDict:
    if "streamlit" in logger_name:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# ==============================
# 动态加载 PACS 相关依赖
# ==============================
try:
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from pynetdicom import AE, VerificationPresentationContexts, StoragePresentationContexts, build_context
    HAS_PACS_LIBS = True
except ImportError:
    HAS_PACS_LIBS = False

# 导入您的 Transformer 网络
from networks import Restormer

# ==========================================
# 0. 网页基础配置与多语言字典
# ==========================================
st.set_page_config(page_title="HAO-Former MSK Denoising", layout="wide", page_icon="🦴")

LANGUAGES = {
    "中文 (Chinese)": {
        "title": "HAO-Former: A Hybrid-loss Aware Ortho-Transformer",
        "subtitle": "肌骨磁共振通用智能降噪与增强诊断平台 v1.0",
        "sidebar_title": "⚙️ 控制面板",
        "device_info": "当前计算设备: ",
        "batch_size": "推理批次大小 (Batch Size):",
        "upload_label": "📂 批量上传影像 (支持多个 .nii.gz 或 DICOM压缩包 .zip)",
        "upload_help": "文件名请包含 'PD' 或 'T1' 以便系统自动匹配对应的模型权重。",
        "run_btn": "🚀 开始批量智能降噪",
        "warn_no_file": "请先上传至少一个文件！",
        "warn_name": "跳过: 文件名中未检测到 'PD' 或 'T1'",
        "status_unzip": "读取与解析原始数据 (Raw Data)...",
        "status_infer": "HAO-Former 全视野动态降噪中...",
        "status_metrics": "计算客观图像质量指标...",
        "status_pack": "打包无损降噪结果...",
        "success": "🎉 批量降噪与评估全部完成！",
        "download_btn": "⬇️ 一键下载所有降噪结果",
        "metrics_title": "📊 图像质量客观评价 (总批次平均值)",
        "m_snr": "绝对质量 (Auto-SNR)",
        "m_cnr": "组织对比度 (Auto-CNR)",
        "m_psnr": "修改幅度 (Diff-PSNR)",
        "m_rna": "噪声剥离占比 (%)",
        "m_ssim": "结构保真度 (Diff-SSIM)",
        "preview_title": "👁️ 实时对比预览: {filename} (中央切片: {slice_idx})",
        "preview_in": "降噪前 (Raw Input)",
        "preview_out": "HAO-Former 降噪后 (Denoised)",
        "model_loaded": "✅ 已自动匹配并加载专属降噪模型: **{seq_type}**",
        "pacs_title": "📡 PACS 归档配置",
        "pacs_send_btn": "📤 将降噪后的 DICOM 结果一键发送至 PACS",
        "pacs_ok": "🎉 成功！已将结果归档至 PACS。",
        "pacs_err": "❌ 无法建立 PACS 连接，请检查网络参数。",
        "pacs_test_ok": "✅ C-ECHO 成功！服务器通讯正常。",
        "pacs_test_err": "❌ C-ECHO 失败或无法建立关联，请检查配置。",
        "pacs_not_dicom": "⚠️ 提示：上传的是 NIfTI，暂不支持直接推送 PACS，请上传 DICOM ZIP。"
    },
    "English": {
        "title": "HAO-Former: A Hybrid-loss Aware Ortho-Transformer",
        "subtitle": "General MSK-MRI Intelligent Denoising & Diagnostic Enhancement Platform v1.0",
        "sidebar_title": "⚙️ Control Panel",
        "device_info": "Compute Device: ",
        "batch_size": "Inference Batch Size:",
        "upload_label": "📂 Batch Upload (supports multiple .nii.gz or .zip DICOMs)",
        "upload_help": "Filename must contain 'PD' or 'T1' for automatic model routing.",
        "run_btn": "🚀 Start Batch Denoising",
        "warn_no_file": "Please upload at least one file!",
        "warn_name": "Skipping: Filename lacks 'PD' or 'T1'",
        "status_unzip": "Parsing raw data...",
        "status_infer": "Denoising with HAO-Former...",
        "status_metrics": "Calculating metrics...",
        "status_pack": "Packaging lossless results...",
        "success": "🎉 Batch Denoising Complete!",
        "download_btn": "⬇️ Download All Results",
        "metrics_title": "📊 Objective Quality Metrics (Batch Average)",
        "m_snr": "Absolute Quality (Auto-SNR)",
        "m_cnr": "Tissue Contrast (Auto-CNR)",
        "m_psnr": "Modification Extent (Diff-PSNR)",
        "m_rna": "Removed Noise (%)",
        "m_ssim": "Structural Fidelity (Diff-SSIM)",
        "preview_title": "👁️ Live Preview: {filename} (Center Slice: {slice_idx})",
        "preview_in": "Raw Input",
        "preview_out": "HAO-Former Denoised",
        "model_loaded": "✅ Automatically matched and loaded specific model: **{seq_type}**",
        "pacs_title": "📡 PACS Archive Config",
        "pacs_send_btn": "📤 Send Denoised DICOMs to PACS",
        "pacs_ok": "🎉 Success! Results archived to PACS.",
        "pacs_err": "❌ Failed to establish PACS connection.",
        "pacs_test_ok": "✅ C-ECHO Success! Server is reachable.",
        "pacs_test_err": "❌ C-ECHO Failed! Check network configuration.",
        "pacs_not_dicom": "⚠️ Note: PACS transmission is for DICOM only."
    },
    "日本語 (Japanese)": {
        "title": "HAO-Former: A Hybrid-loss Aware Ortho-Transformer",
        "subtitle": "筋骨格MRI 汎用インテリジェントノイズ除去・診断支援プラットフォーム v1.0",
        "sidebar_title": "⚙️ コントロールパネル",
        "device_info": "現在の計算デバイス: ",
        "batch_size": "推論バッチサイズ:",
        "upload_label": "📂 一括アップロード (複数 .nii.gz 或者 .zip DICOM)",
        "upload_help": "自動モデルルーティングのため、ファイル名に「PD」または「T1」を含めてください。",
        "run_btn": "🚀 一括ノイズ除去を開始",
        "warn_no_file": "少なくとも1つのファイルをアップロードしてください！",
        "warn_name": "スキップ: ファイル名に「PD」または「T1」がありません",
        "status_unzip": "生データを解析中...",
        "status_infer": "HAO-Formerでノイズ除去中...",
        "status_metrics": "指標を計算中...",
        "status_pack": "可逆結果をパッケージ化中...",
        "success": "🎉 一括ノイズ除去が完了しました！",
        "download_btn": "⬇️ すべての結果をダウンロード",
        "metrics_title": "📊 客観的画质評価 (バッチ平均)",
        "m_snr": "絶対品質 (Auto-SNR)",
        "m_cnr": "組織コントラスト (Auto-CNR)",
        "m_psnr": "変更度合い (Diff-PSNR)",
        "m_rna": "除去ノイズ割合 (%)",
        "m_ssim": "構造の忠実度 (Diff-SSIM)",
        "preview_title": "👁️ プレビュー: {filename} (中央スライス: {slice_idx})",
        "preview_in": "ノイズ除去前 (Raw Input)",
        "preview_out": "HAO-Former ノイズ除去後",
        "model_loaded": "✅ 専用ノイズ除去モデルを自動ロードしました: **{seq_type}**",
        "pacs_title": "📡 PACS アーカイブ設定",
        "pacs_send_btn": "📤 ノイズ除去後のDICOMをPACSに送信",
        "pacs_ok": "🎉 成功！結果がPACSにアーカイブされました。",
        "pacs_err": "❌ PACS接続を確立できません。",
        "pacs_test_ok": "✅ C-ECHO 成功！サーバーと通信可能です。",
        "pacs_test_err": "❌ C-ECHO 失敗！ネットワーク設定を確認してください。",
        "pacs_not_dicom": "⚠️ 注意：NIfTIはPACS送信をサポートしていません。DICOM ZIPをアップロードしてください。"
    }
}

# ==========================================
# 1. 核心算法库 (全对齐最新高频补偿与ITHP评价)
# ==========================================
def load_model_fresh(model_path, device):
    model = Restormer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

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
    2. 原图物理均值乘性对齐 (Multiplicative Gain)
    3. 高频解剖微结构自适应补偿 (High-Frequency Detail Booster)
    4. 降噪强度加权融合 (默认 0.72)
    5. 超高亮信号无损补全
    6. 背景空气底噪单向门控
    """
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
    
    tissue_thresh = torch.quantile(valid_raw, 0.15)
    tissue_mask = smoothed_raw > tissue_thresh
    
    # 组织区域物理亮度乘性对齐
    if tissue_mask.sum() > 100:
        raw_mean = raw_tensor_3d[tissue_mask].mean()
        pred_mean = pred_restored[tissue_mask].mean()
        
        gain = raw_mean / (pred_mean + 1e-8)
        gain = torch.clamp(gain, min=0.5, max=2.0)
        
        pred_restored[tissue_mask] = pred_restored[tissue_mask] * gain

    # 提取骨小梁、软骨间隙与肌束的高频纹理残差
    high_pass_detail = raw_tensor_3d - smoothed_raw

    # 降噪强度融合 + 高频解剖细节定向注入
    refined_output = strength * pred_restored + (1.0 - strength) * raw_tensor_3d
    refined_output[tissue_mask] = refined_output[tissue_mask] + detail_boost * high_pass_detail[tissue_mask]

    # 超高亮信号无损补全
    high_mask = smoothed_raw > p99
    if high_mask.sum() > 0:
        refined_output[high_mask] = raw_tensor_3d[high_mask]

    # 背景空气底噪单向门控 (阈值 0.05)
    bg_thresh = torch.quantile(valid_raw, 0.05)
    bg_mask = smoothed_raw <= bg_thresh
    refined_output[bg_mask] = torch.min(refined_output[bg_mask], raw_tensor_3d[bg_mask])
    
    # 强制钳制在原图真实物理区间
    refined_output = torch.clamp(refined_output, min=raw_min, max=raw_max)

    return refined_output

def get_intra_tissue_noise(vol_tensor, tissue_thresh):
    """
    组织内部同质图块底噪估算法 (ITHP 金标准)
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

def compute_SSIM(img1, img2, data_range=1.0, window_size=11):
    window = create_window(window_size, 1).type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
    sigma1_sq = F.relu(F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq)
    sigma2_sq = F.relu(F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq)
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2
    C1, C2 = (0.01 * data_range)**2, (0.03 * data_range)**2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean().item()

def get_dicom_names(extract_dir):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(extract_dir)
    if series_IDs:
        return sitk.ImageSeriesReader.GetGDCMSeriesFileNames(extract_dir, series_IDs[0])
    for root, _, _ in os.walk(extract_dir):
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
        if series_IDs:
            return sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series_IDs[0])
    return []

def array_to_preview_image(arr):
    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    arr_clipped = np.clip(arr, p1, p99)
    if p99 - p1 == 0:
        return np.zeros_like(arr_clipped, dtype=np.uint8)
    norm_arr = (255.0 * (arr_clipped - p1) / (p99 - p1)).astype(np.uint8)
    return norm_arr

# ==========================================
# 2. 单文件核心推理引擎
# ==========================================
def run_single_pipeline(input_path, output_path, model, device, batch_size, is_dicom_zip, lang_dict, sub_progress):
    dicom_paths_to_send = []
    
    with torch.no_grad(): 
        sub_progress.progress(0.05, text=lang_dict["status_unzip"])
        
        if is_dicom_zip:
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            dicom_names = get_dicom_names(extract_dir)
            if not dicom_names:
                return None
                
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_names)
            con_img = reader.Execute()
            
            dcms = []
            for f in dicom_names:
                d = pydicom.dcmread(f)
                try: d.decompress() 
                except: pass
                dcms.append(d)
            original_dtype = dcms[0].pixel_array.dtype
        else:
            con_img = sitk.ReadImage(input_path)

        con_arr_raw = sitk.GetArrayFromImage(con_img)
        original_shape = con_arr_raw.shape 
        
        con_tensor_3d = torch.from_numpy(con_arr_raw).float().to(device)
        x_model_input, p1, p99, raw_min, raw_max = process_3d_tensor_gpu_infer(con_tensor_3d)
        x_model_input = x_model_input.unsqueeze(1) # [Z, 1, H, W]

        # 动态 Padding 适配 U-Net 跳跃连接 (16 的倍数)
        _, _, H_orig, W_orig = x_model_input.shape
        pad_h = (16 - H_orig % 16) % 16
        pad_w = (16 - W_orig % 16) % 16
        
        if pad_h > 0 or pad_w > 0:
            x_model_input = F.pad(x_model_input, (0, pad_w, 0, pad_h), mode='reflect')

        pred_tensors = []
        for b in range(0, original_shape[0], batch_size):
            batch_x = x_model_input[b : b+batch_size]
            batch_pred = model(batch_x)
            pred_tensors.append(batch_pred)
            sub_progress.progress(0.1 + 0.5 * (b / original_shape[0]), text=lang_dict["status_infer"])
            
        pred_tensor = torch.cat(pred_tensors, dim=0)
        
        if pad_h > 0 or pad_w > 0:
            pred_tensor = pred_tensor[:, :, :H_orig, :W_orig]
        
        # 执行带有高频解剖细节补偿的物理极值还原
        out_physical_tensor = inverse_process_and_refine(
            pred_tensor.squeeze(1), p1, p99, con_tensor_3d, raw_min, raw_max, 
            strength=0.72, detail_boost=0.25
        )
        pred_vol_np = out_physical_tensor.cpu().numpy()

        sub_progress.progress(0.7, text=lang_dict["status_pack"])
        if is_dicom_zip:
            save_dcm_dir = tempfile.mkdtemp()
            new_series_uid = generate_uid() if HAS_PACS_LIBS else pydicom.uid.generate_uid()
            
            def clean_dicom_tag(dcm_obj, tag_name):
                if tag_name in dcm_obj:
                    val = dcm_obj.data_element(tag_name).value
                    if isinstance(val, str):
                        dcm_obj.data_element(tag_name).value = val.replace('\x00', '').strip()
            
            for i, dcm in enumerate(dcms):
                slope = float(dcm.RescaleSlope) if 'RescaleSlope' in dcm else 1.0
                intercept = float(dcm.RescaleIntercept) if 'RescaleIntercept' in dcm else 0.0
                val = (pred_vol_np[i] - intercept) / slope
                val = np.clip(val, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max)
                
                if hasattr(dcm, 'file_meta'):
                    dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                dcm.is_little_endian = True
                dcm.is_implicit_VR = False
                
                dcm.PixelData = val.astype(original_dtype).tobytes()
                
                clean_dicom_tag(dcm, 'PatientName')
                clean_dicom_tag(dcm, 'PatientID')
                clean_dicom_tag(dcm, 'StudyInstanceUID')
                
                if 'SeriesInstanceUID' in dcm:
                    dcm.SeriesInstanceUID = new_series_uid
                
                dcm.SOPInstanceUID = generate_uid() if HAS_PACS_LIBS else pydicom.uid.generate_uid()
                
                if "SeriesNumber" in dcm:
                    try: dcm.SeriesNumber = int(dcm.SeriesNumber) + 1000
                    except ValueError: pass
                
                if 'SeriesDescription' in dcm:
                    old_desc = str(dcm.SeriesDescription).replace('\x00', '').strip()
                    dcm.SeriesDescription = f"{old_desc}_HAO_Denoised"
                else:
                    dcm.SeriesDescription = "HAO_Denoised"
                
                save_file = os.path.join(save_dcm_dir, f"{i:04d}.dcm")
                dcm.save_as(save_file)
                dicom_paths_to_send.append(save_file)
                
            with zipfile.ZipFile(output_path, 'w') as zipf:
                for root, _, files in os.walk(save_dcm_dir):
                    for f in files:
                        zipf.write(os.path.join(root, f), f)
            shutil.rmtree(extract_dir)
        else:
            # 数据类型还原与 Gzip 强力无损压缩存储
            pred_itk = sitk.GetImageFromArray(pred_vol_np)
            pred_itk.CopyInformation(con_img) 
            pred_itk = sitk.Cast(pred_itk, con_img.GetPixelID())
            sitk.WriteImage(pred_itk, output_path, useCompression=True)

        # ========================================================
        # 🎯 核心评价对齐区：组织内部同质图块法 (ITHP)
        # ========================================================
        sub_progress.progress(0.85, text=lang_dict["status_metrics"])
        
        flat_in = con_tensor_3d.view(-1)
        valid_in = flat_in[flat_in > 1.0]
        if len(valid_in) == 0:
            valid_in = flat_in

        # 定义组织主体判定阈值 (完全对齐最新 measure.py)
        tissue_thresh_3d = torch.quantile(valid_in, 0.45)
        
        # 提取组织内部真实底噪
        vol_sigma_in = get_intra_tissue_noise(con_tensor_3d, tissue_thresh_3d)
        vol_sigma_out = get_intra_tissue_noise(out_physical_tensor, tissue_thresh_3d)

        p1_norm = torch.quantile(valid_in, 0.005)
        p99_norm = torch.quantile(valid_in, 0.995)

        snr_in_list, cnr_in_list, snr_out_list, cnr_out_list = [], [], [], []
        psnr_list, ssim_list, rna_list = [], [], []

        for z in range(original_shape[0]):
            in_slice = con_tensor_3d[z]
            out_slice = out_physical_tensor[z]
            
            # 过滤头尾无组织空切片
            t_mask = in_slice > tissue_thresh_3d
            if t_mask.sum() < 500:
                continue  
                
            tissue_in = in_slice[t_mask]
            tissue_out = out_slice[t_mask]
            
            sig_in = torch.mean(tissue_in).item()
            sig_out = torch.mean(tissue_out).item()
            
            snr_in = sig_in / vol_sigma_in.item()
            snr_out = sig_out / vol_sigma_out.item()
            
            # 提取解剖高低对比信号 (80% / 20%)
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
            
            # 映射回统一平等的 [0,1] 空间执行 SSIM/PSNR
            in_slice_norm = (torch.clamp(in_slice, min=p1_norm, max=p99_norm) - p1_norm) / (p99_norm - p1_norm + 1e-8)
            out_slice_norm = (torch.clamp(out_slice, min=p1_norm, max=p99_norm) - p1_norm) / (p99_norm - p1_norm + 1e-8)

            in_slice_4d = in_slice_norm.unsqueeze(0).unsqueeze(0)
            out_slice_4d = out_slice_norm.unsqueeze(0).unsqueeze(0)

            psnr_val = compute_PSNR(out_slice_4d, in_slice_4d, data_range=1.0)
            ssim_val = compute_SSIM(out_slice_4d, in_slice_4d, data_range=1.0)
            rna_val = (10 ** (-psnr_val / 20.0)) * 100
            
            snr_in_list.append(snr_in)
            cnr_in_list.append(cnr_in)
            snr_out_list.append(snr_out)
            cnr_out_list.append(cnr_out)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            rna_list.append(rna_val)

        mid_idx = original_shape[0] // 2
        preview_in_img = array_to_preview_image(con_arr_raw[mid_idx])
        preview_out_img = array_to_preview_image(pred_vol_np[mid_idx])

        sub_progress.progress(1.0, text=lang_dict["success"])
        
        del con_tensor_3d, x_model_input, pred_tensors, pred_tensor, out_physical_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
        if len(snr_in_list) == 0:
            metrics_dict = {
                "snr_in": 0.0, "snr_out": 0.0,
                "cnr_in": 0.0, "cnr_out": 0.0,
                "psnr": 0.0, "rna": 0.0, "ssim": 0.0
            }
        else:
            metrics_dict = {
                "snr_in": np.mean(snr_in_list), "snr_out": np.mean(snr_out_list),
                "cnr_in": np.mean(cnr_in_list), "cnr_out": np.mean(cnr_out_list),
                "psnr": np.mean(psnr_list), "rna": np.mean(rna_list), "ssim": np.mean(ssim_list)
            }
            
        return {
            "metrics": metrics_dict,
            "preview_in": preview_in_img,
            "preview_out": preview_out_img,
            "mid_idx": mid_idx,
            "dicom_paths_to_send": dicom_paths_to_send
        }

# ==========================================
# 3. Streamlit UI 布局与批量管理流
# ==========================================
def main():
    selected_lang = st.sidebar.selectbox("🌐 Language / 语言 / 言語", list(LANGUAGES.keys()))
    t = LANGUAGES[selected_lang]
    
    st.sidebar.markdown(f"### {t['sidebar_title']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"{t['device_info']} **{device_name}**")
    
    batch_size = st.sidebar.slider(t['batch_size'], min_value=1, max_value=32, value=8, step=1)
    
    st.sidebar.markdown("---")
    st.sidebar.header(t["pacs_title"])
    if not HAS_PACS_LIBS:
        st.sidebar.error("缺少 PACS 组件，请在终端执行:\npip install pydicom pynetdicom")
    else:
        pacs_ip = st.sidebar.text_input("Target IP / 目标 IP", value="127.0.0.1")
        pacs_port = st.sidebar.text_input("Target Port / 目标端口", value="104")
        pacs_remote_aet = st.sidebar.text_input("Remote AET / 目标 AET", value="PACS")
        pacs_local_aet = st.sidebar.text_input("Local AET / 本地 AET", value="MR_Denoise")
        
        if st.sidebar.button("📡 C-ECHO", width='stretch'):
            try:
                ae = AE(ae_title=pacs_local_aet.encode('utf-8'))
                ae.requested_contexts = VerificationPresentationContexts
                assoc = ae.associate(pacs_ip, int(pacs_port), ae_title=pacs_remote_aet.encode('utf-8'))
                if assoc.is_established:
                    status = assoc.send_c_echo()
                    assoc.release()
                    if status:
                        st.sidebar.success(t['pacs_test_ok'])
                    else:
                        st.sidebar.error(t['pacs_test_err'])
                else:
                    st.sidebar.error(t['pacs_test_err'])
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    st.markdown(f"<h1 style='text-align: center; color: #4C72B0;'>{t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: gray;'>{t['subtitle']}</h4>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_files = st.file_uploader(t['upload_label'], type=["zip", "nii.gz"], help=t['upload_help'], accept_multiple_files=True)

    if st.button(t['run_btn'], type="primary", width='stretch'):
        if not uploaded_files:
            st.error(t['warn_no_file'])
        else:
            st.session_state['dicoms_to_send'] = []
            
            total_files = len(uploaded_files)
            main_progress = st.progress(0.0, text=f"Batch Progress: 0 / {total_files}")
            
            output_dir = tempfile.mkdtemp()
            all_metrics = []
            loaded_models = {}
            
            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name.upper()
                st.write(f"🔄 Processing [{idx+1}/{total_files}]: **{uploaded_file.name}**")
                
                if "PD" in filename:
                    model_ckpt = "./save/PD_Full_Transformer_best.ckpt"
                    seq_type = "PD"
                elif "T1" in filename:
                    model_ckpt = "./save/T1_Full_Transformer_best.ckpt"
                    seq_type = "T1"
                else:
                    st.warning(f"{t['warn_name']} : {uploaded_file.name}")
                    main_progress.progress((idx + 1) / total_files)
                    continue
                    
                if not os.path.exists(model_ckpt):
                    st.error(f"Missing weights: {model_ckpt}")
                    continue
                
                if model_ckpt not in loaded_models:
                    with st.spinner(f"Loading weights {os.path.basename(model_ckpt)}..."):
                        loaded_models[model_ckpt] = load_model_fresh(model_ckpt, device)
                current_model = loaded_models[model_ckpt]

                st.success(t["model_loaded"].format(seq_type=seq_type))

                temp_input_dir = tempfile.mkdtemp()
                input_ext = ".zip" if uploaded_file.name.endswith(".zip") else ".nii.gz"
                is_dicom_zip = input_ext == ".zip"
                
                in_file_path = os.path.join(temp_input_dir, f"input{input_ext}")
                out_filename = f"denoised_{uploaded_file.name}"
                out_file_path = os.path.join(output_dir, out_filename)
                
                uploaded_file.seek(0)
                with open(in_file_path, "wb") as f:
                    f.write(uploaded_file.read())

                sub_progress = st.progress(0.0)
                
                result_dict = run_single_pipeline(in_file_path, out_file_path, current_model, device, batch_size, is_dicom_zip, t, sub_progress)
                
                if result_dict:
                    metrics = result_dict["metrics"]
                    all_metrics.append(metrics)
                    
                    if is_dicom_zip and result_dict.get("dicom_paths_to_send"):
                        st.session_state['dicoms_to_send'].extend(result_dict["dicom_paths_to_send"])
                    elif not is_dicom_zip:
                        st.info(t["pacs_not_dicom"])
                        
                    with st.container():
                        st.markdown(f"#### {t['preview_title'].format(filename=uploaded_file.name, slice_idx=result_dict['mid_idx'])}")
                        
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(result_dict["preview_in"], caption=t['preview_in'], width='stretch')
                        with col_img2:
                            st.image(result_dict["preview_out"], caption=t['preview_out'], width='stretch')
                            
                        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                        m_col1.metric(label=t['m_snr'], value=f"{metrics['snr_out']:.2f}", delta=f"{metrics['snr_out'] - metrics['snr_in']:.2f}")
                        m_col2.metric(label=t['m_cnr'], value=f"{metrics['cnr_out']:.2f}", delta=f"{metrics['cnr_out'] - metrics['cnr_in']:.2f}")
                        m_col3.metric(label=t['m_psnr'], value=f"{metrics['psnr']:.2f} dB", delta_color="off")
                        m_col4.metric(label=t['m_rna'], value=f"{metrics['rna']:.2f} %", delta_color="off")
                        m_col5.metric(label=t['m_ssim'], value=f"{metrics['ssim']:.4f}", delta_color="off")
                    
                    st.markdown("---")
                    
                shutil.rmtree(temp_input_dir)
                main_progress.progress((idx + 1) / total_files, text=f"Batch Progress: {idx+1} / {total_files}")

            if all_metrics:
                st.markdown(f"### {t['metrics_title']}")
                
                avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(label=t['m_snr'], value=f"{avg_metrics['snr_out']:.2f}", delta=f"{avg_metrics['snr_out'] - avg_metrics['snr_in']:.2f}")
                col2.metric(label=t['m_cnr'], value=f"{avg_metrics['cnr_out']:.2f}", delta=f"{avg_metrics['cnr_out'] - avg_metrics['cnr_in']:.2f}")
                col3.metric(label=t['m_psnr'], value=f"{avg_metrics['psnr']:.2f} dB", delta_color="off")
                col4.metric(label=t['m_rna'], value=f"{avg_metrics['rna']:.2f} %", delta_color="off")
                col5.metric(label=t['m_ssim'], value=f"{avg_metrics['ssim']:.4f}", delta_color="off")

                if len(all_metrics) == 1:
                    single_file = os.listdir(output_dir)[0]
                    with open(os.path.join(output_dir, single_file), "rb") as fp:
                        st.download_button(
                            label=t['download_btn'],
                            data=fp,
                            file_name=single_file,
                            mime="application/zip" if single_file.endswith(".zip") else "application/gzip",
                            width='stretch'
                        )
                else:
                    batch_zip_path = os.path.join(tempfile.mkdtemp(), "HAO_Former_Batch_Results.zip")
                    with zipfile.ZipFile(batch_zip_path, 'w') as zf:
                        for f in os.listdir(output_dir):
                            zf.write(os.path.join(output_dir, f), f)
                    
                    with open(batch_zip_path, "rb") as fp:
                        st.download_button(
                            label=t['download_btn'] + " (ZIP)",
                            data=fp,
                            file_name="HAO_Former_Batch_Results.zip",
                            mime="application/zip",
                            width='stretch'
                        )

    if HAS_PACS_LIBS and st.session_state.get('dicoms_to_send'):
        st.write("---")
        st.subheader("📤 PACS")                
        if st.button(t['pacs_send_btn'], type="primary", width='stretch'):
            try:
                ae = AE(ae_title=pacs_local_aet.encode('utf-8'))
                
                contexts_dict = {}
                for dcm_path in st.session_state['dicoms_to_send']:
                    ds_test = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    sop = ds_test.SOPClassUID
                    tx = ds_test.file_meta.TransferSyntaxUID if hasattr(ds_test, 'file_meta') else pydicom.uid.ImplicitVRLittleEndian
                    
                    if sop not in contexts_dict:
                        contexts_dict[sop] = set()
                    contexts_dict[sop].add(tx)
                    contexts_dict[sop].add(pydicom.uid.ExplicitVRLittleEndian)
                    contexts_dict[sop].add(pydicom.uid.ImplicitVRLittleEndian)

                custom_contexts = []
                for sop_class, tx_list in contexts_dict.items():
                    custom_contexts.append(build_context(sop_class, transfer_syntax=list(tx_list)))
                
                ae.requested_contexts = custom_contexts
                assoc = ae.associate(pacs_ip, int(pacs_port), ae_title=pacs_remote_aet.encode('utf-8'))
                
                if assoc.is_established:
                    sent_count = 0
                    total_dcm = len(st.session_state['dicoms_to_send'])
                    progress_pacs = st.progress(0)
                    
                    for i, ds in enumerate(st.session_state['dicoms_to_send']):
                        status = assoc.send_c_store(pydicom.dcmread(ds))
                        if status and status.Status in [0x0000, 0xB000, 0xB006, 0xB007]:
                            sent_count += 1
                        progress_pacs.progress((i + 1) / total_dcm)
                        
                    assoc.release()
                    
                    if sent_count == total_dcm:
                        st.success(f"{t['pacs_ok']} ({sent_count}/{total_dcm})")
                    else:
                        st.warning(f"⚠️ 传输结束，但仅有 {sent_count}/{total_dcm} 个文件被 PACS 接受。")
                else:
                    st.error(t['pacs_err'])
            except Exception as e:
                st.error(f"❌ PACS传输发生系统异常: {e}")

if __name__ == "__main__":
    main()