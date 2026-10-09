import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
import torch

def register_images(fixed_image, moving_image):
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=100, gradientMagnitudeTolerance=1e-8
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    try:
        final_transform = registration_method.Execute(fixed_image, moving_image)
        resampled_moving = sitk.Resample(
            moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID()
        )
        return resampled_moving
    except Exception as e:
        print(f"\n[警告] 配准遇到问题，已返回未配准的原图: {e}")
        return moving_image

def apply_n4_correction_fast(sitk_img):
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    size = sitk_img.GetSize()
    shrink_factors = [3 if s > 64 else 1 for s in size] 
    
    shrunk_img = sitk.Shrink(sitk_img, shrink_factors)
    mask = sitk.OtsuThreshold(shrunk_img, 0, 1, 200)
    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(shrunk_img, mask)
    
    log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_img)
    bias_field = sitk.Exp(log_bias_field)
    
    corrected_img = sitk.Divide(sitk_img, bias_field)
    return corrected_img

def process_3d_tensor_gpu(tensor_3d):
    """
    【动态尺寸升级版】
    不再要求 target_size，直接提取原图 (Z, H, W) 进行极速归一化，实现 100% 无损原视野输出
    """
    Z, H, W = tensor_3d.shape
    
    flat_tensor = tensor_3d.view(Z, -1)
    p1 = torch.quantile(flat_tensor, 0.01, dim=1, keepdim=True).unsqueeze(-1) 
    p99 = torch.quantile(flat_tensor, 0.99, dim=1, keepdim=True).unsqueeze(-1)
    
    tensor_clipped = torch.max(tensor_3d, p1)
    tensor_clipped = torch.min(tensor_clipped, p99)
    tensor_norm = (tensor_clipped - p1) / (p99 - p1 + 1e-8)
    
    # 直接返回归一化后的原尺寸 Tensor，不再进行任何裁剪和填充！
    return tensor_norm

def save_dataset(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n" + "="*50)
    print(f"[*] 动态尺寸 GPU 预处理引擎启动: {str(device).upper()}")
    print("="*50 + "\n")

    input_files = sorted(glob(os.path.join(args.input_path, '*.nii.gz')))
    total_files = len(input_files)
    if total_files == 0:
        print(f"[!] 在 {args.input_path} 未找到 .nii.gz 文件")
        return

    # 直接确保预处理根目录存在
    os.makedirs(args.save_path, exist_ok=True)

    for i, input_file in enumerate(input_files):
        file_name = os.path.basename(input_file)
        target_file = os.path.join(args.target_path, file_name)

        if not os.path.exists(target_file):
            continue

        input_img = sitk.ReadImage(input_file)
        target_img = sitk.ReadImage(target_file)

        input_img_reg = register_images(fixed_image=target_img, moving_image=input_img)
        
        input_arr = sitk.GetArrayFromImage(apply_n4_correction_fast(input_img_reg))
        target_arr = sitk.GetArrayFromImage(apply_n4_correction_fast(target_img))

        prefix = file_name.replace('.nii.gz', '')
        
        # 目标路径直接设为 preprocess 根目录
        target_dir = args.save_path

        input_tensor_3d = torch.from_numpy(input_arr).float().to(device)
        target_tensor_3d = torch.from_numpy(target_arr).float().to(device)

        input_final_3d = process_3d_tensor_gpu(input_tensor_3d)
        target_final_3d = process_3d_tensor_gpu(target_tensor_3d)

        input_final_cpu = input_final_3d.cpu().numpy()
        target_final_cpu = target_final_3d.cpu().numpy()

        # 直接将 .npy 保存在 preprocess/ 目录下
        for z in range(input_final_cpu.shape[0]):
            np.save(os.path.join(target_dir, f"{prefix}_slice{z:03d}_input.npy"), input_final_cpu[z])
            np.save(os.path.join(target_dir, f"{prefix}_slice{z:03d}_target.npy"), target_final_cpu[z])

        printProgressBar(i + 1, total_files, prefix="预处理进度:", suffix=f"完成 {file_name}", length=40)

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total: print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./Input/', help="存放 Input 图像的文件夹")
    parser.add_argument('--target_path', type=str, default='./Target/', help="存放 Target 图像的文件夹")
    parser.add_argument('--save_path', type=str, default='./preprocess/', help="存放生成的npy的文件夹")
    parser.add_argument('--seed', type=int, default=42)
    # 彻底移除了 val_ratio，因为五折交叉验证不再需要在预处理阶段分割数据
    args = parser.parse_args()
    save_dataset(args)