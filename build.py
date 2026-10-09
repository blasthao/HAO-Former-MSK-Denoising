# -*- coding: utf-8 -*-
import sys
import os
import shutil
import PyInstaller.__main__

# ==========================================
# 1. 突破 Python 默认的递归深度限制
# ==========================================
sys.setrecursionlimit(5000)

# 2. 自动检测环境路径（兼容 Conda 与 Virtualenv）
env_prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV") or sys.prefix
print(f"✅ 正在使用 Python 环境: {env_prefix}")
print("🚀 开始编译 HAO-Former MRI 降噪便携版...\n")

# 3. 动态寻找 site-packages 中的 Streamlit 和 Altair
if os.name == 'nt':
    site_packages = os.path.join(env_prefix, "Lib", "site-packages")
else:
    site_packages = os.path.join(env_prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")

streamlit_path = os.path.join(site_packages, "streamlit") + ";streamlit"
altair_path = os.path.join(site_packages, "altair") + ";altair"

# 4. 执行 PyInstaller 核心编译
PyInstaller.__main__.run([
    'launcher.py',
    '--noconfirm',
    '--onedir',
    '--console',
    '--add-data=app.py;.',
    '--add-data=networks.py;.',
    '--add-data=save;save',  # 打包包含 T1/PD 权重的 save 文件夹[cite: 12, 13]
    f'--add-data={streamlit_path}',
    f'--add-data={altair_path}',
    '--collect-all=pynetdicom',  # 全量收集 PACS 传输依赖
    '--hidden-import=streamlit.runtime.scriptrunner.magic_funcs',
    '--hidden-import=streamlit.runtime.scriptrunner.script_runner',
    '--hidden-import=altair.vegalite.v4',
    '--hidden-import=altair.vegalite.v5',
    '--hidden-import=pydicom.encoders.gdcm',
    '--hidden-import=pydicom.encoders.pylibjpeg',
    '--hidden-import=networks',
    '--hidden-import=torch',
    '--copy-metadata=streamlit'
])

# 5. 自动双保险同步拷贝 save 权重文件夹
print("\n===================================================")
print("正在同步 HAO-Former 模型权重 (save 文件夹)...")
weights_src = "save"
weights_dst = os.path.join("dist", "launcher", "save")

if os.path.exists(weights_src):
    try:
        if os.path.exists(weights_dst):
            shutil.rmtree(weights_dst)
        shutil.copytree(weights_src, weights_dst)
        print("✅ HAO-Former 模型权重同步成功！")
    except Exception as e:
        print(f"⚠️ 拷贝权重时遇到问题: {e}")
else:
    print("⚠️ 未找到 save 文件夹，请确保模型权重已手动存至 dist/launcher/save 目录。")

print("===================================================")
print("🎉 HAO-Former MRI 降噪系统【绿色便携版】编译完成！")
print("请前往 dist/launcher 目录查看您生成的便携式程序。")
print("===================================================")