
# HAO-Former: A Hybrid-loss Aware Ortho-Transformer for MSK-MRI Denoising

**HAO-Former** is an intelligent musculoskeletal (MSK) magnetic resonance imaging (MRI) denoising and quality enhancement platform. Designed for both proton density-weighted (PD) and T1-weighted sequences, the framework integrates transformer-based feature extraction with high-frequency anatomical detail compensation to suppress image noise while preserving trabecular structures and tissue boundaries. The system incorporates an Intra-Tissue Homogeneous Patch (ITHP) objective quality assessment engine and full picture archiving and communication system (PACS) network integration.

You can also try this app on: https://hao-former-msk-denoising.streamlit.app/

---

## 🌟 Key Features

* **Intelligent Sequence Routing**: Automatically detects the presence of `PD` or `T1` tags in the input filenames and routes images to the corresponding specialized model weights.
* **High-Frequency Detail Preservation**: Incorporates a trabecular and microstructural detail booster to prevent over-smoothing and preserve sharp musculoskeletal boundaries.
* **Multi-Format & PACS Compatibility**:
  * Supports 3D NIfTI volumes (`.nii.gz`) and batch DICOM archives (`.zip`).
  * Generates valid DICOM series with newly assigned Series and SOP Instance UIDs, compliant with PACS storage standards.
  * Provides built-in DICOM network communication utilities, including C-ECHO verification and C-STORE push capability.
* **Objective Quality Evaluation (ITHP Engine)**: Automatically calculates intra-tissue signal-to-noise ratio (Auto-SNR), contrast-to-noise ratio (Auto-CNR), differential peak signal-to-noise ratio (Diff-PSNR), structural fidelity (Diff-SSIM), and relative noise attenuation (RNA %)[cite: 10].
* **Interactive Multilingual Interface**: Built on Streamlit with dynamic multilingual support across English, 中文 (Chinese), and 日本語 (Japanese), featuring synchronous side-by-side slice previews.

---

## 📁 Repository Structure

```text
HAO-Former-MSK/
├── app.py                     # Main Streamlit web application
├── networks.py                # Restormer / Transformer architecture definitions
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
└── save/                      # Directory for pretrained checkpoints
    ├── PD_Full_Transformer_best.ckpt
    └── T1_Full_Transformer_best.ckpt

```

> **Note**: When deploying to platforms like Streamlit Community Cloud, verify that checkpoint files uploaded directly to GitHub do not exceed standard file size limits (100 MB). For larger weights, download them dynamically from a remote storage release URL during startup.

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
cd your-repo

```


2. **Install dependencies** (Python 3.10 or higher recommended):
```bash
pip install -r requirements.txt

```


3. **Launch the web application**:
```bash
streamlit run app.py

```



---

## 📖 Usage Pipeline

1. **Configure Parameters**: Select the interface language, verify compute device availability (CUDA GPU or CPU), and adjust the inference batch size in the sidebar.


2. **Upload Medical Images**:
* Upload individual or multiple `.nii.gz` files, or upload DICOM series compressed into `.zip` archives.


* Ensure file names contain either `PD` or `T1` to enable automatic model selection.




3. **Execute Denoising**: Click **"🚀 Start Batch Denoising"** to execute image normalization, transformer inference, anatomical detail recovery, and automated metric extraction.


4. **Export & PACS Archiving**:
* Download the processed images individually or as a complete batch archive (`.zip`).


* If processing DICOM data, configure the target AE Title, IP address, and port in the PACS section, verify connectivity with C-ECHO, and archive the output with C-STORE.





---

## 🙏 Acknowledgements

This project was developed with inspiration and reference from the following open-source contributions:

* **RED-CNN**: Residual Encoder-Decoder Convolutional Neural Network for Low-Dose CT ([https://github.com/SSinyu/RED-CNN.git](https://github.com/SSinyu/RED-CNN.git)). We acknowledge the foundational contributions of this repository to medical image denoising workflows and residual learning architectures.
* **Restormer**: Efficient Transformer for High-Resolution Image Restoration.

---

## ⚖️ Disclaimer

This platform and its associated algorithms are intended strictly for academic research, educational demonstrations, and algorithm benchmarking. They are not certified as medical devices and should not be used as the sole basis for clinical diagnosis or treatment planning.

```

```
<img width="5340" height="1739" alt="statistical_slice_metrics_comparison" src="https://github.com/user-attachments/assets/beb7c02e-2f5c-463f-ae05-ec56ca9b2d1d" />

<img width="2224" height="1440" alt="knee-cor" src="https://github.com/user-attachments/assets/7a2feeeb-3b6f-4921-9976-e57f2c6422df" />

<img width="2188" height="1422" alt="knee-sag" src="https://github.com/user-attachments/assets/8512947d-f0ea-4543-8c06-387d15be9e7e" />

<img width="2276" height="1432" alt="knee-tra" src="https://github.com/user-attachments/assets/86a1f0b8-444b-48a4-8dbf-aa3a75857c73" />

<img width="2322" height="1530" alt="shoulder_cor" src="https://github.com/user-attachments/assets/6675404d-cad7-4626-a194-1c5105523aef" />
