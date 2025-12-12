# 🌐 Vishwash_MLS – Midline Shift Detection (3D UNet)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-3D%20UNet-red)
![Status](https://img.shields.io/badge/Project-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A complete deep-learning pipeline for **Midline Shift (MLS)** detection from brain CT scans using a lightweight **3D UNet** architecture.  
Supports PyTorch + ONNX inference, MLS computation in millimeters, and visualization outputs.

---

## ⭐ Features
- ⚙️ 3D UNet for volumetric segmentation  
- 📈 Computes MLS (Midline Shift) in **mm**  
- 🧪 Training + inference pipeline included  
- 🚀 ONNX export + CPU inference  
- 🖼 Saves masks, overlays, NIfTI files  
- 📂 Supports NIfTI (`.nii/.nii.gz`) and NumPy (`.npy`)  

---

## 🚀 Quickstart

### 1️⃣ Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


**2️⃣ Prepare dataset**

Place your preprocessed CT volumes here:
data/mls/
    ├── vol_001.nii.gz
    ├── vol_002.nii.gz
    └── ...


**3️⃣ Train the 3D UNet**
python src/trainer.py --train_dir data/mls --epochs 10 --out checkpoints/mls3d.pth

**4️⃣ Run inference**
python src/inference.py --model checkpoints/mls3d.pth --input path/to/volume.nii.gz

Outputs include:
MLS value (mm)
segmentation mask
NIfTI results
overlay PNGs

**🧠 What is Midline Shift?**

Midline Shift (MLS) is a key radiological metric used in:
traumatic brain injury
hemorrhage
tumors causing mass effect
edema or swelling

Even 2–5 mm deviation can be clinically significant.
This project automatically measures MLS using predicted midline structures and voxel spacing.

📦 Project Structure:
Vishwash_MLS/
│
├── src/
│   ├── models/              # 3D UNet
│   ├── dataset/             # loading + preprocessing
│   ├── utils/               # MLS measurement
│   ├── trainer.py           # training pipeline
│   └── inference.py         # inference pipeline
│
├── checkpoints/             # trained models
├── data/                    # dataset (ignored by git)
├── export_onnx.py           # ONNX exporter
├── test_onnx_infer_verbose.py
├── app.py                   # API / viewer
└── requirements.txt

⚡ ONNX Export (Optional)
Export: python export_onnx.py

Run ONNX inference: python test_onnx_infer_verbose.py

📲** Web API / Viewer**
Start server: python app.py --serve --onnx checkpoints/mls3d.onnx --host 0.0.0.0 --port 7860

Includes:
Upload volume
Compute MLS
Display mask + overlays





