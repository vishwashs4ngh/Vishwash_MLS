Vishwash_MLS – Midline Shift Detection using 3D UNet

This repository contains a deep-learning pipeline for Midline Shift (MLS) detection from brain CT scans.
It uses a lightweight 3D UNet architecture to segment midline-related structures and estimate MLS for diagnostic support.

📌 Features

3D UNet model designed for low-slice CT volumes

Training, validation and inference scripts

MLS computation (midline deviation in millimeters)

Support for NIfTI (.nii, .nii.gz) and NumPy (.npy) volumes

ONNX export & CPU-friendly inference (optional)

🚀 Quickstart
1. Create virtual environment + install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2. Prepare your dataset

Place preprocessed CT volumes into a directory, for example:

data/mls/
    vol_001.nii.gz
    vol_002.nii.gz
    ...


Each volume should already be resampled/windowed according to your preprocessing pipeline.

3. Train the 3D UNet
python src/trainer.py --train_dir data/mls --epochs 10 --out checkpoints/mls3d.pth


This saves the trained model at:

checkpoints/mls3d.pth

4. Run inference
python src/inference.py --model checkpoints/mls3d.pth --input path/to/volume.nii.gz


The script outputs:

segmentation mask

MLS value (mm)

optional visual overlays

📦 Project Structure
Vishwash_MLS/
│
├── src/
│   ├── models/          # 3D UNet architecture
│   ├── dataset/         # loading & preprocessing
│   ├── utils/           # MLS computation + helpers
│   ├── trainer.py       # model training
│   └── inference.py     # inference pipeline
│
├── checkpoints/         # trained weights (not included)
├── data/                # dataset (ignored in git)
├── requirements.txt
└── README.md

🧠 About Midline Shift

Midline Shift is a crucial indicator in:

traumatic brain injury

intracranial hemorrhage

mass effect conditions

Accurate MLS measurement can guide emergency treatment decisions.

This model estimates MLS by:

segmenting midline-related regions

extracting symmetry deviations

measuring anatomical displacement in millimeters

🔧 Additional Tools (Optional)

You may export the PyTorch model to ONNX for faster CPU inference:

python export_onnx.py


And run ONNX-based inference:

python test_onnx_infer.py

🤝 Contributions

Feel free to open issues or pull requests if you'd like to improve:

model architecture

training pipeline

inference stability

MLS calculation

📬 Contact

Author: Vishwash Singh
GitHub: @vishwashs4ngh
