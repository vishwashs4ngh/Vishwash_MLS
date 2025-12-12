# Vishwash_MLS (Midline Shift) - 3D UNet

Quickstart:
1. Create venv and install: `pip install -r requirements.txt`
2. Put preprocessed NIfTI volumes in a folder (e.g., data/mls/)
3. Train: `python src/trainer.py --train_dir data/mls --epochs 10 --out checkpoints/mls3d.pth`
4. Inference: `python src/inference.py --model checkpoints/mls3d.pth --input path/to/volume.nii.gz`
