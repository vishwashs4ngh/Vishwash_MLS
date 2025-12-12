@echo off
REM -----------------------------
REM run_all.bat (Windows-safe version + auto-launch Streamlit)
REM -----------------------------

REM 1) Activate virtual environment (in this window)
call .venv_local\Scripts\activate.bat

REM 2) Create sample volumes
python create_samples.py

REM 3) Quick training
python run_train.py

REM 4) Run inference on sample
python src\inference.py --model checkpoints\mls3d.pth --input data\mls\train\vol_001.npy --out outputs --threshold 0.5

REM 5) Convert pred_mask.nii.gz to pred_mask.npy (use helper script)
python convert_pred_mask.py outputs\pred_mask.nii.gz

REM 6) Generate overlay PNG
python save_overlay_pngs.py outputs pred_mask.nii.gz

REM 7) Open outputs folder
explorer outputs

REM 8) Launch Streamlit in a new terminal window and keep it open
REM The empty title "" is required by start command syntax
start "" cmd /k "call .venv_local\Scripts\activate.bat & streamlit run app.py"

pause
