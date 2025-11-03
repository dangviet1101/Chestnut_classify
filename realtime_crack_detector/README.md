# Real-time Crack/Defect Detector (YOLOv8)

## Quick Start (Windows)
1. Install Python 3.12.
2. Open **PowerShell** in this folder.
3. Install deps:
   ```powershell
   pip install -r requirements.txt
   ```
4. Put your trained model at **models/best.pt**.
5. Run webcam detection:
   ```powershell
   python predict_rt.py --cfg configs/config.yaml
   ```
   Or double-click **run_camera.bat**.

## Change Inputs
- In `configs/config.yaml` set:
  - `source: 0` for webcam, or `source: path/to/video.mp4` or a folder of images.
  - `weights: models/best.pt` (you can also export ONNX/engine and use a different runtime if needed).
  - `conf`, `iou`, `imgsz` according to your needs.
- You can also override via CLI:
  ```powershell
  python predict_rt.py --cfg configs/config.yaml --source path/to/video.mp4 --conf 0.3 --show
  ```

## Files
- `models/best.pt` – your trained Ultralytics model
- `predict_rt.py` – real-time inference script (Ultralytics backend)
- `configs/config.yaml` – runtime parameters
- `requirements.txt` – pip dependencies
- `class_names.txt` – class names (optional helper if switching to ONNX/TRT later)
- `run_camera.bat` – Windows launcher

## Notes
- Tested with:
  - Python 3.12
  - ultralytics==8.3.222
  - torch==2.6.0+cu124 (for .pt), CUDA-capable GPU recommended
- For ONNX export:
  ```powershell
  yolo mode=export model="models/best.pt" format=onnx opset=12 imgsz=640
  ```
  Then use an ONNX-based script (not included here) with `onnxruntime-gpu`.
