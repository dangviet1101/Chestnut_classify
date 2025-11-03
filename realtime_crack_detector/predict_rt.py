\
import argparse
import yaml
import cv2
from ultralytics import YOLO

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Real-time YOLOv8 detector (crack/defect)")
    ap.add_argument("--cfg", default="configs/config.yaml", help="Path to YAML config")
    # Optional quick overrides:
    ap.add_argument("--source", type=str, default=None, help="0/webcam, path to video/image/folder")
    ap.add_argument("--weights", type=str, default=None, help="Path to .pt model")
    ap.add_argument("--device", type=str, default=None, help="GPU id or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    ap.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    ap.add_argument("--save", action="store_true", help="Save annotated output to disk")
    ap.add_argument("--nosave", action="store_true", help="Do NOT save output")
    ap.add_argument("--show", action="store_true", help="Show live window")
    ap.add_argument("--noshow", action="store_true", help="Do NOT show live window")
    ap.add_argument("--half", action="store_true", help="Use FP16 (if GPU supports)")
    ap.add_argument("--nohalf", action="store_true", help="Disable FP16")
    ap.add_argument("--vid_stride", type=int, default=None, help="Frame stride")
    ap.add_argument("--project", type=str, default=None, help="Project folder for outputs")
    ap.add_argument("--name", type=str, default=None, help="Run name under project")
    args = ap.parse_args()

    C = load_cfg(args.cfg)

    # Apply overrides if given
    def pick(k, default):
        v = getattr(args, k)
        return default if v is None else v

    source     = pick("source", C.get("source", 0))
    weights    = pick("weights", C.get("weights", "models/best.pt"))
    device     = pick("device", C.get("device", 0))
    imgsz      = pick("imgsz", C.get("imgsz", 640))
    conf       = pick("conf", C.get("conf", 0.25))
    iou        = pick("iou", C.get("iou", 0.5))
    vid_stride = pick("vid_stride", C.get("vid_stride", 1))
    project    = pick("project", C.get("project", "runs"))
    name       = pick("name", C.get("name", "realtime"))

    # Booleans respecting --save/--nosave and --show/--noshow/--half
    save = C.get("save", True)
    if args.save: save = True
    if args.nosave: save = False

    show = C.get("show", False)
    if args.show: show = True
    if args.noshow: show = False

    half = C.get("half", True)
    if args.half: half = True
    if args.nohalf: half = False

    model = YOLO(weights)

    # Inference loop
    # Note: for webcam, source=0 (string "0" or int 0 both okay).
    for r in model.predict(
        source=source,
        stream=True,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        half=half,
        vid_stride=vid_stride,
        save=save,
        project=project,
        name=name,
        show=False  # We'll handle display ourselves to allow ESC key
    ):
        frame = r.plot()
        if show:
            cv2.imshow("Crack/Defect Detector", frame)
            # ESC to quit
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
