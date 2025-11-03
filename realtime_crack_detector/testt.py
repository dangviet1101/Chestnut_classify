# Batch test images (no camera)
# YOLOv8 (crack/defect) + White-background minAreaRect aspect ratio (round/not-round) + Grade mapping
# Grade map: round -> 1, not-round -> 2, crack/defect -> 3

import os
import sys
import csv
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

# ---------------- CLI ----------------
def get_args():
    ap = argparse.ArgumentParser("Chestnut hybrid grader (folder only)")
    ap.add_argument("--weights", default=r"D:\An\Final_deadline\realtime_crack_detector\models\best.pt",
                    help="Path to YOLO .pt")
    ap.add_argument("--src", default=r"D:\An\Final_deadline\realtime_crack_detector\test",
                    help="Folder of images to process")
    ap.add_argument("--out", default=r"D:\An\Final_deadline\realtime_crack_detector\results",
                    help="Output folder for annotated images and CSV")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO conf threshold")
    ap.add_argument("--iou",  type=float, default=0.50, help="YOLO IoU threshold")
    # White-background aspect ratio thresholds
    ap.add_argument("--ar-thr", type=float, default=1.18,
                    help="AR=max(w,h)/min(w,h) <= ar_thr => round")
    ap.add_argument("--min-area", type=int, default=300,
                    help="Ignore contours smaller than this area (pixels)")
    ap.add_argument("--csv", action="store_true", help="Export CSV results")
    return ap.parse_args()

# ------------- Shape (white bg + minAreaRect) -------------
def shape_tag_from_crop(crop_bgr, ar_thr=1.18, min_area=300):
    """
    round if AR = max(w,h)/min(w,h) <= ar_thr, measured from rotated minAreaRect
    Assumes white background.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu on gray, invert so object=white
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "unknown"
    c = max(cnts, key=cv2.contourArea)

    if cv2.contourArea(c) < min_area:
        return "unknown"

    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    if w <= 1e-3 or h <= 1e-3:
        return "unknown"
    ar = max(w, h) / min(w, h)

    return "round" if ar <= ar_thr else "not-round"

def grade_from_shape(shape_tag):
    if shape_tag == "round":
        return "1"
    if shape_tag == "not-round":
        return "2"
    return "N/A"

# ---------------- Main ----------------
def main():
    args = get_args()
    REJECT_CLASSES = {"crack", "defect"}  # -> Grade 3

    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights not found: {args.weights}")
        sys.exit(1)
    if not os.path.isdir(args.src):
        print(f"[ERROR] Source folder not found: {args.src}")
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)

    model = YOLO(args.weights)

    csv_writer = None
    csv_f = None
    if args.csv:
        csv_f = open(os.path.join(args.out, "results.csv"), "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["image", "detected_class", "shape", "confidence", "grade", "x1", "y1", "x2", "y2"])

    files = sorted([f for f in os.listdir(args.src)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))])

    if not files:
        print("[WARN] No images found in source folder.")
        return

    for fname in files:
        fpath = os.path.join(args.src, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"[WARN] Cannot read image: {fpath}")
            continue

        # YOLO inference
        try:
            results = model.predict(img, conf=args.conf, iou=args.iou, verbose=False)
        except Exception as e:
            print(f"[ERROR] YOLO failure on {fname}: {e}")
            continue

        detections = []  # (x1,y1,x2,y2, cls_name, conf)
        has_reject = False

        for r in results:
            names = r.names
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_name = names.get(int(b.cls[0]), str(int(b.cls[0])))
                conf = float(b.conf[0])
                detections.append((x1, y1, x2, y2, cls_name, conf))
                if cls_name.lower() in REJECT_CLASSES:
                    has_reject = True

        if has_reject:
            # Case 1: crack/defect -> Grade 3
            for (x1, y1, x2, y2, cls_name, conf) in detections:
                label = f"{cls_name} | Grade 3 | {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if csv_writer:
                    csv_writer.writerow([fname, cls_name, "N/A", f"{conf:.4f}", "3", x1, y1, x2, y2])

        elif detections:
            # Case 2: have bboxes, but none is crack/defect -> evaluate AR on each crop
            H, W = img.shape[:2]
            for (x1, y1, x2, y2, cls_name, conf) in detections:
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(W, x2), min(H, y2)
                crop = img[y1c:y2c, x1c:x2c]
                shape = shape_tag_from_crop(crop, ar_thr=args.ar_thr, min_area=args.min_area)
                grade = grade_from_shape(shape)
                label = f"{shape} | Grade {grade if grade!='N/A' else 'N/A'} | {conf:.2f}"
                cv2.rectangle(img, (x1c, y1c), (x2c, y2c), (0, 200, 0), 2)
                cv2.putText(img, label, (x1c, max(0, y1c - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                if csv_writer:
                    csv_writer.writerow([fname, cls_name, shape, f"{conf:.4f}", grade, x1c, y1c, x2c, y2c])

        else:
            # Case 3: no bbox at all -> evaluate AR on full image
            shape = shape_tag_from_crop(img, ar_thr=args.ar_thr, min_area=args.min_area)
            grade = grade_from_shape(shape)
            label = f"{shape} | Grade {grade if grade!='N/A' else 'N/A'} | no-detect"
            cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            if csv_writer:
                csv_writer.writerow([fname, "no-detect", shape, "N/A", grade, "", "", "", ""])

        # Save annotated image
        out_path = os.path.join(args.out, fname)
        cv2.imwrite(out_path, img)
        print(f"[OK] {fname} -> {out_path}")

    if csv_f:
        csv_f.close()
        print("CSV saved to:", os.path.join(args.out, "results.csv"))
    print("All done. Output folder:", args.out)

if __name__ == "__main__":
    main()
