import os
import glob
import shutil
import random
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_IMAGES_DIR = os.path.join(BASE_DIR, "user_images")
DATASET_DIR = os.path.join(BASE_DIR, "user_trained_dataset")
OUTPUT_WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

def auto_detect_defect_bbox(img, defect_type="hole"):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    median_val = float(np.median(blurred))
    
    diff = cv2.absdiff(blurred, int(median_val))
    _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dilated = cv2.dilate(opened, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_box = None
    max_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < (w * h * 0.40):
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx = bx + bw / 2
            cy = by + bh / 2
            dist_center = np.sqrt(((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2)
            score = area / (1.0 + dist_center * 2.0)
            
            if score > max_score:
                max_score = score
                best_box = (bx, by, bw, bh)
                
    if best_box:
        bx, by, bw, bh = best_box
        return [((bx + bw / 2) / w), ((by + bh / 2) / h), (bw / w), (bh / h)]
    else:
        return [0.5, 0.5, 0.25, 0.25]

def augment_sample(img, boxes):
    samples = []
    
    samples.append((img.copy(), list(boxes)))
    
    h_flipped = cv2.flip(img, 1)
    h_boxes = [(cls_id, 1.0 - cx, cy, bw, bh) for (cls_id, cx, cy, bw, bh) in boxes]
    samples.append((h_flipped, h_boxes))
    
    v_flipped = cv2.flip(img, 0)
    v_boxes = [(cls_id, cx, 1.0 - cy, bw, bh) for (cls_id, cx, cy, bw, bh) in boxes]
    samples.append((v_flipped, v_boxes))
    
    hv_flipped = cv2.flip(img, -1)
    hv_boxes = [(cls_id, 1.0 - cx, 1.0 - cy, bw, bh) for (cls_id, cx, cy, bw, bh) in boxes]
    samples.append((hv_flipped, hv_boxes))
    
    for factor in [0.72, 0.88, 1.14, 1.30]:
        b_img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        samples.append((b_img, list(boxes)))
        b_flip = np.clip(h_flipped.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        samples.append((b_flip, h_boxes))
        
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    samples.append((blur, list(boxes)))
    
    for alpha in [0.85, 1.25]:
        c_img = np.clip(img.astype(np.float32) * alpha + (128 * (1 - alpha)), 0, 255).astype(np.uint8)
        samples.append((c_img, list(boxes)))
        
    return samples

def build_dataset_from_user_files():
    print(f"\n[Scanner] Checking user images in: {USER_IMAGES_DIR}")
    
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    
    hole_files = []
    foreign_files = []
    clean_files = []
    
    for ext in extensions:
        hole_files.extend(glob.glob(os.path.join(USER_IMAGES_DIR, "holes", ext)))
        foreign_files.extend(glob.glob(os.path.join(USER_IMAGES_DIR, "foreign_objects", ext)))
        clean_files.extend(glob.glob(os.path.join(USER_IMAGES_DIR, "clean", ext)))
        for f in glob.glob(os.path.join(USER_IMAGES_DIR, ext)):
            fname = os.path.basename(f).lower()
            if "hole" in fname or "puncture" in fname:
                hole_files.append(f)
            elif "foreign" in fname or "object" in fname:
                foreign_files.append(f)
            else:
                hole_files.append(f)
                
    print(f"Found: {len(hole_files)} Hole images, {len(foreign_files)} Foreign Object images, {len(clean_files)} Clean Belt images.")
    
    if len(hole_files) == 0 and len(foreign_files) == 0:
        print("[Notice] No images found in 'user_images/' yet.")
        print(f"Please place your photos into: {USER_IMAGES_DIR}")
        return None

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        p = os.path.join(DATASET_DIR, sub)
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)

    all_samples = []
    
    for f in hole_files:
        img = cv2.imread(f)
        if img is None:
            continue
        txt_path = os.path.splitext(f)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path) as tf:
                lines = [l.strip().split() for l in tf if l.strip()]
                boxes = [(int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])) for l in lines]
        else:
            box = auto_detect_defect_bbox(img, "hole")
            boxes = [(0, box[0], box[1], box[2], box[3])]
            
        augs = augment_sample(img, boxes)
        all_samples.extend(augs)

    for f in foreign_files:
        img = cv2.imread(f)
        if img is None:
            continue
        txt_path = os.path.splitext(f)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path) as tf:
                lines = [l.strip().split() for l in tf if l.strip()]
                boxes = [(int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])) for l in lines]
        else:
            box = auto_detect_defect_bbox(img, "foreign_object")
            boxes = [(1, box[0], box[1], box[2], box[3])]
            
        augs = augment_sample(img, boxes)
        all_samples.extend(augs)

    for f in clean_files:
        img = cv2.imread(f)
        if img is None:
            continue
        augs = augment_sample(img, [])
        all_samples.extend(augs)

    random.shuffle(all_samples)
    val_count = max(2, int(len(all_samples) * 0.20))
    train_samples = all_samples[val_count:]
    val_samples = all_samples[:val_count]

    print(f"[Augmentor] Multiplied into {len(train_samples)} training samples and {len(val_samples)} validation samples.")

    for i, (img, bxs) in enumerate(train_samples):
        ip = os.path.join(DATASET_DIR, "images", "train", f"usr_train_{i:04d}.jpg")
        lp = os.path.join(DATASET_DIR, "labels", "train", f"usr_train_{i:04d}.txt")
        cv2.imwrite(ip, img)
        with open(lp, "w") as lf:
            for cls_id, cx, cy, bw, bh in bxs:
                lf.write(f"{cls_id} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")

    for i, (img, bxs) in enumerate(val_samples):
        ip = os.path.join(DATASET_DIR, "images", "val", f"usr_val_{i:04d}.jpg")
        lp = os.path.join(DATASET_DIR, "labels", "val", f"usr_val_{i:04d}.txt")
        cv2.imwrite(ip, img)
        with open(lp, "w") as lf:
            for cls_id, cx, cy, bw, bh in bxs:
                lf.write(f"{cls_id} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")

    abs_data_dir = os.path.abspath(DATASET_DIR).replace('\\', '/')
    yaml_path = os.path.join(DATASET_DIR, "user_conveyor.yaml")
    yaml_content = f"""path: {abs_data_dir}
train: images/train
val: images/val

names:
  0: hole
  1: foreign object
  2: crack
"""
    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    return yaml_path

def train_user_model(yaml_path):
    print("\n[AI Trainer] Starting YOLOv8 transfer learning with your images...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    
    model.train(
        data=yaml_path,
        epochs=35,
        imgsz=640,
        batch=16,
        workers=0,
        mosaic=1.0,
        verbose=True
    )
    
    best_weight_source = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(best_weight_source):
        os.makedirs(OUTPUT_WEIGHTS_DIR, exist_ok=True)
        target_best = os.path.join(OUTPUT_WEIGHTS_DIR, "best.pt")
        if os.path.exists(target_best):
            backup_dest = os.path.join(OUTPUT_WEIGHTS_DIR, "best_backup_old.pt")
            shutil.copy2(target_best, backup_dest)
            print(f"[AI Trainer] Backed up old weights to: {backup_dest}")
            
        shutil.copy2(best_weight_source, target_best)
        print(f"[AI Trainer] SUCCESS! New trained weights exported to: {target_best}")

if __name__ == "__main__":
    y_path = build_dataset_from_user_files()
    if y_path:
        train_user_model(y_path)
