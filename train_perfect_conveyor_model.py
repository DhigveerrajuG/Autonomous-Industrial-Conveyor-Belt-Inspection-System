import os
import shutil
import random
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "conveyor_training_data")
OUTPUT_WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

def create_belt_texture(width=640, height=640, dark=False):
    base_lum = random.randint(30, 60) if dark else random.randint(45, 75)
    img = np.full((height, width, 3), base_lum, dtype=np.uint8)
    
    noise = np.random.normal(0, random.uniform(5, 12), (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    groove_spacing = random.randint(18, 36)
    groove_color = max(0, base_lum - random.randint(8, 16))
    for x in range(0, width, groove_spacing):
        cv2.line(img, (x, 0), (x, height), (groove_color, groove_color, groove_color), 1)
        
    if random.random() > 0.4:
        gradient = np.tile(np.linspace(random.uniform(0.75, 0.95), random.uniform(1.05, 1.25), width), (height, 1))
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) * gradient, 0, 255).astype(np.uint8)
            
    return img

def add_synthetic_damage(img):
    h, w = img.shape[:2]
    damage_type = random.choice(['hole', 'hole', 'foreign_object', 'crack', 'tear'])
    boxes = []
    
    if damage_type == 'hole':
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        radius_x = random.randint(18, 55)
        radius_y = random.randint(18, 55)
        
        pts = []
        num_pts = random.randint(10, 16)
        for i in range(num_pts):
            angle = i * (2 * np.pi / num_pts)
            r = random.uniform(0.75, 1.25)
            px = int(cx + np.cos(angle) * radius_x * r)
            py = int(cy + np.sin(angle) * radius_y * r)
            pts.append([px, py])
            
        pts = np.array(pts, np.int32)
        hole_color = (random.randint(120, 210), random.randint(120, 210), random.randint(120, 210)) if random.random() > 0.4 else (8, 10, 12)
        cv2.fillPoly(img, [pts], hole_color)
        cv2.polylines(img, [pts], isClosed=True, color=(160, 150, 140), thickness=2)
        
        for _ in range(random.randint(6, 15)):
            f_angle = random.uniform(0, 2 * np.pi)
            fx1 = int(cx + np.cos(f_angle) * (radius_x * 0.8))
            fy1 = int(cy + np.sin(f_angle) * (radius_y * 0.8))
            fx2 = int(cx + np.cos(f_angle) * (radius_x * 1.2))
            fy2 = int(cy + np.sin(f_angle) * (radius_y * 1.2))
            cv2.line(img, (fx1, fy1), (fx2, fy2), (190, 190, 180), 1)

        x_min = max(4, int(np.min(pts[:, 0])))
        y_min = max(4, int(np.min(pts[:, 1])))
        x_max = min(w - 4, int(np.max(pts[:, 0])))
        y_max = min(h - 4, int(np.max(pts[:, 1])))
        
        boxes.append((0, ((x_min + x_max) / 2) / w, ((y_min + y_max) / 2) / h, (x_max - x_min) / w, (y_max - y_min) / h))

    elif damage_type == 'foreign_object':
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        obj_w = random.randint(28, 65)
        obj_h = random.randint(24, 60)
        
        shadow_pts = np.array([
            [cx - obj_w // 2 + 8, cy + obj_h // 2],
            [cx + obj_w // 2 + 16, cy + obj_h // 2],
            [cx + obj_w // 2 + 8, cy + obj_h // 2 + 10],
            [cx - obj_w // 2, cy + obj_h // 2 + 10]
        ], np.int32)
        cv2.fillPoly(img, [shadow_pts], (18, 20, 24))

        obj_pts = []
        for i in range(8):
            a = i * (2 * np.pi / 8)
            r = random.uniform(0.8, 1.2)
            px = int(cx + np.cos(a) * (obj_w / 2) * r)
            py = int(cy + np.sin(a) * (obj_h / 2) * r)
            obj_pts.append([px, py])
        obj_pts = np.array(obj_pts, np.int32)
        
        rock_color = (random.randint(60, 110), random.randint(70, 130), random.randint(90, 150))
        cv2.fillPoly(img, [obj_pts], rock_color)
        cv2.polylines(img, [obj_pts], True, (30, 35, 40), 2)

        x_min = max(4, int(np.min(obj_pts[:, 0])))
        y_min = max(4, int(np.min(obj_pts[:, 1])))
        x_max = min(w - 4, int(np.max(obj_pts[:, 0])))
        y_max = min(h - 4, int(np.max(obj_pts[:, 1])))

        boxes.append((1, ((x_min + x_max) / 2) / w, ((y_min + y_max) / 2) / h, (x_max - x_min) / w, (y_max - y_min) / h))

    elif damage_type in ('crack', 'tear'):
        cx = random.randint(int(w * 0.25), int(w * 0.75))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        curr_x, curr_y = cx, cy
        pts = [[curr_x, curr_y]]
        
        for _ in range(random.randint(8, 14)):
            curr_x += random.randint(-14, 14)
            curr_y += random.randint(16, 32)
            pts.append([curr_x, curr_y])
            
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(10, 12, 14), thickness=random.randint(4, 7))
        
        x_min = max(4, int(np.min(pts[:, 0]) - 8))
        y_min = max(4, int(np.min(pts[:, 1]) - 8))
        x_max = min(w - 4, int(np.max(pts[:, 0]) + 8))
        y_max = min(h - 4, int(np.max(pts[:, 1]) + 8))
        
        boxes.append((2, ((x_min + x_max) / 2) / w, ((y_min + y_max) / 2) / h, (x_max - x_min) / w, (y_max - y_min) / h))

    return img, boxes

def generate_dataset():
    print("\n[Dataset Generator] Building synthetic Conveyor Belt dataset...")
    
    train_img_dir = os.path.join(DATASET_DIR, "images", "train")
    val_img_dir = os.path.join(DATASET_DIR, "images", "val")
    train_lbl_dir = os.path.join(DATASET_DIR, "labels", "train")
    val_lbl_dir = os.path.join(DATASET_DIR, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    splits = [('train', train_img_dir, train_lbl_dir, 240),
              ('val', val_img_dir, val_lbl_dir, 60)]

    for split_name, img_dir, lbl_dir, count in splits:
        print(f"Generating {count} {split_name} images...")
        for i in range(count):
            img = create_belt_texture()
            
            if random.random() < 0.82:
                img, boxes = add_synthetic_damage(img)
            else:
                boxes = []

            img_path = os.path.join(img_dir, f"conveyor_{split_name}_{i:04d}.jpg")
            lbl_path = os.path.join(lbl_dir, f"conveyor_{split_name}_{i:04d}.txt")

            cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            with open(lbl_path, "w") as f:
                for cls_id, cx, cy, bw, bh in boxes:
                    f.write(f"{cls_id} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")

    abs_data_dir = os.path.abspath(DATASET_DIR).replace('\\', '/')
    yaml_path = os.path.join(DATASET_DIR, "conveyor_belt.yaml")
    yaml_content = f"""path: {abs_data_dir}
train: images/train
val: images/val

names:
  0: hole
  1: foreign object
  2: crack
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[Dataset Generator] Done! Generated dataset at {DATASET_DIR} with 300 labeled images.")
    return yaml_path

def train_model(yaml_path):
    print("\n[AI Trainer] Launching YOLOv8 transfer learning...")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    
    model.train(
        data=yaml_path,
        epochs=35,
        imgsz=640,
        batch=16,
        workers=0,
        mosaic=1.0,
        patience=12,
        verbose=True
    )
    
    print("\n[AI Trainer] Evaluating model on validation split...")
    model.val()
    
    best_weight_source = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(best_weight_source):
        os.makedirs(OUTPUT_WEIGHTS_DIR, exist_ok=True)
        target_best = os.path.join(OUTPUT_WEIGHTS_DIR, "best.pt")
        if os.path.exists(target_best):
            backup_dest = os.path.join(OUTPUT_WEIGHTS_DIR, "best_backup_old.pt")
            shutil.copy2(target_best, backup_dest)
            print(f"[AI Trainer] Backed up original weights to: {backup_dest}")
            
        shutil.copy2(best_weight_source, target_best)
        print(f"[AI Trainer] High-accuracy weights exported directly to: {target_best}")
    else:
        print("[AI Trainer] Could not locate output best.pt weights.")

if __name__ == "__main__":
    yaml_path = generate_dataset()
    train_model(yaml_path)
