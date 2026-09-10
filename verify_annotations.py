import os
import cv2

folder = "user_images/holes"
if os.path.exists(folder):
    for f in os.listdir(folder):
        if f.endswith(".txt"):
            base = os.path.splitext(f)[0]
            img_path = os.path.join(folder, base + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(folder, base + ".jpg")
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            with open(os.path.join(folder, f)) as tf:
                for line in tf:
                    _, cx, cy, bw, bh = map(float, line.strip().split())
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, "HOLE / DAMAGE", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            out_path = os.path.join(folder, "annotated_" + os.path.basename(img_path))
            cv2.imwrite(out_path, img)
            print(f"Saved verified overlay: {out_path}")
