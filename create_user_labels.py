import os

annotations = {
    "cardboard_perforated_tear.png": [
        (0, 462 / 1024.0, 298 / 576.0, 68 / 1024.0, 220 / 576.0)
    ],
    "cardboard_puncture_hole.png": [
        (0, 362 / 1024.0, 168 / 576.0, 56 / 1024.0, 62 / 576.0),
        (0, 444 / 1024.0, 270 / 576.0, 32 / 1024.0, 34 / 576.0)
    ],
    "cardboard_large_gash_hole.png": [
        (0, 318 / 1024.0, 305 / 576.0, 175 / 1024.0, 320 / 576.0)
    ],
    "industrial_belt_longitudinal_rip.jpg": [
        (0, 339 / 550.0, 232 / 399.0, 42 / 550.0, 92 / 399.0)
    ]
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(BASE_DIR, "user_images", "holes")
for filename, boxes in annotations.items():
    img_path = os.path.join(folder, filename)
    if os.path.exists(img_path):
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        with open(txt_path, "w") as f:
            for cls_id, cx, cy, w, h in boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        print(f"Created label for {filename}: {len(boxes)} box(es)")
    else:
        print(f"File not found: {img_path}")
