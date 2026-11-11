"""
Dataset and YOLO Ground-Truth Annotation Generator
Generates synthetic CT images and exact YOLO bounding-box annotations (dataset/labels/).
Saves class 0: kidney_stone annotations for Stone images, and empty label files for Non-Stone images.
"""
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "Original"
LABELS_DIR = BASE_DIR / "dataset" / "labels"

CLASSES = {
    "Stone": 300,
    "Non-Stone": 300,
}


def generate_ct_background(size=224):
    """Generate realistic CT soft-tissue background."""
    rng = np.random.default_rng()
    base = rng.integers(80, 130, size=(size, size), dtype=np.uint8)
    img = Image.fromarray(base, mode="L")
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def generate_stone_image(size=224):
    """
    Generate synthetic CT slice with a bright kidney stone.
    Returns: (PIL.Image, (xc, yc, w, h) normalized coordinates)
    """
    img = generate_ct_background(size)
    draw = ImageDraw.Draw(img)

    # Random oval stone, bright white (200-255 intensity)
    cx = random.randint(size // 4, 3 * size // 4)
    cy = random.randint(size // 4, 3 * size // 4)
    radius_x = random.randint(6, 18)
    radius_y = random.randint(6, 18)
    brightness = random.randint(210, 255)

    draw.ellipse(
        [cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y],
        fill=brightness,
    )

    # Optional halo
    if random.random() > 0.5:
        draw.ellipse(
            [cx - radius_x - 2, cy - radius_y - 2, cx + radius_x + 2, cy + radius_y + 2],
            outline=170,
        )

    # Kidney anatomical border outline
    kx, ky = size // 2, size // 2
    krx, kry = random.randint(55, 80), random.randint(65, 90)
    draw.ellipse(
        [kx - krx, ky - kry, kx + krx, ky + kry],
        outline=50,
    )

    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Normalized YOLO coordinates: class 0, x_center, y_center, width, height
    xc = round(cx / float(size), 6)
    yc = round(cy / float(size), 6)
    w = round((2.0 * radius_x) / float(size), 6)
    h = round((2.0 * radius_y) / float(size), 6)

    return img.convert("RGB"), (xc, yc, w, h)


def generate_non_stone_image(size=224):
    """Generate normal kidney parenchyma with no stone."""
    img = generate_ct_background(size)
    draw = ImageDraw.Draw(img)

    kx, ky = size // 2, size // 2
    krx, kry = random.randint(55, 80), random.randint(65, 90)
    draw.ellipse([kx - krx, ky - kry, kx + krx, ky + kry], outline=50)

    for _ in range(random.randint(2, 5)):
        x = random.randint(kx - krx + 10, kx + krx - 10)
        y = random.randint(ky - kry + 10, ky + kry - 10)
        r = random.randint(5, 14)
        shade = random.randint(95, 145)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=shade)

    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    return img.convert("RGB")


def main():
    print("🏗️  Generating CT Images with Exact YOLO Ground-Truth Annotations...")
    random.seed(42)
    np.random.seed(42)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    for class_name, count in CLASSES.items():
        class_dir = DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, count + 1):
            filename = f"P{i:03d}_{class_name.replace('-', '_')}_CT_{i:04d}.jpg"
            img_path = class_dir / filename
            lbl_path = LABELS_DIR / f"{Path(filename).stem}.txt"

            if class_name == "Stone":
                img, bbox = generate_stone_image()
                img.save(img_path, "JPEG", quality=92)
                # YOLO format: class_id xc yc w h
                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            else:
                img = generate_non_stone_image()
                img.save(img_path, "JPEG", quality=92)
                # Non-Stone negative sample: empty file
                with open(lbl_path, "w", encoding="utf-8") as f:
                    pass

            if i % 100 == 0:
                print(f"  {class_name}: {i}/{count} images & annotations generated")

    print(f"\n✅ All 600 images saved to {DATASET_DIR}")
    print(f"✅ All 600 YOLO label files saved to {LABELS_DIR}")


if __name__ == "__main__":
    main()
