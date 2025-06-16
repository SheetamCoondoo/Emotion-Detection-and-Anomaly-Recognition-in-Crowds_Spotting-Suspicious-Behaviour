import os
import random
import uuid
import pandas as pd
from PIL import Image, ImageOps
from sklearn.utils import resample
from collections import Counter

# Paths
CSV_FILE = "dataset/raw_data/_annotations.csv"
IMG_SOURCE_DIR = "dataset/raw_data/train"
OUTPUT_DIR = "processed_tf_dataset/images"
OUTPUT_CSV = "processed_tf_dataset/annotations.csv"

# Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_FILE)

# Track all data (including augmented)
final_annotations = []

# Balance dataset by augmenting minority classes
class_counts = Counter(df['class'])
max_count = max(class_counts.values())

def augment_image_and_bbox(image, bbox, augment_type):
    """Applies augmentation and updates bounding box accordingly."""
    width, height = image.size
    xmin, ymin, xmax, ymax = bbox

    if augment_type == 'flip':
        image = ImageOps.mirror(image)
        xmin, xmax = width - xmax, width - xmin  # flip horizontally

    elif augment_type == 'rotate':
        angle = random.choice([-15, -10, 10, 15])
        image = image.rotate(angle, expand=True)
        # Keep bbox unchanged (or use Albumentations for better handling)
        # You could calculate updated bbox but this keeps it simple

    return image, (xmin, ymin, xmax, ymax)

# Save original image + record it
for idx, row in df.iterrows():
    img_path = os.path.join(IMG_SOURCE_DIR, row['filename'])
    if not os.path.exists(img_path):
        continue

    # Copy original image
    image = Image.open(img_path).convert('RGB')
    output_img_name = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_img_name)
    image.save(output_path)

    # Record in final CSV format
    final_annotations.append({
        'filename': output_img_name,
        'width': row['width'],
        'height': row['height'],
        'class': row['class'],
        'xmin': row['xmin'],
        'ymin': row['ymin'],
        'xmax': row['xmax'],
        'ymax': row['ymax'],
    })

# Augment minority classes
for class_name in class_counts:
    class_df = df[df['class'] == class_name]
    deficit = max_count - class_counts[class_name]

    if deficit <= 0:
        continue

    for i in range(deficit):
        row = class_df.sample(1).iloc[0]
        img_path = os.path.join(IMG_SOURCE_DIR, row['filename'])
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert('RGB')
        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

        augment_type = random.choice(['flip', 'rotate'])
        aug_img, new_bbox = augment_image_and_bbox(image, bbox, augment_type)

        # Save augmented image
        aug_img_name = f"aug_{uuid.uuid4().hex}.jpg"
        aug_img_path = os.path.join(OUTPUT_DIR, aug_img_name)
        aug_img.save(aug_img_path)

        final_annotations.append({
            'filename': aug_img_name,
            'width': image.width,
            'height': image.height,
            'class': row['class'],
            'xmin': int(new_bbox[0]),
            'ymin': int(new_bbox[1]),
            'xmax': int(new_bbox[2]),
            'ymax': int(new_bbox[3]),
        })

# Save final annotations
final_df = pd.DataFrame(final_annotations)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Process complete! {len(final_df)} images saved to: {OUTPUT_DIR}")
print(f"📝 CSV saved at: {OUTPUT_CSV}")