import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

# --- Configuration Section ---
DATASET_BASE_DIR = 'darts_dataset/800'  # Path to dataset with labels.pkl
OUTPUT_DIR = './darts_yolo_dataset_multiobj'  # New output directory
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800


def calculate_bounding_box(points_norm, padding_percent=0.1):
    """Calculate bounding box from normalized points with padding"""
    if not points_norm or len(points_norm) < 1:
        return 0.5, 0.5, 1.0, 1.0  # Default to full image if no points

    points = np.array(points_norm)
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    padding_x = (max_x - min_x) * padding_percent
    padding_y = (max_y - min_y) * padding_percent

    min_x = max(0.0, min_x - padding_x)
    max_x = min(1.0, max_x + padding_x)
    min_y = max(0.0, min_y - padding_y)
    max_y = min(1.0, max_y + padding_y)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    return center_x, center_y, width, height


def process_dataset(base_dir, output_dir, val_split, random_state):
    """Process dataset with dartboard and individual darts as separate objects"""
    labels_path = os.path.join(base_dir, 'labels.pkl')
    print(f"Loading labels from: {labels_path}")
    try:
        df_labels = pd.read_pickle(labels_path)
    except FileNotFoundError:
        print(f"Error: Label file '{labels_path}' not found. Check DATASET_BASE_DIR.")
        return
    except Exception as e:
        print(f"Error loading label file: {e}")
        return

    print(f"Total dataset size: {len(df_labels)}")
    indices = df_labels.index.tolist()
    train_indices, val_indices = train_test_split(indices,
                                                  test_size=val_split,
                                                  random_state=random_state)
    print(f"Dataset split: {len(train_indices)} training samples, {len(val_indices)} validation samples")

    # Create directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # Process each split
    for split, indices_list in [('train', train_indices), ('val', val_indices)]:
        print(f"\nProcessing {split} set...")
        for index in tqdm(indices_list, desc=f"Processing {split} set"):
            row = df_labels.iloc[index]
            img_folder = row['img_folder']
            img_name = row['img_name']
            xy_normalized = row['xy']  # [[x1, y1], [x2, y2], ...]

            src_image_path = os.path.join(base_dir, img_folder, img_name)
            if not os.path.exists(src_image_path):
                print(f"Warning: Image file not found, skipping: {src_image_path}")
                continue

            base_img_name = os.path.splitext(img_name)[0]
            dst_image_path = os.path.join(output_dir, split, 'images', img_name)
            dst_label_path = os.path.join(output_dir, split, 'labels', base_img_name + '.txt')

            # Copy image file
            shutil.copyfile(src_image_path, dst_image_path)

            # Separate calibration points and dart points
            calibration_points_norm = xy_normalized[:4]
            dart_tip_points_norm = xy_normalized[4:]
            num_darts = len(dart_tip_points_norm)

            # Open label file for writing
            with open(dst_label_path, 'w') as f:
                # Class 0: Dartboard (using calibration points)
                dartboard_x, dartboard_y, dartboard_w, dartboard_h = calculate_bounding_box(calibration_points_norm)
                f.write(f"0 {dartboard_x:.6f} {dartboard_y:.6f} {dartboard_w:.6f} {dartboard_h:.6f}\n")

                # Class 1: Individual darts (each dart gets its own detection)
                for i in range(num_darts):
                    # For each dart point, create a small bounding box around it
                    # Using single point with small fixed size for the dart
                    dart_point = dart_tip_points_norm[i]

                    # Create a small bounding box around the dart point (0.05 = 5% of image width/height)
                    dart_size = 0.05
                    dart_x = dart_point[0]
                    dart_y = dart_point[1]

                    # Write dart as separate object
                    f.write(f"1 {dart_x:.6f} {dart_y:.6f} {dart_size:.6f} {dart_size:.6f}\n")

    # Create data.yaml file
    yaml_path = os.path.join(output_dir, 'dartboard_data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("\n")
        f.write("names:\n")
        f.write("  0: dartboard\n")
        f.write("  1: dart\n")

    print("\nDataset preparation complete!")
    print(f"YOLO format data saved to: {output_dir}")
    print(f"Next: Use the new dartboard_data.yaml file for training.")


# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dart dataset for YOLOv8 object detection training')
    parser.add_argument('--data_dir', type=str, default=DATASET_BASE_DIR,
                        help='Dataset root directory containing labels.pkl and image subfolders')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--val_split', type=float, default=VALIDATION_SPLIT,
                        help='Validation set proportion')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE,
                        help='Random seed for dataset splitting')

    args = parser.parse_args()

    DATASET_BASE_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    VALIDATION_SPLIT = args.val_split
    RANDOM_STATE = args.seed

    if not os.path.isabs(DATASET_BASE_DIR):
        DATASET_BASE_DIR = os.path.abspath(DATASET_BASE_DIR)
    if not os.path.exists(DATASET_BASE_DIR):
        print(f"Error: Dataset directory does not exist: {DATASET_BASE_DIR}")
        exit()

    process_dataset(DATASET_BASE_DIR, OUTPUT_DIR, VALIDATION_SPLIT, RANDOM_STATE)