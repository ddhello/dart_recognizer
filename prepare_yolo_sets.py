import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

# --- 配置区域 ---
# !!! 重要: 请将此路径修改为你存放数据集的根目录 !!!
# 该目录下应包含 labels.pkl 文件以及 d1_02_04_2020 等子文件夹
DATASET_BASE_DIR = 'darts_dataset/800' # <--- 修改这里
OUTPUT_DIR = './darts_yolo_dataset_vis' # <-- 建议使用新的输出目录
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800
NUM_KEYPOINTS = 7
# --- 配置结束 ---

def calculate_bounding_box(calibration_points_norm, img_width, img_height):
    # ... (保持此函数不变) ...
    if not calibration_points_norm or len(calibration_points_norm) != 4:
        print("警告: 校准点不足4个，使用全图作为边界框。")
        return 0.5, 0.5, 1.0, 1.0
    # ... (其余代码不变) ...
    points = np.array(calibration_points_norm)
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    padding_x = (max_x - min_x) * 0.1
    padding_y = (max_y - min_y) * 0.1
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
    """加载数据、划分、转换格式并组织文件"""
    # ... (加载 labels.pkl 和数据集划分部分保持不变) ...
    labels_path = os.path.join(base_dir, 'labels.pkl')
    print(f"加载标注文件: {labels_path}")
    try:
        df_labels = pd.read_pickle(labels_path)
    except FileNotFoundError:
        print(f"错误: 标注文件 '{labels_path}' 未找到。请检查 DATASET_BASE_DIR。")
        return
    except Exception as e:
        print(f"错误: 加载标注文件时出错: {e}")
        return

    print(f"数据集总数: {len(df_labels)}")
    indices = df_labels.index.tolist()
    train_indices, val_indices = train_test_split(indices,
                                                test_size=val_split,
                                                random_state=random_state)
    print(f"划分数据集: {len(train_indices)} 训练样本, {len(val_indices)} 验证样本")

    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    for split, indices_list in [('train', train_indices), ('val', val_indices)]:
        print(f"\n处理 {split} 集...")
        for index in tqdm(indices_list, desc=f"处理 {split} 集"):
            row = df_labels.iloc[index]
            img_folder = row['img_folder']
            img_name = row['img_name']
            xy_normalized = row['xy'] # 归一化坐标列表 [[x1, y1], [x2, y2], ...]

            src_image_path = os.path.join(base_dir, img_folder, img_name)
            if not os.path.exists(src_image_path):
                print(f"警告: 图像文件未找到，跳过: {src_image_path}")
                continue

            base_img_name = os.path.splitext(img_name)[0]
            dst_image_path = os.path.join(output_dir, split, 'images', img_name)
            dst_label_path = os.path.join(output_dir, split, 'labels', base_img_name + '.txt')

            shutil.copyfile(src_image_path, dst_image_path)

            # 准备 YOLO 标签字符串
            class_id = 0

            # 分离校准点和飞镖点
            calibration_points_norm = xy_normalized[:4]
            dart_tip_points_norm = xy_normalized[4:]
            num_darts = len(dart_tip_points_norm)

            # 计算边界框 (基于校准点)
            x_center, y_center, width, height = calculate_bounding_box(calibration_points_norm, IMAGE_WIDTH, IMAGE_HEIGHT)

            # 构建关键点部分: x_norm y_norm visibility
            keypoints_str_parts = []
            # 校准点 (总是可见且已标注)
            for i in range(4):
                if i < len(calibration_points_norm):
                    nx, ny = calibration_points_norm[i]
                    # 添加 x, y, visibility=2
                    keypoints_str_parts.extend([f"{nx:.6f}", f"{ny:.6f}", "2"]) # *** 修改这里 ***
                else:
                    print(f"警告: 图像 {img_name} 缺少校准点 {i + 1}! 使用 (0,0,0)")
                    # 添加 0, 0, visibility=0 作为占位符
                    keypoints_str_parts.extend(["0.0", "0.0", "0"]) # *** 修改这里 ***

            # 飞镖点
            for i in range(3):  # 最多处理3个飞镖 (对应关键点索引 4, 5, 6)
                if i < num_darts:
                    nx, ny = dart_tip_points_norm[i]
                    # 添加 x, y, visibility=2 (已标注且可见)
                    keypoints_str_parts.extend([f"{nx:.6f}", f"{ny:.6f}", "2"]) # *** 修改这里 ***
                else:
                    # 这个飞镖槽位是空的，标记为未标注/不可见
                    # 添加 0, 0, visibility=0
                    keypoints_str_parts.extend(["0.0", "0.0", "0"]) # *** 修改这里 ***

            # 确保关键点部分长度正确 (NUM_KEYPOINTS * 3 for x, y, vis)
            expected_kpt_values = NUM_KEYPOINTS * 3 # 现在是 7 * 3 = 21
            if len(keypoints_str_parts) != expected_kpt_values:
                print(f"错误: 图像 {img_name} 的关键点值数量不匹配 ({len(keypoints_str_parts)} vs {expected_kpt_values})!")
                # 如果出错，用 0.0, 0.0, 0 填充
                while len(keypoints_str_parts) < expected_kpt_values:
                    keypoints_str_parts.extend(["0.0", "0.0", "0"])

            # 组合成完整的 YOLO 标签行 (1 class + 4 box + 21 kpt = 26 列)
            yolo_label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {' '.join(keypoints_str_parts)}\n"

            with open(dst_label_path, 'w') as f:
                f.write(yolo_label_line)

    print("\n数据集准备完成！(带可见性标志)")
    print(f"YOLO 格式数据已保存到: {output_dir}")
    print(f"下一步: 更新 dartboard_data.yaml 文件并重新开始训练。")

# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='准备飞镖数据集用于YOLOv8姿态估计训练 (带可见性)')
    parser.add_argument('--data_dir', type=str, default=DATASET_BASE_DIR,
                        help='包含 labels.pkl 和图像子文件夹的数据集根目录')
    # 建议更改默认输出目录以避免覆盖
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='保存 YOLO 格式数据集的输出目录')
    parser.add_argument('--val_split', type=float, default=VALIDATION_SPLIT,
                        help='验证集所占比例')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE,
                        help='用于数据集划分的随机种子')

    args = parser.parse_args()

    # ... (保持参数处理和路径检查不变) ...
    DATASET_BASE_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    VALIDATION_SPLIT = args.val_split
    RANDOM_STATE = args.seed

    if not os.path.isabs(DATASET_BASE_DIR):
         DATASET_BASE_DIR = os.path.abspath(DATASET_BASE_DIR)
    if not os.path.exists(DATASET_BASE_DIR):
         print(f"错误: 数据集目录不存在: {DATASET_BASE_DIR}")
         exit()

    process_dataset(DATASET_BASE_DIR, OUTPUT_DIR, VALIDATION_SPLIT, RANDOM_STATE)