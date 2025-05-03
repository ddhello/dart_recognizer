from ultralytics import YOLO
import os
import random
import glob
import pandas as pd
import numpy as np
import shutil
import cv2
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def random_predict_and_visualize(model_path,
                                 yaml_path,
                                 num_images=100,
                                 conf_threshold=0.3,
                                 save_dir='prediction_results'):
    """
    使用训练好的模型在数据集中随机选择图像进行预测并可视化结果

    参数:
        model_path: 训练好的模型路径，通常是 best.pt
        yaml_path: dartboard_data.yaml 文件路径
        num_images: 要预测的随机图像数量
        conf_threshold: 置信度阈值
        save_dir: 保存结果的目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 加载数据集配置
    print(f"加载数据集配置: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    # 获取数据集路径
    dataset_path = data_config['path']
    # 类别名称
    class_names = data_config['names']

    # 获取所有训练和验证图像的路径
    train_images = glob.glob(os.path.join(dataset_path, 'train', 'images', '*.jpg')) + \
                   glob.glob(os.path.join(dataset_path, 'train', 'images', '*.png'))
    val_images = glob.glob(os.path.join(dataset_path, 'val', 'images', '*.jpg')) + \
                 glob.glob(os.path.join(dataset_path, 'val', 'images', '*.png'))

    all_images = train_images + val_images
    print(f"找到总共 {len(all_images)} 张图像")

    # 如果图像总数小于请求的数量，调整为实际数量
    if len(all_images) < num_images:
        num_images = len(all_images)
        print(f"可用图像少于请求数量，调整为 {num_images} 张图像")

    # 随机选择图像
    print(f"随机选择 {num_images} 张图像进行预测...")
    selected_images = random.sample(all_images, num_images)

    # 创建结果摘要
    results_summary = []

    # 对每张图像进行预测并可视化
    for i, img_path in enumerate(tqdm(selected_images, desc="处理图像")):
        # 提取图像名称和路径信息
        img_name = os.path.basename(img_path)
        img_base = os.path.splitext(img_name)[0]
        img_dir = os.path.dirname(img_path)

        # 找到对应的标签文件（原始真实标签）
        if "train" in img_dir:
            label_dir = os.path.join(dataset_path, "train", "labels")
        else:
            label_dir = os.path.join(dataset_path, "val", "labels")

        label_path = os.path.join(label_dir, f"{img_base}.txt")

        # 运行预测
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        result = results[0]  # 获取第一个结果（因为只有一张图像）

        # 读取图像用于可视化
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # 创建图像副本用于可视化
        img_pred = img.copy()

        # 在图像上绘制预测结果
        boxes = result.boxes.xyxy.cpu().numpy()  # 获取预测的边界框
        confidences = result.boxes.conf.cpu().numpy()  # 获取置信度
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # 获取类别ID

        # 记录这张图像的dart检测数量
        detected_darts = sum(1 for cls_id in class_ids if class_names[cls_id] == 'dart')

        # 读取原始标签，计算真实的dart数量
        true_darts = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    if class_names[cls_id] == 'dart':
                        true_darts += 1

        # 将预测结果信息添加到摘要中
        results_summary.append({
            'image_name': img_name,
            'predicted_darts': detected_darts,
            'true_darts': true_darts,
            'match': detected_darts == true_darts
        })

        # 绘制预测框
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 1, 1)
        plt.title(f"predict result: {img_name}\ndetected to {detected_darts} darts,truth: {true_darts} darts")
        plt.imshow(img_pred)

        # 添加边界框、类别和置信度
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            color = 'red' if class_names[cls_id] == 'dart' else 'blue'
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x1, y1 - 10, f"{class_names[cls_id]}: {conf:.2f}", color=color, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7))

        # 保存可视化结果
        plt.axis('off')
        save_path = os.path.join(save_dir, f"pred_{i + 1:03d}_{img_base}.jpg")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

    # 创建结果摘要DataFrame并保存
    summary_df = pd.DataFrame(results_summary)
    accuracy = sum(summary_df['match']) / len(summary_df) * 100

    # 添加摘要统计信息
    print("\n预测摘要:")
    print(f"总图像数: {len(summary_df)}")
    print(f"准确预测数量的图像: {sum(summary_df['match'])} ({accuracy:.2f}%)")

    # 分析预测错误的图像
    error_cases = summary_df[~summary_df['match']]
    if len(error_cases) > 0:
        print("\n预测错误的图像:")
        print(error_cases)


    # 创建简单的分析图表
    plt.figure(figsize=(12, 6))

    # 预测数量与真实数量对比
    plt.subplot(1, 2, 1)
    max_value = max(summary_df['predicted_darts'].max(), summary_df['true_darts'].max()) + 1
    plt.hist([summary_df['predicted_darts'], summary_df['true_darts']],
             bins=range(max_value + 1), label=['Predict Count:', 'True Count:'], alpha=0.7)
    plt.xlabel('dart count')
    plt.ylabel('graph count')
    plt.title('dart count distribution')
    plt.legend()
    plt.grid(alpha=0.3)

    # 准确率饼图
    plt.subplot(1, 2, 2)
    plt.pie([sum(summary_df['match']), len(summary_df) - sum(summary_df['match'])],
            labels=['Correct Predict', 'Wrong Predict'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
    plt.title('prediction accuracy')

    # 保存分析图表
    analysis_path = os.path.join(save_dir, "prediction_analysis.jpg")
    plt.savefig(analysis_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"分析图表已保存到 {analysis_path}")

    return summary_df


if __name__ == '__main__':
    # 配置参数
    MODEL_PATH = 'object.pt'  # 模型路径
    YAML_PATH = 'darts_yolo_dataset_multiobj/dartboard_data.yaml'  # 数据配置文件路径
    NUM_IMAGES = 100  # 要预测的随机图像数量
    CONF_THRESHOLD = 0.3  # 置信度阈值
    SAVE_DIR = 'random_predictions'  # 保存预测结果的目录

    # 运行预测和可视化
    summary = random_predict_and_visualize(
        model_path=MODEL_PATH,
        yaml_path=YAML_PATH,
        num_images=NUM_IMAGES,
        conf_threshold=CONF_THRESHOLD,
        save_dir=SAVE_DIR
    )