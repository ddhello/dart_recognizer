from ultralytics import YOLO
import os
import multiprocessing

def main():
    """主训练函数"""
    # 创建实验文件夹
    experiment_name = 'dart_detection_experiment'
    os.makedirs(f'runs/{experiment_name}', exist_ok=True)

    # 加载YOLOv8检测模型（不是姿态估计）
    model = YOLO('yolov8s.pt')  # 使用标准检测模型而非姿态估计模型

    # 训练模型
    results = model.train(
        data='darts_yolo_dataset_multiobj/dartboard_data.yaml',  # 数据配置路径
        epochs=100,
        imgsz=800,                   # 与原始图像大小匹配
        batch=16,
        patience=50,                 # 训练时保持耐心
        name=experiment_name,
        save=True,                   # 训练后保存模型
        pretrained=True,
        optimizer='AdamW',           # 尝试不同的优化器
        cos_lr=True,                 # 余弦学习率
        lr0=0.001,                   # 起始学习率
        lrf=0.01,                    # 最终学习率（相对于lr0的比例）
        weight_decay=0.0005,         # 权重衰减
        warmup_epochs=5,             # 前5个epoch进行预热
        box=7.5,                     # 边界框损失权重
        cls=0.5,                     # 分类损失权重
        dfl=1.5,                     # 分布式焦点损失权重
        mixup=0.1,                   # 应用mixup
        copy_paste=0.1,              # 应用copy-paste
        degrees=10.0,                # 图像旋转增强
        translate=0.1,               # 图像平移增强
        scale=0.5,                   # 缩放增强
        shear=0.0,                   # 剪切增强
        perspective=0.0,             # 透视增强
        flipud=0.0,                  # 上下翻转
        fliplr=0.5,                  # 左右翻转
        mosaic=1.0,                  # Mosaic增强
        conf=0.001,                  # 目标置信度阈值
        iou=0.7,                     # IoU阈值
        max_det=300,                 # 每张图像的最大检测数
        multi_scale=True,            # 多尺度训练
        workers=0,                   # 在Windows上设置为0避免多进程问题
    )

    print("训练完成！结果保存在:", results.save_dir)

    # 可选：在验证集上验证模型
    print("\n验证模型中...")
    metrics = model.val()
    print(f"验证结果: mAP50 = {metrics.box.map50:.4f}, mAP50-95 = {metrics.box.map:.4f}")

    # 示例：在测试图像上运行推理
    print("\n在测试图像上运行推理...")
    test_image_path = 'path/to/test/image.jpg'  # 替换为你的测试图像
    if os.path.exists(test_image_path):
        results = model.predict(test_image_path, conf=0.5, save=True, project=f'runs/{experiment_name}/inference')
        print(f"推理结果保存到: runs/{experiment_name}/inference")
    else:
        print(f"未找到测试图像: {test_image_path}")

    print("\n训练和验证完成！")


if __name__ == '__main__':
    # 这一行在Windows系统上很重要
    multiprocessing.freeze_support()
    # 运行主函数
    main()