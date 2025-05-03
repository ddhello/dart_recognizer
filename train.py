from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolo11s-pose.pt') # 或者 'yolov8s-pose.pt' 等

# 训练模型
results = model.train(
    data='dart_data.yaml',
    epochs=100,
    imgsz=800,
    batch=16,
    name='dartboard_pose_experiment_python' # 结果保存在 runs/pose/...
)

print("训练完成！结果保存在:", results.save_dir)