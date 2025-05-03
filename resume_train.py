from ultralytics import YOLO
import os # 确保导入 os

# --- 配置 ---
# 找到你上一次训练的实验文件夹路径
# 通常在 runs/pose/ 下，名字是你上次指定的 name
# 例如: 'runs/pose/dartboard_pose_experiment_python'
# 如果不确定，可以省略 experiment_path，resume=True 会尝试找最近的
# experiment_path = 'runs/pose/dartboard_pose_experiment_python' # <--- 确认或指定路径
# last_checkpoint = os.path.join(experiment_path, 'weights', 'last.pt') # <--- 确认 last.pt 在这里

# 指定你想要达到的总 epoch 数量
TOTAL_EPOCHS = 200 # 例如，如果你想再训练 100 个，总共达到 200

# 其他训练参数保持一致或按需修改
DATA_YAML = 'dart_data.yaml'
IMAGE_SIZE = 800
BATCH_SIZE = 16 # 最好保持与上次训练一致
EXPERIMENT_NAME = 'dartboard_pose_experiment_python_resume' # 可以用新名字，或用旧名字覆盖

# --- 加载模型 (注意：这里仍然可以从预训练模型开始，resume 会覆盖) ---
# 但为了清晰，可以直接加载 last.pt，或者让 resume=True 处理
# model = YOLO('yolov8s-pose.pt') # 或者下面这行
# model = YOLO(last_checkpoint) # 如果直接加载 last.pt

# *** 使用 resume=True 时，初始加载的模型不那么重要，因为它会被 last.pt 的状态覆盖 ***
# *** 但通常需要提供一个基础模型结构，所以用 'yolov8s-pose.pt' 或上次的 last.pt 都可以 ***
model = YOLO('yolo11s-pose.pt') # 或者 'path/to/runs/pose/dartboard_pose_experiment_python/weights/last.pt'

print(f"尝试从上一次训练继续，目标总 epoch 数: {TOTAL_EPOCHS}")

# --- 训练模型，关键在于 resume=True ---
results = model.train(
    data=DATA_YAML,
    # epochs=TOTAL_EPOCHS, # 使用 resume 时，它会接着计数，设为最终目标
    epochs=TOTAL_EPOCHS,    # 设置为最终想要达到的总轮数
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    name=EXPERIMENT_NAME, # 指定新的或旧的实验名
    resume=True          # <--- 关键参数！
    # device=0,           # 可选：指定 GPU
    # workers=8           # 可选：根据你的 CPU 和 IO 调整
)

print("训练完成！结果保存在:", results.save_dir)