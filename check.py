import pandas as pd
import cv2
import random
import os
import numpy as np

# --- 配置区域 ---
# !!! 重要: 请将此路径修改为你存放数据集的根目录 !!!
# 该目录下应包含 labels.pkl 文件以及 d1_02_04_2020 等子文件夹
DATASET_BASE_DIR = 'darts_dataset/800'  # <--- 修改这里

# 图像的预期尺寸 (根据描述)
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

# 绘图参数
CALIBRATION_POINT_COLOR = (0, 255, 0)  # BGR 格式: 绿色
DART_TIP_COLOR = (0, 0, 255)           # BGR 格式: 红色
POINT_RADIUS = 6
POINT_THICKNESS = -1 # -1 表示填充圆点
TEXT_COLOR = (255, 255, 255) # 白色
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
# --- 配置结束 ---

# 1. 加载标注文件
labels_path = os.path.join(DATASET_BASE_DIR, 'labels.pkl')
print(f"尝试加载标注文件: {labels_path}")

try:
    df_labels = pd.read_pickle(labels_path)
    print(f"成功加载标注文件. 数据集包含 {len(df_labels)} 条记录.")
    # 打印前几行看看格式
    # print("数据框前5行:")
    # print(df_labels.head())
except FileNotFoundError:
    print(f"错误: 标注文件 '{labels_path}' 未找到. 请检查 DATASET_BASE_DIR 是否设置正确。")
    exit()
except Exception as e:
    print(f"错误: 加载标注文件时出错: {e}")
    exit()

# 2. 随机选择一张图像的标注信息
num_images = len(df_labels)
if num_images == 0:
    print("错误: 标注文件中没有数据.")
    exit()

random_index = random.randint(0, num_images - 1)
print(f"\n随机选择索引: {random_index}")

selected_row = df_labels.iloc[random_index]
img_folder = selected_row['img_folder']
img_name = selected_row['img_name']
# bbox = selected_row['bbox'] # bbox 在这里暂时不用
xy_normalized = selected_row['xy'] # 归一化坐标列表 [[x1, y1], [x2, y2], ...]

print(f"  图像文件夹: {img_folder}")
print(f"  图像文件名: {img_name}")
print(f"  归一化坐标点 (xy): {xy_normalized}")

# 3. 构建图像文件的完整路径
# 假设图像子文件夹直接位于 DATASET_BASE_DIR 下
image_path = os.path.join(DATASET_BASE_DIR, img_folder, img_name)
print(f"  图像完整路径: {image_path}")

# 4. 读取图像
image = cv2.imread(image_path)

if image is None:
    print(f"错误: 无法读取图像文件 '{image_path}'. 请检查路径和文件是否存在。")
    exit()

# 检查图像尺寸是否符合预期
h, w, _ = image.shape
if h != IMAGE_HEIGHT or w != IMAGE_WIDTH:
    print(f"警告: 图像尺寸 ({w}x{h}) 与预期的 ({IMAGE_WIDTH}x{IMAGE_HEIGHT}) 不符。")
    # 如果尺寸不符，你可能需要调整下面的坐标转换或考虑调整图像大小
    # 但根据描述，图像应该是 800x800

# 5. 在图像上绘制标注点
if not isinstance(xy_normalized, (list, np.ndarray)) or len(xy_normalized) < 4:
    print(f"警告: 索引 {random_index} 的 'xy' 数据格式不正确或缺少至少4个校准点: {xy_normalized}")
else:
    # 分离校准点和飞镖点
    calibration_points_norm = xy_normalized[:4]
    dart_tip_points_norm = xy_normalized[4:]

    print(f"  找到 {len(calibration_points_norm)} 个校准点和 {len(dart_tip_points_norm)} 个飞镖点。")

    # 绘制校准点 (绿色)
    for i, (nx, ny) in enumerate(calibration_points_norm):
        # 将归一化坐标转换为像素坐标
        px = int(nx * IMAGE_WIDTH)
        py = int(ny * IMAGE_HEIGHT)
        # 绘制圆点
        cv2.circle(image, (px, py), POINT_RADIUS, CALIBRATION_POINT_COLOR, POINT_THICKNESS)
        # 可选：添加文字标签
        cv2.putText(image, f"C{i+1}", (px + 8, py - 8), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    # 绘制飞镖尖端点 (红色)
    for i, (nx, ny) in enumerate(dart_tip_points_norm):
        # 将归一化坐标转换为像素坐标
        px = int(nx * IMAGE_WIDTH)
        py = int(ny * IMAGE_HEIGHT)
        # 绘制圆点
        cv2.circle(image, (px, py), POINT_RADIUS, DART_TIP_COLOR, POINT_THICKNESS)
         # 可选：添加文字标签
        cv2.putText(image, f"D{i+1}", (px + 8, py - 8), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)


# 6. 显示带有标注的图像
window_title = f"Annotated: {img_folder}/{img_name} (Index: {random_index})"
cv2.imshow(window_title, image)

print("\n图像已显示。按键盘任意键关闭窗口...")
cv2.waitKey(0)  # 等待用户按键
cv2.destroyAllWindows() # 关闭所有OpenCV窗口
print("窗口已关闭。")