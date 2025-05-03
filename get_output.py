import cv2
import numpy as np
import math
import os
from ultralytics import YOLO

# --- 配置部分 ---
# !!! 重要：请正确设置以下路径 !!!
POSE_MODEL_PATH = 'best.pt'                         # <--- 你的 POSE 模型路径
DETECTION_MODEL_PATH = 'object.pt'                  # <--- 你的 DETECTION 模型路径
IMAGE_PATH = 'F:\darts_reco\darts_yolo_dataset_multiobj/val\images\DSC_0005.JPG'        # <--- 你的测试图片路径

POSE_CONFIDENCE_THRESHOLD = 0.5  # Pose 模型检测置信度阈值
DETECTION_CONFIDENCE_THRESHOLD = 0.4 # Detection 模型检测置信度阈值 (飞镖)

DARTBOARD_CLASS_NAME = 'dartboard' # Pose模型中飞镖盘的类别名称 (需要与你的模型名称匹配)
DART_CLASS_NAME = 'dart'           # Detection模型中飞镖的类别名称 (需要与你的模型名称匹配)
EXPECTED_KEYPOINTS = 4           # 期望的飞镖盘关键点数量

OUTPUT_DIR = 'combined_output'       # 输出目录
OUTPUT_FILENAME = 'annotated_combined.jpg' # 输出文件名

# --- 辅助函数：计算两条线段的交点 (同上一个脚本) ---
def find_intersection(p1, p2, p3, p4):
    """计算线段 (p1, p2) 和 (p3, p4) 的交点"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return np.array([intersection_x, intersection_y], dtype=np.float32)
    else: return None

# --- 辅助函数：对4个点排序 (左上, 右上, 右下, 左下) ---
# 这个函数对于 getPerspectiveTransform 很重要
def order_points(pts):
    """将4个点按左上、右上、右下、左下的顺序排序"""
    rect = np.zeros((4, 2), dtype="float32")
    # 左上点的 x+y 最小, 右下点的 x+y 最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 右上点的 y-x 最小 (或 x-y 最大), 左下点的 y-x 最大 (或 x-y 最小)
    # diff = np.diff(pts, axis=1) # 这计算的是 x-y
    diff = pts[:, 1] - pts[:, 0] # 计算 y-x
    rect[1] = pts[np.argmin(diff)] # y-x 最小的是右上
    rect[3] = pts[np.argmax(diff)] # y-x 最大的是左下
    return rect

# --- 主处理函数 ---
def process_image_end_to_end(pose_model_path, detection_model_path, image_path,
                             pose_conf, det_conf, output_dir, output_filename):
    """
    加载模型，预测，提取坐标，计算相对位置，并可视化。
    """
    # 1. 检查和加载模型及图像
    if not os.path.exists(pose_model_path): print(f"错误: Pose 模型未找到 {pose_model_path}"); return
    if not os.path.exists(detection_model_path): print(f"错误: Detection 模型未找到 {detection_model_path}"); return
    if not os.path.exists(image_path): print(f"错误: 图片未找到 {image_path}"); return

    print("加载模型...")
    try:
        pose_model = YOLO(pose_model_path)
        detection_model = YOLO(detection_model_path)
    except Exception as e:
        print(f"加载模型时出错: {e}"); return

    print("运行预测...")
    # 运行 Pose 模型
    pose_results = pose_model.predict(image_path, conf=pose_conf, verbose=False)
    pose_result = pose_results[0]
    img_height, img_width = pose_result.orig_shape
    img_orig = pose_result.orig_img # 获取原始图像 numpy 数组

    # 运行 Detection 模型
    detection_results = detection_model.predict(image_path, conf=det_conf, verbose=False)
    detection_result = detection_results[0]

    # 2. 提取飞镖盘关键点 (Pose 模型)
    keypoints_px = None
    best_dartboard_conf = 0.0
    dartboard_instance_index = -1

    if pose_result.keypoints is not None and len(pose_result.boxes) > 0:
        for i in range(len(pose_result.boxes)):
            box = pose_result.boxes[i]
            cls_id = int(box.cls.item())
            class_name = pose_model.names.get(cls_id, "").lower() # 转小写以防万一
            conf = box.conf.item()

            # 寻找置信度最高的飞镖盘实例
            # 寻找置信度最高的飞镖盘实例
            if class_name == DARTBOARD_CLASS_NAME and conf > best_dartboard_conf:
                keypoints_data = pose_result.keypoints[i]
                num_detected_kpts = keypoints_data.shape[1]  # 获取检测到的关键点总数

                # 检查是否至少有 4 个关键点
                if num_detected_kpts >= EXPECTED_KEYPOINTS:
                    # 假设 .xy 直接给出像素坐标
                    all_kpts = keypoints_data.xy.cpu().numpy()[0]  # 获取所有 (num_kpts, 2) 的数组

                    # !!! 关键修改：只取前 EXPECTED_KEYPOINTS (4) 个点 !!!
                    current_kpts = all_kpts[:EXPECTED_KEYPOINTS]

                    # 可选：检查前 4 个点是否包含无效坐标 (例如 0,0)
                    if not np.all(current_kpts == 0):
                        keypoints_px = current_kpts.astype(np.float32)
                        best_dartboard_conf = conf
                        dartboard_instance_index = i
                        # 打印提示信息，说明取了前4个点
                        print(
                            f"信息: Pose模型实例 {i} (dartboard, conf={conf:.4f}) 检测到 {num_detected_kpts} 个关键点，已提取前 {EXPECTED_KEYPOINTS} 个作为角点。")
                    else:
                        print(
                            f"警告: Pose模型实例 {i} (dartboard) 的前 {EXPECTED_KEYPOINTS} 个关键点包含 (0,0)，可能无效。")
                # 如果检测到的点少于4个，则打印警告
                elif num_detected_kpts > 0:
                    print(
                        f"警告: Pose模型实例 {i} (dartboard) 只找到 {num_detected_kpts} 个关键点，少于期望的 {EXPECTED_KEYPOINTS} 个。跳过此实例。")


    if keypoints_px is None:
        print(f"错误：未能从 Pose 模型中找到具有 {EXPECTED_KEYPOINTS} 个有效关键点的 '{DARTBOARD_CLASS_NAME}' 实例。")
        # 可以选择尝试使用 Detection 模型的 bbox 作为备选，但现在先退出
        return

    print(f"从 Pose 模型实例 {dartboard_instance_index} (置信度: {best_dartboard_conf:.4f}) 提取了 {EXPECTED_KEYPOINTS} 个关键点。")

    # !!! 重要：对关键点排序，保证顺序是 左上, 右上, 右下, 左下 !!!
    # 这对于后续的透视变换和中心计算至关重要
    keypoints_px_ordered = order_points(keypoints_px)


    # 3. 提取飞镖中心点 (Detection 模型)
    dart_centers_px = []
    dart_confidences = []
    if detection_result.boxes is not None:
        for i, box in enumerate(detection_result.boxes):
            cls_id = int(box.cls.item())
            class_name = detection_model.names.get(cls_id, "").lower()
            conf = box.conf.item()

            if class_name == DART_CLASS_NAME:
                xywhn = box.xywhn.cpu().numpy()[0] # Normalized [center_x, center_y, width, height]
                center_norm = xywhn[:2]
                center_px = center_norm * np.array([img_width, img_height])
                dart_centers_px.append(center_px.astype(np.float32))
                dart_confidences.append(conf)

    if not dart_centers_px:
        print("信息：Detection 模型未检测到任何飞镖 ('dart')。")
    else:
        print(f"从 Detection 模型提取了 {len(dart_centers_px)} 个飞镖中心点。")

    # 4. 计算飞镖盘中心 (使用排序后的关键点)
    #    使用对角线交点计算中心
    center = find_intersection(keypoints_px_ordered[0], keypoints_px_ordered[2], # 对角线 TL-BR
                                keypoints_px_ordered[1], keypoints_px_ordered[3]) # 对角线 TR-BL

    if center is None:
        print("警告：无法计算对角线交点，使用4个关键点的平均值作为中心。")
        board_center_px = np.mean(keypoints_px_ordered, axis=0)
    else:
        board_center_px = center
    print(f"计算出的飞镖盘中心: (x={board_center_px[0]:.2f}, y={board_center_px[1]:.2f})")
    center_x_int, center_y_int = int(board_center_px[0]), int(board_center_px[1])

    # 5. 计算飞镖相对位置
    dart_relative_info = []
    print("\n--- 飞镖相对位置计算 ---")
    for i, dart_center in enumerate(dart_centers_px):
        dart_x, dart_y = dart_center
        dx = dart_x - board_center_px[0]
        dy = dart_y - board_center_px[1]

        radius = math.sqrt(dx**2 + dy**2)
        angle_rad = math.atan2(-dy, dx) # 图像 Y 轴向下
        angle_deg = math.degrees(angle_rad)
        angle_deg_normalized = (angle_deg + 360) % 360

        info = {
            "index": i,
            "center_px": (dart_x, dart_y),
            "radius_px": radius,
            "angle_deg": angle_deg_normalized,
            "confidence": dart_confidences[i]
        }
        dart_relative_info.append(info)

        print(f"  Dart {i} (Conf: {info['confidence']:.4f}):")
        print(f"    Center (px): (x={dart_x:.2f}, y={dart_y:.2f})")
        print(f"    Relative Radius (px): {radius:.2f}")
        print(f"    Relative Angle (deg): {angle_deg_normalized:.2f} (0=右, 90=上, 180=左, 270=下)")


    # 6. 可视化
    img_display = img_orig.copy()
    # 绘制关键点 (使用排序后的点)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # BGR: TL, TR, BR, BL
    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(keypoints_px_ordered):
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(img_display, (px, py), radius=7, color=colors[i], thickness=-1)
        cv2.putText(img_display, labels[i], (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

    # 绘制中心点
    cv2.circle(img_display, (center_x_int, center_y_int), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.putText(img_display, "Center", (center_x_int + 8, center_y_int - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 绘制对角线
    cv2.line(img_display, tuple(keypoints_px_ordered[0].astype(int)), tuple(keypoints_px_ordered[2].astype(int)), (150, 150, 150), 1)
    cv2.line(img_display, tuple(keypoints_px_ordered[1].astype(int)), tuple(keypoints_px_ordered[3].astype(int)), (150, 150, 150), 1)

    # 绘制飞镖点和信息
    for info in dart_relative_info:
        dart_x_int, dart_y_int = int(info['center_px'][0]), int(info['center_px'][1])
        cv2.circle(img_display, (dart_x_int, dart_y_int), radius=6, color=(0, 165, 255), thickness=-1) # 橙色
        cv2.line(img_display, (center_x_int, center_y_int), (dart_x_int, dart_y_int), (0, 255, 255), 1) # 黄色线
        text = f"D{info['index']}: A={info['angle_deg']:.1f} R={info['radius_px']:.1f}"
        cv2.putText(img_display, text, (dart_x_int + 8, dart_y_int - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)

    # 7. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img_display)
    print(f"\n带有标注的图像已保存到: {output_path}")

    # 可选：显示图像
    # cv2.imshow("Combined Prediction Visualization", img_display)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 返回计算结果供后续使用（例如计分）
    return keypoints_px_ordered, board_center_px, dart_relative_info


# --- 执行主函数 ---
if __name__ == "__main__":
    if IMAGE_PATH == 'path/to/your/test/image.jpg':
        print("错误：请在脚本顶部修改 'IMAGE_PATH' 为你的实际测试图片路径！")
        print("同时检查 POSE_MODEL_PATH 和 DETECTION_MODEL_PATH 是否正确。")
    else:
        # 执行处理流程
        kpts, center, darts_info = process_image_end_to_end(
            POSE_MODEL_PATH,
            DETECTION_MODEL_PATH,
            IMAGE_PATH,
            POSE_CONFIDENCE_THRESHOLD,
            DETECTION_CONFIDENCE_THRESHOLD,
            OUTPUT_DIR,
            OUTPUT_FILENAME
        )

        # 可以在这里添加计分逻辑，使用 kpts, center, darts_info
        if darts_info:
            print("\n下一步：根据以上角度和半径计算分数。")
        else:
             print("\n未检测到飞镖，无法计分。")