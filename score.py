import cv2
import numpy as np
import math
import os
from ultralytics import YOLO

# --- 配置部分 ---
# !!! 重要：请正确设置以下路径 !!!
POSE_MODEL_PATH = 'best.pt'  # <--- 你的 POSE 模型路径
DETECTION_MODEL_PATH = 'object.pt'  # <--- 你的 DETECTION 模型路径
IMAGE_PATH = 'F:\darts_reco\darts_yolo_dataset_multiobj/val\images\DSC_0051.JPG'  # <--- 你的测试图片路径

POSE_CONFIDENCE_THRESHOLD = 0.5  # Pose 模型检测置信度阈值
DETECTION_CONFIDENCE_THRESHOLD = 0.4  # Detection 模型检测置信度阈值 (飞镖)

DARTBOARD_CLASS_NAME = 'dartboard'  # Pose模型中飞镖盘的类别名称
DART_CLASS_NAME = 'dart'  # Detection模型中飞镖的类别名称
EXPECTED_KEYPOINTS = 4  # 期望的飞镖盘关键点数量 (只取前4个)

OUTPUT_DIR = 'scoring_output_rotated'  # 输出目录
OUTPUT_FILENAME = 'scored_darts_rotated.jpg'  # 输出文件名 (原图标注)
CANONICAL_VIS_FILENAME = 'canonical_visualization_rotated.jpg'  # 标准坐标系可视化文件名

# --- 新的标准飞镖盘定义 (归一化，中心为 0,0) ---
CANONICAL_CENTER_X = 0.0
CANONICAL_CENTER_Y = 0.0
CANONICAL_RADIUS_OUTER_DOUBLE = 1.0

R_MM = {'double_outer': 170.0, 'double_inner': 162.0, 'triple_outer': 107.0,
        'triple_inner': 99.0, 'bull_outer': 15.9, 'bull_inner': 6.35}
R_NORM = {name: radius / R_MM['double_outer'] for name, radius in R_MM.items()}
print("归一化半径:", R_NORM)

SEGMENT_VALUES = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
SEGMENT_ANGLE = 360.0 / len(SEGMENT_VALUES)

# --- 关键：定义 4 个校准点在标准归一化坐标系中的【初始假设】位置 ---
# (这个位置可能与最终对齐的坐标系有固定旋转偏差)
angle_45_rad = math.radians(45)
cos_45 = math.cos(angle_45_rad)
sin_45 = math.sin(angle_45_rad)
INITIAL_TARGET_POINTS_NORMALIZED = np.array([
    [-cos_45, sin_45],  # TL (初始假设在 135 度)
    [cos_45, sin_45],  # TR (初始假设在 45 度)
    [cos_45, -sin_45],  # BR (初始假设在 -45 / 315 度)
    [-cos_45, -sin_45]  # BL (初始假设在 -135 / 225 度)
], dtype=np.float32)
print("初始目标归一化坐标点 (TL, TR, BR, BL):\n", INITIAL_TARGET_POINTS_NORMALIZED)

# --- 新增：根据观察到的偏移，旋转目标点以修正坐标系 ---
# !!! 检查并设置正确的旋转角度 !!!
# 如果观察到需要将当前错误的十字位置【顺时针】旋转 N 度才能到正确位置，
# 则这里需要将目标点【逆时针】旋转 N 度来补偿。
rotation_angle_deg = -36.0  # 逆时针旋转 36 度 (补偿顺时针偏移2个分区)
# 如果是补偿顺时针偏移4个分区，则用 72.0
rotation_angle_rad = math.radians(rotation_angle_deg)
cos_rot = math.cos(rotation_angle_rad)
sin_rot = math.sin(rotation_angle_rad)

TARGET_POINTS_NORMALIZED_ROTATED = np.zeros_like(INITIAL_TARGET_POINTS_NORMALIZED)
for i, point in enumerate(INITIAL_TARGET_POINTS_NORMALIZED):
    x, y = point
    x_new = x * cos_rot - y * sin_rot
    y_new = x * sin_rot + y * cos_rot
    TARGET_POINTS_NORMALIZED_ROTATED[i] = [x_new, y_new]
print(f"旋转 {rotation_angle_deg} 度后的最终目标归一化坐标点:\n", TARGET_POINTS_NORMALIZED_ROTATED)
# --- 结束旋转逻辑 ---

# --- 可视化参数 ---
VIS_WIDTH = 500
VIS_HEIGHT = 500
VIS_CENTER_X = VIS_WIDTH / 2
VIS_CENTER_Y = VIS_HEIGHT / 2
VIS_SCALE_FACTOR = min(VIS_WIDTH, VIS_HEIGHT) / 2 * 0.9


# --- 辅助函数：计算交点, 排序点 (同之前) ---
def find_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
    if 0 <= t <= 1 and 0 <= u <= 1:
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float32)
    else:
        return None


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# --- 辅助函数：计算分数 (使用归一化坐标，无额外角度补偿) ---
def calculate_score_normalized(tx, ty, r_norm, segment_values, segment_angle):
    radius_norm = math.sqrt(tx ** 2 + ty ** 2)
    if radius_norm > r_norm['double_outer']: return 0, "Outside Board"
    if radius_norm <= r_norm['bull_inner']: return 50, "Bullseye"
    if radius_norm <= r_norm['bull_outer']: return 25, "Outer Bull"

    angle_rad = math.atan2(ty, tx)
    angle_deg_atan2 = math.degrees(angle_rad)
    clockwise_angle = (90 - angle_deg_atan2 + 360) % 360
    # 使用原始 clockwise_angle 计算分区
    adjusted_clockwise = (clockwise_angle + (segment_angle / 2) + 360) % 360
    segment_index = int(adjusted_clockwise // segment_angle) % len(segment_values)
    base_score = segment_values[segment_index]

    multiplier = 1
    zone = f"Single {base_score}"
    if r_norm['double_inner'] <= radius_norm <= r_norm['double_outer']:
        multiplier = 2; zone = f"Double {base_score}"
    elif r_norm['triple_inner'] <= radius_norm <= r_norm['triple_outer']:
        multiplier = 3; zone = f"Triple {base_score}"
    return base_score * multiplier, zone


# --- 辅助函数：归一化到像素坐标 (同之前) ---
def normalized_to_pixel(nx, ny, vis_center_x, vis_center_y, scale_factor):
    px = vis_center_x + nx * scale_factor
    py = vis_center_y - ny * scale_factor
    return int(px), int(py)


# --- 主处理函数 ---
def process_and_score_image_rotated(pose_model_path, detection_model_path, image_path,
                                    pose_conf, det_conf, output_dir, output_filename, canonical_vis_filename):
    # 1. 检查和加载 (同之前)
    if not all(os.path.exists(p) for p in [pose_model_path, detection_model_path, image_path]): print(
        "错误：路径无效。"); return None, None, None
    print("加载模型...")
    try:
        pose_model = YOLO(pose_model_path)
        detection_model = YOLO(detection_model_path)
    except Exception as e:
        print(f"加载模型出错: {e}"); return None, None, None

    # 2. 运行预测 (同之前)
    print("运行预测...")
    pose_results = pose_model.predict(image_path, conf=pose_conf, verbose=False)
    detection_results = detection_model.predict(image_path, conf=det_conf, verbose=False)
    pose_result = pose_results[0]
    detection_result = detection_results[0]
    img_height, img_width = pose_result.orig_shape
    img_orig = pose_result.orig_img

    # 3. 提取关键点 (同之前)
    keypoints_px = None
    best_dartboard_conf = 0.0
    if pose_result.keypoints is not None and len(pose_result.boxes) > 0:
        for i in range(len(pose_result.boxes)):
            box = pose_result.boxes[i]
            cls_id = int(box.cls.item())
            class_name = pose_model.names.get(cls_id, "").lower()
            conf = box.conf.item()
            if class_name == DARTBOARD_CLASS_NAME and conf > best_dartboard_conf:
                keypoints_data = pose_result.keypoints[i]
                num_detected_kpts = keypoints_data.shape[1]
                if num_detected_kpts >= EXPECTED_KEYPOINTS:
                    all_kpts = keypoints_data.xy.cpu().numpy()[0]
                    current_kpts = all_kpts[:EXPECTED_KEYPOINTS]
                    if not np.all(current_kpts == 0):
                        keypoints_px = current_kpts.astype(np.float32); best_dartboard_conf = conf
                        print(
                            f"信息: Pose实例 {i} 提取前 {EXPECTED_KEYPOINTS} 角点。")
                    else:
                        print(f"警告: Pose实例 {i} 前 {EXPECTED_KEYPOINTS} 关键点含无效坐标。")
                elif num_detected_kpts > 0:
                    print(f"警告: Pose实例 {i} 关键点数 {num_detected_kpts} < {EXPECTED_KEYPOINTS}")
    if keypoints_px is None: print(
        f"错误：未能提取 '{DARTBOARD_CLASS_NAME}' 的 {EXPECTED_KEYPOINTS} 个关键点。"); return None, None, None
    keypoints_px_ordered = order_points(keypoints_px)

    # 4. 提取飞镖中心 (同之前)
    dart_centers_px_list = []
    dart_confidences = []
    if detection_result.boxes is not None:
        for box in detection_result.boxes:
            cls_id = int(box.cls.item())
            class_name = detection_model.names.get(cls_id, "").lower()
            if class_name == DART_CLASS_NAME: center_norm = box.xywhn.cpu().numpy()[0][:2]; dart_centers_px_list.append(
                (center_norm * np.array([img_width, img_height])).astype(np.float32)); dart_confidences.append(
                box.conf.item())
    print(f"提取了 {len(dart_centers_px_list)} 个飞镖中心点。")

    # 5. 计算透视变换矩阵 H (映射到【旋转修正后】的归一化坐标系)
    homography_matrix = None
    try:
        src_pts = keypoints_px_ordered
        dst_pts = TARGET_POINTS_NORMALIZED_ROTATED  # 使用旋转后的目标点
        homography_matrix, status = cv2.findHomography(src_pts, dst_pts)
        if homography_matrix is None:
            print("错误：计算 Homography 失败。")
        else:
            print("计算 Homography 成功 (映射到旋转修正后的归一化坐标系)。")
    except Exception as e:
        print(f"计算 Homography 时出错: {e}")

    # 6. 变换飞镖坐标并计分 (使用【无补偿】的计分函数)
    final_scores_info = []
    transformed_dart_points_norm = []
    if homography_matrix is not None and dart_centers_px_list:
        dart_centers_array = np.array([[center] for center in dart_centers_px_list], dtype=np.float32)
        try:
            transformed_points_array = cv2.perspectiveTransform(dart_centers_array, homography_matrix)
            transformed_dart_points_norm = [tp[0] for tp in transformed_points_array]
            print("\n--- 计算分数 (使用归一化坐标，无额外补偿) ---")
            for i, (nx, ny) in enumerate(transformed_dart_points_norm):
                score, zone = calculate_score_normalized(nx, ny, R_NORM, SEGMENT_VALUES, SEGMENT_ANGLE)  # 调用无补偿版本
                info = {"dart_index": i, "original_center_px": tuple(dart_centers_px_list[i]),
                        "transformed_center_norm": (nx, ny),
                        "score": score, "zone": zone, "confidence": dart_confidences[i]}
                final_scores_info.append(info)
                print(
                    f"  Dart {i} (Conf: {info['confidence']:.3f}): Score={score}, Zone='{zone}' | Normalized=({nx:.3f}, {ny:.3f})")
        except Exception as e:
            print(f"变换坐标或计分时出错: {e}")

    # 7. 可视化 (标注原图 - 代码同之前)
    img_display = img_orig.copy()
    colors_kpt = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    labels_kpt = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(keypoints_px_ordered): px, py = int(pt[0]), int(pt[1]); cv2.circle(img_display, (px, py), 7,
                                                                                              colors_kpt[i],
                                                                                              -1); cv2.putText(
        img_display, labels_kpt[i], (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors_kpt[i], 2)
    center = find_intersection(keypoints_px_ordered[0], keypoints_px_ordered[2], keypoints_px_ordered[1],
                               keypoints_px_ordered[3])
    center = center if center is not None else np.mean(keypoints_px_ordered, axis=0)
    center_x_int, center_y_int = int(center[0]), int(center[1])
    cv2.circle(img_display, (center_x_int, center_y_int), 5, (255, 255, 255), -1)
    cv2.line(img_display, tuple(keypoints_px_ordered[0].astype(int)), tuple(keypoints_px_ordered[2].astype(int)),
             (150, 150, 150), 1)
    cv2.line(img_display, tuple(keypoints_px_ordered[1].astype(int)), tuple(keypoints_px_ordered[3].astype(int)),
             (150, 150, 150), 1)
    total_score = 0
    for info in final_scores_info:
        dart_x_int, dart_y_int = int(info['original_center_px'][0]), int(info['original_center_px'][1])
        cv2.circle(img_display, (dart_x_int, dart_y_int), 6, (0, 165, 255), -1)
        cv2.line(img_display, (center_x_int, center_y_int), (dart_x_int, dart_y_int), (0, 255, 255), 1)
        text = f"S: {info['score']}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_display, (dart_x_int + 8, dart_y_int - text_height - baseline - 2),
                      (dart_x_int + 8 + text_width, dart_y_int), (0, 0, 0), cv2.FILLED)
        cv2.putText(img_display, text, (dart_x_int + 8, dart_y_int - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (50, 255, 50), 1)
        total_score += info['score']
    cv2.putText(img_display, f"Total Score: {total_score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                cv2.LINE_AA)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img_display)
    print(f"\n带有分数标注的图像已保存到: {output_path}")

    # 8. 可视化 (标准归一化坐标系 - 代码同之前，但 H 不同导致结果不同)
    canonical_board_img = np.zeros((VIS_HEIGHT, VIS_WIDTH, 3), dtype=np.uint8) + 30
    vis_center_pt = (int(VIS_CENTER_X), int(VIS_CENTER_Y))
    cv2.line(canonical_board_img, (0, vis_center_pt[1]), (VIS_WIDTH - 1, vis_center_pt[1]), (70, 70, 70), 1)
    cv2.line(canonical_board_img, (vis_center_pt[0], 0), (vis_center_pt[0], VIS_HEIGHT - 1), (70, 70, 70), 1)
    ring_color = (150, 150, 150)
    bull_color = (200, 200, 200)
    for r_norm_val in [R_NORM['double_outer'], R_NORM['double_inner'], R_NORM['triple_outer'],
                       R_NORM['triple_inner']]: cv2.circle(canonical_board_img, vis_center_pt,
                                                           int(r_norm_val * VIS_SCALE_FACTOR), ring_color, 1)
    for r_norm_val in [R_NORM['bull_outer'], R_NORM['bull_inner']]: cv2.circle(canonical_board_img, vis_center_pt,
                                                                               int(r_norm_val * VIS_SCALE_FACTOR),
                                                                               bull_color, 1)
    segment_line_color = (100, 100, 100)
    segment_text_color = (200, 200, 200)
    text_radius_norm = R_NORM['double_outer'] * 1.08
    for i in range(len(SEGMENT_VALUES)):
        boundary_angle_deg_clockwise = (i * SEGMENT_ANGLE) - (SEGMENT_ANGLE / 2)
        boundary_angle_rad_atan2 = math.radians(90 - boundary_angle_deg_clockwise)
        end_nx = CANONICAL_RADIUS_OUTER_DOUBLE * math.cos(boundary_angle_rad_atan2)
        end_ny = CANONICAL_RADIUS_OUTER_DOUBLE * math.sin(boundary_angle_rad_atan2)
        start_px = normalized_to_pixel(0, 0, VIS_CENTER_X, VIS_CENTER_Y, VIS_SCALE_FACTOR)
        end_px = normalized_to_pixel(end_nx, end_ny, VIS_CENTER_X, VIS_CENTER_Y, VIS_SCALE_FACTOR)
        cv2.line(canonical_board_img, start_px, end_px, segment_line_color, 1)
        mid_angle_deg_clockwise = i * SEGMENT_ANGLE
        mid_angle_rad_atan2 = math.radians(90 - mid_angle_deg_clockwise)
        text_nx = text_radius_norm * math.cos(mid_angle_rad_atan2)
        text_ny = text_radius_norm * math.sin(mid_angle_rad_atan2)
        text_px, text_py = normalized_to_pixel(text_nx, text_ny, VIS_CENTER_X, VIS_CENTER_Y, VIS_SCALE_FACTOR)
        segment_value = SEGMENT_VALUES[i]
        (w, h), _ = cv2.getTextSize(str(segment_value), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(canonical_board_img, str(segment_value), (text_px - w // 2, text_py + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, segment_text_color, 1, cv2.LINE_AA)
    dart_point_color = (0, 255, 0)
    dart_text_color = (0, 255, 255)
    for info in final_scores_info:
        nx, ny = info['transformed_center_norm']
        dart_px, dart_py = normalized_to_pixel(nx, ny, VIS_CENTER_X, VIS_CENTER_Y, VIS_SCALE_FACTOR)
        cv2.circle(canonical_board_img, (dart_px, dart_py), 5, dart_point_color, -1)
        cv2.putText(canonical_board_img, str(info['score']), (dart_px + 8, dart_py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    dart_text_color, 1, cv2.LINE_AA)
    # 绘制【修正后】变换的关键点 (检查 H 准确性)
    if keypoints_px_ordered is not None and homography_matrix is not None:
        try:
            transformed_kpts_array_norm = cv2.perspectiveTransform(keypoints_px_ordered.reshape(-1, 1, 2),
                                                                   homography_matrix)
            kpt_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
            target_points_for_vis = TARGET_POINTS_NORMALIZED_ROTATED  # 检查点应该落在旋转后的目标上
            print("\n--- 变换后的归一化关键点 vs 【旋转后】目标点 ---")
            for i, tkpt_norm in enumerate(transformed_kpts_array_norm):
                nx_kpt, ny_kpt = tkpt_norm[0]
                target_nx, target_ny = target_points_for_vis[i]
                print(
                    f"  源关键点 {i}: 变换后 ({nx_kpt:.3f}, {ny_kpt:.3f}) | 目标点: ({target_nx:.3f}, {target_ny:.3f})")
                kpt_px, kpt_py = normalized_to_pixel(nx_kpt, ny_kpt, VIS_CENTER_X, VIS_CENTER_Y, VIS_SCALE_FACTOR)
                cv2.drawMarker(canonical_board_img, (kpt_px, kpt_py), kpt_colors[i], markerType=cv2.MARKER_CROSS,
                               markerSize=12, thickness=2)
        except Exception as e:
            print(f"绘制变换后关键点时出错: {e}")
    canonical_output_path = os.path.join(output_dir, canonical_vis_filename)
    cv2.imwrite(canonical_output_path, canonical_board_img)
    print(f"标准归一化坐标系可视化已保存到: {canonical_output_path}")

    return keypoints_px_ordered, final_scores_info, total_score


# --- 执行主函数 ---
if __name__ == "__main__":
    if IMAGE_PATH == 'path/to/your/test/image.jpg':
        print("错误：请修改 'IMAGE_PATH' 等配置！")
    else:
        keypoints, scores_details, final_total_score = process_and_score_image_rotated(
            POSE_MODEL_PATH, DETECTION_MODEL_PATH, IMAGE_PATH, POSE_CONFIDENCE_THRESHOLD,
            DETECTION_CONFIDENCE_THRESHOLD, OUTPUT_DIR, OUTPUT_FILENAME, CANONICAL_VIS_FILENAME)
        if final_total_score is not None:
            print("\n--- 最终分数总结 ---")
            total_check = 0
            for score_info in scores_details: print(
                f"  Dart {score_info['dart_index']}: Score = {score_info['score']}, Zone = {score_info['zone']}"); total_check += \
            score_info['score']
            print(f"  本轮总分: {final_total_score} (校验和: {total_check})")
        else:
            print("\n处理或计分过程中出现错误。")
