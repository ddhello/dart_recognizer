import os
import random
from pathlib import Path # 使用 pathlib 更方便处理路径
from ultralytics import YOLO
import cv2 # Optional: 仅在你想实时显示结果时需要

# --- 配置区域 ---
# !!! 重要: 修改为你 best.pt 文件的实际路径 !!!
MODEL_PATH = 'best.pt' # 例如: 'runs/pose/dartboard_pose_experiment_python/weights/best.pt'

# !!! 重要: 确保这是你的验证集图像所在的文件夹 !!!
VALIDATION_IMAGE_DIR = './darts_yolo_dataset/val/images' # 指向你的验证集图像文件夹

NUM_SAMPLES = 100     # 要随机选择的图像数量
CONF_THRESHOLD = 0.5  # 检测置信度阈值 (过滤低置信度的检测)
IMAGE_SIZE = 800      # 预测时使用的图像尺寸 (应与训练一致)

# 是否在运行时逐个显示带标注的预测图像？
# 设置为 True 会打开一个窗口显示每个结果，按 'q' 关闭或继续下一个
# 设置为 False 则只将结果保存到文件，不实时显示
SHOW_IMAGES_LIVE = False
# --- 配置结束 ---

def test_random_subset(model_path, img_dir, num_samples, conf, imgsz, show_live):
    """
    使用 YOLOv8 姿态模型在指定图像目录的随机子集上运行预测。
    """

    model_file = Path(model_path)
    val_dir = Path(img_dir)

    # --- 1. 输入验证 ---
    if not model_file.is_file():
        print(f"错误: 模型文件未找到: {model_path}")
        print("请确保 MODEL_PATH 设置正确。")
        return
    if not val_dir.is_dir():
        print(f"错误: 验证图像目录未找到: {img_dir}")
        print("请确保 VALIDATION_IMAGE_DIR 设置正确。")
        return

    # --- 2. 加载模型 ---
    print(f"正在加载模型: {model_path}")
    try:
        model = YOLO(model_file)
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # --- 3. 获取并随机抽样图像文件 ---
    print(f"正在从目录查找图像: {val_dir}")
    try:
        # 查找常见的图像文件后缀
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        all_image_paths = [p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
    except Exception as e:
        print(f"读取图像文件列表时出错: {e}")
        return

    num_available = len(all_image_paths)
    print(f"找到 {num_available} 张图像。")

    if num_available == 0:
        print("错误: 在指定的验证目录中没有找到图像文件。")
        return

    # 检查请求的样本数是否超过可用数量
    if num_samples > num_available:
        print(f"警告: 请求的样本数量 ({num_samples}) 大于可用图像数量 ({num_available})。")
        print(f"将使用所有 {num_available} 张可用图像进行测试。")
        selected_image_paths = all_image_paths
        actual_num_samples = num_available
    elif num_samples <= 0:
         print("错误: 样本数量必须是正数。")
         return
    else:
        print(f"正在随机选择 {num_samples} 张图像进行测试...")
        selected_image_paths = random.sample(all_image_paths, num_samples)
        actual_num_samples = num_samples

    # 将 Path 对象转换为字符串列表，因为 predict 函数通常期望字符串路径
    selected_image_paths_str = [str(p) for p in selected_image_paths]

    # --- 4. 运行预测 ---
    print(f"\n开始对 {actual_num_samples} 张选定图像运行预测...")
    try:
        # 使用 save=True 会自动将带标注的图像保存到 runs/pose/predict* 目录
        # stream=False 对于少量图像（如100张）通常足够快，并返回一个列表
        # stream=True 更适合非常大的文件夹或视频流，返回一个生成器
        results = model.predict(source=selected_image_paths_str,
                                save=True,
                                imgsz=imgsz,
                                conf=conf,
                                stream=False) # Process this batch of images

        save_dir = None # 用于存储结果保存目录
        print("\n--- 处理预测结果 ---")
        processed_count = 0

        # 即使 stream=False, results 也是可迭代的 (通常是一个列表)
        for result in results:
            processed_count += 1
            # 获取保存目录 (通常所有结果都在同一个目录下)
            if save_dir is None and hasattr(result, 'save_dir'):
                 save_dir = result.save_dir

            print(f"({processed_count}/{actual_num_samples}) 处理图像: {Path(result.path).name}") # 只显示文件名

            # 打印一些基本检测信息
            if result.keypoints and result.keypoints.shape[1] > 0: # 检查是否有关键点被检测到
                 num_instances = len(result.boxes) if result.boxes is not None else 0
                 print(f"  检测到 {num_instances} 个飞镖盘实例，包含关键点。")
                 # 你可以在这里添加更详细的打印，比如打印关键点坐标
                 # kpts_data = result.keypoints.cpu().numpy()
                 # print(f"    第一个实例的关键点 (xy): {kpts_data.xy[0]}")
            else:
                 print("  未检测到关键点或飞镖盘。")

            # 可选：实时显示带标注的图像
            if show_live:
                try:
                    annotated_frame = result.plot() # 生成带标注的图像
                    cv2.imshow(f"YOLOv8 Pose Prediction - {Path(result.path).name}", annotated_frame)
                    # 按 'q' 退出整个显示循环，按其他任意键继续下一张
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyWindow(f"YOLOv8 Pose Prediction - {Path(result.path).name}") # 关闭当前窗口
                    if key == ord('q'):
                        print("\n用户请求停止实时显示。")
                        show_live = False # 停止显示后续图像
                except Exception as display_e:
                    print(f"  显示图像时出错: {display_e}")
                    print("  请确保你安装了 OpenCV (pip install opencv-python) 并且有图形环境。")
                    show_live = False # 出错后停止尝试显示

        print(f"\n--- 预测完成 ---")
        print(f"成功处理了 {processed_count} 张图像。")
        if save_dir:
            print(f"带标注的图像已保存到目录: {Path(save_dir).resolve()}") # 显示绝对路径
        else:
            # 如果 save=True 但 save_dir 没获取到，可能是预测过程中断或没有检测结果
            print("预测已运行，但未能确定保存目录。请检查 runs/pose/ 目录下是否有新的 predict* 文件夹。")

    except Exception as e:
        print(f"\n运行预测时发生严重错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细的错误追踪信息

    finally:
        # 确保如果之前打开了窗口，最后能关闭它 (虽然上面是逐个关闭的)
        cv2.destroyAllWindows()

# --- 主程序入口 ---
if __name__ == "__main__":
    # 执行测试函数
    test_random_subset(
        model_path=MODEL_PATH,
        img_dir=VALIDATION_IMAGE_DIR,
        num_samples=NUM_SAMPLES,
        conf=CONF_THRESHOLD,
        imgsz=IMAGE_SIZE,
        show_live=SHOW_IMAGES_LIVE
    )