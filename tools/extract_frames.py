import cv2
import os
import glob

# —— 配置区 —— #
# 要提取帧的视频所在目录
input_dir = r"D:\xyl\emotion\video\video\left"
# 所有子文件夹将创建在此目录下
output_root = r"E:\emotion\frames\left"
# 支持的视频后缀
video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
# ———————— #

# 确保根输出目录存在
os.makedirs(output_root, exist_ok=True)

# 遍历所有视频文件
video_paths = []
for ext in video_extensions:
    video_paths += glob.glob(os.path.join(input_dir, ext))

if not video_paths:
    print(f"在 `{input_dir}` 下没有找到视频文件，请检查后缀或路径。")
    exit()

for video_path in sorted(video_paths):
    # 取文件名（不含扩展）作为子文件夹名
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root, basename)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 无法打开视频：{video_path}")
        continue

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 保存为0000.jpg, 0001.jpg, …
        frame_number = frame_idx + 1
        filename = f"{basename}_frame_{frame_number:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        frame_idx += 1

    cap.release()
    print(f"[完成] `{basename}` → 提取了 {frame_idx} 帧 到 `{output_dir}`")
