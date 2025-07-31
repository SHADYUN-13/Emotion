import os
import cv2
import pandas as pd
import numpy as np
from PIL import ImageFont, ImageDraw, Image

emoji_map = {
    "positive": "😊",
    "negative": "😟",
    "neutral": "😐"
}

img_root = r"E:\emotion\frames\left"
txt_root = r"E:\emotion\frames\txt"
csv_path = r"E:\emotion\frames\crop\val\emotion_result.csv"
output_dir = r"E:\emotion\frames\left_vis"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

for video_id in ["12", "13"]:
    video_folder = f"left_output_{video_id}"
    video_dir = os.path.join(img_root, video_folder)
    if not os.path.isdir(video_dir):
        print(f"缺失: {video_dir}")
        continue
    # 读头框
    txt_path = os.path.join(txt_root, f"left_{video_id}.txt")
    if not os.path.isfile(txt_path):
        print(f"缺失: {txt_path}")
        continue
    # 只收集head（编号从1开始）
    head_boxes = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts[0].startswith("head"):
                pid = int(parts[0][4:]) - 1 # head
                xs = [float(parts[3]), float(parts[5]), float(parts[7]), float(parts[9])]
                ys = [float(parts[4]), float(parts[6]), float(parts[8]), float(parts[10])]
                head_boxes[pid] = (xs, ys)
    # 遍历所有帧
    for fname in os.listdir(video_dir):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        frame_path = os.path.join(video_dir, fname)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"无法读取 {frame_path}")
            continue
        h, w = img.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("seguiemj.ttf", 40)
        except:
            font = ImageFont.truetype("arial.ttf", 36)
        for pid, (xs, ys) in head_boxes.items():
            # 转为像素
            x1, y1 = int(min(xs) * w), int(min(ys) * h)
            x2, y2 = int(max(xs) * w), int(max(ys) * h)
            # 匹配csv
            match = df[(df["video_id"].astype(str)==str(video_id)) & (df["person_id"].astype(str)==str(pid))]
            if match.empty:
                print(f"未找到csv: video_id={video_id}, pid={pid}")
                continue
            pred_label = match.iloc[0]["pred_label"]
            emoji = emoji_map.get(pred_label, "❓")
            draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=3)
            draw.text((x1, y1-45), emoji, font=font, fill=(255,0,0,255))
        img_vis = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        save_path = os.path.join(output_dir, f"{video_folder}_{fname}")
        cv2.imwrite(save_path, img_vis)
        print(f"已保存: {save_path}")

print("全部完成！")
