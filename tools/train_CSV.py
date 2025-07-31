import os
import csv
from pathlib import Path
import re

# crop 根目录
CROP_ROOT = r"E:\emotion\frames\crop\train"
CSV_PATH = r"E:\emotion\frames\crop\train\labels.csv"

# 情绪映射
POSITIVE = {'00', '01', '04', '05', '08', '09'}
NEUTRAL  = {'02', '06', '10'}
NEGATIVE = {'03', '07', '11'}
def get_emotion_label(vid):
    if vid in POSITIVE:
        return 1
    elif vid in NEUTRAL:
        return 0
    elif vid in NEGATIVE:
        return -1
    else:
        return 0  # 默认中性

# 文件名正则匹配
pattern = re.compile(r"(\w+)_frame(\d+)_p(\d+)_(.+)\.jpg")

header = ["img_path", "video_id", "frame_id", "person_id", "object_name", "emotion_label"]

rows = []
for video_dir in Path(CROP_ROOT).iterdir():
    if not video_dir.is_dir():
        continue
    for file in video_dir.glob("*.jpg"):
        match = pattern.match(file.name)
        if match:
            video_id, frame_id, person_id, object_name = match.groups()
            emotion_label = get_emotion_label(video_id)
            row = [
                str(file.resolve()),
                video_id,
                frame_id,
                person_id,
                object_name,
                emotion_label
            ]
            rows.append(row)

with open(CSV_PATH, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"已生成 csv 文件：{CSV_PATH}，共 {len(rows)} 条记录")
