import os
import re
import pandas as pd

crop_val_root = r"E:\emotion\frames\crop\val"
out_csv = r"E:\emotion\frames\crop\val\labels.csv"

rows = []
# 正则提取信息：{video_id}_frame{frame_id}_p{person_id}_{object_name}.jpg
pattern = re.compile(r"(\d+)_frame(\d+)_p(\d+)_(.+)\.jpg")

# 指定标签，key为(video_id, person_id)
special_labels = {
    ("12", "0"): 1,  # head0: positive
    ("12", "1"): -1, # head1: negative
    ("13", "0"): 1,  # head0: positive
    ("13", "1"): 0, # head1: negative
}

for video_id in os.listdir(crop_val_root):
    video_dir = os.path.join(crop_val_root, video_id)
    if not os.path.isdir(video_dir):
        continue
    for img_file in os.listdir(video_dir):
        match = pattern.match(img_file)
        if match:
            vid, frame_id, person_id, object_name = match.groups()
            # 按要求赋标签
            key = (vid, person_id)
            if key in special_labels:
                emotion_label = special_labels[key]
            else:
                emotion_label = -1  # 其他都-1
            row = {
                "img_path": os.path.join(video_dir, img_file),
                "video_id": vid,
                "frame_id": frame_id,
                "person_id": person_id,
                "object_name": object_name,
                "emotion_label": emotion_label
            }
            rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values(['video_id', 'frame_id', 'person_id', 'object_name'])
df.to_csv(out_csv, index=False)
print(f"已生成: {out_csv}，共{len(df)}行")
