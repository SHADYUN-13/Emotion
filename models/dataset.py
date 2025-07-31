import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class ObjectSeqImageDataset(Dataset):
    def __init__(self, label_csv, seq_len=10, img_size=224, transform=None):
        self.seq_len = seq_len
        self.img_size = img_size
        self.transform = transform

        df = pd.read_csv(label_csv)
        # 按 video_id 和 person_id 分组，每组是“某人某视频的一整个物体序列”
        grouped = df.groupby(['video_id', 'person_id'])
        self.samples = []
        for (vid, pid), group in grouped:
            # 按 frame_id 排序，确保顺序信息
            group_sorted = group.sort_values('frame_id')
            img_list = list(group_sorted['img_path'])
            # 假设同一组的标签一致，取第一条即可（默认已处理好）
            label = int(group_sorted.iloc[0]['emotion_label']) + 1  # -1/0/1 → 0/1/2
            self.samples.append((img_list, label))

        # 默认transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        # 用于padding的“空白图片”
        self.blank_img = torch.zeros(3, img_size, img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_list, label = self.samples[idx]
        seq = []
        for img_path in img_list[:self.seq_len]:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            seq.append(img)
        # padding（不足seq_len时补零）
        if len(seq) < self.seq_len:
            seq += [self.blank_img.clone() for _ in range(self.seq_len - len(seq))]
        seq = torch.stack(seq, dim=0)  # [T, 3, H, W]
        return seq, label
