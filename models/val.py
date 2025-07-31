import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ObjectSeqImageDataset
from network import CNNLSTMEmotionNet
import numpy as np

# 参数设置
model_path = r"D:\xyl\emotion\DirectMHP-main\weights\best_model.pth"
label_csv = r"E:\emotion\frames\crop\val\labels.csv"
output_csv = r"E:\emotion\frames\crop\val\emotion_result.csv"

seq_len = 10
img_size = 224
num_classes = 3
batch_size = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# val集 transform
val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# 加载数据集
val_dataset = ObjectSeqImageDataset(label_csv, seq_len, img_size, val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 加载模型
model = CNNLSTMEmotionNet(
    cnn_name='resnet18', lstm_hidden=256, lstm_layers=1, num_classes=num_classes, dropout=0.5, pretrained=False
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 结果记录
results = []

def map_score_to_label(score):
    if score < 0:
        return "negative"
    elif score == 0:
        return "neutral"
    else:
        return "positive"

with torch.no_grad():
    for idx, (seqs, labels) in enumerate(val_loader):
        seqs = seqs.to(device)  # [B, T, 3, H, W]
        logits = model(seqs)
        # logits: [B, num_classes]
        scores = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)   # 0/1/2
        # 把预测映射回标签（-1, 0, 1） → negative, neutral, positive
        for i in range(seqs.size(0)):
            # 对应样本的元信息
            sample_idx = idx * batch_size + i
            if sample_idx >= len(val_dataset):
                continue
            # 获取video_id/person_id（可选，如果你dataset里保存了）
            # 需在dataset加保存self.samples的video_id, person_id
            img_list, _ = val_dataset.samples[sample_idx]
            # 取第一个img_path
            img_path = img_list[0]
            # 用正则或split方式从img_path里提取video_id/person_id
            import re
            match = re.search(r'(\d+)_frame\d+_p(\d+)_', img_path)
            if match:
                video_id = match.group(1)
                person_id = match.group(2)
            else:
                video_id = ""
                person_id = ""
            # 获取情绪分数（也可以直接取最大概率类别分数）
            score = float(scores[i][2] - scores[i][0])  # positive - negative 概率差（可自定义）
            pred_class = preds[i].item()
            if pred_class == 0:
                pred_label = "negative"
            elif pred_class == 1:
                pred_label = "neutral"
            else:
                pred_label = "positive"
            results.append({
                "video_id": video_id,
                "person_id": person_id,
                "score": score,
                "pred_class": pred_class,
                "pred_label": pred_label
            })

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"推理完成，结果保存在: {output_csv}")
