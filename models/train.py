import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from network import CNNLSTMEmotionNet
from dataset import ObjectSeqImageDataset
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def get_class_weights(label_csv, num_classes, device):
    df = pd.read_csv(label_csv)
    labels = df['emotion_label'].astype(int) + 1   # -1/0/1 → 0/1/2
    class_counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    print(f"[INFO] Class counts: {class_counts}, class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float, device=device)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, total_correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return total_loss / total, total_correct / total, np.array(all_labels), np.array(all_preds)

def main():
    # 路径配置
    train_label_csv = r'E:/emotion/frames/crop/train/labels.csv'
    val_label_csv   = r'E:/emotion/frames/crop/val/labels.csv'
    best_model_path = r'D:\xyl\emotion\DirectMHP-main\weights\best_model.pth'

    # 超参数配置
    seq_len = 10
    img_size = 224
    num_classes = 3
    batch_size = 8
    num_epochs = 40
    lr = 3e-4

    # 先定义 device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
    ])
    # 验证集只做Resize+ToTensor
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # 数据集与dataloader
    train_dataset = ObjectSeqImageDataset(train_label_csv, seq_len, img_size, train_transform)
    val_dataset   = ObjectSeqImageDataset(val_label_csv,   seq_len, img_size, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    # 损失加权
    class_weights = get_class_weights(train_label_csv, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 初始化模型
    model = CNNLSTMEmotionNet(
        cnn_name='resnet18',
        lstm_hidden=256,
        lstm_layers=1,
        num_classes=num_classes,
        dropout=0.5,
        pretrained=True
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # 日志
    best_val_loss = float('inf')
    train_loss_log, val_loss_log, train_acc_log, val_acc_log = [], [], [], []

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_true, val_pred = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if epoch % 5 == 0 or epoch == num_epochs:
            cm = confusion_matrix(val_true, val_pred, labels=[0,1,2])
            print("Val Confusion Matrix (neg, norm, pos):\n", cm)

        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        train_acc_log.append(train_acc)
        val_acc_log.append(val_acc)

        # 保存 val loss 最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model updated at epoch {epoch}, val loss: {val_loss:.4f}")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}. Model saved to {best_model_path}")

    # 保存训练日志
    np.savez("train_val_log.npz",
             train_loss=np.array(train_loss_log),
             val_loss=np.array(val_loss_log),
             train_acc=np.array(train_acc_log),
             val_acc=np.array(val_acc_log))

if __name__ == '__main__':
    main()
