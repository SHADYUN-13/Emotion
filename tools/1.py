import numpy as np
import matplotlib.pyplot as plt

logfile = r'D:\xyl\emotion\DirectMHP-main\models\train_val_log.npz'  # 换成你的实际路径
logs = np.load(logfile)

train_loss = logs['train_loss']
val_loss = logs['val_loss']
train_acc = logs['train_acc']
val_acc = logs['val_acc']

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()
