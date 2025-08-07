import os
import zipfile
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from facenet_pytorch import MTCNN
from google.colab import drive

# === MOUNT GOOGLE DRIVE ===
drive.mount('/content/drive')

# === UNZIP DATA ===
zip_path = "/content/drive/MyDrive/processed_3000.zip"
extract_path = "/content/processed_3000"
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/content")
    print("✅ 解壓縮完成")

# === FACE ALIGNMENT (一次性) ===
raw_root = "/content/processed_3000"
aligned_root = "/content/aligned_3000"
mtcnn = MTCNN(image_size=112, margin=0)

for split in ['train', 'val', 'test']:
    print(f"🔄 對齊 {split} 中...")
    raw_dir = os.path.join(raw_root, split)
    aligned_dir = os.path.join(aligned_root, split)
    os.makedirs(aligned_dir, exist_ok=True)
    dataset = datasets.ImageFolder(raw_dir)

    for class_name in dataset.classes:
        os.makedirs(os.path.join(aligned_dir, class_name), exist_ok=True)

    for path, label in tqdm(dataset.samples):
        img = Image.open(path).convert("RGB")
        aligned = mtcnn(img)
        if aligned is not None:
            aligned_img = transforms.ToPILImage()(aligned)
            class_name = dataset.classes[label]
            save_path = os.path.join(aligned_dir, class_name, os.path.basename(path))
            aligned_img.save(save_path)

# === CONFIG ===
data_root = aligned_root
train_dir = os.path.join(data_root, "train")
val_dir = os.path.join(data_root, "val")
test_dir = os.path.join(data_root, "test")

batch_size = 64
epochs = 30
embedding_size = 512
num_workers = 2
lr = 0.01
warmup_epochs = 10
save_every_n_epoch = 5

checkpoint_dir = "/content/drive/MyDrive/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
last_checkpoint_path = os.path.join(checkpoint_dir, "last.pth")
best_checkpoint_path = os.path.join(checkpoint_dir, "best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === AUGMENTATION ===
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_classes = len(train_dataset.classes)

# === MODEL ===
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = cosine * (1 - one_hot) + torch.cos(theta + self.m) * one_hot
        output *= self.s
        return output

class ResNet50Face(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in list(self.backbone.parameters())[:6]:
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        return self.backbone(x)

model = ResNet50Face(embedding_size).to(device)
metric_fc = ArcMarginProduct(embedding_size, num_classes).to(device)

optimizer = optim.SGD([
    {'params': model.parameters()},
    {'params': metric_fc.parameters()}
], lr=lr, momentum=0.9, weight_decay=5e-4)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
criterion = nn.CrossEntropyLoss()

start_epoch = 0
best_val_auc = 0
if os.path.exists(last_checkpoint_path):
    checkpoint = torch.load(last_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    metric_fc.load_state_dict(checkpoint['metric_fc_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_val_auc = checkpoint['best_val_auc']
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

patience = 7
early_stop_counter = 0

for epoch in range(start_epoch, epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    if epoch < warmup_epochs:
        warmup_lr = lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        features = model(imgs)
        outputs = metric_fc(features, labels)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100
    avg_loss = total_loss / len(train_loader)

    # === EVALUATE ===
    model.eval()
    embeddings, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            emb = model(imgs)
            embeddings.append(emb.cpu())
            labels_list.extend(labels.numpy())

    embeddings = torch.cat(embeddings, dim=0)
    labels_list = np.array(labels_list)

    pairs, pair_labels = [], []
    for _ in range(6000):
        idx1 = random.randint(0, len(labels_list) - 1)
        same = np.where(labels_list == labels_list[idx1])[0]
        diff = np.where(labels_list != labels_list[idx1])[0]
        if len(same) > 1:
            idx2 = random.choice(same[same != idx1])
            pairs.append((idx1, idx2))
            pair_labels.append(1)
        idx2 = random.choice(diff)
        pairs.append((idx1, idx2))
        pair_labels.append(0)

    sims = [F.cosine_similarity(embeddings[i], embeddings[j], dim=0).item() for i, j in pairs]
    fpr, tpr, thresholds = roc_curve(pair_labels, sims)
    auc_score = auc(fpr, tpr)
    opt_idx = np.argmax(tpr - fpr)
    opt_threshold = thresholds[opt_idx]

    print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}% | Val AUC: {auc_score:.4f} | Opt Thresh: {opt_threshold:.4f}")

    # Save best
    if auc_score > best_val_auc:
        best_val_auc = auc_score
        early_stop_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metric_fc_state_dict': metric_fc.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc
        }, best_checkpoint_path)
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("⏹️ Early stopping")
            break

    if (epoch + 1) % save_every_n_epoch == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metric_fc_state_dict': metric_fc.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc
        }, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metric_fc_state_dict': metric_fc.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_auc': best_val_auc
    }, last_checkpoint_path)

    scheduler.step()

# === TEST ===
print("測試最佳模型")
checkpoint = torch.load(best_checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_embeddings, test_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Test"):
        imgs = imgs.to(device)
        emb = model(imgs)
        test_embeddings.append(emb.cpu())
        test_labels.extend(labels.numpy())

test_embeddings = torch.cat(test_embeddings, dim=0)
test_labels = np.array(test_labels)

pairs, pair_labels = [], []
for _ in range(6000):
    idx1 = random.randint(0, len(test_labels) - 1)
    same = np.where(test_labels == test_labels[idx1])[0]
    diff = np.where(test_labels != test_labels[idx1])[0]
    if len(same) > 1:
        idx2 = random.choice(same[same != idx1])
        pairs.append((idx1, idx2))
        pair_labels.append(1)
    idx2 = random.choice(diff)
    pairs.append((idx1, idx2))
    pair_labels.append(0)

sims = [torch.dot(F.normalize(test_embeddings[i], dim=0), F.normalize(test_embeddings[j], dim=0)).item() for i, j in pairs]
fpr, tpr, thresholds = roc_curve(pair_labels, sims)
auc_score = auc(fpr, tpr)
opt_idx = np.argmax(tpr - fpr)
opt_threshold = thresholds[opt_idx]
preds = [1 if s > opt_threshold else 0 for s in sims]
acc = np.mean(np.array(preds) == np.array(pair_labels)) * 100

print(f"測試集 Verification Accuracy: {acc:.2f}% | AUC: {auc_score:.4f} | Threshold: {opt_threshold:.4f}")

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
