import torch
import torch.nn as nn
from torchvision import models

# 定義模型架構
class ResNet50Face(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        return self.backbone(x)

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet50Face().to(device)

# 載入 state_dict
checkpoint = torch.load("train_best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 建立假輸入轉 trace
example_input = torch.randn(1, 3, 112, 112).to(device)
traced_model = torch.jit.trace(model, example_input)

# 儲存為真正的 TorchScript 模型
traced_model.save("facenet_model.pt")
