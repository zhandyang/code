import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from thop import profile
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ===================== 中文字体支持 =====================
def setup_chinese_font():
    """强制设置中文字体，确保图表中文正常显示"""
    chinese_fonts = [
        "SimHei", "WenQuanYi Micro Hei", "Heiti TC",
        "SimSun", "Microsoft YaHei", "STCAIYUN"
    ]

    # 解析系统字体名称
    system_fonts = []
    for font_path in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        try:
            font = fm.FontProperties(fname=font_path)
            system_fonts.append(font.get_name())
        except:
            continue

    # 选择可用中文字体
    selected_font = next((f for f in chinese_fonts if f in system_fonts), None)
    if selected_font:
        plt.rcParams["font.family"] = [selected_font]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"已启用中文字体: {selected_font}")
        return True
    else:
        #  fallback到宋体（系统通常存在）
        simsun_path = "C:\\Windows\\Fonts\\simsun.ttc"
        if os.path.exists(simsun_path):
            plt.rcParams["font.family"] = ["SimSun"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"已加载宋体字体: {simsun_path}")
            return True
        else:
            print("警告：未找到可用中文字体，中文可能显示异常")
            return False


setup_chinese_font()

# ===================== 实验配置 =====================
RESULT_DIR = "experiment_results"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "experiment_results/models"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "loss_curves"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "metrics"), exist_ok=True)

# 调整Batch Size
BATCH_SIZES = {
    "ResNet18": 32,
    "ViT-Tiny": 32,
    "MobileViT": 32 # 适当降低MobileViT的Batch Size
}
EPOCHS = 500
PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"使用设备：{DEVICE}，显存：{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
    if torch.cuda.is_available() else "使用CPU（训练会很慢）"
)


# ===================== 模型定义 =====================
class TransposeLayer(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_heads=2, transformer_layers=2, stride=1):
        super().__init__()
        # 残差连接处理：当通道数或尺寸变化时使用1x1卷积调整
        self.residual = None
        if stride > 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU6(),
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1,
                      groups=in_channels * 2, stride=stride, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU6(),
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.proj = nn.Conv2d(out_channels, dim, 1, bias=False)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
                dropout=0.1, batch_first=True
            ),
            num_layers=transformer_layers
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        # 保存残差
        residual = x
        if self.residual is not None:
            residual = self.residual(x)

        local_feat = self.local_conv(x)
        b, c, h, w = local_feat.shape
        proj_feat = self.proj(local_feat).flatten(2).transpose(1, 2)
        global_feat = self.transformer(proj_feat).transpose(1, 2).view(b, -1, h, w)
        return self.se(local_feat + global_feat + residual)


class MobileViTImproved(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.shallow_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6()
        )
        # 第一个Block不进行下采样，但增加通道数
        self.block1 = MobileViTBlock(64, 128, dim=128, num_heads=2, stride=1)
        # 第二个Block进行下采样并增加通道数
        self.block2 = MobileViTBlock(128, 192, dim=192, num_heads=3, stride=2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),  # 添加Dropout正则化
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.shallow_cnn(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ViTTinyBaseline(nn.Module):
    def __init__(self, num_classes=7, patch_size=16, embed_dim=192, num_heads=3, transformer_layers=12):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size) ** 2, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
                dropout=0.1, batch_first=True
            ),
            num_layers=transformer_layers
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x += self.pos_embed
        x = self.norm(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


# ===================== 数据集加载 =====================
class SkinDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# 增强数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  # 新增垂直翻转
    transforms.RandomRotation(20),  # 增加旋转角度范围
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 增强平移和缩放
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 新增颜色抖动
    transforms.RandomGrayscale(p=0.1),  # 新增灰度转换
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.25))  # 新增随机擦除
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def load_ham10000(data_dir):
    metadata = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata.csv"))
    part1_dir = os.path.join(data_dir, "HAM10000_images_part_1")
    part2_dir = os.path.join(data_dir, "HAM10000_images_part_2")

    image_paths, valid_indices = [], []
    for i, img_id in enumerate(metadata["image_id"]):
        p1 = os.path.join(part1_dir, f"{img_id}.jpg")
        p2 = os.path.join(part2_dir, f"{img_id}.jpg")
        if os.path.exists(p1):
            image_paths.append(p1)
            valid_indices.append(i)
        elif os.path.exists(p2):
            image_paths.append(p2)
            valid_indices.append(i)

    labels = LabelEncoder().fit_transform(metadata["dx"].iloc[valid_indices])
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    return (
        SkinDataset(train_paths, train_labels, train_transform),
        SkinDataset(val_paths, val_labels, val_test_transform),
        SkinDataset(test_paths, test_labels, val_test_transform),
        class_weights
    )


def get_dataloaders(data_dir, batch_size):
    train_ds, val_ds, test_ds, class_weights = load_ham10000(data_dir)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        class_weights
    )


# ===================== 训练与评估 =====================
def train_model(model, model_name, train_loader, val_loader, class_weights):
    torch.cuda.empty_cache()  # 训练前清理缓存

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')  # 监控验证损失而非F1作为主要早停指标
    best_val_f1 = 0.0
    best_epoch = 0
    patience_count = 0  # 早停计数器
    history = {
        "train_loss": [], "val_loss": [], "val_f1": []
    }

    model.to(DEVICE)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch + 1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_true, val_preds, average="macro")
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # 早停机制：同时监控损失和F1
        if val_loss < best_val_loss or val_f1 > best_val_f1:
            is_better = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_better = True
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                is_better = True

            if is_better:
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(RESULT_DIR, "experiment_results/models", f"{model_name}_best.pth"))
                print(f"Epoch {epoch + 1}: 最佳验证损失={best_val_loss:.4f}, 最佳验证F1={best_val_f1:.4f}")
                patience_count = 0
        else:
            patience_count += 1
            print(f"Epoch {epoch + 1}: 验证损失未改善，耐心计数={patience_count}/{PATIENCE}")
            if patience_count > PATIENCE:
                print(
                    f"{model_name} 早停于Epoch {epoch + 1}（最佳性能在Epoch {best_epoch + 1}，损失={best_val_loss:.4f}，F1={best_val_f1:.4f}）")
                break

        scheduler.step()
        torch.cuda.empty_cache()  # 每轮清理缓存

    # 绘制曲线
    actual_epochs = len(history["train_loss"])
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), history["train_loss"], label="训练损失")
    plt.plot(range(1, actual_epochs + 1), history["val_loss"], label="验证损失")
    plt.title(f"{model_name} 损失曲线（共{actual_epochs}轮）")
    plt.xlabel("Epoch")
    plt.ylabel("损失值")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULT_DIR, "loss_curves", f"{model_name}_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), history["val_f1"], label="验证集宏平均F1")
    plt.axhline(y=best_val_f1, color='r', linestyle='--', label=f'最佳F1: {best_val_f1:.4f}')
    plt.title(f"{model_name} 验证集F1曲线（共{actual_epochs}轮）")
    plt.xlabel("Epoch")
    plt.ylabel("宏平均F1")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULT_DIR, "loss_curves", f"{model_name}_f1.png"), dpi=300)
    plt.close()

    return history


def evaluate_model(model, model_name, test_loader):
    model.load_state_dict(torch.load(
        os.path.join(RESULT_DIR, "experiment_results/models", f"{model_name}_best.pth"),
        map_location=DEVICE
    ))
    model.to(DEVICE)
    model.eval()

    test_preds, test_true = [], []
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            test_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            test_true.extend(labels.numpy())
    infer_time = time.time() - start_time
    fps = len(test_loader.dataset) / infer_time

    macro_f1 = f1_score(test_true, test_preds, average="macro")
    class_report = classification_report(
        test_true, test_preds, target_names=[f"类别{i}" for i in range(7)],
        zero_division=1, output_dict=True
    )

    input_tensor = torch.randn(1, 3, 224, 224).to(DEVICE)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    params_m = params / 1e6
    flops_g = flops / 1e9

    metrics = {
        "模型名称": model_name, "宏平均F1": macro_f1,
        "参数量(M)": params_m, "FLOPs(G)": flops_g, "推理速度(FPS)": fps,
        "每类精确率": {f"类别{i}": class_report[f"类别{i}"]["precision"] for i in range(7)},
        "每类召回率": {f"类别{i}": class_report[f"类别{i}"]["recall"] for i in range(7)},
        "每类F1": {f"类别{i}": class_report[f"类别{i}"]["f1-score"] for i in range(7)}
    }

    print(f"\n{model_name} 评估结果：")
    print(f"• 宏平均F1: {macro_f1:.4f}")
    print(f"• 参数量: {params_m:.2f} M")
    print(f"• FLOPs: {flops_g:.2f} G")
    print(f"• 推理速度: {fps:.1f} FPS")

    return metrics


# ===================== 主函数 =====================
def main(data_dir):
    if not os.path.exists(os.path.join(data_dir, "HAM10000_metadata.csv")):
        print(f"错误：未找到数据集文件，请检查路径: {data_dir}")
        return

    # 模型选择功能
    print("\n可用模型：ResNet18, ViT-Tiny, MobileViT")
    model_choice = input("请输入要训练的模型名称: ").strip()
    valid_models = ["ResNet18", "ViT-Tiny", "MobileViT"]
    if model_choice not in valid_models:
        print(f"错误：模型名称无效，请从{valid_models}中选择")
        return

    # 仅初始化选中的模型
    models = {
        "ResNet18": ResNet18Baseline(),
        "ViT-Tiny": ViTTinyBaseline(),
        "MobileViT": MobileViTImproved()
    }
    model = models[model_choice]
    model_name = model_choice

    # 清理缓存并加载数据
    torch.cuda.empty_cache()
    batch_size = BATCH_SIZES[model_name]
    print(f"\n===== 开始训练 {model_name}（计划训练{EPOCHS}轮，Batch Size={batch_size}） =====")

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(data_dir, batch_size)
    train_model(model, model_name, train_loader, val_loader, class_weights)

    # 评估
    metrics = evaluate_model(model, model_name, test_loader)
    print(f"{model_name} 评估完成：宏平均F1={metrics['宏平均F1']:.4f}")

    # 保存指标
    metrics_df = pd.DataFrame([{
        "模型名称": metrics["模型名称"],
        "宏平均F1": metrics["宏平均F1"],
        "参数量(M)": metrics["参数量(M)"],
        "FLOPs(G)": metrics["FLOPs(G)"],
        "推理速度(FPS)": metrics["推理速度(FPS)"]
    }])
    metrics_df.to_csv(os.path.join(RESULT_DIR, "metrics", f"{model_name}_summary.csv"), index=False,
                      encoding="utf-8-sig")

    details_df = pd.DataFrame({
        "指标": ["精确率", "召回率", "F1分数"],
        **{f"类别{i}": [
            metrics["每类精确率"][f"类别{i}"],
            metrics["每类召回率"][f"类别{i}"],
            metrics["每类F1"][f"类别{i}"]
        ] for i in range(7)}
    })
    details_df.to_csv(os.path.join(RESULT_DIR, "metrics", f"{model_name}_details.csv"), index=False,
                      encoding="utf-8-sig")

    print(f"\n{model_name} 训练完成！结果保存至：{RESULT_DIR}")


if __name__ == "__main__":
    DATA_DIR = r"HAM10000"  # 替换为实际数据集路径
    main(DATA_DIR)