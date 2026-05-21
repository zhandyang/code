import os
import sys
import torch
import torch.nn as nn
from code import (
    train_model, evaluate_model, get_dataloaders,
    RESULT_DIR, DEVICE, EPOCHS, setup_chinese_font
)
import pandas as pd

setup_chinese_font()

# ===================== 新模型定义 =====================

class StandardCNN(nn.Module):
    """标准卷积神经网络"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ConvViT(nn.Module):
    """混合卷积-Transformer架构（ConvViT）"""
    def __init__(self, num_classes=7, embed_dim=256, num_heads=4, transformer_layers=6):
        super().__init__()

        # CNN特征提取器（更深的卷积层）
        self.conv_stem = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56

            # Stage 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # 28x28

            # Stage 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # 14x14
        )

        # 将CNN特征映射到Transformer维度
        self.patch_size = 14  # 14x14 feature map
        self.num_patches = (224 // 16) ** 2  # 196 patches

        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer编码器
        self.norm_pre = nn.LayerNorm(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=transformer_layers
        )
        self.norm_post = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        # CNN特征提取
        x = self.conv_stem(x)  # [B, 256, 14, 14]

        # 投影到Transformer维度
        x = self.proj(x)  # [B, embed_dim, 14, 14]

        # 展平为序列
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, 196, embed_dim]

        # 添加位置编码
        x = x + self.pos_embed

        # Transformer处理
        x = self.norm_pre(x)
        x = self.transformer(x)
        x = self.norm_post(x)

        # 全局平均池化
        x = x.mean(dim=1)  # [B, embed_dim]

        # 分类
        x = self.head(x)
        return x


# ===================== 批次大小配置 =====================
BATCH_SIZES = {
    "CNN": 32,
    "ConvViT": 24,  # Transformer层需要更多内存
}


# ===================== 主训练函数 =====================
def main(data_dir, model_name):
    if not os.path.exists(os.path.join(data_dir, "HAM10000_metadata.csv")):
        print(f"错误：未找到数据集文件，请检查路径: {data_dir}")
        return

    # 模型字典
    models = {
        "CNN": StandardCNN(),
        "ConvViT": ConvViT()
    }

    if model_name not in models:
        print(f"错误：模型名称无效，请从 {list(models.keys())} 中选择")
        return

    model = models[model_name]
    batch_size = BATCH_SIZES[model_name]

    print(f"\n===== 开始训练 {model_name}（计划训练{EPOCHS}轮，Batch Size={batch_size}） =====")
    print(f"设备：{DEVICE}")

    # 加载数据
    torch.cuda.empty_cache()
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(data_dir, batch_size)

    # 训练模型
    train_model(model, model_name, train_loader, val_loader, class_weights)

    # 评估模型
    metrics = evaluate_model(model, model_name, test_loader)
    print(f"\n{model_name} 评估完成：宏平均F1={metrics['宏平均F1']:.4f}")

    # 保存指标
    metrics_df = pd.DataFrame([{
        "模型名称": metrics["模型名称"],
        "宏平均F1": metrics["宏平均F1"],
        "参数量(M)": metrics["参数量(M)"],
        "FLOPs(G)": metrics["FLOPs(G)"],
        "推理速度(FPS)": metrics["推理速度(FPS)"]
    }])
    metrics_df.to_csv(
        os.path.join(RESULT_DIR, "metrics", f"{model_name}_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    details_df = pd.DataFrame({
        "指标": ["精确率", "召回率", "F1分数"],
        **{f"类别{i}": [
            metrics["每类精确率"][f"类别{i}"],
            metrics["每类召回率"][f"类别{i}"],
            metrics["每类F1"][f"类别{i}"]
        ] for i in range(7)}
    })
    details_df.to_csv(
        os.path.join(RESULT_DIR, "metrics", f"{model_name}_details.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n{model_name} 训练完成！结果保存至：{RESULT_DIR}")


if __name__ == "__main__":
    DATA_DIR = r"HAM10000"  # 替换为实际数据集路径

    # 交互式选择模型
    print("\n可用模型：CNN, ConvViT")
    model_choice = input("请输入要训练的模型名称（或输入 'both' 训练两个模型）: ").strip()

    if model_choice.lower() == 'both':
        for model_name in ["CNN", "ConvViT"]:
            print(f"\n{'='*60}")
            main(DATA_DIR, model_name)
            torch.cuda.empty_cache()
    else:
        main(DATA_DIR, model_choice)
