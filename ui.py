import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import numpy as np
from PIL import Image, ImageTk
import torchvision.transforms as transforms
from torch.nn import functional as F
from torchvision import models
import torch.nn as nn

# 确保中文显示正常
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# ===================== 【将训练代码复制过来即可】 =====================
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
        residual = x
        if self.residual is not None:
            residual = self.residual(x)

        local_feat = self.local_conv(x)
        b, c, h, w = local_feat.shape
        proj_feat = self.proj(local_feat).flatten(2).transpose(1, 2)
        global_feat = self.transformer(proj_feat).transpose(1, 2).view(b, -1, h, w)
        return self.se(local_feat + global_feat + residual)

# ===================== 你训练时的 MobileViT 原封不动 =====================
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
        self.block1 = MobileViTBlock(64, 128, dim=128, num_heads=2, stride=1)
        self.block2 = MobileViTBlock(128, 192, dim=192, num_heads=3, stride=2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
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
        return self.head(x)

class StandardCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
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
    def __init__(self, num_classes=7, embed_dim=256, num_heads=4, transformer_layers=6):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.num_patches = (224 // 16) ** 2
        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm_pre = nn.LayerNorm(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=0.1, activation='gelu', batch_first=True),
            num_layers=transformer_layers
        )
        self.norm_post = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, num_classes)
        )

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.norm_pre(x)
        x = self.transformer(x)
        x = self.norm_post(x)
        x = x.mean(dim=1)
        return self.head(x)

# 皮肤病类别映射
CLASS_NAMES = {
    0: "光化性角化病 (akiec)",
    1: "基底细胞癌 (bcc)",
    2: "良性角化病 (bkl)",
    3: "皮肤纤维瘤 (df)",
    4: "黑色素瘤 (mel)",
    5: "色素痣 (nv)",
    6: "血管病变 (vasc)"
}

class SkinDiseaseClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("皮肤病图像分类器")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        self.selected_model = tk.StringVar(value="ResNet18")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.uploaded_image = None
        self.image_path = None

        self.create_widgets()
        self.load_selected_model()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="皮肤病图像分类系统", font=("SimHei",14,"bold"))
        title_label.pack(pady=(0,20))

        # 模型选择 —— 已添加 CNN 和 ConvViT
        model_frame = ttk.LabelFrame(main_frame, text="模型选择", padding=10)
        model_frame.pack(fill=tk.X, pady=(0,15))
        ttk.Label(model_frame, text="模型:").pack(side=tk.LEFT, padx=5)
        model_cb = ttk.Combobox(model_frame, textvariable=self.selected_model, 
                               values=["ResNet18","ViT-Tiny","MobileViT","CNN","ConvViT"], state="readonly", width=18)
        model_cb.pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="加载模型", command=self.load_selected_model).pack(side=tk.LEFT)
        ttk.Label(model_frame, text=f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}",foreground="blue").pack(side=tk.RIGHT)

        # 图像预览
        img_frame = ttk.LabelFrame(main_frame, text="图像预览", padding=10)
        img_frame.pack(fill=tk.BOTH, expand=True, pady=(0,15))
        self.image_display = ttk.Label(img_frame, text="请上传图像")
        self.image_display.pack(fill=tk.BOTH, expand=True)

        # 按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0,15))
        ttk.Button(btn_frame, text="上传图像", command=self.upload_image).pack(side=tk.LEFT,padx=5)
        ttk.Button(btn_frame, text="开始分类", command=self.classify_image).pack(side=tk.LEFT)

        # 结果
        res_frame = ttk.LabelFrame(main_frame, text="分类结果", padding=10)
        res_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(res_frame, height=8, font=("SimHei",10))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)

        # 状态栏
        self.status_label = ttk.Label(self.root, text="就绪", anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=10, pady=3)

    def load_selected_model(self):
        name = self.selected_model.get()
        self.status_label.config(text=f"加载 {name}...")
        self.root.update()

        try:
            if name == "ResNet18":
                self.model = ResNet18Baseline()
            elif name == "ViT-Tiny":
                self.model = ViTTinyBaseline()
            elif name == "MobileViT":
                self.model = MobileViTImproved()
            elif name == "CNN":
                self.model = StandardCNN()
            elif name == "ConvViT":
                self.model = ConvViT()

            # ===================== 固定路径 =====================
            path = os.path.join(r"D:\code\experiment_results\models", f"{name}_best.pth")

            if not os.path.exists(path):
                messagebox.showerror("错误", f"找不到模型：\n{path}")
                return

            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device).eval()

            self.status_label.config(text=f"✅ {name} 加载成功")
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"模型：{name}\n可上传图像")
            self.result_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("失败", str(e))

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("图像","*.jpg;*.png;*.jpeg")])
        if not path: return
        self.image_path = path
        img = Image.open(path).convert("RGB")
        img.thumbnail((550,350))
        tkimg = ImageTk.PhotoImage(img)
        self.uploaded_image = tkimg
        self.image_display.config(image=tkimg)
        self.status_label.config(text=f"已加载：{os.path.basename(path)}")

    def classify_image(self):
        if not self.model or not self.image_path:
            messagebox.showwarning("提示","先加载模型和图像")
            return

        self.status_label.config(text="预测中...")
        self.root.update()

        try:
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])

            img = Image.open(self.image_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(x)
                prob = F.softmax(out, dim=1)
                
                # ===== 最可能类别 + Top3 类别 =====
                top3_probs, top3_indices = torch.topk(prob, 3)
                top3_probs = top3_probs.cpu().numpy()[0] * 100
                top3_indices = top3_indices.cpu().numpy()[0]

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"模型：{self.selected_model.get()}\n\n")
            self.result_text.insert(tk.END, f"🔥 最有可能的类别：{CLASS_NAMES[top3_indices[0]]}\n")
            self.result_text.insert(tk.END, f"置信度：{top3_probs[0]:.2f}%\n\n")
            self.result_text.insert(tk.END, "📊 前3名可能结果：\n")
            for i, (idx, p) in enumerate(zip(top3_indices, top3_probs)):
                self.result_text.insert(tk.END, f"{i+1}. {CLASS_NAMES[idx]}：{p:.2f}%\n")
            self.result_text.config(state=tk.DISABLED)
            self.status_label.config(text="✅ 预测完成")

        except Exception as e:
            messagebox.showerror("错误", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SkinDiseaseClassifierUI(root)
    root.mainloop()