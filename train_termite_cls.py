# train_termite_cls.py
import os
import math
import argparse
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import timm

from mobilenet_v1 import mobilenet_v1  # 你的自定义 MobileNetV1
# 从你刚刚创建的 models_builder.py 文件中，跨文件导入 build_densenet_model 函数
from models_builder import build_densenet_model

# ================== 自定义数据增强 ================== #
class RandomJPEGCompression(object):
    """随机 JPEG 压缩，模拟压缩带来的糊/块效应"""
    def __init__(self, quality_range=(40, 80), p=0.3):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        import io
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomGaussianNoise(object):
    """随机加一点高斯噪声"""
    def __init__(self, sigma_range=(2, 8), p=0.3):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        sigma = random.uniform(*self.sigma_range)
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, sigma, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)


class RandomSaltPepperNoise(object):
    """
    随机椒盐噪声增强器（SCI 论文稳健优化版）
    作用: 模拟野外摄像头传感器过热、无线丢包造成的单色离散黑白噪点
    """
    def __init__(self, noise_ratio_range=(0.01, 0.05), p=0.3):
        self.noise_ratio_range = noise_ratio_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        # 1. 动态确定本次 Batch 样本的噪点密度
        noise_ratio = random.uniform(*self.noise_ratio_range)

        # 2. 转化为 NumPy 矩阵
        np_img = np.array(img)
        h, w, c = np_img.shape
        num_noise_pixels = int(noise_ratio * h * w)

        if num_noise_pixels < 2:
            return img

        # 3. 随机抽取不重复的像素空间坐标
        rand_h = np.random.randint(0, h, num_noise_pixels)
        rand_w = np.random.randint(0, w, num_noise_pixels)

        half_noise = num_noise_pixels // 2

        # 通过显式多维索引，确保 RGB/三通道同时被强行污染，彻底封锁彩色混淆
        # 盐噪声：前一半赋予纯白 [255, 255, 255]
        h_salt, w_salt = rand_h[:half_noise], rand_w[:half_noise]
        np_img[h_salt, w_salt, :] = 255

        # 椒噪声：后一半赋予纯黑 [0, 0, 0]
        h_pepper, w_pepper = rand_h[half_noise:], rand_w[half_noise:]
        np_img[h_pepper, w_pepper, :] = 0

        return Image.fromarray(np_img)
# ================== 数据加载 ================== #
def get_dataloaders(
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    data_root 下包含 train / val / test
    无缝集成了高斯噪声与椒盐噪声的混合物理退化管道
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # 强力升级后的训练增强管道：形成“空间-颜色-成像质量”三维立体抗压扰动
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # 几何空间扰动：±15° 旋转
        transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),  # 尺度多变扰动

        # 光照/颜色表型增强
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        # 频域退化：高斯模糊
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.3
        ),

        # 成像质量级复合物理扰动：模拟真实的林业恶劣传感器
        RandomJPEGCompression(quality_range=(40, 80), p=0.3),
        RandomGaussianNoise(sigma_range=(2, 8), p=0.3),

        # 挂载经过 SCI 稳健优化的椒盐离散单色噪点器
        RandomSaltPepperNoise(noise_ratio_range=(0.01, 0.05), p=0.3),

        # 归一化收尾
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # 验证/测试集严格遵循确定性预处理（严禁引入任何随机噪声，保证基准评阅的科学公正）
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # 载入数据集
    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_set = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_names = train_set.classes

    # 构建并行的工业级数据加载器
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names


# ================== 颜色注意力模块（用于 mbv2_color / mbv2_tl2） ================== #
class ColorAttentionBlock(nn.Module):
    """
    轻量颜色注意力模块：
    - 利用通道间的颜色对比（去掉灰度分量）
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        color_feat = x - x.mean(dim=1, keepdim=True)
        y = torch.mean(color_feat, dim=(2, 3), keepdim=True)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


# ================== ECA 模块 & mbv2_eca ================== #
class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                          # [N,C,1,1]
        y = y.squeeze(-1).transpose(-1, -2)           # [N,1,C]
        y = self.conv(y)                              # [N,1,C]
        y = y.transpose(-1, -2).unsqueeze(-1)         # [N,C,1,1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def add_eca_to_mobilenet_v2(model: nn.Module, k_size: int = 3) -> nn.Module:
    """
    自动识别 MobileNetV2 中的 InvertedResidual block，
    在每个 block 后串接一个 ECA 模块
    """
    InvertedResidual = None
    for m in model.features:
        if m.__class__.__name__ == "InvertedResidual":
            InvertedResidual = m.__class__
            break

    if InvertedResidual is None:
        raise RuntimeError("未在 MobileNetV2.features 中找到 InvertedResidual")

    for i, m in enumerate(model.features):
        if isinstance(m, InvertedResidual):
            out_ch = None
            for layer in reversed(list(m.modules())):
                if isinstance(layer, nn.Conv2d):
                    out_ch = layer.out_channels
                    break
            if out_ch is None:
                raise RuntimeError(f"无法从 block {i} 中推断输出通道")

            eca = ECALayer(out_ch, k_size)
            model.features[i] = nn.Sequential(m, eca)

    return model
# ================== MixUp / CutMix 工具函数 ================== #
def mixup_data_intra_class(x, y, alpha=0.4):
    """
    类内 MixUp：只在同一类别内部随机配对
    x: [B, C, H, W]
    y: [B]  (类别索引)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B = x.size(0)
    device = x.device
    index = torch.arange(B, device=device)

    # 对每个类别单独洗牌
    for cls in y.unique():
        cls_mask = (y == cls)
        cls_idx = torch.where(cls_mask)[0]
        if len(cls_idx) <= 1:
            continue
        perm = cls_idx[torch.randperm(len(cls_idx))]
        index[cls_idx] = perm

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(W, H, lam):
    """CutMix 的 bbox 生成"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0):
    """
    标准 CutMix（可跨类混合）
    x: [B, C, H, W]
    y: [B]
    """
    B, C, H, W = x.size()
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    device = x.device
    index = torch.randperm(B).to(device)

    x1, y1, x2, y2 = rand_bbox(W, H, lam)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # 边界裁剪后重新算真实 lam（混合比例）
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    MixUp / CutMix 通用损失计算方式：
    L = lam * CE(pred, y_a) + (1 - lam) * CE(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ================== 各模型构建函数 ================== #
def build_mobilenet_v2(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v2_color(num_classes: int) -> nn.Module:
    """MobileNetV2 + 颜色注意力（插在第一层 block 之后）"""
    try:
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
    except Exception:
        backbone = models.mobilenet_v2(pretrained=True)

    in_features = backbone.classifier[1].in_features
    backbone.classifier[1] = nn.Linear(in_features, num_classes)

    features = list(backbone.features)
    first_block_channels = 16
    cab = ColorAttentionBlock(first_block_channels, reduction=8)

    new_features = []
    new_features.append(features[0])   # stem conv
    new_features.append(features[1])   # 第一个 block
    new_features.append(cab)           # 颜色注意力
    new_features.extend(features[2:])  # 其余层

    backbone.features = nn.Sequential(*new_features)
    return backbone


def build_mobilenet_v2_eca(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    model = add_eca_to_mobilenet_v2(model)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """
    模型工厂函数（已无缝扩展现代化轻量级模型分支）
    mbv2_tl ：结构和 mbv2 一样，只是训练策略不同（TL：分层 LR）
    mbv2_tl2：结构 = mbv2_color，训练策略 = TL
    """
    name = model_name.lower()

    # --- 原有模型分支（保持绝对不改动） ---
    if name == "mbv2":
        return build_mobilenet_v2(num_classes)
    elif name == "mbv2_tl":
        return build_mobilenet_v2(num_classes)
    elif name == "mbv2_color":
        return build_mobilenet_v2_color(num_classes)
    elif name == "mbv2_tl2":
        return build_mobilenet_v2_color(num_classes)
    elif name == "mbv2_eca":
        return build_mobilenet_v2_eca(num_classes)
    elif name == "res18":
        return build_resnet18(num_classes)
    elif name == "effb0":
        return build_efficientnet_b0(num_classes)
    elif name == "mbv1":
        return mobilenet_v1(num_classes)

    # --- 新增现代化轻量级模型分支（完美回应审稿人） ---
    elif name == "mbv3":
        print(">> [timm] 正在加载预训练 MobileNetV3-Large 并无缝对接分类头...")
        # mobilenetv3_large_100 是标准的 V3 大版，timm 会自动根据 num_classes 修改最后一层 Linear
        return timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=num_classes)

    elif name == "mobilevit":
        print(">> [timm] 正在加载预训练 MobileViT-XXS (轻量化 Transformer) 并无缝对接分类头...")
        # mobilevit_xxs 是专门用于移动端算力受限环境的视觉 Transformer 变体
        return timm.create_model('mobilevit_xxs', pretrained=True, num_classes=num_classes)
        # --- 3. 终极新增：标准密集连接网络分支（单独文件模块化调用） ---
    elif name == "dense121":
        # 跨文件呼叫 models_builder.py 里的特定构建器，自动适配你的类别数
        return build_densenet_model(model_name="densenet121", num_classes=num_classes, pretrained=True)

    elif name == "dense161":
        return build_densenet_model(model_name="densenet161", num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"未知模型: {model_name}")

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    use_mixup: bool = False,
    use_cutmix: bool = False,
    mixup_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
) -> Tuple[float, float]:

    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", ncols=100, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ====== 1) 动态随机路由：支持 MixUp 和 CutMix 联合演进 ====== #
        current_mode = "none"
        if use_mixup and use_cutmix:
            # 如果两个开关都开了，每个 Batch 随机各分配一半的概率，让网络经受复合揉捏
            current_mode = "mixup" if random.random() < 0.5 else "cutmix"
        elif use_mixup:
            current_mode = "mixup"
        elif use_cutmix:
            current_mode = "cutmix"

        # ====== 2) 执行对应的底层张量混合 ====== #
        if current_mode == "mixup":
            # 调用你自定义的“类内 MixUp”，语义绝对纯净
            images_mix, targets_a, targets_b, lam = mixup_data_intra_class(
                images, labels, alpha=mixup_alpha
            )
            outputs = model(images_mix)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # 1：因为是类内混合，targets_a 永远等于 targets_b，直接累加绝对精确
            _, preds = outputs.max(1)
            running_correct += preds.eq(targets_a).sum().item()

        elif current_mode == "cutmix":
            # 调用你的标准 CutMix（包含跨类区域挖补）
            images_mix, targets_a, targets_b, lam = cutmix_data(
                images, labels, alpha=cutmix_alpha
            )
            outputs = model(images_mix)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            #  2：应对标准 CutMix 跨类突变，采用加权决策期望计算真实 Acc，消灭统计粉饰
            _, preds = outputs.max(1)
            correct_a = preds.eq(targets_a).sum().item()
            correct_b = preds.eq(targets_b).sum().item()
            # 严格按照物理割补比例 λ 进行加权积分累加
            running_correct += int(lam * correct_a + (1.0 - lam) * correct_b)

        else:
            # ====== 不用混合增强的普通 Baseline 训练 ====== #
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = outputs.max(1)
            running_correct += preds.eq(labels).sum().item()

        # ====== 3) 反向传播与显存安全退出 ====== #
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{running_correct / total:.4f}"
        })

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = outputs.max(1)
            running_loss += loss.item() * labels.size(0)
            running_correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    """
    先 warmup_epochs 线性升高学习率，再做余弦退火到 0
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(
            max(1, epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


# ================== 训练主逻辑 ================== #
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 记录每一轮的 loss / acc（后面画曲线用）
    history_train_loss = []
    history_val_loss = []
    history_train_acc = []
    history_val_acc = []

    # DataLoader
    train_loader, val_loader, _, class_names = get_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    num_classes = len(class_names)
    print("类别:", class_names)

    # 构建模型
    model = get_model(args.model, num_classes=num_classes)
    model.to(device)

    # TL 模型：只做“分层学习率”，不做任何冻结
    model_lower = args.model.lower()
    TL_MODELS = {"mbv2_tl", "mbv2_tl2"}
    is_tl = model_lower in TL_MODELS

    if is_tl:
        print(
            f"\n>> 使用 TL 训练策略（{model_lower}）："
            f"不冻结 backbone，分层学习率训练全网络 "
            f"(head lr={args.lr}, backbone lr={args.tl_backbone_lr})"
        )
    else:
        print(f"\n>> 使用 baseline 配置训练 {model_lower}（统一学习率 lr={args.lr}）")

    # 构建优化器
    if is_tl:
        backbone_params = []
        head_params = list(model.classifier.parameters())
        ca_params = []  # ColorAttention 的参数单独一组

        for name, param in model.named_parameters():
            # 1) 先分出 ColorAttention（features.2 是我们插入 CAB 的位置）
            if "features.2" in name:
                ca_params.append(param)
            # 2) 再把其他 features 都当作 backbone
            elif "features" in name:
                backbone_params.append(param)

        optimizer = optim.AdamW(
            [
                {"params": head_params, "lr": args.lr},  # 分类头，大 LR
                {"params": ca_params, "lr": args.lr},  # ColorAttention，用统一 / 大 LR
                {"params": backbone_params, "lr": args.tl_backbone_lr},  # 其他 backbone，小 LR
            ],
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #label_smoothing=0.1
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    # EarlyStopping
    patience = args.patience
    no_improve = 0
    best_val_acc = 0.0

    model_name = args.model.lower()
    best_model_path = f"best_{model_name}.pth"

    for epoch in range(args.epochs):
        print(f"\n===== Epoch [{epoch + 1}/{args.epochs}] =====")

        # 1) train
        # train_loss, train_acc = train_one_epoch(
        #     model, train_loader, criterion, optimizer, device
        # )
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_mixup=args.use_mixup,
            use_cutmix=args.use_cutmix,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
        )

        # 2) val
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        # 3) scheduler
        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc * 100:.2f}%"
        )

        # 记录本轮的指标
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        # 4) EarlyStopping & 保存 best
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f">>> Val 精度提升，保存新最佳模型到: {best_model_path}")
        else:
            no_improve += 1
            print(f"Val 未提升，no_improve = {no_improve}/{patience}")

        if no_improve >= patience:
            print(
                f"Val 精度连续 {patience} 个 epoch 未提升，"
                f"提前停止训练（EarlyStopping）。"
            )
            break

        # 每个 epoch 结束清一次缓存，稍微省一点显存
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n训练结束！最佳 Val Acc = {best_val_acc * 100:.2f}%")
    print(f"最终最佳模型保存在: {best_model_path}")
    # ====== 保存训练过程曲线（方便后面画图） ====== #
    history = {
        "train_loss": np.array(history_train_loss, dtype=np.float32),
        "val_loss": np.array(history_val_loss, dtype=np.float32),
        "train_acc": np.array(history_train_acc, dtype=np.float32),
        "val_acc": np.array(history_val_acc, dtype=np.float32),
    }
    history_path = f"history_{model_name}.npz"
    np.savez(history_path, **history)
    print(f"训练曲线已保存到: {history_path}")

# ================== main & 参数 ================== #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Termite classification training"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="dataset 根目录（包含 train/val/test）"
    )
    parser.add_argument(
        "--model", type=str, default="mbv2",
        choices=[
            "mbv1",
            "mbv2",         # baseline +（统一学习率）
            "mbv2_tl",      # TL：普通 MobileNetV2 + 分层 LR
            "mbv2_color",   # baseline + ColorAttention
            "mbv2_tl2",     # ColorAttention + TL 训练（你要的增强版）
            "mbv2_eca",
            "res18",
            "effb0",
            "mbv3",
            "mobilevit",
        ],
        help="选择模型"
    )
    parser.add_argument(
        "--epochs", type=int, default=80,
        help="最大训练轮数（配合 EarlyStopping）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="batch size（显存不够就调小）"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="分类头学习率 / 普通模型学习率"
    )
    parser.add_argument(
        "--tl-backbone-lr", type=float, default=5e-5,
        help="TL 模型中 backbone 的学习率（只对 mbv2_tl / mbv2_tl2 生效）"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW 的 weight decay"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5,
        help="学习率 warmup 轮数"
    )
    parser.add_argument(
        "--patience", type=int, default=8,
        help="EarlyStopping 容忍轮数"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader num_workers"
    )
    # ===== MixUp / CutMix 设置 =====
    parser.add_argument(
        "--use-mixup", action="store_true",
        help="在训练阶段启用类内 MixUp 增强"
    )
    parser.add_argument(
        "--use-cutmix", action="store_true",
        help="在训练阶段启用 CutMix 增强（与 MixUp 二选一）"
    )
    parser.add_argument(
        "--mixup-alpha", type=float, default=0.4,
        help="MixUp 的 Beta 分布参数 alpha"
    )
    parser.add_argument(
        "--cutmix-alpha", type=float, default=1.0,
        help="CutMix 的 Beta 分布参数 alpha"
    )

    return parser.parse_args()



if __name__ == "__main__":如果__name__ == "__main__"；
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 保持标准训练注释
    # train_model(args)

    print("\n" + "=" * 70)print("\n"   "=" * 70)
    print("=" * 70)   print("=" * 70)print("=" * 70)   print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")Device = torch.device("cuda"；如果torch.cuda；Is_available () else "cpu")

    # 3. 严格构建与之前完全一致的极限受损测试集管道
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    robust_tf = transforms.Compose([Robust_tf =变换。组成([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        RandomSaltPepperNoise(noise_ratio_range=(0.08, 0.08), p=1.0),RandomSaltPepperNoise(noise_ratio_range=(0.08, 0.08), p=1.0)，
        transforms.ToTensor(),   变换。ToTensor (),
        # 连续区域特征断裂核心阻断
        transforms.RandomErasing(p=1.0, scale=(0.10, 0.20), ratio=(0.5, 2.0), value=0.0),transform . randomerasing (p=1.0, scale=(0.10, 0.20), ratio=(0.5, 2.0), value=0.0)，
        transforms.Normalize(imagenet_mean, imagenet_std),变换。正常化(imagenet_mean imagenet_std),
    ])

    test_dir = os.path.join(args.data, "test")test_dir = os.path.join(args.data, "test")
    test_set = datasets.ImageFolder(test_dir, transform=robust_tf)Test_set =数据集。ImageFolder (test_dir变换= robust_tf)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)test_loader = DataLoader（test_set, batch_size=args）。batch_size, shuffle=False, num_workers=0)
    num_classes = len(test_set.classes)num_classes = len（test_set.classes）

    #
    EVAL_MODELS_MATRIX = {   Eval_models_matrix = {
        "MobileNetV1 (远古组)": ("mbv1", "best_mbv1.pth"),"MobileNetV1 (远古组)": ("mbv1", "best_mbv1.pth"),
        "MobileNetV2 (Vanilla)": ("mbv2", "best_mbv2.pth"),
        "MobileNetV2+ECA (通用通道)": ("mbv2_eca", "best_mbv2_eca.pth"),
        "MobileNetV2+Color (Ours单模块)": ("mbv2_color", "best_mbv2_color.pth"),
        "MobileNetV2+TL (标准迁移)": ("mbv2_tl", "best_mbv2_tl.pth"),
        "MobileNetV3 (Modern Lightweight)": ("mbv3", "best_mbv3.pth"),
        "MobileViT (轻量化Transformer)": ("mobilevit", "best_mobilevit.pth"),
        "ResNet18 (重型参数巨兽)": ("res18", "best_res18.pth"),
        "Ours 完全体 (CAB + TL2)": ("mbv2_tl2", "best_mbv2_tl2.pth"),
    }

    results_report = {}

    print(f">> 原生数据流就绪，包含 {num_classes} 个白蚁类别，共 {len(test_set)} 张极限受损评测图像。\n")

    # 5. 循环自动化推理擂台
    for display_name, (model_tag, weight_file) in EVAL_MODELS_MATRIX.items():
        if os.path.exists(weight_file):
            print(f"[ 进程评估] 正在加载并评估 {display_name} ...")

            # 动态实例化对应的模型拓扑
            model = get_model(model_tag, num_classes=num_classes)

            # 严格原生对齐载入权重
            model.load_state_dict(torch.load(weight_file, map_location=device))
            model.to(device).eval()

            correct = 0
            with torch.no_grad():
                for imgs, lbls in tqdm(test_loader, desc=f"{model_tag} 盲测中", leave=False):
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    _, preds = model(imgs).max(1)
                    correct += preds.eq(lbls).sum().item()

            acc = (correct / len(test_set)) * 100.0
            results_report[display_name] = acc

            # 及时释放显存，防止全量大擂台时 OOM
            del model   争取利比里亚民主运动
            if device.type == "cuda":if device.type == "cuda":
                torch.cuda.empty_cache()
        else:   其他:
            print(f"  未在当前目录下找到 [{weight_file}]，跳过该组评测。")print(f"  未在当前目录下找到 [{weight_file}]，跳过该组评测。")

    # ========================================================
    #  终极多模型学术战报综合大面板输出
    # ========================================================
    print("\n" + "=" * 70)print("\n"   "=" * 70)
    print(" 垂直领域复合型极限物理退化对抗评测全量总决战面板")
    print("=" * 70)   print("=" * 70)
    print(f"{'模型名称 (Architecture Models)':<38} | {'极限抗压准确率 (Robustness Acc)'}")print(f"{'模型名称 (Architecture Models)':<38} | {'极限抗压准确率 (Robustness Acc)'}")
    print("-" * 70)   打印（"-" * 70）

    for name, score in results_report.items():对于名称，results_report.items（）中的分数：
        if "Ours" in name:   if &; our &；
            print(f" {name:<35} | \033[1;31m{score:.2f}%\033[0m")打印(f"{名称:& lt; 35} | 033年\[1;31 033{得分:.2f} % \ [0 m")
        else:   其他:
            print(f" {name:<35} | {score:.2f}%")打印(f"{名称:& lt; 35} |{得分:.2f} %“)

    print("=" * 70)   print("=" * 70)print("=" * 70)   print("=" * 70)
    print(" Table 4 (Robustness Matrix) 中。")print("Table 4 (Robustness Matrix) 中。")
    print("=" * 70)   print("=" * 70)print("=" * 70)   print("=" * 70)
