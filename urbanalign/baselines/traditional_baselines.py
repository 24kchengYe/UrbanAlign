"""
UrbanAlign 2.0 - Stage 7: Traditional Baselines for Comparative Evaluation

三种传统/简化 baseline, 用于论文消融对比:
  C1: End-to-End Siamese Network (端到端深度学习孪生网络, 纯视觉特征)
  C2: Segmentation Feature Regression (基于要素分割的回归模型, 传统城市计算)
  C3: Zero-shot VLM (零样本大模型直接感知, 无维度定义/多智能体)

消融实验矩阵:
  C0: 2016 Legacy (ResNet50 + Siamese) - 传统的物体特征对比
  C1: Modern Visual (CLIP + Siamese) - 现代的通用视觉对比
  C2: Urban Analytics (Segmentation + Reg) - 传统的城市要素统计
  C3: Raw Intelligence (Zero-shot VLM) - 大模型的原生感官

运行方式:
python abc_stage7_traditional_baselines.py
"""
import pandas as pd
import numpy as np
import os
import json
import base64
from io import BytesIO
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import random


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 关键：确保卷积/线性层计算结果确定
    torch.backends.cudnn.benchmark = False


seed_everything(42)  # 固定种子为 42

from urbanalign.config import (
    OUTPUT_DIR, CLIP_CACHE,
    STAGE2_MODE, PCA_DIMS,
    CATEGORIES, IMAGE_DIR,
    call_llm_api, get_split_data,
    get_trueskill_cache, get_stage2_output, get_stage7_output,
)

# 分割特征缓存路径
SEGMENTATION_CACHE = os.path.join(OUTPUT_DIR, "segmentation_features.npz")
# 新增 ResNet 特征缓存路径
RESNET_CACHE = os.path.join(OUTPUT_DIR, "resnet50_features.npz")

# DeepLabV3 PASCAL VOC 21类 → 城市场景分组
# 将21类合并为8个城市相关要素
# URBAN_ELEMENT_GROUPS = {
#     'person':      [15],          # person
#     'vehicle':     [2, 6, 7, 14, 19],  # bicycle, bus, car, motorbike, train
#     'furniture':   [9, 11, 18, 20],    # chair, diningtable, sofa, tvmonitor
#     'plant':       [16],          # pottedplant
#     'animal':      [3, 4, 8, 10, 12, 13, 17],  # bird, boat, cat, cow, dog, horse, sheep
#     'bottle':      [5],           # bottle
#     'aeroplane':   [1],           # aeroplane
#     'background':  [0],           # __background__ (包含建筑、道路、天空等未标注区域)
# }
# N_URBAN_FEATURES = len(URBAN_ELEMENT_GROUPS)

# ==============================================================================
# 2.0 配置增强：Cityscapes 标准要素
# ==============================================================================

# Cityscapes 19类标准定义 (按索引排序)
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]

# 为了回归更稳健，我们将19类归类为7个城市功能组 (Urban Functional Groups)
URBAN_ELEMENT_GROUPS = {
    'Flat':         [0, 1],          # 道路、人行道
    'Construction': [2, 3, 4],       # 建筑、墙、围栏
    'Object':       [5, 6, 7],       # 电线杆、信号灯、指示牌
    'Nature':       [8, 9],          # 植被、地面
    'Sky':          [10],            # 天空
    'Human':        [11, 12],        # 行人、骑行者
    'Vehicle':      [13, 14, 15, 16, 17, 18] # 各类车辆
}
N_URBAN_FEATURES = len(URBAN_ELEMENT_GROUPS)

# ==============================================================================
# ==============================================================================
# 0. 辅助函数: ResNet50 特征提取 (2016 范式)
def _extract_resnet_features():
    """提取 ImageNet 预训练的 ResNet50 特征 (2048维)"""
    import torch
    from torchvision import models, transforms
    from PIL import Image

    if os.path.exists(RESNET_CACHE):
        print(f"  [缓存] 加载 ResNet50 特征: {os.path.basename(RESNET_CACHE)}")
        data = np.load(RESNET_CACHE, allow_pickle=True)
        return dict(zip(data['image_ids'].tolist(), data['features']))

    print(f"  [计算] 提取 ResNet50 特征 (ImageNet Pretrained)...")

    # 采集全量图片 ID (确保在函数内执行)
    all_image_ids = set()
    for cat in CATEGORIES:
        df_ref, df_pool = get_split_data(category=cat)
        for df in [df_ref, df_pool]:
            all_image_ids.update(df['left_id'].astype(str).tolist())
            all_image_ids.update(df['right_id'].astype(str).tolist())

    if not all_image_ids:
        print("[错误] 未能在 DataFrames 中找到任何图片 ID。")
        return {}

    # 初始化模型
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except:
        model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 路径诊断
    sample_id = list(all_image_ids)[0]
    if _find_image(sample_id) is None:
        print(f"\n[路径诊断失败] IMAGE_DIR: {IMAGE_DIR}")
        print(f"无法找到样例图片 ID: {sample_id}。请检查磁盘挂载或 config.py 中的路径。")
        return {}

    res_features = {}
    for img_id in tqdm(all_image_ids, desc="ResNet Extraction"):
        img_path = _find_image(img_id)
        if not img_path: continue
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor).flatten().cpu().numpy()
            res_features[img_id] = feat
        except:
            continue

    if res_features:
        ids = list(res_features.keys())
        feats = np.array([res_features[i] for i in ids])
        np.savez(RESNET_CACHE, image_ids=np.array(ids), features=feats)
        print(f"  [完成] 已成功保存 {len(res_features)} 张图片的 ResNet 特征。")
    return res_features


# ==============================================================================
# 辅助函数：兼容两种列名 (winner / human_winner)
# ==============================================================================
def _get_winner(row):
    """从row中获取winner值，兼容两种列名"""
    return row.get('human_winner') or row.get('winner')


# ==============================================================================
# 1. Baseline C0: Siamese ResNet (2016 Legacy 完整实现)
# ==============================================================================
def run_baseline_c0_resnet_siamese():
    print(f"\n{'=' * 60}\nBaseline C0: 2016 Legacy (ResNet50 Siamese Network)\n{'=' * 60}")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    resnet_map = _extract_resnet_features()
    if not resnet_map:
        print("[ERROR] 无法加载 ResNet 特征")
        return None

    # 定义双塔结构 (Siamese Network)
    class SiameseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = nn.Sequential(
                nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 128), nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 3, 64), nn.ReLU(),
                nn.Linear(64, 3),  # [left, right, equal]
            )

        def forward(self, x_l, x_r):
            z_l, z_r = self.tower(x_l), self.tower(x_r)
            combined = torch.cat([z_l, z_r, torch.abs(z_l - z_r)], dim=1)
            return self.classifier(combined)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []
    label_map = {'left': 0, 'right': 1, 'equal': 2}
    inv_map = {0: 'left', 1: 'right', 2: 'equal'}

    for cat in CATEGORIES:
        df_ref, df_pool = get_split_data(category=cat)

        def prepare_batch(df):
            Ls, Rs, Ys, raw = [], [], [], []
            for _, r in df.iterrows():
                fl, fr = resnet_map.get(str(r['left_id'])), resnet_map.get(str(r['right_id']))
                if fl is not None and fr is not None:
                    Ls.append(fl);
                    Rs.append(fr)
                    Ys.append(label_map.get(str(_get_winner(r)).lower(), 2))
                    raw.append(r)
            return (np.array(Ls), np.array(Rs), np.array(Ys), raw) if Ls else (None, None, None, [])

        ref_L, ref_R, ref_Y, _ = prepare_batch(df_ref)
        pool_L, pool_R, pool_Y, pool_raw = prepare_batch(df_pool)
        if ref_L is None or pool_L is None: continue

        # Mirror augmentation: swap left/right with flipped label (doubles training data)
        flip_map = {0: 1, 1: 0, 2: 2}
        ref_L_orig, ref_R_orig, ref_Y_orig = ref_L.copy(), ref_R.copy(), ref_Y.copy()
        ref_L = np.concatenate([ref_L_orig, ref_R_orig])
        ref_R = np.concatenate([ref_R_orig, ref_L_orig])
        ref_Y = np.concatenate([ref_Y_orig, np.array([flip_map[y] for y in ref_Y_orig])])

        model = SiameseNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(torch.FloatTensor(ref_L), torch.FloatTensor(ref_R), torch.LongTensor(ref_Y)),
                            batch_size=32, shuffle=True)

        model.train()
        for _ in range(50):
            for bl, br, by in loader:
                bl, br, by = bl.to(device), br.to(device), by.to(device)
                optimizer.zero_grad();
                criterion(model(bl, br), by).backward();
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_c0 = model(torch.FloatTensor(pool_L).to(device), torch.FloatTensor(pool_R).to(device))
            probs_c0 = torch.softmax(logits_c0, dim=1).cpu().numpy()  # (n, 3): [left=0, right=1, equal=2]
            preds = logits_c0.argmax(1).cpu().numpy()

        for i, row in enumerate(pool_raw):
            all_results.append({
                'left_id': str(row['left_id']), 'right_id': str(row['right_id']), 'category': cat,
                'human_winner': str(_get_winner(row) or '').lower(), 'synthetic_winner': inv_map[preds[i]],
                'prob_left': float(probs_c0[i, 0]), 'prob_right': float(probs_c0[i, 1]),
                'prob_equal': float(probs_c0[i, 2]),
            })
        print(f"  [{cat}] 准确率: {accuracy_score(pool_Y, preds) * 100:.2f}%")

    df_out = pd.DataFrame(all_results)
    for cat in df_out['category'].unique():
        out_file = get_stage7_output('c0', cat)
        df_out[df_out['category'] == cat].to_csv(out_file, index=False)
        print(f"  输出: {os.path.basename(out_file)}")
    return df_out
# ==============================================================================
# 1. 共享资源加载
# ==============================================================================
def load_clip_features():
    if not os.path.exists(CLIP_CACHE):
        print(f"[ERROR] CLIP缓存缺失: {CLIP_CACHE}")
        return {}
    data = np.load(CLIP_CACHE)
    return {os.path.splitext(os.path.basename(str(p)))[0]: v
            for p, v in zip(data['paths'], data['embeddings'])}


def image_to_base64(img_path):
    """图片转base64 (带缩放优化)"""
    from PIL import Image
    with Image.open(img_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail((512, 512))
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def calculate_metrics(df, exclude_equal=False):
    temp = df.copy()
    if exclude_equal:
        # 仅排除人类标注为equal的对, 模型预测equal在非equal对上算预测错误
        temp = temp[temp['human_winner'] != 'equal']
    if len(temp) == 0:
        return 0, 0
    acc = accuracy_score(temp['human_winner'], temp['synthetic_winner'])
    try:
        kappa = cohen_kappa_score(temp['human_winner'], temp['synthetic_winner'])
    except:
        kappa = 0
    return acc, kappa


def _find_image(image_id):
    """查找图片文件 (支持多种后缀)"""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        path = os.path.join(IMAGE_DIR, f"{image_id}{ext}")
        if os.path.exists(path):
            return path
    for f in os.listdir(IMAGE_DIR) if os.path.isdir(IMAGE_DIR) else []:
        if f.startswith(str(image_id) + '.'):
            return os.path.join(IMAGE_DIR, f)
    return None


# ==============================================================================
# 2. Baseline C1: End-to-End Siamese Network (端到端深度学习孪生网络)
# ==============================================================================
def run_baseline_c1_siamese_network():
    """
    端到端深度学习孪生网络:
    - 共享权重的双塔结构 (Shared-weight Twin Tower)
    - 每张图通过相同的 CLIP 视觉编码器 (frozen) 提取特征
    - 共享的可训练投影层将两塔特征映射到比较空间
    - 拼接 [proj(left), proj(right), |proj(left)-proj(right)|] 后分类
    - 训练集: Ref, 测试集: Pool
    """
    print(f"\n{'='*60}")
    print("Baseline C1: End-to-End Siamese Network")
    print(f"{'='*60}")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    clip_map = load_clip_features()
    if not clip_map:
        print("[ERROR] 无法加载CLIP特征")
        return None

    clip_dim = len(next(iter(clip_map.values())))
    proj_dim = 128

    # 定义 Siamese 网络
    class SiameseNet(nn.Module):
        def __init__(self, input_dim, proj_dim):
            super().__init__()
            # 共享投影塔 (双塔共享权重)
            self.tower = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, proj_dim),
                nn.ReLU(),
            )
            # 比较与分类头: [proj_left, proj_right, |diff|] → 3类
            self.classifier = nn.Sequential(
                nn.Linear(proj_dim * 3, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 3),
            )

        def forward(self, x_left, x_right):
            # 双塔共享权重
            z_left = self.tower(x_left)
            z_right = self.tower(x_right)
            # 拼接: [左投影, 右投影, 差的绝对值]
            combined = torch.cat([z_left, z_right, torch.abs(z_left - z_right)], dim=1)
            return self.classifier(combined)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []
    label_map = {'left': 0, 'right': 1, 'equal': 2}
    inv_label_map = {0: 'left', 1: 'right', 2: 'equal'}

    for cat in CATEGORIES:
        df_ref, df_pool = get_split_data(category=cat)
        print(f"\n  [{cat}] Ref={len(df_ref)}对, Pool={len(df_pool)}对")

        # 构建特征
        def build_pairs(df):
            lefts, rights, labels, rows = [], [], [], []
            for _, r in df.iterrows():
                l_id, r_id = str(r['left_id']), str(r['right_id'])
                f_l, f_r = clip_map.get(l_id), clip_map.get(r_id)
                if f_l is None or f_r is None:
                    continue
                lefts.append(f_l)
                rights.append(f_r)
                labels.append(label_map.get(str(_get_winner(r)).lower(), 2))
                rows.append(r)
            if not lefts:
                return None, None, None, []
            return np.array(lefts), np.array(rights), np.array(labels), rows

        ref_L, ref_R, ref_y, _ = build_pairs(df_ref)
        pool_L, pool_R, pool_y, pool_rows = build_pairs(df_pool)

        if ref_L is None or pool_L is None or len(ref_L) < 10:
            print(f"    跳过: 数据不足")
            continue

        # Mirror augmentation: swap left/right with flipped label (doubles training data)
        flip_map = {0: 1, 1: 0, 2: 2}
        ref_L_orig, ref_R_orig, ref_y_orig = ref_L.copy(), ref_R.copy(), ref_y.copy()
        ref_L = np.concatenate([ref_L_orig, ref_R_orig])
        ref_R = np.concatenate([ref_R_orig, ref_L_orig])
        ref_y = np.concatenate([ref_y_orig, np.array([flip_map[y] for y in ref_y_orig])])
        print(f"    Mirror augmentation: {len(ref_y_orig)} → {len(ref_y)} 训练对")

        # 训练 Siamese 网络
        model = SiameseNet(clip_dim, proj_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.FloatTensor(ref_L), torch.FloatTensor(ref_R), torch.LongTensor(ref_y)
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        model.train()
        for epoch in range(50):
            for batch_L, batch_R, batch_y in train_loader:
                batch_L, batch_R, batch_y = batch_L.to(device), batch_R.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_L, batch_R)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        # 预测
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.FloatTensor(pool_L).to(device),
                torch.FloatTensor(pool_R).to(device)
            )
            probs_c1 = torch.softmax(logits, dim=1).cpu().numpy()  # (n, 3): [left=0, right=1, equal=2]
            preds = logits.argmax(dim=1).cpu().numpy()

        for i, row in enumerate(pool_rows):
            all_results.append({
                'left_id': str(row['left_id']),
                'right_id': str(row['right_id']),
                'category': cat,
                'human_winner': str(_get_winner(row) or '').lower(),
                'synthetic_winner': inv_label_map[preds[i]],
                'prob_left': float(probs_c1[i, 0]), 'prob_right': float(probs_c1[i, 1]),
                'prob_equal': float(probs_c1[i, 2]),
            })

        y_true = [str(_get_winner(r)).lower() for r in pool_rows]
        y_pred = [inv_label_map[p] for p in preds]
        acc = accuracy_score(y_true, y_pred)
        print(f"    准确率: {acc*100:.2f}% (n={len(y_true)})")

    if all_results:
        df_out = pd.DataFrame(all_results)
        for cat in df_out['category'].unique():
            out_file = get_stage7_output('c1', cat)
            df_out[df_out['category'] == cat].to_csv(out_file, index=False)
            print(f"  输出: {os.path.basename(out_file)}")
        return df_out
    return None


# ==============================================================================
# 3. Baseline C2: Segmentation Feature Regression (基于要素分割的回归模型)
# ==============================================================================
def _extract_segmentation_features():
    """
    使用 SegFormer (B0) 在 Cityscapes 上预训练的模型提取 19 类要素比例。
    SegFormer 是目前街景分析的行业标准，比 DeepLabV3 更精准。
    """
    if os.path.exists(SEGMENTATION_CACHE):
        print(f"  [缓存] 加载 Cityscapes 分割特征: {os.path.basename(SEGMENTATION_CACHE)}")
        data = np.load(SEGMENTATION_CACHE, allow_pickle=True)
        return dict(zip(data['image_ids'].tolist(), data['features']))

    print(f"  [计算] 提取 Cityscapes 要素特征 (SegFormer-B0)...")

    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from PIL import Image

    # 加载预训练模型 (自动下载 Cityscapes 权重)
    # 修改模型 ID
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 收集图片ID
    all_image_ids = set()
    for cat in CATEGORIES:
        df_ref, df_pool = get_split_data(category=cat)
        for df in [df_ref, df_pool]:
            all_image_ids.update(df['left_id'].astype(str).tolist())
            all_image_ids.update(df['right_id'].astype(str).tolist())

    seg_features = {}
    group_names = sorted(URBAN_ELEMENT_GROUPS.keys())

    for img_id in tqdm(all_image_ids, desc="Cityscapes Segmentation"):
        img_path = _find_image(img_id)
        if img_path is None: continue
        try:
            img = Image.open(img_path).convert('RGB')
            # 预处理
            inputs = processor(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # shape (1, 19, H/4, W/4)
                # 插值回原始尺寸并取最大概率类别
                upsampled_logits = torch.nn.functional.interpolate(
                    logits, size=img.size[::-1], mode='bilinear', align_corners=False
                )
                pred = upsampled_logits.argmax(1)[0].cpu().numpy()

            # 计算各功能组的像素比例
            total_pixels = pred.size
            proportions = []
            for g_name in group_names:
                g_indices = URBAN_ELEMENT_GROUPS[g_name]
                mask = np.isin(pred, g_indices)
                proportions.append(mask.sum() / total_pixels)

            seg_features[img_id] = np.array(proportions, dtype=np.float32)

        except Exception as e:
            print(f"图片 {img_id} 处理失败: {e}")
            continue

    # 保存缓存
    if seg_features:
        ids = list(seg_features.keys())
        feats = np.array([seg_features[i] for i in ids])
        np.savez(SEGMENTATION_CACHE, image_ids=np.array(ids), features=feats)
        print(f"  [要素组] {', '.join(group_names)}")

    return seg_features


def run_baseline_c2_segmentation_regression():
    """
    基于要素分割+CLIP的混合回归模型:
    - SegFormer 语义分割提取 Cityscapes 19类→7组城市要素比例
    - CLIP ViT-L/14 视觉特征 (PCA降维)
    - 特征差向量: [seg_diff, clip_pca_diff] → GradientBoosting 分类
    - 代表传统城市计算方法的增强版: 域特征+通用视觉特征
    """
    print(f"\n{'='*60}")
    print("Baseline C2: Segmentation + CLIP Feature Regression")
    print(f"{'='*60}")

    seg_map = _extract_segmentation_features()
    if not seg_map:
        print("[ERROR] 无法提取分割特征")
        return None

    clip_map = load_clip_features()
    if not clip_map:
        print("[WARNING] CLIP特征不可用, 仅使用分割特征")

    # PCA降维CLIP特征
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    clip_pca = None
    if clip_map:
        all_clip_ids = list(clip_map.keys())
        all_clip_feats = np.array([clip_map[k] for k in all_clip_ids])
        clip_pca = PCA(n_components=PCA_DIMS, random_state=42)
        clip_pca.fit(all_clip_feats)
        print(f"  CLIP PCA: {all_clip_feats.shape[1]}d → {PCA_DIMS}d (explained var: {clip_pca.explained_variance_ratio_.sum():.1%})")

    all_results = []

    for cat in CATEGORIES:
        df_ref, df_pool = get_split_data(category=cat)
        print(f"\n  [{cat}] Ref={len(df_ref)}对, Pool={len(df_pool)}对")

        def build_features(df):
            X, y, rows = [], [], []
            for _, r in df.iterrows():
                l_id, r_id = str(r['left_id']), str(r['right_id'])
                s_l, s_r = seg_map.get(l_id), seg_map.get(r_id)
                if s_l is None or s_r is None:
                    continue
                # Segmentation difference
                seg_diff = s_l - s_r
                # CLIP PCA difference (if available)
                if clip_pca is not None and l_id in clip_map and r_id in clip_map:
                    c_l = clip_pca.transform(clip_map[l_id].reshape(1, -1))[0]
                    c_r = clip_pca.transform(clip_map[r_id].reshape(1, -1))[0]
                    feat = np.concatenate([seg_diff, c_l - c_r])
                else:
                    feat = seg_diff
                X.append(feat)
                y.append(str(_get_winner(r)).lower())
                rows.append(r)
            return np.array(X) if X else None, y, rows

        X_ref, y_ref, _ = build_features(df_ref)
        X_pool, y_pool, pool_rows = build_features(df_pool)

        if X_ref is None or X_pool is None or len(X_ref) < 10:
            print(f"    跳过: 特征构建不足")
            continue

        # Mirror augmentation for training
        X_ref_mirror = -X_ref  # negating difference = swapping left/right
        y_ref_mirror = ['right' if y == 'left' else 'left' if y == 'right' else y for y in y_ref]
        X_ref_aug = np.concatenate([X_ref, X_ref_mirror])
        y_ref_aug = y_ref + y_ref_mirror

        scaler = StandardScaler()
        X_ref_s = scaler.fit_transform(X_ref_aug)
        X_pool_s = scaler.transform(X_pool)

        le = LabelEncoder()
        le.fit(['left', 'right', 'equal'])
        y_ref_enc = le.transform(y_ref_aug)

        n_feat = X_ref.shape[1]
        print(f"    特征维度: {n_feat} (seg={N_URBAN_FEATURES}" +
              (f"+clip_pca={PCA_DIMS})" if clip_pca else ")") +
              f", 训练样本: {len(y_ref_aug)} (含镜像)")

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(X_ref_s, y_ref_enc)

        y_proba = model.predict_proba(X_pool_s)  # (n, n_classes) — classes in le.classes_ order
        y_pred_enc = y_proba.argmax(axis=1)
        y_pred = le.inverse_transform(y_pred_enc)

        # le.classes_ is sorted alphabetically: ['equal', 'left', 'right']
        cls_list = list(le.classes_)
        idx_left = cls_list.index('left')
        idx_right = cls_list.index('right')
        idx_equal = cls_list.index('equal')

        for i, row in enumerate(pool_rows):
            all_results.append({
                'left_id': str(row['left_id']),
                'right_id': str(row['right_id']),
                'category': cat,
                'human_winner': str(_get_winner(row) or '').lower(),
                'synthetic_winner': y_pred[i],
                'prob_left': float(y_proba[i, idx_left]),
                'prob_right': float(y_proba[i, idx_right]),
                'prob_equal': float(y_proba[i, idx_equal]),
            })

        acc = accuracy_score(y_pool, y_pred)
        print(f"    准确率: {acc*100:.2f}% (n={len(y_pool)})")

    if all_results:
        df_out = pd.DataFrame(all_results)
        for cat in df_out['category'].unique():
            out_file = get_stage7_output('c2', cat)
            df_out[df_out['category'] == cat].to_csv(out_file, index=False)
            print(f"  输出: {os.path.basename(out_file)}")
        return df_out
    return None


# ==============================================================================
# 4. Baseline C3: Zero-shot VLM (零样本大模型直接感知)
# ==============================================================================
def run_baseline_c3_zeroshot_vlm():
    """
    零样本VLM baseline: 简单prompt直接问 "Which image looks more {category}?"
    无维度定义、无observer/debater, 仅直接比较。
    注意: 有API成本。
    """
    print(f"\n{'='*60}")
    print("Baseline C3: Zero-shot VLM (Direct Comparison)")
    print(f"{'='*60}")
    print("  注意: 此baseline需要API调用, 有成本!")

    all_results = []

    for cat in CATEGORIES:
        _, df_pool = get_split_data(category=cat)
        print(f"\n  [{cat}] Pool={len(df_pool)}对")

        success, fail = 0, 0
        for _, row in tqdm(df_pool.iterrows(), total=len(df_pool),
                           desc=f"C3 ZeroShot {cat}"):
            l_id, r_id = str(row['left_id']), str(row['right_id'])

            img_a_path = _find_image(l_id)
            img_b_path = _find_image(r_id)

            if img_a_path is None or img_b_path is None:
                fail += 1
                continue

            try:
                b64_a = image_to_base64(img_a_path)
                b64_b = image_to_base64(img_b_path)
            except Exception:
                fail += 1
                continue

            prompt = (
                f"Look at Image A (first) and Image B (second). "
                f"Which image looks more {cat}? "
                f"You MUST choose one. Answer with exactly one word: "
                f"'left' if Image A looks more {cat}, or 'right' if Image B looks more {cat}. "
                f"Do not say 'equal' or 'same' — pick the one that is even slightly more {cat}."
            )

            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}},
                ]}
            ]

            response = call_llm_api(messages, temperature=0, max_tokens=50)
            winner = _parse_winner(response)

            all_results.append({
                'left_id': l_id,
                'right_id': r_id,
                'category': cat,
                'human_winner': str(_get_winner(row) or '').lower(),
                'synthetic_winner': winner
            })
            success += 1

        print(f"    完成: {success}对, 失败: {fail}对")

        if all_results:
            df_temp = pd.DataFrame([r for r in all_results if r['category'] == cat])
            if len(df_temp) > 0:
                acc, kappa = calculate_metrics(df_temp)
                print(f"    准确率: {acc*100:.2f}%, Kappa: {kappa:.3f}")

    if all_results:
        df_out = pd.DataFrame(all_results)
        for cat in df_out['category'].unique():
            out_file = get_stage7_output('c3', cat)
            df_out[df_out['category'] == cat].to_csv(out_file, index=False)
            print(f"  输出: {os.path.basename(out_file)}")
        return df_out
    return None


def _parse_winner(response):
    """从VLM响应中解析winner"""
    if response is None:
        return 'equal'
    resp = response.strip().lower()
    if 'left' in resp and 'right' not in resp:
        return 'left'
    elif 'right' in resp and 'left' not in resp:
        return 'right'
    elif 'equal' in resp or 'same' in resp or 'tie' in resp:
        return 'equal'
    for word in resp.split():
        if word in ('left', 'a', 'first'):
            return 'left'
        if word in ('right', 'b', 'second'):
            return 'right'
    return 'equal'


# ==============================================================================
# 5. 主函数
# ==============================================================================
def run_all_baselines():
    print("\n" + "=" * 80)
    print("UrbanAlign 2.0 - Stage 7: Traditional Baselines")
    print("=" * 80)
    print("  C0: End-to-End Siamese Network (端到端深度学习孪生网络 ResNet50-ImageNet)")
    print("  C1: End-to-End Siamese Network (端到端深度学习孪生网络 CLIP)")
    print("  C2: Segmentation Feature Regression (基于要素分割的回归模型)")
    print("  C3: Zero-shot VLM (零样本大模型直接感知)")

    results_summary = []

    # C0: 2016 Legacy
    df_c0 = run_baseline_c0_resnet_siamese()
    if df_c0 is not None:
        acc, kappa = calculate_metrics(df_c0)
        results_summary.append(('C0-ResNet-2016', acc, kappa, len(df_c0)))

    # C1: Siamese Network
    df_c1 = run_baseline_c1_siamese_network()
    if df_c1 is not None:
        acc, kappa = calculate_metrics(df_c1)
        results_summary.append(('C1-Siamese', acc, kappa, len(df_c1)))

    # C2: Segmentation Regression
    df_c2 = run_baseline_c2_segmentation_regression()
    if df_c2 is not None:
        acc, kappa = calculate_metrics(df_c2)
        results_summary.append(('C2-SegReg', acc, kappa, len(df_c2)))

    # C3: Zero-shot VLM
    df_c3 = run_baseline_c3_zeroshot_vlm()
    if df_c3 is not None:
        acc, kappa = calculate_metrics(df_c3)
        results_summary.append(('C3-ZeroShot', acc, kappa, len(df_c3)))

    # 汇总
    print(f"\n{'='*60}")
    print("Baseline Results Summary")
    print(f"{'='*60}")
    print(f"  {'Baseline':<20} {'Accuracy':>10} {'Kappa':>8} {'N':>6}")
    print(f"  {'-'*44}")
    for name, acc, kappa, n in results_summary:
        print(f"  {name:<20} {acc*100:>9.2f}% {kappa:>7.3f} {n:>6}")

    print(f"\n  输出文件:")
    for key in ['c0', 'c1', 'c2', 'c3']:
        for cat in CATEGORIES:
            path = get_stage7_output(key, cat)
            exists = os.path.exists(path)
            status = "OK" if exists else "SKIP"
            print(f"    [{status}] {os.path.basename(path)}")

    print(f"\n{'='*60}")
    print("Baselines完成! 运行 Stage 4 可自动纳入对比。")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_baselines()
