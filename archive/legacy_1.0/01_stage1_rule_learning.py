"""
UrbanAlign Stage 1: Rule Learning via TrueSkill Stratification
从Place Pulse 2.0众包数据中通过分层采样提取感知评估规则
"""
import pandas as pd
import numpy as np
import os
import trueskill
import json
import base64
import requests
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. 导入配置
# ==============================================================================
# 导入全局配置
from config import (
    API_KEY, BASE_URL, MODEL_NAME, call_llm_api,
    DATA_DIR, IMAGE_DIR, HUMAN_CHOICES4trueskill_CSV,
    OUTPUT_DIR, CLIP_CACHE, TRUESKILL_CACHE, ID_MAPPING_CSV,
    STAGE1_RULES, STAGE1_PREVIEW,
    CATEGORIES, N_STRATIFIED_SAMPLES, PCA_DIMS
)

HUMAN_CHOICES_CSV = HUMAN_CHOICES4trueskill_CSV
# 为了兼容性保留的别名
CLIP_CACHE_PATH = CLIP_CACHE
RULES_OUTPUT = STAGE1_RULES
PREVIEW_IMAGE = STAGE1_PREVIEW

# ==============================================================================
# 2. ID映射模块(防止token偏见)
# ==============================================================================
def create_id_mapping(df_raw):
    """创建简化的ID映射"""
    if os.path.exists(ID_MAPPING_CSV):
        print(f"[INFO] 加载已有ID映射: {ID_MAPPING_CSV}")
        mapping_df = pd.read_csv(ID_MAPPING_CSV)
        return dict(zip(mapping_df['original_id'].astype(str), mapping_df['alias_id']))

    print("[INFO] 创建新的ID映射...")
    all_ids = pd.concat([df_raw['left_id'], df_raw['right_id']]).unique()
    mapping = {str(orig_id): f"IMG_{i+1:05d}" for i, orig_id in enumerate(all_ids)}

    pd.DataFrame(list(mapping.items()),
                 columns=['original_id', 'alias_id']).to_csv(ID_MAPPING_CSV, index=False)
    print(f"[INFO] ID映射已保存: {ID_MAPPING_CSV}")
    return mapping

# ==============================================================================
# 3. TrueSkill评分计算
# ==============================================================================
def compute_trueskill_ratings(df_raw):
    """使用TrueSkill算法计算每张图片的得分和不确定性"""
    if os.path.exists(TRUESKILL_CACHE):
        print(f"[INFO] 加载已有TrueSkill评分: {TRUESKILL_CACHE}")
        return pd.read_csv(TRUESKILL_CACHE)

    print("[INFO] 计算TrueSkill评分...")
    env = trueskill.TrueSkill(mu=25.0, sigma=8.333, draw_probability=0.10)
    ratings = {cat: {} for cat in CATEGORIES}

    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="TrueSkill Rating"):
        cat = row['category']
        if cat not in CATEGORIES:
            continue

        left_id, right_id = str(row['left_id']), str(row['right_id'])
        winner = row['winner']

        # 初始化评分
        if left_id not in ratings[cat]:
            ratings[cat][left_id] = env.create_rating()
        if right_id not in ratings[cat]:
            ratings[cat][right_id] = env.create_rating()

        # 更新评分
        r1, r2 = ratings[cat][left_id], ratings[cat][right_id]
        if winner == 'left':
            ranks = [0, 1]
        elif winner == 'right':
            ranks = [1, 0]
        else:
            ranks = [0, 0]

        (new_r1,), (new_r2,) = env.rate([(r1,), (r2,)], ranks=ranks)
        ratings[cat][left_id], ratings[cat][right_id] = new_r1, new_r2

    # 转换为DataFrame
    records = []
    for cat in CATEGORIES:
        for img_id, rating in ratings[cat].items():
            records.append({
                'category': cat,
                'image_id': img_id,
                'mu': rating.mu,
                'sigma': rating.sigma
            })

    df_ratings = pd.DataFrame(records)
    df_ratings.to_csv(TRUESKILL_CACHE, index=False)
    print(f"[INFO] TrueSkill评分已保存: {TRUESKILL_CACHE}")
    return df_ratings

# ==============================================================================
# 4. CLIP特征提取和降维
# ==============================================================================
def load_clip_features(id_mapping):
    """加载CLIP特征并进行PCA降维"""
    if not os.path.exists(CLIP_CACHE_PATH):
        print(f"[WARN] CLIP特征文件不存在: {CLIP_CACHE_PATH}")
        return {}

    print(f"[INFO] 加载CLIP特征并降维至{PCA_DIMS}维...")
    data = np.load(CLIP_CACHE_PATH)
    paths, embeddings = data['paths'], data['embeddings']

    # PCA降维
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_DIMS)
    reduced_embs = pca.fit_transform(scaler.fit_transform(embeddings))

    # 构建映射
    clip_map = {}
    for path_item, emb in zip(paths, reduced_embs):
        path_str = str(path_item[0]) if isinstance(path_item, np.ndarray) else str(path_item)
        orig_id = os.path.splitext(os.path.basename(path_str))[0]
        if orig_id in id_mapping:
            clip_map[orig_id] = emb.tolist()

    print(f"[INFO] 成功加载{len(clip_map)}个CLIP特征")
    return clip_map

# ==============================================================================
# 5. 分层采样与可视化
# ==============================================================================
def stratified_sampling(df_ratings, n_samples=N_STRATIFIED_SAMPLES):
    """为每个类别执行三层分层采样:高分/低分/模糊"""
    print(f"[INFO] 执行分层采样,每层{n_samples}个样本...")

    stratified_samples = {}
    for cat in CATEGORIES:
        df_cat = df_ratings[df_ratings['category'] == cat]

        # 高分样本:  μ > 75百分位 且 σ 较小
        mu_75 = df_cat['mu'].quantile(0.75)
        high_samples = df_cat[(df_cat['mu'] >= mu_75)].copy()
        high_samples = high_samples.nsmallest(n_samples * 2, 'sigma').nlargest(n_samples, 'mu')

        # 低分样本: μ < 25百分位 且 σ 较小
        mu_25 = df_cat['mu'].quantile(0.25)
        low_samples = df_cat[(df_cat['mu'] <= mu_25)].copy()
        low_samples = low_samples.nsmallest(n_samples * 2, 'sigma').nsmallest(n_samples, 'mu')

        # 模糊样本: σ > 75百分位
        sigma_75 = df_cat['sigma'].quantile(0.75)
        ambig_samples = df_cat[df_cat['sigma'] >= sigma_75].copy()
        ambig_samples = ambig_samples.nlargest(n_samples, 'sigma')

        stratified_samples[cat] = {
            'high': high_samples,
            'low': low_samples,
            'ambiguous': ambig_samples
        }

        print(f"  {cat}: 高分{len(high_samples)} | 低分{len(low_samples)} | 模糊{len(ambig_samples)}")

    return stratified_samples

def visualize_stratified_samples(stratified_samples):
    """可视化每个维度的分层采样样本并保存"""
    print(f"[INFO] 生成采样预览图: {PREVIEW_IMAGE}")

    num_cats = len(CATEGORIES)
    tiers = ['high', 'low', 'ambiguous']
    num_tiers = len(tiers)

    # 动态计算子图布局: 每一行是一个维度，每一列是不同层级的样本
    # 总列数 = 层级数 * 每层样本数
    fig, axes = plt.subplots(num_cats, num_tiers * N_STRATIFIED_SAMPLES,
                             figsize=(4 * num_tiers, 2 * num_cats))

    # 如果只有一个分类，axes 会是一维数组，需要调整为二维
    if num_cats == 1:
        axes = np.expand_dims(axes, axis=0)

    plt.subplots_adjust(wspace=0.05, hspace=0.4)

    for c_idx, cat in enumerate(CATEGORIES):
        cat_samples = stratified_samples[cat]

        for t_idx, tier in enumerate(tiers):
            tier_df = cat_samples[tier]

            for s_idx in range(N_STRATIFIED_SAMPLES):
                col_idx = t_idx * N_STRATIFIED_SAMPLES + s_idx
                ax = axes[c_idx, col_idx]

                if s_idx < len(tier_df):
                    row = tier_df.iloc[s_idx]
                    img_id = str(row['image_id'])
                    img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")

                    try:
                        img = Image.open(img_path).convert('RGB')
                        ax.imshow(img)
                        # 设置子图标题
                        if c_idx == 0 and s_idx == 0:
                            ax.set_title(tier.upper(), fontsize=14, fontweight='bold', color='#2c3e50')
                        # 在第一列显示维度名称
                        if col_idx == 0:
                            ax.set_ylabel(cat.capitalize(), fontsize=14, fontweight='bold')

                        # 显示μ和σ在图下方
                        ax.set_xlabel(f"μ={row['mu']:.1f}\nσ={row['sigma']:.1f}", fontsize=9)
                    except:
                        ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'Empty', ha='center', va='center', color='gray')

                ax.set_xticks([])
                ax.set_yticks([])
                # 隐藏边框
                for spine in ax.spines.values():
                    spine.set_visible(False)

    plt.savefig(PREVIEW_IMAGE, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  采样预览图已成功保存!")

# ==============================================================================
# 6. 规则提取Prompt构建
# ==============================================================================
def build_rule_extraction_prompt(stratified_samples, id_mapping, clip_map):
    """构建带视觉和CLIP特征的规则提取prompt"""

    content = [{
        "type": "text",
        "text": f"""Analyze Place Pulse 2.0 crowdsourced perception data.

Task: Extract rules predicting perception from CLIP vectors + visual features.

You'll see stratified samples with TrueSkill μ (human rating scores).
Learn CLIP patterns correlating with high/low μ scores.

Dimensions: {', '.join(CATEGORIES)}"""
    }]

    for cat in CATEGORIES:
        samples = stratified_samples[cat]

        content.append({
            "type": "text",
            "text": f"\n=== CATEGORY: {cat.upper()} ===\n"
        })

        for tier in ['high', 'low', 'ambiguous']:
            tier_label = f"{cat.upper()} - {tier.upper()}"
            content.append({"type": "text", "text": f"\n[{tier_label}]"})

            for idx, (_, row) in enumerate(samples[tier].iterrows()):
                orig_id = str(row['image_id'])
                alias_id = id_mapping.get(orig_id, "Unknown")
                clip_vec = clip_map.get(orig_id, [0.0]*PCA_DIMS)

                # 添加文本信息
                info_text = f"""
Image: {alias_id}
TrueSkill: μ={row['mu']:.2f}, σ={row['sigma']:.2f}
CLIP Vector: {[f'{v:.2f}' for v in clip_vec]}
"""
                content.append({"type": "text", "text": info_text})

                # 添加图片
                try:
                    img_path = os.path.join(IMAGE_DIR, f"{orig_id}.jpg")
                    with open(img_path, 'rb') as f:
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
                except:
                    pass

    # 添加任务说明
    task_prompt = f"""
Extract rules for: {', '.join(CATEGORIES)}

Output JSON:
{{
  "{CATEGORIES[0]}": {{
    "high_indicators": [{{"feature": "Modern buildings", "weight": 0.9}}, {{"feature": "CLIP dim3>0.5", "weight": 0.8}}],
    "low_indicators": [{{"feature": "Deteriorated buildings", "weight": 0.9}}, {{"feature": "CLIP dim3<-0.5", "weight": 0.8}}],
    "evaluation_protocol": ["1. Check CLIP dims", "2. Count visual features", "3. Combine scores"],
    "equal_threshold": "CLIP L2 distance < 0.25"
  }}
}}

Rules must use CLIP + visual features (NOT TrueSkill μ/σ values).
"""

    content.append({"type": "text", "text": task_prompt})
    return content

# ==============================================================================
# 7. 主函数
# ==============================================================================
def run_stage1_rule_learning():
    """Stage 1主流程"""
    print("\n" + "="*80)
    print("UrbanAlign Stage 1: Rule Learning")
    print("="*80 + "\n")

    # 1. 加载数据
    print("[STEP 1] 加载数据...")
    df_raw = pd.read_csv(HUMAN_CHOICES_CSV)
    print(f"  加载{len(df_raw)}条人类判断数据")

    # 2. ID映射
    print("\n[STEP 2] 创建ID映射...")
    id_mapping = create_id_mapping(df_raw)

    # 3. TrueSkill评分
    print("\n[STEP 3] 计算TrueSkill评分...")
    df_ratings = compute_trueskill_ratings(df_raw)

    # 4. 加载CLIP特征
    print("\n[STEP 4] 加载CLIP特征...")
    clip_map = load_clip_features(id_mapping)

    # 5. 分层采样
    print("\n[STEP 5] 分层采样与可视化...")
    stratified_samples = stratified_sampling(df_ratings)
    # 调用新增的可视化函数
    visualize_stratified_samples(stratified_samples)

    # 6. 调用LLM提取规则
    print("\n[STEP 6] 调用LLM提取评估规则...")
    prompt_content = build_rule_extraction_prompt(stratified_samples, id_mapping, clip_map)

    # 使用config中的统一API调用函数
    messages = [{"role": "user", "content": prompt_content}]

    print(f"  发送请求到 {MODEL_NAME}...")
    rules_text = call_llm_api(
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        timeout=300
    )

    if rules_text:
        # 保存规则
        with open(RULES_OUTPUT, 'w', encoding='utf-8') as f:
            f.write(rules_text)

        print(f"  规则提取成功!")
        print(f"  输出文件: {RULES_OUTPUT}")
        print("\n规则预览:")
        print(rules_text[:500] + "...")
        return rules_text
    else:
        print(f"  规则提取失败")
        return None

if __name__ == "__main__":
    run_stage1_rule_learning()