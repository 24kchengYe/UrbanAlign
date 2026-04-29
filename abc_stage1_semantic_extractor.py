"""
UrbanAlign 2.0 - Stage 1: Semantic Dimension Extractor
语义维度定义：利用大模型从少量共识样本中提炼通用评价维度

核心思想：
- 不再让LLM学习"规则"，而是定义"评价维度"
- 输入：感知类别（如wealthy）+ 少量高/低共识样本
- 输出：5-8个通用视觉评价维度（如Facade Quality, Vegetation Maintenance等）

理论依据：
- 大模型是"语义翻译器"，不是"分类器"
- 通过维度解耦，提升可解释性和迁移性
"""
import pandas as pd
import numpy as np
import os
import json
import base64
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# 1. 导入配置
# ==============================================================================
from config import (
    API_KEY, BASE_URL, MODEL_NAME, call_llm_api, get_split_data,
    DATA_DIR, IMAGE_DIR,
    OUTPUT_DIR, CLIP_CACHE, get_id_mapping_csv,
    CATEGORIES, N_CONSENSUS_SAMPLES, PCA_DIMS,
    N_DIMENSIONS_MIN, N_DIMENSIONS_MAX, DIMENSION_EXAMPLES_FROM_AI,
    get_trueskill_cache, get_stage1_dimensions, get_stage1_dimension_log,
    get_human_choices_csv
)

# ==============================================================================
# 2. 维度示例配置
# ==============================================================================
_HARDCODED_DIMENSION_EXAMPLES = {
    'wealthy': ['Facade Quality', 'Vegetation Maintenance', 'Pavement Integrity',
               'Vehicle Quality', 'Building Modernity', 'Infrastructure Condition'],
    'safety': ['Lighting Adequacy', 'Visibility Clarity', 'Pedestrian Infrastructure',
              'Building Maintenance', 'Street Activity', 'Informal Surveillance'],
    'beautiful': ['Architectural Harmony', 'Natural Elements', 'Color Coordination',
                 'Landscape Design', 'Historic Preservation', 'Visual Complexity'],
    'lively': ['Human Activity', 'Commercial Density', 'Street Furniture',
              'Public Amenities', 'Temporal Dynamism', 'Social Interactions'],
    'boring': ['Visual Monotony', 'Functional Diversity', 'Architectural Variety',
              'Landscape Diversity', 'Activity Sparsity', 'Color Palette'],
    'depressing': ['Decay Indicators', 'Neglect Signs', 'Environmental Quality',
                  'Social Vitality', 'Aesthetic Degradation', 'Atmospheric Mood']
}


def _log_dimension_examples(category, dimensions, source="ai"):
    """将生成的维度示例追加保存到累积日志CSV"""
    from datetime import datetime
    log_file = get_stage1_dimension_log(category)
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'category': category,
        'source': source,
        'n_dimensions': len(dimensions),
        'dimensions': json.dumps(dimensions, ensure_ascii=False),
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(log_file):
        df_row.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_row.to_csv(log_file, index=False)


def generate_dimension_examples_from_ai(category):
    """
    用LLM生成维度示例名称列表（纯文本调用，无图片，成本极低）。
    返回维度名称列表，失败时返回None（调用方会回退到硬编码）。
    每次生成结果会追加保存到 DIMENSION_EXAMPLES_LOG CSV。
    """
    # 取硬编码示例作为格式参考
    hardcoded = _HARDCODED_DIMENSION_EXAMPLES.get(category, [])
    format_ref = ', '.join(f'"{d}"' for d in hardcoded[:3]) if hardcoded else '"Facade Quality", "Vegetation Maintenance", "Pavement Integrity"'

    prompt = (
        f'You are an urban perception researcher. '
        f'List {N_DIMENSIONS_MIN} visual dimensions for evaluating "{category}" '
        f'perception in street-view images.\n\n'
        f'Requirements:\n'
        f'- Each dimension name must be a SHORT PHRASE (2-3 words, e.g. "Facade Quality")\n'
        f'- Must be visually observable and scorable on a 1-10 scale\n'
        f'- Each dimension should be independent (low mutual correlation)\n\n'
        f'Format reference: {format_ref}\n\n'
        f'Return JSON exactly as:\n'
        f'{{"dimensions": ["Short Phrase 1", "Short Phrase 2", ...]}}'
    )
    resp = call_llm_api(
        [{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=256,
        response_format={"type": "json_object"}
    )
    if resp:
        try:
            data = json.loads(resp)
            dims = data.get('dimensions', [])
            if isinstance(dims, list) and len(dims) >= 3:
                _log_dimension_examples(category, dims, source="ai")
                return dims
        except Exception:
            pass
    return None


# ==============================================================================
# 2b. ID映射模块(防止token偏见)
# ==============================================================================
def create_id_mapping(df_raw, category):
    """创建简化的ID映射（按类别保存）"""
    mapping_path = get_id_mapping_csv(category)
    if os.path.exists(mapping_path):
        print(f"[INFO] 加载已有ID映射: {mapping_path}")
        mapping_df = pd.read_csv(mapping_path)
        return dict(zip(mapping_df['original_id'].astype(str), mapping_df['alias_id']))

    print(f"[INFO] 创建新的ID映射 ({category})...")
    all_ids = pd.concat([df_raw['left_id'], df_raw['right_id']]).unique()
    mapping = {str(orig_id): f"IMG_{i+1:05d}" for i, orig_id in enumerate(all_ids)}

    pd.DataFrame(list(mapping.items()),
                 columns=['original_id', 'alias_id']).to_csv(mapping_path, index=False)
    print(f"[INFO] ID映射已保存: {mapping_path}")
    return mapping

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
def load_trueskill_ratings(category):
    """加载指定类别的TrueSkill评分（用于筛选高/低共识样本）"""
    ts_file = get_trueskill_cache(category)
    if not os.path.exists(ts_file):
        print(f"[ERROR] TrueSkill评分文件不存在: {ts_file}")
        print("请先运行 abc_preprocess3 生成TrueSkill评分")
        return None
    return pd.read_csv(ts_file)

def sample_consensus_images(df_ratings, category, n_samples=5):
    """
    为每个类别采样高/低共识样本

    高共识 = 高μ + 低σ（大家都认为很好）
    低共识 = 低μ + 低σ（大家都认为很差）
    """
    df_cat = df_ratings[df_ratings['category'] == category]

    # 高共识样本: μ > 75百分位 且 σ < 中位数
    mu_75 = df_cat['mu'].quantile(0.6)
    sigma_median = df_cat['sigma'].median()
    high_consensus = df_cat[(df_cat['mu'] > mu_75) & (df_cat['sigma'] < sigma_median)]
    high_consensus = high_consensus.nlargest(n_samples, 'mu')

    # 低共识样本: μ < 25百分位 且 σ < 中位数
    mu_25 = df_cat['mu'].quantile(0.3)
    low_consensus = df_cat[(df_cat['mu'] < mu_25) & (df_cat['sigma'] < sigma_median)]
    low_consensus = low_consensus.nsmallest(n_samples, 'mu')

    return high_consensus, low_consensus

def image_to_base64(img_path):
    """图片转base64"""
    with Image.open(img_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        from io import BytesIO
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ==============================================================================
# 3. 核心Prompt构建
# ==============================================================================
def build_dimension_extraction_prompt(category, high_samples, low_samples, id_mapping):
    """
    构建语义维度提取prompt

    目标：让LLM定义5-8个通用的视觉评价维度
    """

    # 维度示例：AI生成 或 硬编码回退
    fallback = ['Dimension 1', 'Dimension 2', 'Dimension 3']
    if DIMENSION_EXAMPLES_FROM_AI:
        ai_examples = generate_dimension_examples_from_ai(category)  # 成功时内部已记录日志
        if ai_examples:
            dimension_examples_list = ai_examples
        else:
            dimension_examples_list = _HARDCODED_DIMENSION_EXAMPLES.get(category, fallback)
            _log_dimension_examples(category, dimension_examples_list, source="hardcoded_fallback")
    else:
        dimension_examples_list = _HARDCODED_DIMENSION_EXAMPLES.get(category, fallback)
        _log_dimension_examples(category, dimension_examples_list, source="hardcoded")

    # 每次调用随机选取目标维度数, 在多trial下产生多样性
    import random
    target_n = random.randint(N_DIMENSIONS_MIN, N_DIMENSIONS_MAX)

    content = [{
        "type": "text",
        "text": f"""You are an urban design expert defining evaluation dimensions for perception assessment.

Task: Define exactly {target_n} universal visual dimensions for evaluating "{category}" perception in street environments.

**Context**:
You'll see examples of street scenes rated HIGHLY and POORLY on "{category}" by crowdsourced volunteers.
Your job is to identify what underlying visual dimensions distinguish these two groups.

**Dimension Requirements**:
1. Observable: Must be visible in street-view images
2. Measurable: Can be scored 1-10 by visual inspection
3. Universal: Applicable across different cities/cultures
4. Independent: Each dimension captures distinct aspects
5. Interpretable: Clear semantic meaning

**Reference Dimensions (for format reference only)**:
{', '.join(dimension_examples_list)}

**Now observe the consensus samples**:
"""
    }]

    # 添加高共识样本
    content.append({"type": "text", "text": f"\n=== HIGH {category.upper()} CONSENSUS (高评分共识样本) ==="})
    for idx, (_, row) in enumerate(high_samples.iterrows(), 1):
        img_id = str(row['image_id'])
        alias = id_mapping.get(img_id, "Unknown")
        content.append({
            "type": "text",
            "text": f"\nHigh Sample {idx}: {alias} (μ={row['mu']:.2f}, σ={row['sigma']:.2f})"
        })
        try:
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
            b64 = image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        except:
            pass

    # 添加低共识样本
    content.append({"type": "text", "text": f"\n=== LOW {category.upper()} CONSENSUS (低评分共识样本) ==="})
    for idx, (_, row) in enumerate(low_samples.iterrows(), 1):
        img_id = str(row['image_id'])
        alias = id_mapping.get(img_id, "Unknown")
        content.append({
            "type": "text",
            "text": f"\nLow Sample {idx}: {alias} (μ={row['mu']:.2f}, σ={row['sigma']:.2f})"
        })
        try:
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
            b64 = image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        except:
            pass

    # 添加输出要求
    task_prompt = f"""
Based on the visual differences between HIGH and LOW consensus samples, define evaluation dimensions.

Output Format (JSON):
{{
  "category": "{category}",
  "dimensions": [
    {{
      "name": "Dimension Name (e.g., Facade Quality)",
      "description": "What this dimension measures (1-2 sentences)",
      "high_indicators": ["Indicator 1 (e.g., Well-maintained paint)", "Indicator 2"],
      "low_indicators": ["Indicator 1 (e.g., Peeling walls)", "Indicator 2"],
    }}
  ],
  "rationale": "Why these dimensions capture '{category}' perception (2-3 sentences)"
}}

CRITICAL:
- You MUST define exactly {target_n} dimensions — no more, no fewer.
- Each dimension must be visually measurable (1-10 scale)
- Dimensions should be independent (low correlation)
- Focus on visual features, not CLIP embeddings
"""

    content.append({"type": "text", "text": task_prompt})
    return content

# ==============================================================================
# 4. 主函数
# ==============================================================================
def run_dimension_extraction():
    """Stage 1主流程：为每个类别提取语义维度"""
    print("\n" + "="*80)
    print("UrbanAlign 2.0 - Stage 1: Semantic Dimension Extraction")
    print("="*80 + "\n")

    # 1. 数据集物理隔离
    print("[STEP 1] 数据集物理隔离...")
    print("  使用Reference集采样（防止数据泄露）")

    # 2. 为每个类别提取维度
    for category in CATEGORIES:
        output_file = get_stage1_dimensions(category)

        # 跳过已完成的类别
        if os.path.exists(output_file):
            print(f"[SKIP] {category}: {os.path.basename(output_file)} already exists")
            continue

        print(f"\n{'='*80}")
        print(f"处理类别: {category.upper()}")
        print(f"{'='*80}")

        # 加载该类别的TrueSkill评分
        df_ratings = load_trueskill_ratings(category)
        if df_ratings is None:
            continue

        # 加载/创建ID映射
        cat_csv = get_human_choices_csv(category)
        if os.path.exists(cat_csv):
            df_tmp = pd.read_csv(cat_csv)
            id_mapping = create_id_mapping(df_tmp, category)
        else:
            print(f"[ERROR] {category}: 无法创建ID映射，缺少人类标注文件")
            continue

        # 获取Reference集（物理隔离）
        df_ref, df_pool = get_split_data(category=category)
        print(f"  Reference集: {len(df_ref)}对（用于维度定义）")
        print(f"  Pool集: {len(df_pool)}对（保留给Stage 2）")

        # 从Reference集的图片中筛选TrueSkill评分
        ref_image_ids = set()
        ref_image_ids.update(df_ref['left_id'].astype(str))
        ref_image_ids.update(df_ref['right_id'].astype(str))

        df_ratings_ref = df_ratings[
            (df_ratings['category'] == category) &
            (df_ratings['image_id'].astype(str).isin(ref_image_ids))
        ]

        print(f"  Reference图片评分: {len(df_ratings_ref)}张")

        # 采样共识样本（仅从Reference集）
        high_samples, low_samples = sample_consensus_images(df_ratings_ref, category, n_samples=N_CONSENSUS_SAMPLES)
        print(f"  高共识样本: {len(high_samples)}个")
        print(f"  低共识样本: {len(low_samples)}个")

        # 构建prompt
        prompt_content = build_dimension_extraction_prompt(
            category, high_samples, low_samples, id_mapping
        )

        # 调用LLM
        print(f"  调用LLM提取语义维度...")
        messages = [{"role": "user", "content": prompt_content}]

        response = call_llm_api(
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"},
            timeout=300
        )

        if response:
            try:
                dimensions_data = json.loads(response)

                # 显示提取的维度
                print(f"\n  成功提取{category}的语义维度:")
                if 'dimensions' in dimensions_data:
                    for dim in dimensions_data['dimensions']:
                        print(f"    - {dim.get('name', 'Unknown')}: {dim.get('description', '')[:60]}...")

                # 保存为 per-category JSON
                dim_output = {category: dimensions_data}
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dim_output, f, indent=2, ensure_ascii=False)
                print(f"  已保存: {os.path.basename(output_file)}")

            except Exception as e:
                print(f"  解析失败: {e}")
        else:
            print(f"  LLM调用失败")

    print(f"\n{'='*80}")
    print(f"总计处理{len(CATEGORIES)}个类别")
    print("下一步：运行 abc_stage2_multi_mode_synthesis.py 进行特征蒸馏")

if __name__ == "__main__":
    run_dimension_extraction()

