"""
UrbanAlign Stage 2: Visual In-Context Alignment (V-ICA)
功能：利用基于 TrueSkill 筛选的高共识“金标准”参考对，引导 LLM 进行类比推理。
优化：
1. 视觉范例注入：将 3 组最相似的、高分差对子的图片送入 LLM。
2. 修复评估模块：修正了变量名错误，补全了 Kappa 和分维度统计。
3. 纯净 RAG：默认不加载 Stage 1 文本规则，避免认知干扰。
"""
import pandas as pd
import numpy as np
import os
import json
import base64
import time
import re
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.neighbors import NearestNeighbors

# ==============================================================================
# 1. 配置与控制参数
# ==============================================================================
from config import (
    API_KEY, BASE_URL, MODEL_NAME, call_llm_api,
    DATA_DIR, IMAGE_DIR, HUMAN_CHOICES_CSV,
    OUTPUT_DIR, CLIP_CACHE, ID_MAPPING_CSV,
    STAGE1_RULES, TRUESKILL_CACHE,
    STAGE2_SYNTHETIC_POOL_RAG, STAGE2_SYNTHETIC_POOL_RULE, STAGE2_SYNTHETIC_POOL_NO_RULE,
    STAGE2_SAMPLED_PAIRS,
    CATEGORIES, N_POOL_MULTIPLIER, PCA_DIMS,
    EXCLUDE_EQUAL_IN_EVAL
)

# --- 核心实验开关 ---
USE_TRIPLE_RAG = True       # 开启 3 组视觉参考模式
REFERENCE_K = 3             # 检索 3 组参考
EXCLUDE_EQUAL = EXCLUDE_EQUAL_IN_EVAL

# 路径管理
OUTPUT_FILE = STAGE2_SYNTHETIC_POOL_RAG if USE_TRIPLE_RAG else STAGE2_SYNTHETIC_POOL_NO_RULE
METADATA_FILE = STAGE2_SAMPLED_PAIRS

PERCEPTION_MAP = {
    'safety': 'safer', 'beautiful': 'more beautiful', 'lively': 'livelier',
    'wealthy': 'wealthier', 'boring': 'more boring', 'depressing': 'more depressing'
}

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
def image_to_base64(img_path, size=(320, 320)):
    """图片缩放优化以平衡视觉特征与 Token 成本"""
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img.thumbnail(size)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=75)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except: return None

def extract_json_data(text):
    try:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        start, end = text.find('{'), text.rfind('}')
        return json.loads(text[start:end+1]) if start != -1 else None
    except: return None

def load_clip_features():
    data = np.load(CLIP_CACHE)
    scaler = StandardScaler(); pca = PCA(n_components=PCA_DIMS)
    reduced = pca.fit_transform(scaler.fit_transform(data['embeddings']))
    return {os.path.splitext(os.path.basename(str(p[0]) if isinstance(p, np.ndarray) else str(p)))[0]: vec.tolist()
            for p, vec in zip(data['paths'], reduced)}

# ==============================================================================
# 3. 核心 Prompt 模块 (修复了之前缺失的部分)
# ==============================================================================
def build_synthesis_prompt(category, ref_data_list):
    """
    构建包含 3 组参考案例的复合提示词结构
    """
    target = PERCEPTION_MAP[category]

    # 1. 任务背景
    content = [
        {"type": "text", "text": f"You are an expert in urban perception analysis. Task: Decide which image looks {target}."}
    ]

    # 2. 注入 3 组视觉范例 (In-Context Alignment)
    content.append({"type": "text", "text": "\n--- [REFERENCE EXAMPLES: HUMAN CONSENSUS] ---"})
    for i, ref in enumerate(ref_data_list):
        content.append({"type": "text", "text": f"Example {i+1}: In the following pair, humans consistently decided the Winner is {ref['winner'].upper()}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref['l_b64']}"}})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref['r_b64']}"}})

    # 3. 注入当前任务指令
    content.append({"type": "text", "text": "\n--- [TARGET TASK] ---"})
    content.append({"type": "text", "text": f"Evaluate the final pair. Calibrate your judgment scale based on the examples above. Compare building quality, greenery, and infrastructure."})

    # 4. JSON 约束
    instruction = f"\nRespond in valid JSON format:\n{{ \"{category}\": {{ \"comparison_reasoning\": \"Briefly relate current task to examples\", \"winner\": \"left/right/equal\" }} }}"
    content.append({"type": "text", "text": instruction})

    return content

# ==============================================================================
# 4. 金标准检索引擎 (Consensus-based Retrieval)
# ==============================================================================
def build_golden_indices(df_human, clip_map, df_ratings):
    """筛选 TrueSkill Delta Mu 显著的对子作为金标准库"""
    print("[INFO] 正在从 36 万对全量数据中筛选高共识金标准...")
    indices = {}
    mu_lookup = df_ratings.set_index(['category', 'image_id'])['mu'].to_dict()

    for cat in CATEGORIES:
        df_cat = df_human[df_human['category'] == cat].copy()
        df_cat = df_cat[df_cat['left_id'].astype(str).isin(clip_map) & df_cat['right_id'].astype(str).isin(clip_map)]

        # 计算 Delta Mu (衡量共识强度)
        df_cat['delta_mu'] = df_cat.apply(lambda r: abs(mu_lookup.get((cat, str(r['left_id'])), 25) -
                                                     mu_lookup.get((cat, str(r['right_id'])), 25)), axis=1)
        # 只保留共识度最高的 25% 对子
        threshold = df_cat['delta_mu'].quantile(0.75)
        df_gold = df_cat[df_cat['delta_mu'] >= threshold].copy()

        if df_gold.empty: continue

        # 检索特征：视觉差值向量 abs(V_left - V_right)
        diff_vectors = [np.abs(np.array(clip_map[str(r['left_id'])]) - np.array(clip_map[str(r['right_id'])]))
                        for _, r in df_gold.iterrows()]

        nn = NearestNeighbors(n_neighbors=REFERENCE_K + 1, metric='cosine')
        nn.fit(diff_vectors)
        indices[cat] = {"nn": nn, "data": df_gold}
    return indices

# ==============================================================================
# 5. 执行流程
# ==============================================================================
def run_stage2_synthesis():
    print("\n" + "="*80)
    print(f"UrbanAlign Stage 2 | Triple Visual In-Context Alignment")
    print(f"Output: {os.path.basename(OUTPUT_FILE)}")
    print("="*80)

    # 1. 资源准备
    df_human_full = pd.read_csv(HUMAN_CHOICES_CSV)
    df_ratings = pd.read_csv(TRUESKILL_CACHE)
    clip_map = load_clip_features()
    indices = build_golden_indices(df_human_full, clip_map, df_ratings)
    sampled_pairs = pd.read_csv(METADATA_FILE)

    if os.path.exists(OUTPUT_FILE):
        df_old = pd.read_csv(OUTPUT_FILE); results = df_old.to_dict('records')
        done = set(zip(df_old['left_id'].astype(str), df_old['right_id'].astype(str), df_old['category']))
    else:
        done, results = set(), []

    tasks = [row for _, row in sampled_pairs.iterrows() if (str(row['left_id']), str(row['right_id']), row['category']) not in done]

    for i, row in enumerate(tqdm(tasks, desc="Synthesizing")):
        l_id, r_id, cat = str(row['left_id']), str(row['right_id']), row['category']

        # 检索并准备 3 组参考对子的视觉数据
        target_diff = np.abs(np.array(clip_map[l_id]) - np.array(clip_map[r_id]))
        _, idxs = indices[cat]["nn"].kneighbors([target_diff])

        ref_data_list = []
        for ref_idx in idxs[0][:REFERENCE_K]:
            ref_row = indices[cat]["data"].iloc[ref_idx]
            if str(ref_row['left_id']) == l_id: continue

            ref_data_list.append({
                'l_b64': image_to_base64(os.path.join(IMAGE_DIR, f"{ref_row['left_id']}.jpg"), size=(256, 256)),
                'r_b64': image_to_base64(os.path.join(IMAGE_DIR, f"{ref_row['right_id']}.jpg"), size=(256, 256)),
                'winner': ref_row['winner']
            })

        # 准备当前任务图
        task_l_b64 = image_to_base64(os.path.join(IMAGE_DIR, f'{l_id}.jpg'))
        task_r_b64 = image_to_base64(os.path.join(IMAGE_DIR, f'{r_id}.jpg'))

        # 组装 Prompt
        prompt_content = build_synthesis_prompt(cat, ref_data_list)
        prompt_content.extend([
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{task_l_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{task_r_b64}"}}
        ])

        msgs = [{"role": "system", "content": "You are a perception expert. Perform visual analogy. JSON ONLY."},
                {"role": "user", "content": prompt_content}]

        res = call_llm_api(messages=msgs, temperature=0, response_format={"type": "json_object"})
        if res:
            try:
                data = extract_json_data(res).get(cat, {})
                winner = data.get('winner', 'error').lower()
                if winner in ['left', 'right', 'equal']:
                    results.append({
                        'left_id': l_id, 'right_id': r_id, 'category': cat,
                        'human_winner': row['human_winner'], 'synthetic_winner': winner,
                        'agreement': 1 if winner == row['human_winner'] else 0
                    })
            except: pass
        if (i + 1) % 10 == 0: pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    # ==============================================================================
    # 6. 学术评估报告 (全量扫描对比)
    # ==============================================================================
    df_final = pd.DataFrame(results)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*80 + "\nUrbanAlign 学术评估报告 (V-ICA)\n" + "="*80)
    eval_files = [STAGE2_SYNTHETIC_POOL_RAG, STAGE2_SYNTHETIC_POOL_RULE, STAGE2_SYNTHETIC_POOL_NO_RULE]
    for pf in eval_files:
        if os.path.exists(pf):
            df_e = pd.read_csv(pf)
            if df_e.empty: continue
            if EXCLUDE_EQUAL:
                df_e = df_e[(df_e['human_winner'] != 'equal') & (df_e['synthetic_winner'] != 'equal')]
            if df_e.empty: continue

            print(f"\n>> 文件: {os.path.basename(pf)}")
            acc = accuracy_score(df_e['human_winner'], df_e['synthetic_winner'])
            kappa = cohen_kappa_score(df_e['human_winner'], df_e['synthetic_winner']) if len(df_e) > 1 else 0

            print(f"   [指标] Accuracy: {acc*100:5.2f}% | Cohen's Kappa: {kappa:5.3f} | N={len(df_e)}")
            for c in df_e['category'].unique():
                sub = df_e[df_e['category'] == c]
                if len(sub) < 2: continue
                c_acc = accuracy_score(sub['human_winner'], sub['synthetic_winner'])
                c_kappa = cohen_kappa_score(sub['human_winner'], sub['synthetic_winner'])
                print(f"     * {c:10s}: Acc {c_acc*100:5.2f}% | Kappa {c_kappa:5.3f}")
    print("="*80)

if __name__ == "__main__":
    run_stage2_synthesis()