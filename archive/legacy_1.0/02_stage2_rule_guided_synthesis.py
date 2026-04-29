"""
UrbanAlign Stage 2: Rule-Guided & Zero-shot Synthesis (Robust Version)
修复了 JSON 读取报错，支持规则清洗与学术消融实验。
新增：控制参数 EXCLUDE_EQUAL，支持过滤 'equal' 样本进行纯对立性评估。
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

# ==============================================================================
# 1. 配置与消融实验控制
# ==============================================================================
from config import (
    API_KEY, BASE_URL, MODEL_NAME, call_llm_api,
    DATA_DIR, IMAGE_DIR, HUMAN_CHOICES_CSV,
    OUTPUT_DIR, CLIP_CACHE, ID_MAPPING_CSV,
    STAGE1_RULES,
    STAGE2_SYNTHETIC_POOL_RULE, STAGE2_SYNTHETIC_POOL_NO_RULE,
    STAGE2_SAMPLED_PAIRS,
    CATEGORIES, N_POOL_MULTIPLIER, PCA_DIMS,
    EXCLUDE_EQUAL_IN_EVAL  # 从 config 导入新参数
)

# --- 实验控制参数 ---
USE_RULE_GUIDANCE = True  # True: 注入规则(实验组) | False: 纯视觉直觉(对照组)
EXCLUDE_EQUAL = EXCLUDE_EQUAL_IN_EVAL  # 评估时是否排除 'equal' 数据

# 动态切换路径
RULES_FILE = STAGE1_RULES
METADATA_FILE = STAGE2_SAMPLED_PAIRS
OUTPUT_FILE = STAGE2_SYNTHETIC_POOL_RULE if USE_RULE_GUIDANCE else STAGE2_SYNTHETIC_POOL_NO_RULE

PERCEPTION_MAP = {
    'safety': 'safer',
    'beautiful': 'more beautiful',
    'lively': 'livelier',
    'wealthy': 'wealthier',
    'boring': 'more boring',
    'depressing': 'more depressing'
}

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
def extract_json_from_text(text):
    """从包含 Markdown 或解释性文字的文本中提取纯 JSON 对象"""
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx+1])
        return json.loads(text)
    except Exception as e:
        print(f"[ERROR] JSON 解析失败: {e}")
        return None

def load_clip_features():
    if not os.path.exists(CLIP_CACHE): return {}
    data = np.load(CLIP_CACHE)
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_DIMS)
    reduced = pca.fit_transform(scaler.fit_transform(data['embeddings']))
    clip_map = {}
    for path_item, vec in zip(data['paths'], reduced):
        path_str = str(path_item[0]) if isinstance(path_item, np.ndarray) else str(path_item)
        img_id = os.path.splitext(os.path.basename(path_str))[0]
        clip_map[img_id] = [round(v, 2) for v in vec]
    return clip_map

def image_to_base64(img_path):
    with Image.open(img_path) as img:
        if img.mode != 'RGB': img = img.convert('RGB')
        img.thumbnail((512, 512))
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ==============================================================================
# 3. 核心 Prompt 工程
# ==============================================================================
def build_comparison_prompt(alias_a, alias_b, vec_a, vec_b, category, rules_obj=None):
    target = PERCEPTION_MAP[category]
    cat_rules = ""
    if USE_RULE_GUIDANCE and rules_obj and category in rules_obj:
        cat_rules = json.dumps(rules_obj[category], indent=2, ensure_ascii=False)

    base_prompt = f"You are a local resident participating in an urban environment survey."
    rule_section = ""
    if USE_RULE_GUIDANCE and cat_rules:
        rule_section = f"\n[CONSENSUS GUIDELINES]\nThese rules reflect common human perceptions for '{category}':\n{cat_rules}\n"

    prompt = f"""{base_prompt}
Task: Compare the LEFT and RIGHT street images and decide which one feels {target}.

{rule_section}
[DATA]
- Left Image ({alias_a}) CLIP: {vec_a}
- Right Image ({alias_b}) CLIP: {vec_b}

[INSTRUCTIONS]
1. Observe building style, maintenance, greenery, and cleanliness.
2. {"Apply the GUIDELINES to inform your judgment." if USE_RULE_GUIDANCE else "Use your immediate visual intuition."}
3. If the two scenes are nearly identical in terms of being {target}, choose 'equal'.

[OUTPUT FORMAT]
Respond ONLY in valid JSON:
{{
  "{category}": {{
    "reasoning": "one sentence visual reason",
    "winner": "left", "right", or "equal",
    "confidence": 1-5
  }}
}}"""
    return prompt

# ==============================================================================
# 4. 执行逻辑
# ==============================================================================
def run_stage2_synthesis():
    print("\n" + "="*80)
    print(f"UrbanAlign Stage 2: {'Rule-Guided' if USE_RULE_GUIDANCE else 'Zero-shot'} Mode")
    print(f"Output: {os.path.basename(OUTPUT_FILE)}")
    print(f"Exclude 'Equal' in Eval: {EXCLUDE_EQUAL}")
    print("="*80 + "\n")

    # 1. 健壮的规则加载
    rules_obj = {}
    if USE_RULE_GUIDANCE:
        if not os.path.exists(RULES_FILE):
            print(f"[ERROR] 找不到规则文件: {RULES_FILE}"); return
        with open(RULES_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            rules_obj = extract_json_from_text(content)
        print("[STEP 1] 规则库加载并清洗成功。")
    else:
        print("[STEP 1] 消融模式：跳过规则加载。")

    # 2. 基础资源加载
    id_mapping_df = pd.read_csv(ID_MAPPING_CSV)
    id_to_alias = dict(zip(id_mapping_df['original_id'].astype(str), id_mapping_df['alias_id']))
    clip_map = load_clip_features()

    # 3. 采样逻辑
    if os.path.exists(METADATA_FILE):
        sampled_pairs = pd.read_csv(METADATA_FILE)
    else:
        print("[STEP 3] 正在生成采样数据...")
        df_human = pd.read_csv(HUMAN_CHOICES_CSV)
        temp_list = []
        for cat in CATEGORIES:
            cat_data = df_human[df_human['category'] == cat]
            n = min(len(cat_data), int(len(cat_data) * N_POOL_MULTIPLIER))
            temp_list.append(cat_data.sample(n=n, random_state=42))
        sampled_pairs = pd.concat(temp_list, ignore_index=True)
        sampled_pairs.rename(columns={'winner': 'human_winner'}, inplace=True)
        sampled_pairs.to_csv(METADATA_FILE, index=False)

    # 4. 执行标注
    if os.path.exists(OUTPUT_FILE):
        df_old = pd.read_csv(OUTPUT_FILE)
        done = set(zip(df_old['left_id'].astype(str), df_old['right_id'].astype(str), df_old['category']))
        results = df_old.to_dict('records')
    else:
        done = set(); results = []

    tasks = [row for _, row in sampled_pairs.iterrows() if (str(row['left_id']), str(row['right_id']), row['category']) not in done]
    print(f"[STEP 4] 开始合成任务... 待处理: {len(tasks)} 对")

    for i, row in enumerate(tqdm(tasks, desc="Synthesizing")):
        l_id, r_id, cat = str(row['left_id']), str(row['right_id']), row['category']
        img_a, img_b = os.path.join(IMAGE_DIR, f"{l_id}.jpg"), os.path.join(IMAGE_DIR, f"{r_id}.jpg")
        if not (os.path.exists(img_a) and os.path.exists(img_b)): continue

        prompt = build_comparison_prompt(id_to_alias.get(l_id, "Unknown"), id_to_alias.get(r_id, "Unknown"),
                                         clip_map.get(l_id, []), clip_map.get(r_id, []), cat, rules_obj)

        messages = [{"role": "system", "content": "You are an intuitive urban resident. Respond with JSON only."},
                    {"role": "user", "content": [{"type": "text", "text": prompt},
                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(img_a)}"}},
                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(img_b)}"}} ]}]

        response = call_llm_api(messages=messages, temperature=0.1, response_format={"type": "json_object"})

        if response:
            try:
                res_data = json.loads(response).get(cat, {})
                winner = res_data.get('winner', 'error').lower()
                if winner in ['left', 'right', 'equal']:
                    results.append({
                        'left_id': l_id, 'right_id': r_id, 'category': cat,
                        'human_winner': row['human_winner'],
                        'synthetic_winner': winner,
                        'reasoning': res_data.get('reasoning', ''),
                        'agreement': 1 if winner == row['human_winner'] else 0
                    })
            except: pass
        if (i + 1) % 50 == 0: pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    # 5. 性能评估报告
    df_final = pd.DataFrame(results)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*80)
    print(f"STAGE 2 最终性能评估报告 {'(已排除 Equal 样本)' if EXCLUDE_EQUAL else ''}")
    print("="*80)

    pool_files = [STAGE2_SYNTHETIC_POOL_RULE, STAGE2_SYNTHETIC_POOL_NO_RULE]

    for pf in pool_files:
        if os.path.exists(pf):
            df_eval = pd.read_csv(pf)
            if df_eval.empty: continue

            # --- 核心修改：过滤 'equal' 样本 ---
            if EXCLUDE_EQUAL:
                # 排除人类标注为 equal 或 AI 标注为 equal 的行
                df_eval = df_eval[(df_eval['human_winner'] != 'equal') & (df_eval['synthetic_winner'] != 'equal')]
                if df_eval.empty:
                    print(f"\n{os.path.basename(pf)}: 排除 'equal' 后无剩余数据。")
                    continue

            mode_name = "【实验组: 规则引导】" if "with_rule" in pf else "【对照组: 纯视觉】"
            print(f"\n{mode_name}")
            print(f"  - 图像对总数: {len(df_eval)}")

            # 总体指标
            acc = accuracy_score(df_eval['human_winner'], df_eval['synthetic_winner'])
            kappa = cohen_kappa_score(df_eval['human_winner'], df_eval['synthetic_winner'])
            print(f"  - 总体准确率 (Accuracy): {acc*100:5.2f}%")
            print(f"  - 总体一致性 (Cohen's Kappa): {kappa:5.3f}")

            # 分维度指标
            print("  - 分维度详情:")
            for cat in df_eval['category'].unique():
                sub = df_eval[df_eval['category'] == cat]
                if len(sub) < 2: continue # 样本太少无法计算一致性
                c_acc = accuracy_score(sub['human_winner'], sub['synthetic_winner'])
                c_kappa = cohen_kappa_score(sub['human_winner'], sub['synthetic_winner'])
                print(f"    * {cat:10s}: 准确率 {c_acc*100:5.2f}% | Kappa {c_kappa:5.3f} | 样本数 {len(sub)}")

    print("\n" + "="*80)

if __name__ == "__main__":
    run_stage2_synthesis()