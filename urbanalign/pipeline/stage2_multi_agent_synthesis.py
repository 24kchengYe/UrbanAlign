"""
UrbanAlign 2.0 - Stage 2: Multi-Mode Synthesis (逻辑增强完整版)

4种模式对比实验：
- Mode 1: 单张图片直接打分（Single Image Direct Scoring）
- Mode 2: 图片对直接打分（Pairwise Direct Scoring）
- Mode 3: 单张图片多智能体打分（Single Image Multi-Agent）
- Mode 4: 图片对多智能体博弈（Pairwise Multi-Agent Deliberation）

核心优化：
1. 参数配置化：从 config 导入显著性阈值与采样比例，消除硬编码。
2. 鲁棒解析：新增 robust_parse 逻辑，当 AI 遗漏整体强度分时，利用维度分加权补偿，防止数据丢失。
3. 统一接口：将所有 Mode 的权重、强度、分数统一存储，为 Stage 3 局部加权回归提供规整的 X 矩阵。
4. 闭环评估：运行结束自动生成分类别、含/不含平局的准确率报告。
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import base64
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score

# ==============================================================================
# 1. 导入配置
# ==============================================================================
from urbanalign.config import (
    API_KEY, BASE_URL, MODEL_NAME, call_llm_api,
    DATA_DIR, IMAGE_DIR,
    OUTPUT_DIR, CLIP_CACHE,
    STAGE2_MODE,
    CATEGORIES, N_POOL_MULTIPLIER, MAX_PAIRS_PER_CATEGORY,
    ST2_INTENSITY_SIG_THRESH, EVAL_EXCLUDE_EQUAL,
    get_stage1_dimensions, get_stage2_output, get_stage2_sampled_pairs,
    get_human_choices_csv, get_split_data, get_split_cache_paths,
    get_stage3_output, get_stage4_output, get_stage5_output
)

# Mode选择
SYNTHESIS_MODE = STAGE2_MODE

if SYNTHESIS_MODE == 5:
    print("\n" + "=" * 80 + "\nMode 5: 自动运行全部4种模式\n" + "=" * 80)
    import subprocess

    subprocess.run(['python', 'abc_stage2_auto_all_modes.py'])
    sys.exit(0)

# 文件路径现在通过 get_stage2_output(mode, cat) 和 get_stage2_sampled_pairs(cat) 动态生成


# ==============================================================================
# 2. 辅助函数与鲁棒解析
# ==============================================================================
def load_dimensions(category):
    """加载指定类别的语义维度定义"""
    dim_file = get_stage1_dimensions(category)
    if not os.path.exists(dim_file):
        return None
    with open(dim_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_clip_features():
    if not os.path.exists(CLIP_CACHE): return {}
    data = np.load(CLIP_CACHE)
    return {os.path.splitext(os.path.basename(str(p)))[0]: v for p, v in zip(data['paths'], data['embeddings'])}


def image_to_base64(img_path):
    with Image.open(img_path) as img:
        if img.mode != 'RGB': img = img.convert('RGB')
        img.thumbnail((512, 512))
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def robust_parse_vlm_output(data, prefix=""):
    """
    鲁棒性解析：
    1. 统一维度权重字段名为 'dimension_weights'。
    2. 如果 AI 遗漏 overall_intensity，则通过维度分的平均值进行补偿。
    """
    if data is None: return None, 50, {}

    # 提取权重
    weights = data.get(f'dimension_weights{prefix}') or data.get('weights') or data.get('dimension_weights') or {}

    # 提取得分向量
    scores = data.get(f'image_{prefix[-1].lower()}_scores') if prefix else (data.get('scores') or data)

    # 提取/补偿 综合强度 (0-100)
    intensity = data.get(f'overall_intensity{prefix}') or data.get('overall_intensity')
    if intensity is None and isinstance(scores, dict):
        # 降级方案：1-10 均值映射到 0-100
        intensity = np.mean([float(v) for v in scores.values()]) * 10.0
    elif intensity is None:
        intensity = 50.0

    return scores, float(intensity), weights


# ==============================================================================
# 3. 推理核心：模式函数实现
# ==============================================================================

def mode1_single_direct(img_path, category, dimensions):
    dim_list = dimensions.get('dimensions', [])
    dim_names = [d['name'] for d in dim_list]
    dim_desc = '\n'.join([f"- {d['name']}: {d['description']}" for d in dim_list])

    prompt = f"""Rate this image on "{category}" perception. Use the specific dimensions provided.
[DIMENSIONS]
{dim_desc}
[TASK]
1. Rate each dimension (1-10).
2. Assign weights (0.0-1.0) showing each dimension's impact on overall "{category}" (sum must be 1.0).
3. Provide an overall intensity score (0-100).

Output JSON:
{{
  "scores": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
  "dimension_weights": {{{', '.join([f'"{d}": weight' for d in dim_names])}}},
  "overall_intensity": 0-100
}}"""
    b64 = image_to_base64(img_path)
    return call_llm_api([{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url",
                                                                                         "image_url": {
                                                                                             "url": f"data:image/jpeg;base64,{b64}"}}]}],
                        temperature=0.1, response_format={"type": "json_object"})


def mode2_pair_direct(img_a_path, img_b_path, category, dimensions):
    dim_list = dimensions.get('dimensions', [])
    dim_desc = '\n'.join([f"- {d['name']}: {d['description']}" for d in dim_list])

    prompt = f"""Compare two images on "{category}" perception using these dimensions:
{dim_desc}
[TASK]
Rate BOTH images (1-10), assign importance weights, and provide final overall intensities (0-100).

Output JSON:
{{
  "image_a_scores": {{...}}, "image_b_scores": {{...}},
  "dimension_weights": {{...}},
  "overall_intensity_a": 0-100, "overall_intensity_b": 0-100,
  "winner": "left"/"right"/"equal"
}}"""
    b64_a, b64_b = image_to_base64(img_a_path), image_to_base64(img_b_path)
    return call_llm_api([{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "text", "text": "Image A:"}, {"type": "image_url",
                                                                                             "image_url": {
                                                                                                 "url": f"data:image/jpeg;base64,{b64_a}"}},
                                                      {"type": "text", "text": "Image B:"}, {"type": "image_url",
                                                                                             "image_url": {
                                                                                                 "url": f"data:image/jpeg;base64,{b64_b}"}}]}],
                        temperature=0.1, response_format={"type": "json_object"})


# ==============================================================================
# 3b. Mode 3: Single Image Multi-Agent (Observer→Debater→Judge)
# ==============================================================================
def mode3_single_multiagent(img_path, category, dimensions):
    """Mode 3: 单张图片的多智能体打分"""
    dim_list = dimensions.get('dimensions', [])
    dim_names = [d['name'] for d in dim_list]
    dim_desc = '\n'.join([f"- {d['name']}: {d['description']}" for d in dim_list])
    b64 = image_to_base64(img_path)

    # Step 1: Observer
    obs_prompt = f"""Describe visual details for "{category}" assessment.

[DIMENSIONS]
{dim_desc}

Describe objectively (3-5 sentences).
"""
    observation = call_llm_api([
        {"role": "system", "content": "Detail-oriented observer."},
        {"role": "user", "content": [
            {"type": "text", "text": obs_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}
    ], temperature=0.3, max_tokens=300)

    if not observation:
        return None
    time.sleep(0.5)

    # Step 2: Debater
    deb_prompt = f"""Argue for HIGH and LOW for each dimension.

[OBSERVATION]
{observation}

[DIMENSIONS]
{dim_desc}

For each, argue both sides (1-2 sentences).
"""
    debate = call_llm_api([
        {"role": "system", "content": "Critical debater."},
        {"role": "user", "content": [
            {"type": "text", "text": deb_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}
    ], temperature=0.5, max_tokens=400)

    if not debate:
        return None
    time.sleep(0.5)

    # Step 3: Judge — output scores + weights + intensity
    judge_prompt = f"""Final scores (1-10) for each dimension, with importance weights and overall intensity.

[OBSERVATION]
{observation}

[DEBATE]
{debate}

[DIMENSIONS]
{dim_desc}

Output JSON:
{{
  "scores": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
  "dimension_weights": {{{', '.join([f'"{d}": weight' for d in dim_names])}}},
  "overall_intensity": 0-100
}}"""
    return call_llm_api([
        {"role": "system", "content": "Impartial judge. JSON only."},
        {"role": "user", "content": [
            {"type": "text", "text": judge_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}
    ], temperature=0.1, response_format={"type": "json_object"}, max_tokens=300)


# ==============================================================================
# 3c. Mode 4: Pairwise Multi-Agent Deliberation (Observer→Debater→Judge)
# ==============================================================================
def mode4_pair_multiagent(img_a_path, img_b_path, category, dimensions):
    """Mode 4: 图片对的多智能体对比博弈"""
    dim_list = dimensions.get('dimensions', [])
    dim_names = [d['name'] for d in dim_list]
    dim_desc = '\n'.join([f"- {d['name']}: {d['description']}" for d in dim_list])
    b64_a, b64_b = image_to_base64(img_a_path), image_to_base64(img_b_path)

    # Step 1: Observer
    obs_prompt = f"""Compare VISUAL DIFFERENCES between A and B for "{category}".

[DIMENSIONS]
{dim_desc}

Describe contrasts (A has X, B has Y) in 3-5 sentences.
"""
    observation = call_llm_api([
        {"role": "system", "content": "Observer focusing on differences."},
        {"role": "user", "content": [
            {"type": "text", "text": obs_prompt},
            {"type": "text", "text": "Image A:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
            {"type": "text", "text": "Image B:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}}
        ]}
    ], temperature=0.3, max_tokens=400)

    if not observation:
        return None
    time.sleep(0.5)

    # Step 2: Debater
    deb_prompt = f"""Argue for BOTH A and B on each dimension.

[COMPARISON]
{observation}

[DIMENSIONS]
{dim_desc}

For each dimension:
- Why A scores HIGH? Why A scores LOW?
- Why B scores HIGH? Why B scores LOW?

Be concise (2-3 sentences per dimension).
"""
    debate = call_llm_api([
        {"role": "system", "content": "Debater arguing both sides."},
        {"role": "user", "content": [
            {"type": "text", "text": deb_prompt},
            {"type": "text", "text": "Image A:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
            {"type": "text", "text": "Image B:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}}
        ]}
    ], temperature=0.5, max_tokens=600)

    if not debate:
        return None
    time.sleep(0.5)

    # Step 3: Judge
    judge_prompt = f"""Final comparative judgment for "{category}".

[OBSERVATION]
{observation}

[DEBATE]
{debate}

[DIMENSIONS]
{dim_desc}

Rate BOTH images, assign weights, and determine winner.

Output JSON:
{{
  "image_a_scores": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
  "image_b_scores": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
  "dimension_weights": {{{', '.join([f'"{d}": weight' for d in dim_names])}}},
  "overall_intensity_a": 0-100,
  "overall_intensity_b": 0-100,
  "winner": "left"/"right"/"equal"
}}"""
    return call_llm_api([
        {"role": "system", "content": "Impartial judge. JSON only."},
        {"role": "user", "content": [
            {"type": "text", "text": judge_prompt},
            {"type": "text", "text": "Image A:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
            {"type": "text", "text": "Image B:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}}
        ]}
    ], temperature=0.1, response_format={"type": "json_object"}, max_tokens=600)


# ==============================================================================
# 4. 合成引擎：处理逻辑单元 (统一 Pool 与 Ref)
# ==============================================================================
def process_one_pair(row, category, dimensions_all):
    l_id, r_id = str(row['left_id']), str(row['right_id'])
    img_a, img_b = os.path.join(IMAGE_DIR, f"{l_id}.jpg"), os.path.join(IMAGE_DIR, f"{r_id}.jpg")
    dims = dimensions_all.get(category, {})

    try:
        if SYNTHESIS_MODE == 1:
            # Mode 1: 单图直接打分
            ra = mode1_single_direct(img_a, category, dims)
            rb = mode1_single_direct(img_b, category, dims)
            sa, ia, wa = robust_parse_vlm_output(json.loads(ra))
            sb, ib, wb = robust_parse_vlm_output(json.loads(rb))
            win = 'left' if ia > ib + ST2_INTENSITY_SIG_THRESH else (
                'right' if ib > ia + ST2_INTENSITY_SIG_THRESH else 'equal')
            return {'image_a_scores': json.dumps(sa), 'image_b_scores': json.dumps(sb),
                    'overall_intensity_a': ia, 'overall_intensity_b': ib,
                    'ai_dimension_weights': json.dumps(wa), 'synthetic_winner': win}

        elif SYNTHESIS_MODE == 3:
            # Mode 3: 单图多智能体 (Observer→Debater→Judge)
            ra = mode3_single_multiagent(img_a, category, dims)
            rb = mode3_single_multiagent(img_b, category, dims)
            sa, ia, wa = robust_parse_vlm_output(json.loads(ra))
            sb, ib, wb = robust_parse_vlm_output(json.loads(rb))
            win = 'left' if ia > ib + ST2_INTENSITY_SIG_THRESH else (
                'right' if ib > ia + ST2_INTENSITY_SIG_THRESH else 'equal')
            return {'image_a_scores': json.dumps(sa), 'image_b_scores': json.dumps(sb),
                    'overall_intensity_a': ia, 'overall_intensity_b': ib,
                    'ai_dimension_weights': json.dumps(wa), 'synthetic_winner': win}

        elif SYNTHESIS_MODE == 2:
            # Mode 2: 成对直接打分
            resp = mode2_pair_direct(img_a, img_b, category, dims)
            data = json.loads(resp)
            sa, ia, wa = robust_parse_vlm_output(data, prefix="_a")
            sb, ib, _ = robust_parse_vlm_output(data, prefix="_b")
            return {'image_a_scores': json.dumps(sa), 'image_b_scores': json.dumps(sb),
                    'overall_intensity_a': ia, 'overall_intensity_b': ib,
                    'ai_dimension_weights': json.dumps(wa), 'synthetic_winner': data.get('winner', 'equal')}

        elif SYNTHESIS_MODE == 4:
            # Mode 4: 成对多智能体博弈 (Observer→Debater→Judge)
            resp = mode4_pair_multiagent(img_a, img_b, category, dims)
            data = json.loads(resp)
            sa, ia, wa = robust_parse_vlm_output(data, prefix="_a")
            sb, ib, _ = robust_parse_vlm_output(data, prefix="_b")
            return {'image_a_scores': json.dumps(sa), 'image_b_scores': json.dumps(sb),
                    'overall_intensity_a': ia, 'overall_intensity_b': ib,
                    'ai_dimension_weights': json.dumps(wa), 'synthetic_winner': data.get('winner', 'equal')}
    except:
        return None


# ==============================================================================
# 5. 主流程控制（解耦架构：对全量采样数据统一打分，不区分Ref/Pool）
# ==============================================================================
def run_multi_mode_synthesis():
    MODE_NAMES = {1: "Mode 1: Single Direct", 2: "Mode 2: Pair Direct", 3: "Mode 3: Single Multi-Agent",
                  4: "Mode 4: Pair Multi-Agent"}
    print(f"\n{'='*80}")
    print(f"UrbanAlign 2.0 - Stage 2: {MODE_NAMES[SYNTHESIS_MODE]}")
    print(f"{'='*80}")
    print(f"  解耦架构: 对全量采样数据统一打分 (不区分Ref/Pool)")
    print(f"  Stage 3 将动态拆分为 Ref 和 Pool\n")

    for cat in CATEGORIES:
        output_file = get_stage2_output(SYNTHESIS_MODE, cat)
        sampled_cache = get_stage2_sampled_pairs(cat)

        # ── 第1步: 确保采样缓存存在且符合当前配置 ──
        cache_updated = False

        # 计算当前配置的期望采样数
        cat_csv = get_human_choices_csv(cat)
        if not os.path.exists(cat_csv):
            print(f"[ERROR] {cat}: 人类标注文件不存在: {cat_csv}")
            continue
        df_cat = pd.read_csv(cat_csv)
        expected_n = max(1, int(len(df_cat) * N_POOL_MULTIPLIER))
        if MAX_PAIRS_PER_CATEGORY > 0:
            expected_n = min(expected_n, MAX_PAIRS_PER_CATEGORY)
        expected_n = min(expected_n, len(df_cat))

        if os.path.exists(sampled_cache):
            sampled_pairs = pd.read_csv(sampled_cache)
            if len(sampled_pairs) == expected_n:
                print(f"[LOAD] {cat}: 采样缓存 {len(sampled_pairs)}对 (与配置一致)")
            elif len(sampled_pairs) > expected_n:
                print(f"[CAP] {cat}: 已有缓存 {len(sampled_pairs)}对 > 期望 {expected_n}, 缩小...")
                sampled_pairs = sampled_pairs.sample(n=expected_n, random_state=42)
                sampled_pairs.to_csv(sampled_cache, index=False)
                cache_updated = True
                print(f"  缓存已更新: {len(sampled_pairs)}对")
            else:
                print(f"[EXPAND] {cat}: 已有缓存 {len(sampled_pairs)}对 < 期望 {expected_n}, 重新采样...")
                sampled_pairs = df_cat.sample(n=expected_n, random_state=42)
                if 'winner' in sampled_pairs.columns and 'human_winner' not in sampled_pairs.columns:
                    sampled_pairs.rename(columns={'winner': 'human_winner'}, inplace=True)
                sampled_pairs.to_csv(sampled_cache, index=False)
                cache_updated = True
                print(f"  缓存已更新: {len(sampled_pairs)}对")
        else:
            sampled_pairs = df_cat.sample(n=expected_n, random_state=42)
            if 'winner' in sampled_pairs.columns and 'human_winner' not in sampled_pairs.columns:
                sampled_pairs.rename(columns={'winner': 'human_winner'}, inplace=True)
            sampled_pairs.to_csv(sampled_cache, index=False)
            cache_updated = True
            cap_info = f", cap={MAX_PAIRS_PER_CATEGORY}" if MAX_PAIRS_PER_CATEGORY > 0 else ""
            print(f"[NEW] {cat}: 新建采样缓存 {len(sampled_pairs)}对 (比例={N_POOL_MULTIPLIER}{cap_info})")

        # ── 第1.5步: 触发 Ref/Pool 分割缓存 (供下游 Stage 3/6 使用) ──
        if cache_updated:
            print(f"  采样缓存变更, 重新划分 Ref/Pool...")
            get_split_data(category=cat, force_reload=True)
            # 自动清理下游 Stage 3/4/5 旧输出 (采样数据已变, 旧结果不再有效)
            stale_files = []
            for m in [1, 2, 3, 4]:
                stale_files.append(get_stage3_output(m, cat))
            for s4_name in ['all_modes_comparison', 'dimension_discriminability']:
                stale_files.append(get_stage4_output(s4_name, cat))
            stale_files.append(get_stage5_output(SYNTHESIS_MODE, cat))
            removed = []
            for f in stale_files:
                if os.path.exists(f):
                    os.remove(f)
                    removed.append(os.path.basename(f))
            if removed:
                print(f"  清理下游旧输出: {', '.join(removed)}")
        else:
            # 确保 split 缓存存在 (首次运行时生成)
            ref_cache, pool_cache = get_split_cache_paths(cat)
            if not os.path.exists(ref_cache) or not os.path.exists(pool_cache):
                print(f"  Split缓存不存在, 生成 Ref/Pool...")
                get_split_data(category=cat)

        # ── 第2步: 若缓存缩小, 裁剪已有打分输出使其只保留缓存内的对 ──
        sampled_keys = set(zip(
            sampled_pairs['left_id'].astype(str), sampled_pairs['right_id'].astype(str)
        ))
        if cache_updated and os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                if len(existing_df) > 0:
                    before_len = len(existing_df)
                    existing_df = existing_df[
                        [( str(r['left_id']), str(r['right_id']) ) in sampled_keys
                         for _, r in existing_df.iterrows()]
                    ]
                    if len(existing_df) < before_len:
                        existing_df.to_csv(output_file, index=False)
                        print(f"  打分输出裁剪: {before_len} → {len(existing_df)}对 (匹配缩小后的采样缓存)")
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                print(f"[WARN] {cat}: 输出文件损坏, 将重新开始")
                os.remove(output_file)

        # ── 第3步: 跳过检查 & 断点续传 ──
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                if len(existing_df) == 0:
                    raise pd.errors.EmptyDataError("No data rows")
                done = set(zip(existing_df['left_id'].astype(str), existing_df['right_id'].astype(str)))
                remaining = sum(1 for k in sampled_keys if k not in done)
                if remaining == 0:
                    print(f"[SKIP] {cat}: {os.path.basename(output_file)} already complete ({len(existing_df)}对)")
                    continue
                results = existing_df.to_dict('records')
                print(f"[RESUME] {cat}: {len(results)}对已完成, {remaining}对待处理")
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                print(f"[WARN] {cat}: 输出文件损坏或为空, 将重新开始")
                os.remove(output_file)
                results, done = [], set()
        else:
            results, done = [], set()

        print(f"\n{'='*60}")
        print(f"处理类别: {cat}")
        print(f"{'='*60}")

        # 加载该类别的维度定义
        dims_all = load_dimensions(cat)
        if not dims_all:
            print(f"[ERROR] {cat}: 维度文件缺失，请先运行 Stage 1")
            continue

        tasks = [r for _, r in sampled_pairs.iterrows()
                 if (str(r['left_id']), str(r['right_id'])) not in done]
        print(f"  待处理: {len(tasks)}对")

        for i, r in enumerate(tqdm(tasks, desc=f"Mode {SYNTHESIS_MODE} ({cat})")):
            res = process_one_pair(r, cat, dims_all)
            if res:
                res.update({
                    'left_id': r['left_id'], 'right_id': r['right_id'], 'category': cat,
                    'human_winner': r['human_winner'], 'mode': SYNTHESIS_MODE
                })
                results.append(res)
            if (i + 1) % 20 == 0:
                pd.DataFrame(results).to_csv(output_file, index=False)

        df_final = pd.DataFrame(results)
        df_final.to_csv(output_file, index=False)
        print(f"  输出: {os.path.basename(output_file)} ({len(df_final)}对)")

        if len(df_final) > 0 and 'synthetic_winner' in df_final.columns:
            acc = accuracy_score(df_final['human_winner'], df_final['synthetic_winner'])
            kappa = cohen_kappa_score(df_final['human_winner'], df_final['synthetic_winner'])
            print(f"  快速评估 (含Equal): 准确率={acc*100:.2f}% | Kappa={kappa:.3f}")

    print(f"\n{'='*80}")
    print(f"Mode {SYNTHESIS_MODE} 全部类别打分完成！")
    print(f"{'='*80}")
    print(f"\n  下一步: python abc_stage3_hybrid_vrm.py (动态拆分Ref/Pool并对齐)")


if __name__ == "__main__":
    run_multi_mode_synthesis()

# """
# import pandas as pd
# import numpy as np
# import os
# import sys
# import json
# import base64
# import time
# from PIL import Image
# from io import BytesIO
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, cohen_kappa_score
# from sklearn.metrics.pairwise import cosine_similarity  # 用于一对一寻找最近邻
#
# # ==============================================================================
# # 1. 导入配置
# # ==============================================================================
# from urbanalign.config import (
#     API_KEY, BASE_URL, MODEL_NAME, call_llm_api, get_split_data,
#     DATA_DIR, IMAGE_DIR, HUMAN_CHOICES_CSV,
#     OUTPUT_DIR, CLIP_CACHE, ID_MAPPING_CSV,
#     STAGE1_DIMENSIONS, STAGE2_MODE,
#     CATEGORIES, N_POOL_MULTIPLIER, PCA_DIMS, STAGE2_SAMPLED_PAIRS,
#     VRM_SAMPLE_RATIO
# )
#
# # Mode选择（1-5）
# SYNTHESIS_MODE = STAGE2_MODE
#
# # Mode 5特殊处理：自动运行全部模式
# if SYNTHESIS_MODE == 5:
#     print("\n" + "="*80)
#     print("Mode 5: 自动运行全部4种模式")
#     print("="*80)
#     print("\n正在启动自动运行脚本...")
#     import subprocess
#     subprocess.run(['python', 'abc_stage2_auto_all_modes.py'])
#     sys.exit(0)
#
# # 输出文件（按Mode分开）
# MODE_OUTPUT_MAP = {
#     1: os.path.join(OUTPUT_DIR, "stage2_mode1_single_direct.csv"),
#     2: os.path.join(OUTPUT_DIR, "stage2_mode2_pair_direct.csv"),
#     3: os.path.join(OUTPUT_DIR, "stage2_mode3_single_multiagent.csv"),
#     4: os.path.join(OUTPUT_DIR, "stage2_mode4_pair_multiagent.csv")
# }
#
# OUTPUT_FILE = MODE_OUTPUT_MAP[SYNTHESIS_MODE]
# SAMPLED_PAIRS_FILE = STAGE2_SAMPLED_PAIRS
# DIMENSIONS_FILE = STAGE1_DIMENSIONS
#
# # 全局映射与模式专用标尺路径
# GLOBAL_MAPPING_FILE = os.path.join(OUTPUT_DIR, "stage2_vrm_global_mapping.csv")
# UNIFIED_VRM_REF_FILE = os.path.join(OUTPUT_DIR, f"stage2_mode{SYNTHESIS_MODE}_vrm_reference_scored.csv")
#
# # ==============================================================================
# # 2. 辅助函数
# # ==============================================================================
# def load_dimensions():
#     if not os.path.exists(DIMENSIONS_FILE):
#         print(f"[ERROR] 维度文件不存在: {DIMENSIONS_FILE}")
#         return None
#     with open(DIMENSIONS_FILE, 'r', encoding='utf-8') as f:
#         return json.load(f)
#
# def load_clip_features():
#     if not os.path.exists(CLIP_CACHE):
#         print(f"[ERROR] CLIP缓存不存在: {CLIP_CACHE}")
#         return {}
#     data = np.load(CLIP_CACHE)
#     clip_map = {}
#     for path_item, vec in zip(data['paths'], data['embeddings']):
#         path_str = str(path_item[0]) if isinstance(path_item, np.ndarray) else str(path_item)
#         img_id = os.path.splitext(os.path.basename(path_str))[0]
#         clip_map[img_id] = vec
#     return clip_map
#
# def image_to_base64(img_path):
#     with Image.open(img_path) as img:
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
#         img.thumbnail((512, 512))
#         buffered = BytesIO()
#         img.save(buffered, format="JPEG", quality=85)
#         return base64.b64encode(buffered.getvalue()).decode('utf-8')
#
# # ==============================================================================
# # 3. Mode 1: Single Image Direct Scoring
# # ==============================================================================
# def mode1_single_direct(img_path, category, dimensions):
#     """
#     Mode 1: 单张图片直接打分
#     """
#     dim_list = dimensions.get('dimensions', [])
#     dim_names = [d['name'] for d in dim_list]
#     dim_desc = '\n'.join([f"- {d['name']}: {d['description']}" for d in dim_list])
#
#     prompt = f"""Rate this street-view image on "{category}" perception.
#
# [DIMENSIONS]
# {dim_desc}
#
# [TASK]
# Rate each dimension (1-10 scale).
#
# Output JSON:
# {{{', '.join([f'"{d}": score' for d in dim_names])}}}
# """
#
#     b64 = image_to_base64(img_path)
#     messages = [
#         {"role": "system", "content": "You are a visual assessment expert. Respond with JSON only."},
#         {"role": "user", "content": [
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
#         ]}
#     ]
#
#     return call_llm_api(messages, temperature=0.1, response_format={"type": "json_object"}, max_tokens=300)
#
# # ==============================================================================
# # 4. Mode 2: Pairwise Direct Scoring
# # ==============================================================================
# def mode2_pair_direct(img_a_path, img_b_path, category, dimensions):
#     """
#     Mode 2: 图片对直接对比打分
#     """
#     dim_list = dimensions.get('dimensions', [])
#     dim_names = [d['name'] for d in dim_list]
#     dim_desc = '\n'.join([f"- {d['name']}: {d['description']}" for d in dim_list])
#
#     prompt = f"""Compare two images on "{category}" perception.
#
# [DIMENSIONS]
# {dim_desc}
#
# [TASK]
# Rate BOTH images on each dimension (1-10), then judge winner.
#
# Output JSON:
# {{
#   "image_a_scores": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
#   "image_b_scores": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
#   "winner": "left"/"right"/"equal"
# }}
# """
#
#     b64_a, b64_b = image_to_base64(img_a_path), image_to_base64(img_b_path)
#     messages = [
#         {"role": "system", "content": "Comparative assessment expert. Respond with JSON only."},
#         {"role": "user", "content": [
#             {"type": "text", "text": prompt},
#             {"type": "text", "text": "Image A:"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
#             {"type": "text", "text": "Image B:"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}}
#         ]}
#     ]
#
#     return call_llm_api(messages, temperature=0.1, response_format={"type": "json_object"}, max_tokens=500)
#
# # ==============================================================================
# # 5. Mode 3: Single Image Multi-Agent
# # ==============================================================================
# def mode3_single_multiagent(img_path, category, dimensions):
#     """
#     Mode 3: 单张图片的多智能体打分
#     """
#     dim_list = dimensions.get('dimensions', [])
#     dim_names = [d['name'] for d in dim_list]
#     dim_desc = '\n'.join([f"- {d['name']}: {d['description']}" for d in dim_list])
#     b64 = image_to_base64(img_path)
#
#     # Step 1: Observer
#     obs_prompt = f"""Describe visual details for "{category}" assessment.
#
# [DIMENSIONS]
# {dim_desc}
#
# Describe objectively (3-5 sentences).
# """
#
#     observation = call_llm_api([
#         {"role": "system", "content": "Detail-oriented observer."},
#         {"role": "user", "content": [
#             {"type": "text", "text": obs_prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
#         ]}
#     ], temperature=0.3, max_tokens=300)
#
#     if not observation:
#         return None
#     time.sleep(0.5)
#
#     # Step 2: Debater
#     deb_prompt = f"""Argue for HIGH and LOW for each dimension.
#
# [OBSERVATION]
# {observation}
#
# [DIMENSIONS]
# {dim_desc}
#
# For each, argue both sides (1-2 sentences).
# """
#
#     debate = call_llm_api([
#         {"role": "system", "content": "Critical debater."},
#         {"role": "user", "content": [
#             {"type": "text", "text": deb_prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
#         ]}
#     ], temperature=0.5, max_tokens=400)
#
#     if not debate:
#         return None
#     time.sleep(0.5)
#
#     # Step 3: Judge
#     judge_prompt = f"""Final scores (1-10) for each dimension.
#
# [OBSERVATION]
# {observation}
#
# [DEBATE]
# {debate}
#
# [DIMENSIONS]
# {dim_desc}
#
# Output JSON:
# {{{', '.join([f'"{d}": score' for d in dim_names])}}}
# """
#
#     return call_llm_api([
#         {"role": "system", "content": "Impartial judge. JSON only."},
#         {"role": "user", "content": [
#             {"type": "text", "text": judge_prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
#         ]}
#     ], temperature=0.1, response_format={"type": "json_object"}, max_tokens=300)
#
# # ==============================================================================
# # 6. Mode 4: Pairwise Multi-Agent Deliberation
# # ==============================================================================
# def mode4_pair_multiagent(img_a_path, img_b_path, category, dimensions):
#     """
#     Mode 4: 图片对的多智能体对比博弈
#     """
#     dim_list = dimensions.get('dimensions', [])
#     dim_names = [d['name'] for d in dim_list]
#     dim_desc = '\n'.join([f"- {d['name']}: {d['description']}" for d in dim_list])
#     b64_a, b64_b = image_to_base64(img_a_path), image_to_base64(img_b_path)
#
#     # Step 1: Observer
#     obs_prompt = f"""Compare VISUAL DIFFERENCES between A and B for "{category}".
#
# [DIMENSIONS]
# {dim_desc}
#
# Describe contrasts (A has X, B has Y) in 3-5 sentences.
# """
#
#     observation = call_llm_api([
#         {"role": "system", "content": "Observer focusing on differences."},
#         {"role": "user", "content": [
#             {"type": "text", "text": obs_prompt},
#             {"type": "text", "text": "Image A:"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
#             {"type": "text", "text": "Image B:"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}}
#         ]}
#     ], temperature=0.3, max_tokens=400)
#
#     if not observation:
#         return None
#     time.sleep(0.5)
#
#     # Step 2: Debater
#     deb_prompt = f"""Argue for BOTH A and B on each dimension.
#
# [COMPARISON]
# {observation}
#
# [DIMENSIONS]
# {dim_desc}
#
# For each dimension:
# - Why A scores HIGH? Why A scores LOW?
# - Why B scores HIGH? Why B scores LOW?
#
# Be concise (2-3 sentences per dimension).
# """
#
#     debate = call_llm_api([
#         {"role": "system", "content": "Debater arguing both sides."},
#         {"role": "user", "content": [
#             {"type": "text", "text": deb_prompt},
#             {"type": "text", "text": "Image A:"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
#             {"type": "text", "text": "Image B:"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}}
#         ]}
#     ], temperature=0.5, max_tokens=600)
#
#     if not debate:
#         return None
#     time.sleep(0.5)
#
#     # Step 3: Judge
#     judge_prompt = f"""Final comparative judgment for "{category}".
#
# [OBSERVATION]
# {observation}
#
# [DEBATE]
# {debate}
#
# [DIMENSIONS]
# {dim_desc}
#
# Rate BOTH images, determine winner.
#
# Output JSON:
# {{
#   "image_a_scores": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
#   "image_b_scores": {{{', '.join([f'"{d}": score' for d in dim_names])}}},
#   "winner": "left"/"right"/"equal",
#   "confidence": 1-5
# }}
# """
#
#     return call_llm_api([
#         {"role": "system", "content": "Impartial judge. JSON only."},
#         {"role": "user", "content": [
#             {"type": "text", "text": judge_prompt},
#             {"type": "text", "text": "Image A:"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
#             {"type": "text", "text": "Image B:"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}}
#         ]}
#     ], temperature=0.1, response_format={"type": "json_object"}, max_tokens=600)
#
# # ==============================================================================
# # 7. 主函数
# # ==============================================================================
# def run_multi_mode_synthesis():
#     """Stage 2主流程：支持4种模式"""
#     MODE_NAMES = {
#         1: "Mode 1: Single Image Direct Scoring",
#         2: "Mode 2: Pairwise Direct Scoring",
#         3: "Mode 3: Single Image Multi-Agent",
#         4: "Mode 4: Pairwise Multi-Agent Deliberation"
#     }
#
#     print("\n" + "="*80)
#     print(f"UrbanAlign 2.0 - Stage 2: Multi-Mode Synthesis")
#     print(f"当前模式: {MODE_NAMES[SYNTHESIS_MODE]}")
#     print("="*80 + "\n")
#
#     # 1. 加载维度定义
#     print("[STEP 1] 加载语义维度...")
#     dimensions_all = load_dimensions()
#     if not dimensions_all:
#         return
#
#     # 2. 加载CLIP特征
#     print("\n[STEP 2] 加载CLIP特征...")
#     clip_map = load_clip_features()
#
#     # 3. 加载ID映射
#     print("\n[STEP 3] 加载ID映射...")
#     id_mapping_df = pd.read_csv(ID_MAPPING_CSV)
#     id_to_alias = dict(zip(id_mapping_df['original_id'].astype(str),
#                           id_mapping_df['alias_id']))
#
#     # 4. 数据集采样与物理隔离
#     print("\n[STEP 4] 数据集采样与物理隔离...")
#
#     # 4.1 采样Synthesis Pool (用于本阶段实验)
#     if os.path.exists(SAMPLED_PAIRS_FILE):
#         sampled_pairs = pd.read_csv(SAMPLED_PAIRS_FILE)
#         print(f"  加载已有采样Pool: {len(sampled_pairs)}对")
#     else:
#         temp_list = []
#         for cat in CATEGORIES:
#             df_ref, df_pool = get_split_data(category=cat)
#             n = min(len(df_pool), int(len(df_pool) * N_POOL_MULTIPLIER))
#             sampled_cat = df_pool.sample(n=n, random_state=42)
#             temp_list.append(sampled_cat)
#         sampled_pairs = pd.concat(temp_list, ignore_index=True)
#         sampled_pairs.rename(columns={'winner': 'human_winner'}, inplace=True)
#         sampled_pairs.to_csv(SAMPLED_PAIRS_FILE, index=False)
#         print(f"  ✓ 已创建Pool采样: {len(sampled_pairs)}对")
#
#     # 4.2 统一VRM标尺采样与标注 (抽取最符合Pool分布的Reference)
#     if os.path.exists(UNIFIED_VRM_REF_FILE):
#         print(f"  加载当前模式专用的VRM标尺结果: {UNIFIED_VRM_REF_FILE}")
#         df_vrm_ref = pd.read_csv(UNIFIED_VRM_REF_FILE)
#     else:
#         # 4.2.1 确定全局共享的映射关系 (基于CLIP拼接特征 [L, R] 的双向精准匹配)
#         if not os.path.exists(GLOBAL_MAPPING_FILE):
#             print(f"  正在生成全局一致的映射关系 (基于CLIP拼接特征 [L, R] 的双向匹配)...")
#             vrm_ref_indices_total = []
#
#             for cat in CATEGORIES:
#                 df_ref, _ = get_split_data(category=cat)
#                 pool_cat = sampled_pairs[sampled_pairs['category'] == cat]
#
#                 if len(pool_cat) == 0: continue
#
#                 # 【核心增强】：双向特征提取，保留镜像相似性
#                 def get_bidirectional_features(df):
#                     feats_std = []  # 正向 [f_l, f_r]
#                     feats_flip = [] # 镜像 [f_r, f_l]
#                     idxs = []
#                     for idx, row in df.iterrows():
#                         f_l = clip_map.get(str(row['left_id']))
#                         f_r = clip_map.get(str(row['right_id']))
#                         if f_l is not None and f_r is not None:
#                             # 归一化后拼接，确保长度一致且语义对齐
#                             v_l, v_r = np.array(f_l), np.array(f_r)
#                             v_l = v_l / np.linalg.norm(v_l)
#                             v_r = v_r / np.linalg.norm(v_r)
#
#                             feats_std.append(np.concatenate([v_l, v_r]))
#                             feats_flip.append(np.concatenate([v_r, v_l]))
#                             idxs.append(idx)
#                     return np.array(feats_std), np.array(feats_flip), idxs
#
#                 ref_std, ref_flip, ref_idxs = get_bidirectional_features(df_ref)
#                 pool_std, _, _ = get_bidirectional_features(pool_cat)
#
#                 if len(ref_std) > 0 and len(pool_std) > 0:
#                     # 计算 Pool(A,B) 对 Ref(L,R) 和 Ref(R,L) 的相似度
#                     sim_std = cosine_similarity(pool_std, ref_std)
#                     sim_flip = cosine_similarity(pool_std, ref_flip)
#
#                     # 取两者最大值，即找到了视觉关系最像的邻居
#                     final_sim_matrix = np.maximum(sim_std, sim_flip)
#
#                     best_ref_local_indices = np.argmax(final_sim_matrix, axis=1)
#                     # 映射回原始 ID 并去重
#                     unique_best_ref_idxs = np.unique([ref_idxs[i] for i in best_ref_local_indices])
#                     vrm_ref_indices_total.append(df_ref.loc[unique_best_ref_idxs])
#
#             df_vrm_to_score = pd.concat(vrm_ref_indices_total, ignore_index=True)
#             df_vrm_to_score.to_csv(GLOBAL_MAPPING_FILE, index=False)
#             print(f"  ✓ 全局映射已保存: {len(df_vrm_to_score)}对参考样本 (含镜像逻辑)")
#         else:
#             df_vrm_to_score = pd.read_csv(GLOBAL_MAPPING_FILE)
#             print(f"  加载全局共享映射关系: {len(df_vrm_to_score)}对")
#
#         # 4.2.2 使用“当前模式”的逻辑为选中的标尺对进行打分
#         print(f"  正在使用 {MODE_NAMES[SYNTHESIS_MODE]} 为VRM标尺打分...")
#         vrm_results = []
#         for _, row in tqdm(df_vrm_to_score.iterrows(), total=len(df_vrm_to_score), desc=f"Mode {SYNTHESIS_MODE} Ref Scoring"):
#             l_id, r_id, cat = str(row['left_id']), str(row['right_id']), row['category']
#             img_a, img_b = os.path.join(IMAGE_DIR, f"{l_id}.jpg"), os.path.join(IMAGE_DIR, f"{r_id}.jpg")
#             dimensions = dimensions_all.get(cat, {})
#
#             # 严格根据当前模式调用函数，确保标尺与测试工具一致
#             try:
#                 if SYNTHESIS_MODE == 1:
#                     resp_a = mode1_single_direct(img_a, cat, dimensions)
#                     resp_b = mode1_single_direct(img_b, cat, dimensions)
#                     if resp_a and resp_b:
#                         s_a, s_b = json.loads(resp_a), json.loads(resp_b)
#                         # 基于语义向量判定
#                         val_a, val_b = sum(s_a.values()), sum(s_b.values())
#                         vrm_results.append({
#                             'left_id': l_id, 'right_id': r_id, 'category': cat,
#                             'human_winner': row['winner'],
#                             'image_a_scores': json.dumps(s_a), 'image_b_scores': json.dumps(s_b),
#                             'synthetic_winner': 'left' if val_a > val_b else ('right' if val_a < val_b else 'equal')
#                         })
#                 elif SYNTHESIS_MODE == 2:
#                     resp = mode2_pair_direct(img_a, img_b, cat, dimensions)
#                     if resp:
#                         res = json.loads(resp)
#                         vrm_results.append({
#                             'left_id': l_id, 'right_id': r_id, 'category': cat,
#                             'human_winner': row['winner'],
#                             'image_a_scores': json.dumps(res.get('image_a_scores', {})),
#                             'image_b_scores': json.dumps(res.get('image_b_scores', {})),
#                             'synthetic_winner': res.get('winner', 'equal')
#                         })
#                 elif SYNTHESIS_MODE == 3:
#                     resp_a = mode3_single_multiagent(img_a, cat, dimensions)
#                     resp_b = mode3_single_multiagent(img_b, cat, dimensions)
#                     if resp_a and resp_b:
#                         s_a, s_b = json.loads(resp_a), json.loads(resp_b)
#                         val_a, val_b = sum(s_a.values()), sum(s_b.values())
#                         vrm_results.append({
#                             'left_id': l_id, 'right_id': r_id, 'category': cat,
#                             'human_winner': row['winner'],
#                             'image_a_scores': json.dumps(s_a), 'image_b_scores': json.dumps(s_b),
#                             'synthetic_winner': 'left' if val_a > val_b else ('right' if val_a < val_b else 'equal')
#                         })
#                 elif SYNTHESIS_MODE == 4:
#                     resp = mode4_pair_multiagent(img_a, img_b, cat, dimensions)
#                     if resp:
#                         res = json.loads(resp)
#                         vrm_results.append({
#                             'left_id': l_id, 'right_id': r_id, 'category': cat,
#                             'human_winner': row['winner'],
#                             'image_a_scores': json.dumps(res.get('image_a_scores', {})),
#                             'image_b_scores': json.dumps(res.get('image_b_scores', {})),
#                             'synthetic_winner': res.get('winner', 'equal')
#                         })
#             except: pass
#
#         pd.DataFrame(vrm_results).to_csv(UNIFIED_VRM_REF_FILE, index=False)
#         print(f"  ✓ 当前模式标尺已保存: {os.path.basename(UNIFIED_VRM_REF_FILE)}")
#
#     # 5. 断点续传
#     print(f"\n[STEP 5] 开始正式合成（{MODE_NAMES[SYNTHESIS_MODE]}）...")
#     if os.path.exists(OUTPUT_FILE):
#         df_old = pd.read_csv(OUTPUT_FILE)
#         done = set(zip(df_old['left_id'].astype(str), df_old['right_id'].astype(str), df_old['category']))
#         results = df_old.to_dict('records')
#         print(f"  已完成: {len(df_old)}对，继续...")
#     else:
#         done = set()
#         results = []
#
#     tasks = [row for _, row in sampled_pairs.iterrows()
#              if (str(row['left_id']), str(row['right_id']), row['category']) not in done]
#     print(f"  待处理: {len(tasks)}对")
#
#     # 6. 根据Mode执行正式任务
#     for i, row in enumerate(tqdm(tasks, desc=f"Mode {SYNTHESIS_MODE} Pool")):
#         l_id, r_id, cat = str(row['left_id']), str(row['right_id']), row['category']
#         img_a, img_b = os.path.join(IMAGE_DIR, f"{l_id}.jpg"), os.path.join(IMAGE_DIR, f"{r_id}.jpg")
#
#         if not (os.path.exists(img_a) and os.path.exists(img_b)): continue
#         dimensions = dimensions_all.get(cat, {})
#         if 'dimensions' not in dimensions: continue
#
#         # 调用对应模式逻辑
#         if SYNTHESIS_MODE == 1:
#             response_a = mode1_single_direct(img_a, cat, dimensions)
#             response_b = mode1_single_direct(img_b, cat, dimensions)
#             if response_a and response_b:
#                 try:
#                     s_a, s_b = json.loads(response_a), json.loads(response_b)
#                     delta = sum(s_a.values()) - sum(s_b.values())
#                     # 采用软阈值 1.0 判定胜者，其余设为 equal
#                     winner = 'left' if delta > 1.0 else ('right' if delta < -1.0 else 'equal')
#                     results.append({
#                         'left_id': l_id, 'right_id': r_id, 'category': cat,
#                         'human_winner': row['human_winner'], 'synthetic_winner': winner,
#                         'image_a_scores': json.dumps(s_a), 'image_b_scores': json.dumps(s_b),
#                         'agreement': 1 if winner == row['human_winner'] else 0, 'mode': 1
#                     })
#                 except: pass
#
#         elif SYNTHESIS_MODE == 2:
#             response = mode2_pair_direct(img_a, img_b, cat, dimensions)
#             if response:
#                 try:
#                     res = json.loads(response)
#                     winner = res.get('winner', 'equal').lower()
#                     results.append({
#                         'left_id': l_id, 'right_id': r_id, 'category': cat,
#                         'human_winner': row['human_winner'], 'synthetic_winner': winner,
#                         'image_a_scores': json.dumps(res.get('image_a_scores', {})),
#                         'image_b_scores': json.dumps(res.get('image_b_scores', {})),
#                         'agreement': 1 if winner == row['human_winner'] else 0, 'mode': 2
#                     })
#                 except: pass
#
#         elif SYNTHESIS_MODE == 3:
#             response_a = mode3_single_multiagent(img_a, cat, dimensions)
#             response_b = mode3_single_multiagent(img_b, cat, dimensions)
#             if response_a and response_b:
#                 try:
#                     s_a, s_b = json.loads(response_a), json.loads(response_b)
#                     delta = sum(s_a.values()) - sum(s_b.values())
#                     winner = 'left' if delta > 1.0 else ('right' if delta < -1.0 else 'equal')
#                     results.append({
#                         'left_id': l_id, 'right_id': r_id, 'category': cat,
#                         'human_winner': row['human_winner'], 'synthetic_winner': winner,
#                         'image_a_scores': json.dumps(s_a), 'image_b_scores': json.dumps(s_b),
#                         'agreement': 1 if winner == row['human_winner'] else 0, 'mode': 3
#                     })
#                 except: pass
#
#         elif SYNTHESIS_MODE == 4:
#             response = mode4_pair_multiagent(img_a, img_b, cat, dimensions)
#             if response:
#                 try:
#                     res = json.loads(response)
#                     winner = res.get('winner', 'equal').lower()
#                     results.append({
#                         'left_id': l_id, 'right_id': r_id, 'category': cat,
#                         'human_winner': row['human_winner'], 'synthetic_winner': winner,
#                         'image_a_scores': json.dumps(res.get('image_a_scores', {})),
#                         'image_b_scores': json.dumps(res.get('image_b_scores', {})),
#                         'confidence': res.get('confidence', 3),
#                         'agreement': 1 if winner == row['human_winner'] else 0, 'mode': 4
#                     })
#                 except: pass
#
#         # 每 50 对保存，防止程序意外中断
#         if (i + 1) % 50 == 0:
#             pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
#         time.sleep(1.0 if SYNTHESIS_MODE in [1, 2] else 2.0)
#
#     # 最终保存与统计
#     df_final = pd.DataFrame(results)
#     df_final.to_csv(OUTPUT_FILE, index=False)
#
#     print(f"\n{'='*80}\nStage 2完成！（{MODE_NAMES[SYNTHESIS_MODE]}）\n{'='*80}")
#     print(f"  测试集输出: {os.path.basename(OUTPUT_FILE)}")
#     print(f"  标尺集输出: {os.path.basename(UNIFIED_VRM_REF_FILE)}")
#
#     if len(df_final) > 0 and 'agreement' in df_final.columns:
#         acc = accuracy_score(df_final['human_winner'], df_final['synthetic_winner'])
#         print(f"\n  【性能统计】\n    准确率: {acc*100:.2f}%\n    样本总数: {len(df_final)}对")
#
#     print(f"\n下一步: python abc_stage3_hybrid_vrm.py")
#
# if __name__ == "__main__":
#     run_multi_mode_synthesis()