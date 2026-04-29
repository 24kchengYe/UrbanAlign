"""
UrbanAlign 2.0 - Stage 3: Hybrid Visual Relationship Mapping (Local Weight Solver 深度增强版)
混合特征空间的局部语义权重解析与对齐

学术核心逻辑：
1. 混合寻邻：在 [CLIP_L, CLIP_R, AI_Delta_Vector] 构成的 N-维混合差分空间寻找视觉与理由双重近邻。
2. 局部权重拟合 (LWRR)：通过 RBF 加权岭回归，在每一个局部流形区域解算维度权重 w，
   实现 Human_TrueSkill_Delta ≈ Σ (w_i * AI_Semantic_Delta_i)。
3. 判定重构 (Re-inference)：结合拟合值（连续信号）与邻居共识（统计信号）重新划分胜负平。
4. 深度审计：输出包含解释力、一致性、标签分布的多维评估报告。
"""
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 导入配置
# ==============================================================================
from config import (
    OUTPUT_DIR, CLIP_CACHE,
    STAGE2_MODE,
    CATEGORIES, SELECTION_RATIO, ALPHA_HYBRID,
    K_MAX_ST3, TAU_KERNEL_ST3, RIDGE_ALPHA_ST3, EQUAL_EPS_ST3, EQUAL_CONSENSUS_MIN,
    get_split_data, get_optimal_lwrr_params,
    get_trueskill_cache, get_stage2_output, get_stage3_output,
    get_stage2_sampled_pairs
)

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
def parse_semantic_vector(score_json):
    """解析语义分数为有序向量 (支持 N-维动态适配)"""
    try:
        if pd.isna(score_json): return None
        data = json.loads(score_json) if isinstance(score_json, str) else score_json
        if not data: return None
        # 严格按键名排序，确保向量空间维度对齐
        keys = sorted(data.keys())
        return np.array([float(data[k]) for k in keys])
    except:
        return None

def load_trueskill_map(category):
    """加载预计算的人类直觉 Mu 值标尺（per-category）"""
    ts_file = get_trueskill_cache(category)
    if not os.path.exists(ts_file):
        print(f"[ERROR] TrueSkill 缓存缺失: {ts_file}")
        return {}
    df = pd.read_csv(ts_file)
    return dict(zip(zip(df['category'], df['image_id'].astype(str)), df['mu']))

def load_clip_features():
    """加载并解析 CLIP 特征映射"""
    if not os.path.exists(CLIP_CACHE):
        print(f"[ERROR] CLIP 缓存缺失: {CLIP_CACHE}")
        return {}
    data = np.load(CLIP_CACHE)
    return {os.path.splitext(os.path.basename(str(p)))[0]: v for p, v in zip(data['paths'], data['embeddings'])}

def calculate_metrics(df, label_col='synthetic_winner', gt_col='human_winner', exclude_equal=False):
    """多维性能指标计算"""
    temp_df = df.copy()
    if exclude_equal:
        temp_df = temp_df[(temp_df[label_col] != 'equal') & (temp_df[gt_col] != 'equal')]

    if len(temp_df) == 0: return 0, 0

    acc = accuracy_score(temp_df[gt_col], temp_df[label_col])
    try:
        kappa = cohen_kappa_score(temp_df[gt_col], temp_df[label_col])
    except:
        kappa = 0
    return acc, kappa

# ==============================================================================
# 3. 主程序：Local Weight Alignment (LWRR)
# ==============================================================================
def run_local_weight_alignment():
    print("\n" + "="*80)
    print(f"UrbanAlign 2.0 - Stage 3: Local Weight Mapping (Mode {STAGE2_MODE})")
    print("="*80)
    print(f"  解耦架构: 从统一打分文件动态拆分 Ref/Pool")

    # 1. 资源准备（CLIP特征全局共享）
    clip_map = load_clip_features()

    # 2. 按类别执行：动态拆分 + 解构对齐
    for cat in CATEGORIES:
        stage2_scored_file = get_stage2_output(STAGE2_MODE, cat)
        stage3_output = get_stage3_output(STAGE2_MODE, cat)

        # 跳过已完成的类别 (自动检测上游数据是否更新)
        if os.path.exists(stage3_output):
            stage3_mtime = os.path.getmtime(stage3_output)
            upstream_newer = False
            # 检查 Stage 2 打分文件是否更新
            if os.path.exists(stage2_scored_file) and os.path.getmtime(stage2_scored_file) > stage3_mtime:
                upstream_newer = True
            # 检查采样缓存是否更新 (Ref/Pool 划分可能变了)
            sampled_cache = get_stage2_sampled_pairs(cat)
            if os.path.exists(sampled_cache) and os.path.getmtime(sampled_cache) > stage3_mtime:
                upstream_newer = True
            # 检查 Stage 5 最优参数是否更新 (参数寻优后需重新对齐)
            best_params_file = os.path.join(OUTPUT_DIR, f'stage5_best_params_{cat}.json')
            if os.path.exists(best_params_file) and os.path.getmtime(best_params_file) > stage3_mtime:
                upstream_newer = True
            if upstream_newer:
                print(f"\n[STALE] {cat}: 上游数据已更新, 删除旧输出重新对齐...")
                os.remove(stage3_output)
            else:
                print(f"\n[SKIP] {cat}: {os.path.basename(stage3_output)} already exists")
                continue

        if not os.path.exists(stage2_scored_file):
            print(f"\n[SKIP] {cat}: Stage 2 打分文件缺失 {os.path.basename(stage2_scored_file)}")
            continue

        # 加载该类别的最优参数 (Stage 5 优化结果 或 全局默认)
        cat_params, params_source = get_optimal_lwrr_params(cat)
        cat_K_MAX = cat_params['K_MAX_ST3']
        cat_TAU = cat_params['TAU_KERNEL_ST3']
        cat_RIDGE = cat_params['RIDGE_ALPHA_ST3']
        cat_EPS = cat_params['EQUAL_EPS_ST3']
        cat_CONSENSUS = cat_params['EQUAL_CONSENSUS_MIN']
        cat_ALPHA = cat_params['ALPHA_HYBRID']
        cat_SELECTION = cat_params['SELECTION_RATIO']

        ts_map = load_trueskill_map(cat)
        df_scored = pd.read_csv(stage2_scored_file)
        print(f"\n  [{cat}] Stage 2 打分数据: {len(df_scored)}对 (params: {params_source})")
        if params_source == 'optimized':
            print(f"    优化参数: K={cat_K_MAX}, τ={cat_TAU}, λ={cat_RIDGE}, ε={cat_EPS}, α={cat_ALPHA}, sel={cat_SELECTION}")

        final_aligned_results = []
        df_scored_cat = df_scored[df_scored['category'] == cat].copy() if 'category' in df_scored.columns else df_scored.copy()
        if len(df_scored_cat) == 0:
            continue

        # 2.0 动态拆分：用 get_split_data() 的索引匹配已打分数据
        df_ref_split, df_pool_split = get_split_data(category=cat)

        # 构建匹配键集合
        ref_keys = set(zip(df_ref_split['left_id'].astype(str), df_ref_split['right_id'].astype(str)))
        pool_keys = set(zip(df_pool_split['left_id'].astype(str), df_pool_split['right_id'].astype(str)))

        # 从已打分数据中匹配出 Ref 和 Pool
        scored_keys = list(zip(df_scored_cat['left_id'].astype(str), df_scored_cat['right_id'].astype(str)))
        ref_mask = [k in ref_keys for k in scored_keys]
        pool_mask = [k in pool_keys for k in scored_keys]

        r_cat = df_scored_cat[ref_mask].copy()
        p_cat = df_scored_cat[pool_mask].copy()

        if len(r_cat) == 0 or len(p_cat) == 0:
            print(f"\n  [{cat}] Ref={len(r_cat)}对, Pool={len(p_cat)}对 — 数据不足，跳过")
            continue

        print(f"\n  [{cat}] Ref={len(r_cat)}对 (LWRR标尺), Pool={len(p_cat)}对 (待校准)")

        # ----------------------------------------------------------------------
        # 2.1 数据流形构建 (Manifold Data Preparation)
        # ----------------------------------------------------------------------
        def prepare_manifold_data(df, is_ref=False):
            coords, sem_deltas, targets, meta = [], [], [], []
            valid_iloc_indices = []  # 记录成功构建流形数据的行在df中的iloc位置
            for iloc_pos, (idx, r) in enumerate(df.iterrows()):
                l_id, r_id = str(r['left_id']), str(r['right_id'])
                f_l, f_r = clip_map.get(l_id), clip_map.get(r_id)
                s_l = parse_semantic_vector(r.get('image_a_scores'))
                s_r = parse_semantic_vector(r.get('image_b_scores'))

                if any(x is None for x in [f_l, f_r, s_l, s_r]): continue

                # 理由向量：N-维语义分差
                s_diff = s_l - s_r

                # 混合空间坐标：[CLIP_L, CLIP_R, S_diff]，按cat_ALPHA加权
                clip_part = np.concatenate([
                    normalize(f_l.reshape(1,-1))[0],
                    normalize(f_r.reshape(1,-1))[0]
                ])
                sem_part = normalize(s_diff.reshape(1,-1))[0]
                v_coord = np.concatenate([
                    cat_ALPHA * clip_part,
                    (1 - cat_ALPHA) * sem_part
                ])

                if is_ref:
                    # 引导信号：TrueSkill Delta
                    mu_l, mu_r = ts_map.get((cat, l_id), 25.0), ts_map.get((cat, r_id), 25.0)
                    h_delta = mu_l - mu_r

                    # 存储标准对
                    coords.append(v_coord)
                    sem_deltas.append(s_diff)
                    targets.append(h_delta)
                    meta.append({'orig_idx': idx, 'label': str(r['human_winner']).lower()})

                    # 镜像增强 (Mirror Invariance Symmetry)
                    clip_flip = np.concatenate([
                        normalize(f_r.reshape(1,-1))[0],
                        normalize(f_l.reshape(1,-1))[0]
                    ])
                    sem_flip = normalize((-s_diff).reshape(1,-1))[0]
                    v_flip = np.concatenate([
                        cat_ALPHA * clip_flip,
                        (1 - cat_ALPHA) * sem_flip
                    ])
                    coords.append(v_flip)
                    sem_deltas.append(-s_diff)
                    targets.append(-h_delta)
                    # 翻转人类标签
                    h_win_flip = 'right' if r['human_winner'] == 'left' else ('left' if r['human_winner'] == 'right' else 'equal')
                    meta.append({'orig_idx': idx, 'label': h_win_flip})
                else:
                    coords.append(v_coord)
                    sem_deltas.append(s_diff)
                    valid_iloc_indices.append(iloc_pos)
            return np.array(coords), np.array(sem_deltas), np.array(targets), meta, valid_iloc_indices

        ref_coords, ref_S_diff, ref_y_ts, ref_meta, _ = prepare_manifold_data(r_cat, is_ref=True)
        pool_coords, pool_S_diff, _, _, pool_valid_indices = prepare_manifold_data(p_cat, is_ref=False)

        if len(ref_coords) == 0 or len(pool_coords) == 0: continue

        # ----------------------------------------------------------------------
        # 2.2 局部加权岭回归 (Locally Weighted Ridge Regression)
        # ----------------------------------------------------------------------
        sim_matrix = cosine_similarity(pool_coords, ref_coords)

        cat_aligned_rows = []
        for i in tqdm(range(len(pool_coords)), desc=f"Aligning {cat}"):
            # 搜索 K 个近邻 (邻居数量参数化)
            k = min(cat_K_MAX, len(ref_coords))
            top_k_idx = np.argsort(sim_matrix[i])[-k:][::-1]

            # 获取局部理由空间与引导信号
            local_X = ref_S_diff[top_k_idx]
            local_y = ref_y_ts[top_k_idx]

            # 核权重 (RBF Kernel)：距离越近，对逻辑权重的贡献越大
            neighbor_sims = sim_matrix[i][top_k_idx]
            weights = np.exp(neighbor_sims / cat_TAU)

            # 拟合局部投影函数
            model = Ridge(alpha=cat_RIDGE)
            model.fit(local_X, local_y, sample_weight=weights)

            # 解构出的该区域感知权重向量 (w)
            real_weights = model.coef_

            # 拟合 Pool 样本的 TrueSkill 分差
            fitted_delta = model.predict(pool_S_diff[i].reshape(1, -1))[0]

            # --- 统计稳健型重新判定 (Refined Re-inference) ---
            neighbor_labels = [ref_meta[idx]['label'] for idx in top_k_idx]
            equal_consensus = neighbor_labels.count('equal') / len(neighbor_labels)

            # 逻辑：如果拟合值过小，或者邻域内大部分人类判定为平局，则强制判定为 Equal
            if abs(fitted_delta) < cat_EPS or equal_consensus > cat_CONSENSUS:
                new_winner = 'equal'
            else:
                new_winner = 'left' if fitted_delta > 0 else 'right'

            # 置信度复合指标 (Confidence Composite)
            # NAR: 邻居中与预测标签一致的比例
            neighbor_agreement = sum(1 for lbl in neighbor_labels if lbl == new_winner) / len(neighbor_labels)
            # PM: 拟合分差相对于平局阈值的归一化边距
            prediction_margin = min(1.0, abs(fitted_delta) / (cat_EPS * 3.0))
            confidence = 0.5 * neighbor_agreement + 0.5 * prediction_margin

            # 封装结果 (用 pool_valid_indices 映射回 p_cat 的正确行)
            row_data = p_cat.iloc[pool_valid_indices[i]].to_dict()
            row_data.update({
                'synthetic_winner': new_winner,
                'fitted_ts_delta': fitted_delta,
                'confidence_score': confidence,
                'neighbor_agreement': neighbor_agreement,
                'prediction_margin': prediction_margin,
                'local_weights': json.dumps(real_weights.tolist()),
                'equal_consensus': equal_consensus
            })
            cat_aligned_rows.append(row_data)

        # ----------------------------------------------------------------------
        # 2.3 高解释力筛选 (Fidelity Filtering)
        # ----------------------------------------------------------------------
        df_cat_aligned = pd.DataFrame(cat_aligned_rows)
        # 仅保留置信度最高的样本
        n_keep = max(1, int(len(df_cat_aligned) * cat_SELECTION))
        df_cat_final = df_cat_aligned.sort_values('confidence_score', ascending=False).head(n_keep)
        final_aligned_results.append(df_cat_final)

        # 3. 保存该类别的对齐结果
        if not final_aligned_results: continue
        df_final = pd.concat(final_aligned_results, ignore_index=True)

        # 物理保存
        df_save = df_final.copy()
        for col in ['image_a_scores', 'image_b_scores']:
            if col in df_save.columns:
                df_save[col] = df_save[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
        df_save.to_csv(stage3_output, index=False)

        # --- 生成评估总结报告 ---
        print(f"\n  " + "═"*60)
        print(f"  Stage 3 评估总结 ({cat} - Mode {STAGE2_MODE})")
        print(f"  " + "═"*60)

        overall_acc, overall_kappa = calculate_metrics(df_final)
        pure_acc, _ = calculate_metrics(df_final, exclude_equal=True)

        print(f"    总样本数: {len(df_final):<6} | 置信度 (Avg Confidence): {df_final['confidence_score'].mean():.3f}")
        print(f"    总体准确率: {overall_acc*100:.2f}% | 排除平局准确率: {pure_acc*100:.2f}%")
        print(f"    一致性 (Kappa): {overall_kappa:.3f}")

        dist = df_final['synthetic_winner'].value_counts(normalize=True)
        for label in ['left', 'right', 'equal']:
            print(f"      - {label.capitalize():<6}: {dist.get(label, 0)*100:5.1f}%")

        print(f"    已保存: {os.path.basename(stage3_output)}")

if __name__ == "__main__":
    run_local_weight_alignment()