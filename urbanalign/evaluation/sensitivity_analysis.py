"""
UrbanAlign 2.0 - Stage 5: 参数敏感性分析 (Sensitivity Analysis)

核心思想:
  所有对齐阶段参数的调优都不需要重跑 Stage 2 的 API 调用 (成本为0)。
  本模块对关键参数做网格搜索, 输出每种参数组合的准确率/Kappa,
  本身就构成论文中的参数敏感性分析表。

支持两类参数搜索:
  1. ST2_INTENSITY_SIG_THRESH: 从已存储的 overall_intensity_a/b 重新计算 winner
     (甚至不需要重跑 Stage 3, 仅重新判定)
  2. LWRR 参数组 (K_MAX, TAU, ALPHA, RIDGE_ALPHA, EQUAL_EPS, EQUAL_CONSENSUS, SELECTION_RATIO):
     需要重跑 Stage 3 对齐循环 (纯CPU计算, 无API调用)

运行方式:
  python abc_stage5_sensitivity_analysis.py
"""
import pandas as pd
import numpy as np
import os
import json
import itertools
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

from urbanalign.config import (
    OUTPUT_DIR, CLIP_CACHE,
    STAGE2_MODE,
    CATEGORIES, SENSITIVITY_GRID,
    ALPHA_HYBRID, K_MAX_ST3, TAU_KERNEL_ST3, RIDGE_ALPHA_ST3,
    EQUAL_EPS_ST3, EQUAL_CONSENSUS_MIN, SELECTION_RATIO,
    ST2_INTENSITY_SIG_THRESH,
    N_RANDOM_SEARCH,
    LABELED_SET_RATIO,
    get_split_data,
    get_trueskill_cache, get_stage2_output, get_human_choices_csv,
    get_stage5_output,
)

# ==============================================================================
# 1. 辅助函数 (复用 Stage 3 的核心逻辑)
# ==============================================================================
def parse_semantic_vector(score_json):
    try:
        if pd.isna(score_json): return None
        data = json.loads(score_json) if isinstance(score_json, str) else score_json
        if not data: return None
        keys = sorted(data.keys())
        return np.array([float(data[k]) for k in keys])
    except:
        return None

def load_trueskill_map():
    ts_map = {}
    for cat in CATEGORIES:
        ts_file = get_trueskill_cache(cat)
        if os.path.exists(ts_file):
            df = pd.read_csv(ts_file)
            ts_map.update(dict(zip(zip(df['category'], df['image_id'].astype(str)), df['mu'])))
    return ts_map

def load_clip_features():
    if not os.path.exists(CLIP_CACHE):
        return {}
    data = np.load(CLIP_CACHE)
    return {os.path.splitext(os.path.basename(str(p)))[0]: v for p, v in zip(data['paths'], data['embeddings'])}

def calculate_metrics(df, exclude_equal=False):
    temp = df.copy()
    if exclude_equal:
        temp = temp[(temp['synthetic_winner'] != 'equal') & (temp['human_winner'] != 'equal')]
    if len(temp) == 0:
        return 0, 0
    acc = accuracy_score(temp['human_winner'], temp['synthetic_winner'])
    try:
        kappa = cohen_kappa_score(temp['human_winner'], temp['synthetic_winner'])
    except:
        kappa = 0
    return acc, kappa


def bootstrap_ci(df, exclude_equal=False, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap 95% confidence interval for accuracy and kappa."""
    temp = df.copy()
    if exclude_equal:
        temp = temp[(temp['synthetic_winner'] != 'equal') & (temp['human_winner'] != 'equal')]
    if len(temp) < 2:
        return (0, 0, 0), (0, 0, 0)  # (acc_low, acc_mid, acc_high), (kappa_low, kappa_mid, kappa_high)

    rng = np.random.RandomState(seed)
    accs, kappas = [], []
    n = len(temp)
    y_true = temp['human_winner'].values
    y_pred = temp['synthetic_winner'].values

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        try:
            kappas.append(cohen_kappa_score(y_true[idx], y_pred[idx]))
        except:
            kappas.append(0)

    alpha = 1 - ci
    acc_arr = np.array(accs)
    kappa_arr = np.array(kappas)
    acc_ci = (np.percentile(acc_arr, alpha/2*100), np.median(acc_arr), np.percentile(acc_arr, (1-alpha/2)*100))
    kappa_ci = (np.percentile(kappa_arr, alpha/2*100), np.median(kappa_arr), np.percentile(kappa_arr, (1-alpha/2)*100))
    return acc_ci, kappa_ci


# ==============================================================================
# 2. ST2_INTENSITY_SIG_THRESH 敏感性分析
#    直接从存储的 overall_intensity_a/b 重新计算 winner, 无需重跑任何模型
# ==============================================================================
def analyze_st2_threshold(df_scored, thresholds):
    """
    对 Stage 2 输出数据, 用不同阈值重新判定 winner, 评估准确率。
    仅对 Mode 1/3 (单张模式) 有效 — 这些模式存储了 overall_intensity_a/b。
    Mode 2/4 的 winner 由 VLM 直接输出, 不受此阈值影响。
    """
    if 'overall_intensity_a' not in df_scored.columns or 'overall_intensity_b' not in df_scored.columns:
        print("[WARN] 数据中无 overall_intensity_a/b 列, 跳过 ST2 阈值分析")
        return []

    results = []
    for thresh in thresholds:
        df_test = df_scored.copy()
        ia = df_test['overall_intensity_a'].astype(float)
        ib = df_test['overall_intensity_b'].astype(float)

        df_test['synthetic_winner'] = np.where(
            ia > ib + thresh, 'left',
            np.where(ib > ia + thresh, 'right', 'equal')
        )

        for cat in df_test['category'].unique():
            df_cat = df_test[df_test['category'] == cat]
            acc, kappa = calculate_metrics(df_cat)
            acc_ne, kappa_ne = calculate_metrics(df_cat, exclude_equal=True)
            equal_pct = (df_cat['synthetic_winner'] == 'equal').mean()
            acc_ci, kappa_ci = bootstrap_ci(df_cat, exclude_equal=True)

            results.append({
                'param': 'ST2_INTENSITY_SIG_THRESH',
                'value': thresh,
                'category': cat,
                'accuracy': acc,
                'kappa': kappa,
                'accuracy_no_equal': acc_ne,
                'kappa_no_equal': kappa_ne,
                'acc_ci_low': acc_ci[0],
                'acc_ci_high': acc_ci[2],
                'kappa_ci_low': kappa_ci[0],
                'kappa_ci_high': kappa_ci[2],
                'equal_pct': equal_pct,
                'n_samples': len(df_cat)
            })

    return results


# ==============================================================================
# 3. LWRR 参数敏感性分析
#    重跑 Stage 3 对齐循环 (纯CPU, 无API)
# ==============================================================================
def build_manifold_data(df, clip_map, ts_map, cat, alpha_hybrid, is_ref=False):
    """构建混合空间流形数据 (可参数化 alpha_hybrid)

    Returns:
        coords, sem_deltas, targets, meta, valid_iloc_indices
        valid_iloc_indices: pool行在原始df中的iloc位置 (用于run_lwrr_with_params回查p_cat)
    """
    coords, sem_deltas, targets, meta = [], [], [], []
    valid_iloc_indices = []
    for iloc_pos, (idx, r) in enumerate(df.iterrows()):
        l_id, r_id = str(r['left_id']), str(r['right_id'])
        f_l, f_r = clip_map.get(l_id), clip_map.get(r_id)
        s_l = parse_semantic_vector(r.get('image_a_scores'))
        s_r = parse_semantic_vector(r.get('image_b_scores'))

        if any(x is None for x in [f_l, f_r, s_l, s_r]):
            continue

        s_diff = s_l - s_r
        clip_part = np.concatenate([
            normalize(f_l.reshape(1, -1))[0],
            normalize(f_r.reshape(1, -1))[0]
        ])
        sem_part = normalize(s_diff.reshape(1, -1))[0]
        v_coord = np.concatenate([
            alpha_hybrid * clip_part,
            (1 - alpha_hybrid) * sem_part
        ])

        if is_ref:
            mu_l = ts_map.get((cat, l_id), 25.0)
            mu_r = ts_map.get((cat, r_id), 25.0)
            h_delta = mu_l - mu_r

            coords.append(v_coord)
            sem_deltas.append(s_diff)
            targets.append(h_delta)
            meta.append({'orig_idx': idx, 'label': str(r['human_winner']).lower()})

            # 镜像增强
            clip_flip = np.concatenate([
                normalize(f_r.reshape(1, -1))[0],
                normalize(f_l.reshape(1, -1))[0]
            ])
            sem_flip = normalize((-s_diff).reshape(1, -1))[0]
            v_flip = np.concatenate([
                alpha_hybrid * clip_flip,
                (1 - alpha_hybrid) * sem_flip
            ])
            coords.append(v_flip)
            sem_deltas.append(-s_diff)
            targets.append(-h_delta)
            h_win_flip = 'right' if r['human_winner'] == 'left' else (
                'left' if r['human_winner'] == 'right' else 'equal')
            meta.append({'orig_idx': idx, 'label': h_win_flip})
        else:
            coords.append(v_coord)
            sem_deltas.append(s_diff)
            valid_iloc_indices.append(iloc_pos)

    if len(coords) == 0:
        return np.array([]), np.array([]), np.array([]), [], []
    return np.array(coords), np.array(sem_deltas), np.array(targets) if targets else np.array([]), meta, valid_iloc_indices


def run_lwrr_with_params(ref_coords, ref_S_diff, ref_y_ts, ref_meta,
                         pool_coords, pool_S_diff, p_cat,
                         k_max, tau, ridge_alpha, equal_eps, equal_consensus_min,
                         selection_ratio, pool_valid_indices=None):
    """
    用指定参数运行 LWRR 对齐, 返回对齐后的 DataFrame。
    核心循环与 abc_stage3_hybrid_vrm.py 一致, 但参数从外部传入。

    pool_valid_indices: build_manifold_data返回的有效iloc索引列表。
        当build_manifold_data过滤掉缺失特征的行时, pool_coords[i]对应p_cat.iloc[pool_valid_indices[i]]。
        若为None, 退化为p_cat.iloc[i] (假设无行被过滤)。
    """
    sim_matrix = cosine_similarity(pool_coords, ref_coords)
    aligned_rows = []

    for i in range(len(pool_coords)):
        k = min(k_max, len(ref_coords))
        top_k_idx = np.argsort(sim_matrix[i])[-k:][::-1]

        local_X = ref_S_diff[top_k_idx]
        local_y = ref_y_ts[top_k_idx]

        neighbor_sims = sim_matrix[i][top_k_idx]
        weights = np.exp(neighbor_sims / tau)

        model = Ridge(alpha=ridge_alpha)
        model.fit(local_X, local_y, sample_weight=weights)

        fitted_delta = model.predict(pool_S_diff[i].reshape(1, -1))[0]

        neighbor_labels = [ref_meta[idx]['label'] for idx in top_k_idx]
        equal_consensus = neighbor_labels.count('equal') / len(neighbor_labels)

        if abs(fitted_delta) < equal_eps or equal_consensus > equal_consensus_min:
            new_winner = 'equal'
        else:
            new_winner = 'left' if fitted_delta > 0 else 'right'

        # 置信度复合指标
        neighbor_agreement = sum(1 for lbl in neighbor_labels if lbl == new_winner) / len(neighbor_labels)
        prediction_margin = min(1.0, abs(fitted_delta) / (equal_eps * 3.0))
        confidence = 0.5 * neighbor_agreement + 0.5 * prediction_margin

        iloc_idx = pool_valid_indices[i] if pool_valid_indices is not None else i
        row_data = p_cat.iloc[iloc_idx].to_dict()
        row_data['synthetic_winner'] = new_winner
        row_data['confidence_score'] = confidence
        aligned_rows.append(row_data)

    if not aligned_rows:
        return pd.DataFrame()

    df_aligned = pd.DataFrame(aligned_rows)

    # confidence filtering
    n_keep = max(1, int(len(df_aligned) * selection_ratio))
    df_aligned = df_aligned.sort_values('confidence_score', ascending=False).head(n_keep)

    return df_aligned


def analyze_single_lwrr_param(param_name, param_values,
                              precomputed_data, default_params):
    """
    对单个 LWRR 参数做敏感性分析 (其他参数固定为默认值)。

    param_name: 参数名
    param_values: 待搜索的值列表
    precomputed_data: dict, 包含每个category的预计算流形数据
    default_params: dict, 默认参数值
    """
    results = []

    for val in tqdm(param_values, desc=f"Sweep {param_name}"):
        params = default_params.copy()
        params[param_name] = val

        # ALPHA_HYBRID 变化需要重建流形 (坐标依赖alpha)
        need_rebuild = (param_name == 'ALPHA_HYBRID')

        for cat, cat_data in precomputed_data.items():
            if need_rebuild:
                # 重建流形数据
                ref_coords, ref_S_diff, ref_y_ts, ref_meta, _ = build_manifold_data(
                    cat_data['r_cat'], cat_data['clip_map'], cat_data['ts_map'],
                    cat, val, is_ref=True
                )
                pool_coords, pool_S_diff, _, _, pool_valid_indices = build_manifold_data(
                    cat_data['p_cat'], cat_data['clip_map'], cat_data['ts_map'],
                    cat, val, is_ref=False
                )
                if len(ref_coords) == 0 or len(pool_coords) == 0:
                    continue
            else:
                ref_coords = cat_data['ref_coords']
                ref_S_diff = cat_data['ref_S_diff']
                ref_y_ts = cat_data['ref_y_ts']
                ref_meta = cat_data['ref_meta']
                pool_coords = cat_data['pool_coords']
                pool_S_diff = cat_data['pool_S_diff']
                pool_valid_indices = cat_data.get('pool_valid_indices')

            df_aligned = run_lwrr_with_params(
                ref_coords, ref_S_diff, ref_y_ts, ref_meta,
                pool_coords, pool_S_diff, cat_data['p_cat'],
                k_max=params['K_MAX_ST3'],
                tau=params['TAU_KERNEL_ST3'],
                ridge_alpha=params['RIDGE_ALPHA_ST3'],
                equal_eps=params['EQUAL_EPS_ST3'],
                equal_consensus_min=params['EQUAL_CONSENSUS_MIN'],
                selection_ratio=params['SELECTION_RATIO'],
                pool_valid_indices=pool_valid_indices
            )

            if len(df_aligned) == 0:
                continue

            acc, kappa = calculate_metrics(df_aligned)
            acc_ne, kappa_ne = calculate_metrics(df_aligned, exclude_equal=True)
            equal_pct = (df_aligned['synthetic_winner'] == 'equal').mean()
            avg_confidence = df_aligned['confidence_score'].mean()
            acc_ci, kappa_ci = bootstrap_ci(df_aligned, exclude_equal=True)

            results.append({
                'param': param_name,
                'value': val,
                'category': cat,
                'accuracy': acc,
                'kappa': kappa,
                'accuracy_no_equal': acc_ne,
                'kappa_no_equal': kappa_ne,
                'acc_ci_low': acc_ci[0],
                'acc_ci_high': acc_ci[2],
                'kappa_ci_low': kappa_ci[0],
                'kappa_ci_high': kappa_ci[2],
                'equal_pct': equal_pct,
                'avg_confidence': avg_confidence,
                'n_samples': len(df_aligned)
            })

    return results


# ==============================================================================
# 3b. 组合参数随机搜索
# ==============================================================================
def analyze_combined_lwrr_params(precomputed_data, default_params, sensitivity_grid,
                                  n_random, seed=42):
    """
    对所有 LWRR 参数做组合随机搜索。
    全组合空间通常很大 (50000+), 用随机采样 n_random 个组合近似。

    关键优化: 预缓存每个 ALPHA_HYBRID 值的流形数据 (避免重复 rebuild)。
    """
    lwrr_param_names = ['K_MAX_ST3', 'TAU_KERNEL_ST3', 'RIDGE_ALPHA_ST3',
                        'EQUAL_EPS_ST3', 'EQUAL_CONSENSUS_MIN',
                        'ALPHA_HYBRID', 'SELECTION_RATIO']

    # 构建每个参数的候选值列表 (无候选值时用默认值)
    param_lists = []
    active_params = []
    for p in lwrr_param_names:
        vals = sensitivity_grid.get(p, [])
        if vals:
            param_lists.append(vals)
            active_params.append(p)
        else:
            param_lists.append([default_params[p]])
            active_params.append(p)

    # 全组合空间大小
    total_combos = 1
    for pl in param_lists:
        total_combos *= len(pl)

    # 生成采样索引
    all_combos = list(itertools.product(*param_lists))
    rng = np.random.RandomState(seed)
    if n_random >= total_combos:
        sampled_combos = all_combos
    else:
        indices = rng.choice(total_combos, size=n_random, replace=False)
        sampled_combos = [all_combos[i] for i in indices]

    print(f"  全组合空间: {total_combos}, 随机采样: {len(sampled_combos)}")

    # 预缓存每个 ALPHA_HYBRID 值的流形数据
    alpha_values = set()
    alpha_idx = active_params.index('ALPHA_HYBRID')
    for combo in sampled_combos:
        alpha_values.add(combo[alpha_idx])

    alpha_cache = {}  # alpha -> {cat: (ref_coords, ref_S_diff, ref_y_ts, ref_meta, pool_coords, pool_S_diff, pool_valid_indices)}
    for alpha in sorted(alpha_values):
        alpha_cache[alpha] = {}
        for cat, cat_data in precomputed_data.items():
            if alpha == default_params['ALPHA_HYBRID']:
                # 复用已有的默认流形数据
                alpha_cache[alpha][cat] = (
                    cat_data['ref_coords'], cat_data['ref_S_diff'],
                    cat_data['ref_y_ts'], cat_data['ref_meta'],
                    cat_data['pool_coords'], cat_data['pool_S_diff'],
                    cat_data.get('pool_valid_indices')
                )
            else:
                ref_coords, ref_S_diff, ref_y_ts, ref_meta, _ = build_manifold_data(
                    cat_data['r_cat'], cat_data['clip_map'], cat_data['ts_map'],
                    cat, alpha, is_ref=True
                )
                pool_coords, pool_S_diff, _, _, pool_valid_indices = build_manifold_data(
                    cat_data['p_cat'], cat_data['clip_map'], cat_data['ts_map'],
                    cat, alpha, is_ref=False
                )
                if len(ref_coords) > 0 and len(pool_coords) > 0:
                    alpha_cache[alpha][cat] = (
                        ref_coords, ref_S_diff, ref_y_ts, ref_meta,
                        pool_coords, pool_S_diff, pool_valid_indices
                    )

    print(f"  已预缓存 {len(alpha_values)} 个 ALPHA_HYBRID 值的流形数据")

    results = []
    for combo in tqdm(sampled_combos, desc="Combined search"):
        params_dict = dict(zip(active_params, combo))
        alpha = params_dict['ALPHA_HYBRID']

        combo_accs = []
        for cat, cat_data in precomputed_data.items():
            if alpha not in alpha_cache or cat not in alpha_cache[alpha]:
                continue

            ref_coords, ref_S_diff, ref_y_ts, ref_meta, pool_coords, pool_S_diff, pool_valid_indices = alpha_cache[alpha][cat]

            df_aligned = run_lwrr_with_params(
                ref_coords, ref_S_diff, ref_y_ts, ref_meta,
                pool_coords, pool_S_diff, cat_data['p_cat'],
                k_max=int(params_dict['K_MAX_ST3']),
                tau=params_dict['TAU_KERNEL_ST3'],
                ridge_alpha=params_dict['RIDGE_ALPHA_ST3'],
                equal_eps=params_dict['EQUAL_EPS_ST3'],
                equal_consensus_min=params_dict['EQUAL_CONSENSUS_MIN'],
                selection_ratio=params_dict['SELECTION_RATIO'],
                pool_valid_indices=pool_valid_indices
            )

            if len(df_aligned) == 0:
                continue

            acc, kappa = calculate_metrics(df_aligned)
            acc_ne, kappa_ne = calculate_metrics(df_aligned, exclude_equal=True)
            equal_pct = (df_aligned['synthetic_winner'] == 'equal').mean()
            avg_confidence = df_aligned['confidence_score'].mean()
            acc_ci, kappa_ci = bootstrap_ci(df_aligned, exclude_equal=True)
            combo_accs.append(acc)

            results.append({
                'param': 'COMBINED',
                'value': json.dumps(params_dict),
                'category': cat,
                'accuracy': acc,
                'kappa': kappa,
                'accuracy_no_equal': acc_ne,
                'kappa_no_equal': kappa_ne,
                'acc_ci_low': acc_ci[0],
                'acc_ci_high': acc_ci[2],
                'kappa_ci_low': kappa_ci[0],
                'kappa_ci_high': kappa_ci[2],
                'equal_pct': equal_pct,
                'avg_confidence': avg_confidence,
                'n_samples': len(df_aligned)
            })

    return results


# ==============================================================================
# 3c. LABELED_SET_RATIO 敏感性分析
#     改变数据拆分本身 (Ref/Pool比例), 与LWRR算法参数不同
# ==============================================================================
def _split_data_with_ratio(df_cat, ratio, random_state=42):
    """
    按指定ratio将单类别数据拆分为Ref/Pool, 不写缓存文件, 不影响全局状态。
    复制 get_split_data() 的拆分逻辑但仅处理单个类别DataFrame。
    """
    ref_size = max(1, min(int(len(df_cat) * ratio), len(df_cat)))
    df_ref = df_cat.sample(n=ref_size, random_state=random_state)
    df_pool = df_cat.drop(df_ref.index)
    return df_ref, df_pool


def analyze_labeled_set_ratio(precomputed_base, default_params, ratio_values,
                               clip_map, ts_map, df_scored, df_human_all):
    """
    对 LABELED_SET_RATIO 做敏感性分析。
    每个ratio值: 重新拆分Ref/Pool → 匹配已打分数据 → build_manifold → LWRR → 评估。
    """
    results = []

    for ratio in tqdm(ratio_values, desc="Sweep LABELED_SET_RATIO"):
        for cat in CATEGORIES:
            # 从全量人类数据中按ratio拆分
            df_human_cat = df_human_all[df_human_all['category'] == cat]
            if len(df_human_cat) == 0:
                continue

            df_ref_split, df_pool_split = _split_data_with_ratio(df_human_cat, ratio)

            # 构建匹配键
            ref_keys = set(zip(df_ref_split['left_id'].astype(str), df_ref_split['right_id'].astype(str)))
            pool_keys = set(zip(df_pool_split['left_id'].astype(str), df_pool_split['right_id'].astype(str)))

            # 从已打分数据中匹配
            df_scored_cat = df_scored[df_scored['category'] == cat].copy()
            scored_keys = list(zip(df_scored_cat['left_id'].astype(str), df_scored_cat['right_id'].astype(str)))
            ref_mask = [k in ref_keys for k in scored_keys]
            pool_mask = [k in pool_keys for k in scored_keys]

            r_cat = df_scored_cat[ref_mask].copy().reset_index(drop=True)
            p_cat = df_scored_cat[pool_mask].copy().reset_index(drop=True)

            if len(r_cat) < 2 or len(p_cat) < 2:
                continue

            # 构建流形数据
            ref_coords, ref_S_diff, ref_y_ts, ref_meta, _ = build_manifold_data(
                r_cat, clip_map, ts_map, cat, default_params['ALPHA_HYBRID'], is_ref=True
            )
            pool_coords, pool_S_diff, _, _, pool_valid_indices = build_manifold_data(
                p_cat, clip_map, ts_map, cat, default_params['ALPHA_HYBRID'], is_ref=False
            )

            if len(ref_coords) == 0 or len(pool_coords) == 0:
                continue

            # 运行LWRR
            df_aligned = run_lwrr_with_params(
                ref_coords, ref_S_diff, ref_y_ts, ref_meta,
                pool_coords, pool_S_diff, p_cat,
                k_max=default_params['K_MAX_ST3'],
                tau=default_params['TAU_KERNEL_ST3'],
                ridge_alpha=default_params['RIDGE_ALPHA_ST3'],
                equal_eps=default_params['EQUAL_EPS_ST3'],
                equal_consensus_min=default_params['EQUAL_CONSENSUS_MIN'],
                selection_ratio=default_params['SELECTION_RATIO'],
                pool_valid_indices=pool_valid_indices
            )

            if len(df_aligned) == 0:
                continue

            acc, kappa = calculate_metrics(df_aligned)
            acc_ne, kappa_ne = calculate_metrics(df_aligned, exclude_equal=True)
            equal_pct = (df_aligned['synthetic_winner'] == 'equal').mean()
            avg_confidence = df_aligned['confidence_score'].mean()
            acc_ci, kappa_ci = bootstrap_ci(df_aligned, exclude_equal=True)

            results.append({
                'param': 'LABELED_SET_RATIO',
                'value': ratio,
                'category': cat,
                'accuracy': acc,
                'kappa': kappa,
                'accuracy_no_equal': acc_ne,
                'kappa_no_equal': kappa_ne,
                'acc_ci_low': acc_ci[0],
                'acc_ci_high': acc_ci[2],
                'kappa_ci_low': kappa_ci[0],
                'kappa_ci_high': kappa_ci[2],
                'equal_pct': equal_pct,
                'avg_confidence': avg_confidence,
                'n_samples': len(df_aligned),
                'ref_size': len(r_cat),
                'pool_size': len(p_cat),
            })

    return results


# ==============================================================================
# 4. 主流程
# ==============================================================================
def run_sensitivity_analysis():
    print("\n" + "=" * 80)
    print(f"UrbanAlign 2.0 - Stage 5: Parameter Sensitivity Analysis (Mode {STAGE2_MODE})")
    print("=" * 80)

    # 加载所有类别的Stage 2数据
    dfs_scored = []
    for cat in CATEGORIES:
        stage2_file = get_stage2_output(STAGE2_MODE, cat)
        if os.path.exists(stage2_file):
            dfs_scored.append(pd.read_csv(stage2_file))
    if not dfs_scored:
        print(f"[ERROR] 未找到任何类别的 Stage 2 文件 (Mode {STAGE2_MODE})")
        return
    df_scored = pd.concat(dfs_scored, ignore_index=True)
    print(f"  Stage 2 数据: {len(df_scored)}对 ({len(dfs_scored)}个类别)")

    all_results = []

    # ──────────────────────────────────────────────────────────
    # Part A: ST2_INTENSITY_SIG_THRESH 敏感性 (无需Stage 3)
    # ──────────────────────────────────────────────────────────
    st2_thresholds = SENSITIVITY_GRID.get('ST2_INTENSITY_SIG_THRESH', [])
    if st2_thresholds:
        print(f"\n{'─'*60}")
        print(f"Part A: ST2_INTENSITY_SIG_THRESH 敏感性分析")
        print(f"  搜索范围: {st2_thresholds}")
        print(f"  注意: 仅对 Mode 1/3 (单张模式) 有效")
        print(f"{'─'*60}")

        st2_results = analyze_st2_threshold(df_scored, st2_thresholds)
        all_results.extend(st2_results)

        if st2_results:
            df_st2 = pd.DataFrame(st2_results)
            print("\n  ST2_INTENSITY_SIG_THRESH 结果:")
            for cat in df_st2['category'].unique():
                print(f"\n    [{cat}]")
                sub = df_st2[df_st2['category'] == cat]
                for _, row in sub.iterrows():
                    print(f"      thresh={row['value']:5.1f} → "
                          f"Acc={row['accuracy']*100:5.1f}%  "
                          f"Kappa={row['kappa']:.3f}  "
                          f"Equal%={row['equal_pct']*100:4.1f}%")

    # ──────────────────────────────────────────────────────────
    # Part B: LWRR 参数敏感性 (需重跑Stage 3对齐循环)
    # ──────────────────────────────────────────────────────────
    lwrr_params = ['K_MAX_ST3', 'TAU_KERNEL_ST3', 'RIDGE_ALPHA_ST3',
                   'EQUAL_EPS_ST3', 'EQUAL_CONSENSUS_MIN',
                   'ALPHA_HYBRID', 'SELECTION_RATIO']

    # 检查是否有需要搜索的LWRR参数
    has_lwrr = any(SENSITIVITY_GRID.get(p) for p in lwrr_params)

    if has_lwrr:
        print(f"\n{'─'*60}")
        print(f"Part B: LWRR 参数敏感性分析 (Stage 3 对齐循环)")
        print(f"{'─'*60}")

        # 加载预计算资源
        clip_map = load_clip_features()
        ts_map = load_trueskill_map()

        # 默认参数
        default_params = {
            'K_MAX_ST3': K_MAX_ST3,
            'TAU_KERNEL_ST3': TAU_KERNEL_ST3,
            'RIDGE_ALPHA_ST3': RIDGE_ALPHA_ST3,
            'EQUAL_EPS_ST3': EQUAL_EPS_ST3,
            'EQUAL_CONSENSUS_MIN': EQUAL_CONSENSUS_MIN,
            'ALPHA_HYBRID': ALPHA_HYBRID,
            'SELECTION_RATIO': SELECTION_RATIO,
        }
        print(f"  默认参数: {default_params}")

        # 预计算流形数据 (对默认 ALPHA_HYBRID)
        precomputed = {}
        for cat in CATEGORIES:
            df_scored_cat = df_scored[df_scored['category'] == cat].copy()
            if len(df_scored_cat) == 0:
                continue

            df_ref_split, df_pool_split = get_split_data(category=cat)
            scored_keys = list(zip(
                df_scored_cat['left_id'].astype(str),
                df_scored_cat['right_id'].astype(str)
            ))
            ref_keys = set(zip(
                df_ref_split['left_id'].astype(str),
                df_ref_split['right_id'].astype(str)
            ))
            pool_keys = set(zip(
                df_pool_split['left_id'].astype(str),
                df_pool_split['right_id'].astype(str)
            ))
            ref_mask = [k in ref_keys for k in scored_keys]
            pool_mask = [k in pool_keys for k in scored_keys]

            r_cat = df_scored_cat[ref_mask].copy().reset_index(drop=True)
            p_cat = df_scored_cat[pool_mask].copy().reset_index(drop=True)

            if len(r_cat) == 0 or len(p_cat) == 0:
                continue

            ref_coords, ref_S_diff, ref_y_ts, ref_meta, _ = build_manifold_data(
                r_cat, clip_map, ts_map, cat, ALPHA_HYBRID, is_ref=True
            )
            pool_coords, pool_S_diff, _, _, pool_valid_indices = build_manifold_data(
                p_cat, clip_map, ts_map, cat, ALPHA_HYBRID, is_ref=False
            )

            if len(ref_coords) == 0 or len(pool_coords) == 0:
                continue

            precomputed[cat] = {
                'r_cat': r_cat, 'p_cat': p_cat,
                'ref_coords': ref_coords, 'ref_S_diff': ref_S_diff,
                'ref_y_ts': ref_y_ts, 'ref_meta': ref_meta,
                'pool_coords': pool_coords, 'pool_S_diff': pool_S_diff,
                'pool_valid_indices': pool_valid_indices,
                'clip_map': clip_map, 'ts_map': ts_map,
            }

            print(f"  [{cat}] Ref={len(r_cat)}对 (流形={len(ref_coords)}含镜像), "
                  f"Pool={len(p_cat)}对 (流形={len(pool_coords)})")

        if not precomputed:
            print("[WARN] 无有效数据, 跳过LWRR敏感性分析")
        else:
            # 逐参数搜索 (单变量敏感性)
            for param_name in lwrr_params:
                param_values = SENSITIVITY_GRID.get(param_name, [])
                if not param_values:
                    continue

                print(f"\n  === {param_name} ===")
                print(f"  搜索范围: {param_values}")

                param_results = analyze_single_lwrr_param(
                    param_name, param_values, precomputed, default_params
                )
                all_results.extend(param_results)

                if param_results:
                    df_p = pd.DataFrame(param_results)
                    for cat in df_p['category'].unique():
                        print(f"\n    [{cat}]")
                        sub = df_p[df_p['category'] == cat]
                        for _, row in sub.iterrows():
                            fid_str = f"  Conf={row['avg_confidence']:.3f}" if 'avg_confidence' in row and not pd.isna(row.get('avg_confidence')) else ""
                            print(f"      {param_name}={row['value']:<6} → "
                                  f"Acc={row['accuracy']*100:5.1f}%  "
                                  f"Kappa={row['kappa']:.3f}  "
                                  f"Equal%={row['equal_pct']*100:4.1f}%"
                                  f"{fid_str}")

    # ──────────────────────────────────────────────────────────
    # Part C: 组合参数随机搜索 (联合优化所有LWRR参数)
    # ──────────────────────────────────────────────────────────
    if has_lwrr and precomputed and N_RANDOM_SEARCH > 0:
        print(f"\n{'─'*60}")
        print(f"Part C: 组合参数随机搜索 (N={N_RANDOM_SEARCH})")
        print(f"{'─'*60}")

        combined_results = analyze_combined_lwrr_params(
            precomputed, default_params, SENSITIVITY_GRID,
            n_random=N_RANDOM_SEARCH
        )
        all_results.extend(combined_results)

        if combined_results:
            df_comb = pd.DataFrame(combined_results)
            # 按组合取跨类别平均准确率
            df_comb_avg = df_comb.groupby('value').agg({
                'accuracy': 'mean', 'kappa': 'mean'
            }).reset_index().sort_values('accuracy', ascending=False)

            print(f"\n  Top-5 参数组合:")
            for rank, (_, row) in enumerate(df_comb_avg.head(5).iterrows(), 1):
                params = json.loads(row['value'])
                param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
                print(f"    #{rank}: Acc={row['accuracy']*100:.2f}%  Kappa={row['kappa']:.3f}")
                print(f"         {param_str}")

            # 准确率区间
            acc_min = df_comb_avg['accuracy'].min()
            acc_max = df_comb_avg['accuracy'].max()
            print(f"\n  准确率区间: [{acc_min*100:.1f}%, {acc_max*100:.1f}%]")

            # 最优组合详情
            best_combo = json.loads(df_comb_avg.iloc[0]['value'])
            print(f"\n  最优组合详情:")
            for k, v in best_combo.items():
                current = default_params.get(k, '?')
                marker = " ★" if v != current else ""
                print(f"    {k}: {v} (当前={current}){marker}")

    # ──────────────────────────────────────────────────────────
    # Part D: LABELED_SET_RATIO 敏感性分析 (改变数据拆分)
    # ──────────────────────────────────────────────────────────
    ratio_values = SENSITIVITY_GRID.get('LABELED_SET_RATIO', [])
    if ratio_values:
        print(f"\n{'─'*60}")
        print(f"Part D: LABELED_SET_RATIO 敏感性分析 (数据拆分比例)")
        print(f"  搜索范围: {ratio_values}")
        print(f"  当前值: {LABELED_SET_RATIO}")
        print(f"  注意: 此参数改变Ref/Pool拆分, 不纳入Part C组合搜索")
        print(f"{'─'*60}")

        # 加载资源 (如果Part B已加载则复用, 否则新加载)
        if 'clip_map' not in dir() or clip_map is None:
            clip_map = load_clip_features()
        if 'ts_map' not in dir() or ts_map is None:
            ts_map = load_trueskill_map()

        # 加载全量人类数据 (按类别合并)
        dfs_human = []
        for cat in CATEGORIES:
            f = get_human_choices_csv(cat)
            if os.path.exists(f):
                dfs_human.append(pd.read_csv(f))
        df_human_all = pd.concat(dfs_human, ignore_index=True)

        # 默认LWRR参数
        if 'default_params' not in dir():
            default_params = {
                'K_MAX_ST3': K_MAX_ST3,
                'TAU_KERNEL_ST3': TAU_KERNEL_ST3,
                'RIDGE_ALPHA_ST3': RIDGE_ALPHA_ST3,
                'EQUAL_EPS_ST3': EQUAL_EPS_ST3,
                'EQUAL_CONSENSUS_MIN': EQUAL_CONSENSUS_MIN,
                'ALPHA_HYBRID': ALPHA_HYBRID,
                'SELECTION_RATIO': SELECTION_RATIO,
            }

        ratio_results = analyze_labeled_set_ratio(
            precomputed_base=None,
            default_params=default_params,
            ratio_values=ratio_values,
            clip_map=clip_map,
            ts_map=ts_map,
            df_scored=df_scored,
            df_human_all=df_human_all
        )
        all_results.extend(ratio_results)

        if ratio_results:
            df_ratio = pd.DataFrame(ratio_results)
            print("\n  LABELED_SET_RATIO 结果:")
            for cat in df_ratio['category'].unique():
                print(f"\n    [{cat}]")
                sub = df_ratio[df_ratio['category'] == cat]
                for _, row in sub.iterrows():
                    print(f"      ratio={row['value']:<5} → "
                          f"Acc={row['accuracy']*100:5.1f}%  "
                          f"Kappa={row['kappa']:.3f}  "
                          f"Ref={row['ref_size']}  Pool={row['pool_size']}  "
                          f"n_aligned={row['n_samples']}")

    # ──────────────────────────────────────────────────────────
    # 汇总输出
    # ──────────────────────────────────────────────────────────
    if all_results:
        df_all = pd.DataFrame(all_results)

        # 按类别分别保存
        saved_files = []
        for cat in df_all['category'].unique():
            output_file = get_stage5_output(STAGE2_MODE, cat)
            df_all[df_all['category'] == cat].to_csv(output_file, index=False)
            saved_files.append(output_file)

        # 生成最优参数建议
        print(f"\n{'='*80}")
        print("敏感性分析汇总")
        print(f"{'='*80}")

        current_val_map = {
            'ST2_INTENSITY_SIG_THRESH': ST2_INTENSITY_SIG_THRESH,
            'K_MAX_ST3': K_MAX_ST3,
            'TAU_KERNEL_ST3': TAU_KERNEL_ST3,
            'RIDGE_ALPHA_ST3': RIDGE_ALPHA_ST3,
            'EQUAL_EPS_ST3': EQUAL_EPS_ST3,
            'EQUAL_CONSENSUS_MIN': EQUAL_CONSENSUS_MIN,
            'ALPHA_HYBRID': ALPHA_HYBRID,
            'SELECTION_RATIO': SELECTION_RATIO,
            'LABELED_SET_RATIO': LABELED_SET_RATIO,
        }

        for param in df_all['param'].unique():
            df_param = df_all[df_all['param'] == param]

            if param == 'COMBINED':
                # 组合搜索: 打印最优组合 (详细结果已在Part C输出)
                avg_by_val = df_param.groupby('value').agg({
                    'accuracy': 'mean', 'kappa': 'mean'
                }).reset_index().sort_values('accuracy', ascending=False)
                best_row = avg_by_val.iloc[0]
                best_combo = json.loads(best_row['value'])
                print(f"\n  COMBINED (组合随机搜索):")
                print(f"    搜索组合数: {len(avg_by_val)}")
                print(f"    最优准确率: {best_row['accuracy']*100:.1f}% (Kappa={best_row['kappa']:.3f})")
                print(f"    最优参数:")
                for k, v in best_combo.items():
                    current = current_val_map.get(k, '?')
                    marker = " ★" if v != current else ""
                    print(f"      {k}: {v} (当前={current}){marker}")
                continue

            # 单参数: 按参数值取平均 (跨类别)
            avg_by_val = df_param.groupby('value').agg({
                'accuracy': 'mean',
                'kappa': 'mean',
                'accuracy_no_equal': 'mean',
            }).reset_index()

            best_row = avg_by_val.loc[avg_by_val['accuracy'].idxmax()]
            current_val = current_val_map.get(param, '?')

            print(f"\n  {param}:")
            print(f"    当前值: {current_val}")
            print(f"    最优值: {best_row['value']} (Acc={best_row['accuracy']*100:.1f}%)")

            # 范围
            acc_range = avg_by_val['accuracy'].max() - avg_by_val['accuracy'].min()
            print(f"    准确率变化范围: {acc_range*100:.1f}% ({'敏感' if acc_range > 0.03 else '不敏感'})")

        print(f"\n  结果保存: {len(saved_files)} 个类别文件")
        for f in saved_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  此表可直接用于论文的参数敏感性分析 (Sensitivity Analysis)")

        # ──────────────────────────────────────────────────────────
        # 保存 per-category 最优参数 JSON (供 Stage 3 自动调用)
        # ──────────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("Per-Category 最优参数保存")
        print(f"{'─'*60}")

        # 默认参数基线
        default_lwrr = {
            'K_MAX_ST3': K_MAX_ST3,
            'TAU_KERNEL_ST3': TAU_KERNEL_ST3,
            'RIDGE_ALPHA_ST3': RIDGE_ALPHA_ST3,
            'EQUAL_EPS_ST3': EQUAL_EPS_ST3,
            'EQUAL_CONSENSUS_MIN': EQUAL_CONSENSUS_MIN,
            'ALPHA_HYBRID': ALPHA_HYBRID,
            'SELECTION_RATIO': SELECTION_RATIO,
        }

        for cat in df_all['category'].unique():
            df_cat = df_all[df_all['category'] == cat]
            best_params = default_lwrr.copy()
            best_source = 'default'

            # 优先使用 COMBINED 搜索的 per-category 最优
            df_combined = df_cat[df_cat['param'] == 'COMBINED']
            if len(df_combined) > 0:
                best_row = df_combined.sort_values('accuracy_no_equal', ascending=False).iloc[0]
                combo_params = json.loads(best_row['value'])
                # 仅更新 LWRR 参数 (不包含 LABELED_SET_RATIO 等)
                for k in default_lwrr:
                    if k in combo_params:
                        best_params[k] = combo_params[k]
                best_source = 'combined_search'
            else:
                # fallback: 从单参数 sweep 中逐个取 per-category 最优
                for param_name in default_lwrr:
                    df_p = df_cat[(df_cat['param'] == param_name)]
                    if len(df_p) > 0:
                        best_row = df_p.sort_values('accuracy_no_equal', ascending=False).iloc[0]
                        best_params[param_name] = best_row['value']
                        best_source = 'single_param_sweep'

            # 确保类型正确
            best_params['K_MAX_ST3'] = int(best_params['K_MAX_ST3'])

            # 保存 JSON
            best_params_out = {
                'category': cat,
                'source': best_source,
                'mode': STAGE2_MODE,
                'params': best_params,
                'defaults': default_lwrr,
            }
            out_path = os.path.join(OUTPUT_DIR, f'stage5_best_params_{cat}.json')
            with open(out_path, 'w') as f:
                json.dump(best_params_out, f, indent=2)

            # 打印差异
            diffs = [k for k in default_lwrr if best_params[k] != default_lwrr[k]]
            if diffs:
                diff_str = ', '.join([f"{k}: {default_lwrr[k]}→{best_params[k]}" for k in diffs])
                print(f"  [{cat}] 最优参数已保存 ({best_source}), 变更: {diff_str}")
            else:
                print(f"  [{cat}] 最优参数 = 默认参数 ({best_source})")

        print(f"\n  Stage 3 重跑时将自动加载这些最优参数")

    else:
        print("\n[WARN] 无分析结果")

    print(f"\n{'='*80}")
    print("敏感性分析完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_sensitivity_analysis()
