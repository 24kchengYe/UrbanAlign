"""
UrbanAlign 2.0 - Stage 6: End-to-End Dimension Optimization

核心思想:
  不同语义维度集 → 不同Stage 2打分 → 不同对齐准确率。
  本模块自动生成多套维度定义, 用小样本端到端评估每套的准确率,
  找出最优维度集, 最终覆盖到 STAGE1_DIMENSIONS。

优化策略 (Explore → Converge):
  阶段1 (探索): 较高temperature, 独立生成, 各类别分别积累最优维度
  阶段2 (收敛): 较低temperature, 保留好维度仅变异差维度, 精细优化

运行方式:
  python abc_stage6_e2e_dimension_optimization.py
"""
import pandas as pd
import numpy as np
import os
import json
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score

from urbanalign.config import (
    API_KEY, BASE_URL, MODEL_NAME, call_llm_api,
    DATA_DIR, IMAGE_DIR,
    OUTPUT_DIR, CLIP_CACHE,
    CATEGORIES, MAX_PAIRS_PER_CATEGORY,
    N_CONSENSUS_SAMPLES, N_DIMENSIONS_MIN, N_DIMENSIONS_MAX,
    ALPHA_HYBRID, K_MAX_ST3, TAU_KERNEL_ST3, RIDGE_ALPHA_ST3,
    EQUAL_EPS_ST3, EQUAL_CONSENSUS_MIN, SELECTION_RATIO,
    N_DIMENSION_TRIALS, E2E_POOL_MULTIPLIER, E2E_STAGE2_MODE,
    E2E_PATIENCE, E2E_ELITE_SEED, E2E_SAMPLE_RATIO, E2E_EXPLORE_RATIO,
    get_trueskill_cache, get_human_choices_csv, get_stage1_dimensions,
    get_stage2_sampled_pairs, get_split_data,
    get_stage6_trial_dims, get_stage6_trial_scored, get_stage6_summary,
)

# 复用现有模块的函数
from abc_stage1_semantic_extractor import (
    sample_consensus_images, load_trueskill_ratings,
    build_dimension_extraction_prompt, create_id_mapping,
)
from abc_stage2_multi_mode_synthesis import (
    mode2_pair_direct, image_to_base64, robust_parse_vlm_output,
)
from abc_stage5_sensitivity_analysis import (
    build_manifold_data, run_lwrr_with_params, calculate_metrics,
    load_clip_features, load_trueskill_map,
)


# ==============================================================================
# 1. 维度生成 (支持两种模式: 独立生成 / 变异优化)
# ==============================================================================
def generate_new_dimensions(category, df_ratings, id_mapping, temperature=0.7,
                            elite_dims=None, mutation_mode=False):
    """为一个类别生成新的语义维度集

    Args:
        elite_dims: 当前最优维度集 (dict), 用于引导生成变体
                    None = 独立生成 (Trial 0 或 E2E_ELITE_SEED=False)
        mutation_mode: True = 收敛阶段, 保留大部分好维度仅替换1-2个差维度
                       False = 探索阶段, 完全或大幅重新生成
    """
    high_samples, low_samples = sample_consensus_images(
        df_ratings, category, n_samples=N_CONSENSUS_SAMPLES
    )

    if len(high_samples) == 0 or len(low_samples) == 0:
        print(f"    [WARN] {category}: 共识样本不足, 跳过")
        return None

    prompt_content = build_dimension_extraction_prompt(
        category, high_samples, low_samples, id_mapping
    )

    # 根据模式追加不同的引导信息
    if elite_dims and E2E_ELITE_SEED:
        elite_dim_list = elite_dims.get('dimensions', [])
        if elite_dim_list:
            seed_names = [d['name'] for d in elite_dim_list]

            if mutation_mode:
                # 收敛模式: 强调保留大部分, 仅微调1-2个
                seed_text = {
                    "type": "text",
                    "text": (
                        f"\n\n**REFINEMENT MODE (converge phase for {category}):**\n"
                        f"The following {len(seed_names)} dimensions are the CURRENT BEST set "
                        f"with highest alignment accuracy:\n"
                        f"{', '.join(seed_names)}\n\n"
                        f"**IMPORTANT**: Keep MOST of these dimensions (at least {max(1, len(seed_names)-2)} out of {len(seed_names)}). "
                        f"Only replace or modify 1-2 dimensions that you think could be improved. "
                        f"Small, targeted changes are preferred over large rewrites. "
                        f"You may also fine-tune dimension descriptions/examples without changing names. "
                        f"Remember: the prompt specifies the exact number of dimensions to output — follow it strictly."
                    )
                }
            else:
                # 探索模式: 鼓励创新但提供参考
                seed_text = {
                    "type": "text",
                    "text": (
                        f"\n\n**REFERENCE (current best dimensions for {category}):**\n"
                        f"The following {len(seed_names)} dimensions achieved the highest alignment accuracy so far:\n"
                        f"{', '.join(seed_names)}\n\n"
                        f"You may keep, modify, replace, or reorder these dimensions. "
                        f"Your goal is to IMPROVE upon this set. "
                        f"Feel free to introduce new dimensions or drop underperforming ones. "
                        f"Remember: the prompt specifies the exact number of dimensions to output — follow it strictly."
                    )
                }
            prompt_content.append(seed_text)

    response = call_llm_api(
        messages=[{"role": "user", "content": prompt_content}],
        temperature=temperature,
        max_tokens=4096,
        response_format={"type": "json_object"},
        timeout=300
    )

    if response:
        try:
            return json.loads(response)
        except Exception as e:
            print(f"    [WARN] {category}: 解析失败 - {e}")
    return None


# ==============================================================================
# 2. 小样本打分 (Mode 2, 轻量版)
# ==============================================================================
def score_small_sample(df_pairs, category, dimensions_all):
    """用 Mode 2 对小样本数据打分, 返回带分数的 DataFrame"""
    dims = dimensions_all.get(category, {})
    if not dims:
        print(f"    [WARN] {category}: 无维度定义, 跳过")
        return pd.DataFrame()

    results = []
    for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs),
                       desc=f"    Scoring {category}"):
        l_id, r_id = str(row['left_id']), str(row['right_id'])
        img_a = os.path.join(IMAGE_DIR, f"{l_id}.jpg")
        img_b = os.path.join(IMAGE_DIR, f"{r_id}.jpg")

        if not os.path.exists(img_a) or not os.path.exists(img_b):
            continue

        try:
            resp = mode2_pair_direct(img_a, img_b, category, dims)
            data = json.loads(resp)
            sa, ia, wa = robust_parse_vlm_output(data, prefix="_a")
            sb, ib, _ = robust_parse_vlm_output(data, prefix="_b")

            row_data = row.to_dict()
            row_data.update({
                'image_a_scores': json.dumps(sa),
                'image_b_scores': json.dumps(sb),
                'overall_intensity_a': ia,
                'overall_intensity_b': ib,
                'ai_dimension_weights': json.dumps(wa),
                'synthetic_winner': data.get('winner', 'equal'),
            })
            results.append(row_data)
        except Exception as e:
            continue

        time.sleep(0.1)  # 避免API限流

    df_out = pd.DataFrame(results) if results else pd.DataFrame()
    # 原始数据列名为'winner', 下游build_manifold_data需要'human_winner'
    if 'winner' in df_out.columns and 'human_winner' not in df_out.columns:
        df_out = df_out.rename(columns={'winner': 'human_winner'})
    return df_out


# ==============================================================================
# 3. LWRR对齐 + 评估 (返回per-category结果)
# ==============================================================================
def evaluate_alignment(df_scored, clip_map, ts_map):
    """对已打分数据做LWRR对齐并返回per-category准确率

    使用与 Stage 3 相同的 get_split_data() 进行 Ref/Pool 划分,
    保持全流水线数据一致性。

    Returns:
        cat_results: dict {category: (acc, kappa)}
        mean_acc: float 全类别平均准确率
        mean_kappa: float 全类别平均kappa
    """
    cat_results = {}

    for cat in df_scored['category'].unique():
        df_cat = df_scored[df_scored['category'] == cat].copy()
        if len(df_cat) == 0:
            continue

        # 使用统一的 get_split_data() 获取 Ref/Pool 划分
        df_ref_split, df_pool_split = get_split_data(category=cat)

        # 用 (left_id, right_id) 键匹配已打分数据
        ref_keys = set(zip(df_ref_split['left_id'].astype(str), df_ref_split['right_id'].astype(str)))
        pool_keys = set(zip(df_pool_split['left_id'].astype(str), df_pool_split['right_id'].astype(str)))

        scored_keys = list(zip(df_cat['left_id'].astype(str), df_cat['right_id'].astype(str)))
        ref_mask = [k in ref_keys for k in scored_keys]
        pool_mask = [k in pool_keys for k in scored_keys]

        r_cat = df_cat[ref_mask].reset_index(drop=True)
        p_cat = df_cat[pool_mask].reset_index(drop=True)

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

        df_aligned = run_lwrr_with_params(
            ref_coords, ref_S_diff, ref_y_ts, ref_meta,
            pool_coords, pool_S_diff, p_cat,
            k_max=K_MAX_ST3, tau=TAU_KERNEL_ST3, ridge_alpha=RIDGE_ALPHA_ST3,
            equal_eps=EQUAL_EPS_ST3, equal_consensus_min=EQUAL_CONSENSUS_MIN,
            selection_ratio=SELECTION_RATIO,
            pool_valid_indices=pool_valid_indices
        )

        if len(df_aligned) == 0:
            continue

        acc, kappa = calculate_metrics(df_aligned)
        cat_results[cat] = (acc, kappa)

    if not cat_results:
        return {}, 0, 0

    all_acc = [v[0] for v in cat_results.values()]
    all_kappa = [v[1] for v in cat_results.values()]
    return cat_results, np.mean(all_acc), np.mean(all_kappa)


# ==============================================================================
# 4. 温度调度 (Explore → Converge)
# ==============================================================================
def get_temperature(trial, n_trials, explore_boundary):
    """根据当前trial和阶段返回temperature

    探索阶段 (trial < explore_boundary):
        temperature = 0.85 ~ 1.0 (较高, 鼓励多样性)
    收敛阶段 (trial >= explore_boundary):
        temperature = 0.7 → 0.5 (递减, 精细优化)
    """
    if trial < explore_boundary:
        # 探索阶段: 0.85 → 1.0 线性递增
        if explore_boundary <= 1:
            return 0.9
        progress = trial / (explore_boundary - 1)
        return 0.85 + 0.15 * progress
    else:
        # 收敛阶段: 0.7 → 0.5 线性递减
        converge_trials = n_trials - explore_boundary
        if converge_trials <= 1:
            return 0.6
        progress = (trial - explore_boundary) / (converge_trials - 1)
        return 0.7 - 0.2 * progress


# ==============================================================================
# 5. 主流程 (两阶段: Explore → Converge, Per-category最优跟踪)
# ==============================================================================
def run_e2e_dimension_optimization():
    explore_boundary = max(1, int(N_DIMENSION_TRIALS * E2E_EXPLORE_RATIO))

    print("\n" + "=" * 80)
    print(f"UrbanAlign 2.0 - Stage 6: E2E Dimension Optimization")
    print(f"  Trials: {N_DIMENSION_TRIALS} (Explore: {explore_boundary}, Converge: {N_DIMENSION_TRIALS - explore_boundary})")
    print(f"  Pool multiplier: {E2E_POOL_MULTIPLIER}, Mode: {E2E_STAGE2_MODE}")
    cap_info = f", cap={MAX_PAIRS_PER_CATEGORY}" if MAX_PAIRS_PER_CATEGORY > 0 else ""
    ratio_info = f", sample_ratio={E2E_SAMPLE_RATIO}" if E2E_SAMPLE_RATIO < 1.0 else ""
    print(f"  Elite seed: {E2E_ELITE_SEED}, Patience: {E2E_PATIENCE}{cap_info}{ratio_info}")
    print("=" * 80)

    # 加载共享资源 (按类别合并)
    dfs_ratings = []
    for cat in CATEGORIES:
        df_r = load_trueskill_ratings(cat)
        if df_r is not None:
            dfs_ratings.append(df_r)
    if not dfs_ratings:
        print("[ERROR] 无法加载任何类别的TrueSkill评分")
        return
    df_ratings = pd.concat(dfs_ratings, ignore_index=True)

    dfs_raw = []
    id_mapping = {}
    for cat in CATEGORIES:
        f = get_human_choices_csv(cat)
        if os.path.exists(f):
            df_cat_raw = pd.read_csv(f)
            dfs_raw.append(df_cat_raw)
            id_mapping.update(create_id_mapping(df_cat_raw, cat))
    if not dfs_raw:
        print("[ERROR] 无法加载任何类别的人类选择数据")
        return
    df_raw = pd.concat(dfs_raw, ignore_index=True)
    clip_map = load_clip_features()
    ts_map = load_trueskill_map()

    # 加载采样缓存 (与 Stage 2/3 共用同一份数据)
    sampled_pairs = {}
    for cat in CATEGORIES:
        stage2_cache = get_stage2_sampled_pairs(cat)
        if os.path.exists(stage2_cache):
            df_cached = pd.read_csv(stage2_cache)
            full_size = len(df_cached)
            # 应用 E2E_SAMPLE_RATIO 二次采样 (不修改缓存文件)
            # 注意: 必须用不同于 get_split_data() 的随机种子 (42),
            #   否则 sample(n_small, seed=42) ⊂ sample(n_large, seed=42),
            #   导致E2E子样本全部落入Ref集, Pool为空, 无法评估
            if E2E_SAMPLE_RATIO < 1.0:
                n_use = max(1, int(full_size * E2E_SAMPLE_RATIO))
                sampled_pairs[cat] = df_cached.sample(n=n_use, random_state=2024)
                print(f"  [{cat}] 加载Stage2采样缓存: {full_size}对 → 使用{n_use}对 ({E2E_SAMPLE_RATIO*100:.0f}%)")
            else:
                sampled_pairs[cat] = df_cached
                print(f"  [{cat}] 加载Stage2采样缓存: {full_size}对")
        else:
            # Stage 2 缓存不存在, 从全量人类数据生成 (与 Stage 2 相同逻辑)
            df_source = df_raw[df_raw['category'] == cat]
            n_sample = max(1, int(len(df_source) * E2E_POOL_MULTIPLIER))
            if MAX_PAIRS_PER_CATEGORY > 0:
                n_sample = min(n_sample, MAX_PAIRS_PER_CATEGORY)
            n_sample = min(n_sample, len(df_source))
            df_new = df_source.sample(n=n_sample, random_state=42)
            # 统一列名
            if 'winner' in df_new.columns and 'human_winner' not in df_new.columns:
                df_new = df_new.rename(columns={'winner': 'human_winner'})
            df_new.to_csv(stage2_cache, index=False)
            cap_info = f", cap={MAX_PAIRS_PER_CATEGORY}" if MAX_PAIRS_PER_CATEGORY > 0 else ""
            # 对新建缓存也应用 E2E_SAMPLE_RATIO
            if E2E_SAMPLE_RATIO < 1.0:
                n_use = max(1, int(n_sample * E2E_SAMPLE_RATIO))
                sampled_pairs[cat] = df_new.sample(n=n_use, random_state=2024)
                print(f"  [{cat}] 新建Stage2采样缓存: {n_sample}对 → 使用{n_use}对 ({E2E_SAMPLE_RATIO*100:.0f}%) (源=全量{len(df_source)}对{cap_info})")
            else:
                sampled_pairs[cat] = df_new
                print(f"  [{cat}] 新建Stage2采样缓存: {n_sample}对 (源=全量{len(df_source)}对{cap_info})")

        # 确保 Ref/Pool split 缓存存在 (与 Stage 3 共用)
        get_split_data(category=cat)

    # ──────────────────────────────────────────────────────────
    # Per-category 最优跟踪
    # ──────────────────────────────────────────────────────────
    best_cat_dims = {}      # {cat: dims_dict} 各类别的最优维度
    best_cat_acc = {}       # {cat: float} 各类别的最优准确率
    best_cat_kappa = {}     # {cat: float} 各类别的最优kappa
    best_cat_trial = {}     # {cat: int} 各类别最优所在的trial

    trial_results = []
    best_global_acc = -1.0
    no_improve_count = 0    # 全局无提升计数

    for trial in range(N_DIMENSION_TRIALS):
        is_explore = trial < explore_boundary
        phase_name = "EXPLORE" if is_explore else "CONVERGE"
        temperature = get_temperature(trial, N_DIMENSION_TRIALS, explore_boundary)
        mutation = not is_explore and E2E_ELITE_SEED  # 收敛阶段启用变异模式

        print(f"\n{'─'*60}")
        print(f"Trial {trial+1}/{N_DIMENSION_TRIALS}  [{phase_name}]  τ={temperature:.2f}")
        if best_global_acc >= 0:
            print(f"  (全局最优Acc={best_global_acc*100:.1f}%, "
                  f"无提升={no_improve_count}/{E2E_PATIENCE})")
        if best_cat_acc:
            cat_acc_str = ", ".join(f"{c}:{a*100:.0f}%" for c, a in best_cat_acc.items())
            print(f"  Per-cat最优: {cat_acc_str}")
        print(f"{'─'*60}")

        # ① 生成新维度集
        mode_str = "mutation" if mutation else ("elite-ref" if (is_explore and trial > 0 and E2E_ELITE_SEED and best_cat_dims) else "independent")
        print(f"  [Step 1] 生成维度集 (τ={temperature:.2f}, mode={mode_str})...")
        trial_dims = {}
        for cat in CATEGORIES:
            # 确定elite参考
            elite = best_cat_dims.get(cat) if (E2E_ELITE_SEED and best_cat_dims.get(cat)) else None

            # 探索阶段: Trial 0 独立生成, 之后可用elite作为参考(非变异)
            # 收敛阶段: 必须有elite才做变异, 否则用elite参考
            dims = generate_new_dimensions(
                cat, df_ratings, id_mapping, temperature,
                elite_dims=elite,
                mutation_mode=mutation
            )
            if dims:
                trial_dims[cat] = dims
                n_dims = len(dims.get('dimensions', []))
                print(f"    {cat}: {n_dims}维度")
            else:
                # 生成失败时, 收敛阶段沿用最优
                if mutation and cat in best_cat_dims:
                    trial_dims[cat] = best_cat_dims[cat]
                    print(f"    {cat}: 生成失败, 沿用最优")
                else:
                    print(f"    {cat}: 失败, 跳过")

        if not trial_dims:
            print(f"  [SKIP] 所有类别维度生成失败")
            no_improve_count += 1
            if E2E_PATIENCE > 0 and no_improve_count >= E2E_PATIENCE:
                print(f"  [EARLY STOP] 连续{no_improve_count}次无有效结果, 停止搜索")
                break
            continue

        # 保存trial维度 (按类别分别保存)
        for cat, dims in trial_dims.items():
            trial_dim_file = get_stage6_trial_dims(trial, cat)
            with open(trial_dim_file, 'w', encoding='utf-8') as f:
                json.dump({cat: dims}, f, indent=2, ensure_ascii=False)

        # ② 小样本打分
        print(f"  [Step 2] Mode {E2E_STAGE2_MODE} 小样本打分...")
        scored_dfs = []
        for cat in CATEGORIES:
            if cat not in trial_dims:
                continue
            df_scored = score_small_sample(sampled_pairs[cat], cat, trial_dims)
            if len(df_scored) > 0:
                df_scored['category'] = cat
                scored_dfs.append(df_scored)

        if not scored_dfs:
            print(f"  [SKIP] 无有效打分结果")
            no_improve_count += 1
            if E2E_PATIENCE > 0 and no_improve_count >= E2E_PATIENCE:
                print(f"  [EARLY STOP] 连续{no_improve_count}次无有效结果, 停止搜索")
                break
            continue

        df_all_scored = pd.concat(scored_dfs, ignore_index=True)

        # 保存trial打分 (按类别分别保存)
        for cat in df_all_scored['category'].unique():
            trial_scored_file = get_stage6_trial_scored(trial, cat)
            df_all_scored[df_all_scored['category'] == cat].to_csv(trial_scored_file, index=False)
        print(f"    已保存 {len(df_all_scored['category'].unique())} 个类别打分文件 ({len(df_all_scored)}对)")

        # ③ LWRR对齐 + 评估 (per-category)
        print(f"  [Step 3] LWRR对齐 + 准确率评估...")
        cat_results, mean_acc, mean_kappa = evaluate_alignment(df_all_scored, clip_map, ts_map)

        # 提取维度名称 (robust: LLM返回格式可能不一致)
        dim_names = {}
        for cat, dims in trial_dims.items():
            dim_list = dims.get('dimensions', []) if isinstance(dims, dict) else []
            dim_names[cat] = [d.get('name', 'unnamed') if isinstance(d, dict) else str(d)
                              for d in dim_list]

        trial_results.append({
            'trial': trial,
            'phase': phase_name,
            'temperature': temperature,
            'accuracy': mean_acc,
            'kappa': mean_kappa,
            'n_scored': len(df_all_scored),
            'cat_results': cat_results,
            'dimensions': dim_names,
            'trial_dims': trial_dims,
        })

        # ④ Per-category 最优更新
        any_cat_improved = False
        for cat, (cat_acc, cat_kappa) in cat_results.items():
            prev_best = best_cat_acc.get(cat, -1.0)
            if cat_acc > prev_best:
                improvement = cat_acc - prev_best if prev_best >= 0 else cat_acc
                best_cat_acc[cat] = cat_acc
                best_cat_kappa[cat] = cat_kappa
                best_cat_dims[cat] = trial_dims[cat]
                best_cat_trial[cat] = trial
                any_cat_improved = True
                print(f"    {cat}: {cat_acc*100:.1f}% ★ 新最优 (+{improvement*100:.1f}%)")
            else:
                print(f"    {cat}: {cat_acc*100:.1f}% (最优={prev_best*100:.1f}% @Trial {best_cat_trial.get(cat, '?')})")

        # 全局准确率 (基于per-cat最优的组合)
        if best_cat_acc:
            assembled_acc = np.mean(list(best_cat_acc.values()))
        else:
            assembled_acc = mean_acc

        # 早停判断 (基于是否有任何类别提升)
        if any_cat_improved:
            no_improve_count = 0
            best_global_acc = assembled_acc
            print(f"  → 组合最优Acc={assembled_acc*100:.1f}% (来自{len(best_cat_acc)}个类别的per-cat最优)")
        else:
            no_improve_count += 1
            print(f"  → 本轮Acc={mean_acc*100:.1f}%, 无类别提升 ({no_improve_count}/{E2E_PATIENCE})")
            if E2E_PATIENCE > 0 and no_improve_count >= E2E_PATIENCE:
                print(f"  [EARLY STOP] 连续{no_improve_count}次无任何类别提升, "
                      f"停止搜索 (已运行{trial+1}/{N_DIMENSION_TRIALS})")
                break

    # ──────────────────────────────────────────────────────────
    # 汇总对比
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("E2E Dimension Optimization 汇总")
    print(f"  策略: explore→converge (explore={explore_boundary}, converge={N_DIMENSION_TRIALS - explore_boundary}), "
          f"elite_seed={'ON' if E2E_ELITE_SEED else 'OFF'}, "
          f"patience={E2E_PATIENCE if E2E_PATIENCE > 0 else 'OFF'}")
    print(f"  实际运行 {len(trial_results)}/{N_DIMENSION_TRIALS} trials")
    print(f"{'='*80}")

    if not trial_results:
        print("[WARN] 无有效trial结果")
        return

    # 对比表
    print(f"\n  {'Trial':<8} {'Phase':<10} {'Temp':<8} {'Accuracy':<12} {'Kappa':<10} {'N_Scored':<10}")
    print(f"  {'─'*58}")
    for r in trial_results:
        # 标记: 如果这个trial在某个类别上是最优的
        is_best_for_any = any(
            best_cat_trial.get(cat) == r['trial'] for cat in r.get('cat_results', {})
        )
        marker = " ★" if is_best_for_any else ""
        print(f"  {r['trial']:<8} {r['phase']:<10} {r['temperature']:<8.2f} "
              f"{r['accuracy']*100:<12.1f} {r['kappa']:<10.3f} {r['n_scored']:<10}{marker}")

    # Per-category 最优汇总
    print(f"\n  Per-category 最优维度 (组装自各trial的最优):")
    print(f"  {'─'*58}")
    if best_cat_acc:
        assembled_acc_list = []
        assembled_kappa_list = []
        for cat in CATEGORIES:
            if cat in best_cat_acc:
                acc = best_cat_acc[cat]
                kappa = best_cat_kappa[cat]
                t = best_cat_trial[cat]
                dims = best_cat_dims[cat]
                dim_names = [d.get('name', 'unnamed') if isinstance(d, dict) else str(d)
                             for d in (dims.get('dimensions', []) if isinstance(dims, dict) else [])]
                assembled_acc_list.append(acc)
                assembled_kappa_list.append(kappa)
                print(f"    [{cat}] Acc={acc*100:.1f}%, κ={kappa:.3f} (Trial {t})")
                print(f"      维度: {', '.join(dim_names)}")

        assembled_acc = np.mean(assembled_acc_list) if assembled_acc_list else 0
        assembled_kappa = np.mean(assembled_kappa_list) if assembled_kappa_list else 0
        print(f"\n  组装后全局: Acc={assembled_acc*100:.1f}%, κ={assembled_kappa:.3f}")

        # 与单个trial的全局最优对比
        best_single_trial = max(trial_results, key=lambda x: x['accuracy'])
        print(f"  单trial最优: Trial {best_single_trial['trial']} "
              f"(Acc={best_single_trial['accuracy']*100:.1f}%, κ={best_single_trial['kappa']:.3f})")
        gain = assembled_acc - best_single_trial['accuracy']
        if gain > 0:
            print(f"  Per-cat组装增益: +{gain*100:.1f}%")
    else:
        print("    无有效结果")

    # 复制per-category最优维度到各类别的 STAGE1_DIMENSIONS
    print(f"\n  将per-category最优维度覆盖到各类别维度文件:")
    for cat in CATEGORIES:
        if cat in best_cat_dims:
            dims = best_cat_dims[cat]
            cat_dim_file = get_stage1_dimensions(cat)
            with open(cat_dim_file, 'w', encoding='utf-8') as f:
                json.dump({cat: dims}, f, indent=2, ensure_ascii=False)
            print(f"    已覆盖: {os.path.basename(cat_dim_file)} (来自Trial {best_cat_trial[cat]})")
    print(f"  完成!")

    # 保存汇总CSV (按类别分别保存)
    summary_rows = []
    for r in trial_results:
        row = {k: v for k, v in r.items() if k not in ('dimensions', 'trial_dims', 'cat_results')}
        # 添加per-cat accuracy列
        for cat, (cat_acc, cat_kappa) in r.get('cat_results', {}).items():
            row[f'acc_{cat}'] = cat_acc
            row[f'kappa_{cat}'] = cat_kappa
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)

    saved_summaries = []
    for cat in CATEGORIES:
        summary_file = get_stage6_summary(cat)
        df_summary.to_csv(summary_file, index=False)
        saved_summaries.append(summary_file)
    print(f"  汇总保存: {len(saved_summaries)} 个类别文件")
    for f in saved_summaries:
        print(f"    - {os.path.basename(f)}")

    print(f"\n{'='*80}")
    print("E2E维度优化完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_e2e_dimension_optimization()
