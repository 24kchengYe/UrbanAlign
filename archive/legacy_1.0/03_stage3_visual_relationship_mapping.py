"""
UrbanAlign 2.0 - Stage 3: Hybrid Visual Relationship Mapping (Kernel-based VRM 完整版)
混合特征空间的非线性视觉关系映射

学术核心：
1. 混合空间对齐：融合 CLIP 视觉特征与多维语义分差向量，构建高维感知特征坐标。
2. 核函数加权：利用 RBF 核函数对 K 个近邻标尺进行加权偏差修正，从分布层面最小化 MMD 偏移。
3. 镜像逻辑对齐：支持双向特征匹配，自动修正镜像样本的感知偏见符号（Bias Sign）。

输入：
- Stage 2 的 Pool 结果 (测试集)
- Stage 2 的模式专用标尺结果 (Ref Scored)

输出：
- 对齐后且经过标签修正（Synthetic Winner Updated）的高保真数据集
"""
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 导入配置
# ==============================================================================
from config import (
    DATA_DIR, IMAGE_DIR, OUTPUT_DIR, CLIP_CACHE,
    STAGE2_MODE, STAGE2_OUTPUT_MAP, STAGE3_OUTPUT_MAP,
    CATEGORIES, SELECTION_RATIO,
    K_MAX, EQUAL_PENALTY
)

# 输入文件定义（严格对接 Stage 2 产出）
STAGE2_POOL_FILE = STAGE2_OUTPUT_MAP[STAGE2_MODE]
UNIFIED_VRM_REF_FILE = os.path.join(OUTPUT_DIR, f"stage2_mode{STAGE2_MODE}_vrm_reference_scored.csv")
STAGE3_OUTPUT = STAGE3_OUTPUT_MAP[STAGE2_MODE]

# 算法超参数
TAU = 0.5  # 核函数带宽 (Kernel Bandwidth)，用于控制权重的平滑度

# ==============================================================================
# 2. 核心辅助函数
# ==============================================================================
def parse_semantic_vector(score_json):
    """
    解析 JSON 格式的语义维得分。
    通过 sorted(keys) 确保不同样本间的向量维度顺序完全一致。
    """
    try:
        if pd.isna(score_json): return None
        data = json.loads(score_json)
        # 支持动态维度数量 (如 7维或 8维)
        return np.array([float(data[k]) for k in sorted(data.keys())])
    except:
        return None

def load_clip_features():
    """加载预计算的 CLIP 特征"""
    if not os.path.exists(CLIP_CACHE):
        print(f"[ERROR] 找不到 CLIP 缓存: {CLIP_CACHE}")
        return {}
    data = np.load(CLIP_CACHE)
    clip_map = {}
    for path_item, vec in zip(data['paths'], data['embeddings']):
        path_str = str(path_item[0]) if isinstance(path_item, np.ndarray) else str(path_item)
        img_id = os.path.splitext(os.path.basename(path_str))[0]
        clip_map[img_id] = vec
    return clip_map

# ==============================================================================
# 3. 主程序：Hybrid Kernel VRM Alignment
# ==============================================================================
def run_kernel_vrm_alignment():
    print("\n" + "="*80)
    print(f"UrbanAlign 2.0 - Stage 3: Hybrid Kernel VRM Alignment (Mode {STAGE2_MODE})")
    print("="*80)

    # 1. 数据准备
    clip_map = load_clip_features()
    if not os.path.exists(STAGE2_POOL_FILE) or not os.path.exists(UNIFIED_VRM_REF_FILE):
        print("✗ 错误: Stage 2 产出文件缺失。请确保顺序执行 Stage 2 各模式。")
        return

    df_pool = pd.read_csv(STAGE2_POOL_FILE)
    df_vrm_ref = pd.read_csv(UNIFIED_VRM_REF_FILE)

    aligned_results = []

    # 2. 分类别对齐流程
    for cat in CATEGORIES:
        pool_cat = df_pool[df_pool['category'] == cat].copy()
        ref_cat = df_vrm_ref[df_vrm_ref['category'] == cat].copy()
        if len(pool_cat) == 0 or len(ref_cat) == 0: continue

        print(f"\n[处理类别: {cat}] 执行多维语义对齐...")

        # ----------------------------------------------------------------------
        # 2.1 映射到混合差分空间 (Feature Concatenation)
        # ----------------------------------------------------------------------
        def build_hybrid_features(df, is_ref=False):
            hybrid_feats = []
            info_list = [] # 存储对应的 Bias 或分数信息

            for idx, r in df.iterrows():
                f_l, f_r = clip_map.get(str(r['left_id'])), clip_map.get(str(r['right_id']))
                if f_l is None or f_r is None: continue

                # A. 提取语义分差向量
                s_vec_a = parse_semantic_vector(r.get('image_a_scores'))
                s_vec_b = parse_semantic_vector(r.get('image_b_scores'))
                if s_vec_a is None or s_vec_b is None: continue

                s_diff = s_vec_a - s_vec_b
                ai_delta_scalar = np.mean(s_diff)

                # B. 构建特征：Concatenate([L_clip, R_clip], Semantic_Diff)
                # 语义分差也进行归一化，防止量纲影响相似度计算
                v_std = np.concatenate([
                    normalize(f_l.reshape(1,-1))[0],
                    normalize(f_r.reshape(1,-1))[0],
                    normalize(s_diff.reshape(1,-1))[0]
                ])

                if is_ref:
                    # 确定人类真值分差标尺 (Left=+2, Right=-2, Equal=0)
                    h_win = str(r['human_winner']).lower()
                    h_delta = 2.0 if h_win == 'left' else (-2.0 if h_win == 'right' else 0.0)

                    # 存储正向特征及其 Bias
                    hybrid_feats.append(v_std)
                    info_list.append({
                        'bias': ai_delta_scalar - h_delta,
                        'is_flipped': False,
                        'original_idx': idx
                    })

                    # --- 镜像增强：处理方向逻辑的核心 ---
                    v_flip = np.concatenate([
                        normalize(f_r.reshape(1,-1))[0],
                        normalize(f_l.reshape(1,-1))[0],
                        normalize((-s_diff).reshape(1,-1))[0]
                    ])
                    hybrid_feats.append(v_flip)
                    # 镜像偏见符号翻转：(-AI) - (-Human)
                    info_list.append({
                        'bias': (-ai_delta_scalar) - (-h_delta),
                        'is_flipped': True,
                        'original_idx': idx
                    })
                else:
                    hybrid_feats.append(v_std)
                    info_list.append({'ai_delta': ai_delta_scalar})

            return np.array(hybrid_feats), info_list

        ref_vecs, ref_info = build_hybrid_features(ref_cat, is_ref=True)
        pool_vecs, pool_info = build_hybrid_features(pool_cat, is_ref=False)

        if len(ref_vecs) == 0 or len(pool_vecs) == 0: continue

        # ----------------------------------------------------------------------
        # 2.2 执行核函数回归对齐 (Kernel Alignment)
        # ----------------------------------------------------------------------
        # 计算 Pool 与 Ref（含镜像）的相似度矩阵 [N_pool, N_ref_aug]
        sim_matrix = cosine_similarity(pool_vecs, ref_vecs)

        cat_aligned_rows = []
        for i in range(len(pool_vecs)):
            # 选取 K 个近邻 (K_MAX 建议为 5)
            k = min(K_MAX, len(ref_vecs))
            top_k_idx = np.argsort(sim_matrix[i])[-k:][::-1]

            similarities = sim_matrix[i][top_k_idx]
            biases = np.array([ref_info[j]['bias'] for j in top_k_idx])

            # --- RBF 核权重计算 ---
            weights = np.exp(similarities / TAU)
            weights /= np.sum(weights) # 归一化

            # 计算局部加权偏见 (Local Weighted Bias)
            local_bias = np.dot(weights, biases)

            # 执行校准
            raw_ai_delta = pool_info[i]['ai_delta']
            calibrated_delta = raw_ai_delta - local_bias

            # --- 核心修正：基于校准后的分数重新判定胜者 ---
            # 设置灵敏度阈值 0.5
            if calibrated_delta > 0.5:
                updated_winner = 'left'
            elif calibrated_delta < -0.5:
                updated_winner = 'right'
            else:
                updated_winner = 'equal'

            # --- 计算保真度 (Fidelity) ---
            # 逻辑：考察近邻标尺中，AI 原有的判断与人类的一致性
            agreement_list = []
            for idx in top_k_idx:
                orig_r = ref_cat.iloc[ref_info[idx]['original_idx']]
                is_correct = 1.0 if str(orig_r['synthetic_winner']).lower() == str(orig_r['human_winner']).lower() else 0.0
                agreement_list.append(is_correct)

            # 加权得到保真度得分
            fidelity = np.dot(weights, agreement_list)

            # 更新数据行
            row_data = pool_cat.iloc[i].to_dict()
            row_data.update({
                'synthetic_winner': updated_winner,   # 对齐后的标签
                'fidelity_score': fidelity,           # 逻辑保真度
                'calibrated_delta': calibrated_delta,  # 校准后的分差
                'vrm_bias': local_bias                # 计算出的偏见量
            })
            cat_aligned_rows.append(row_data)

        # ----------------------------------------------------------------------
        # 2.3 高保真筛选 (Selection)
        # ----------------------------------------------------------------------
        df_cat_aligned = pd.DataFrame(cat_aligned_rows)
        n_keep = max(1, int(len(df_cat_aligned) * SELECTION_RATIO))
        # 根据保真度进行硬筛选，仅保留逻辑最一致的“高真数据”
        df_cat_final = df_cat_aligned.sort_values('fidelity_score', ascending=False).head(n_keep)
        aligned_results.append(df_cat_final)

    # 3. 保存与验证
    if not aligned_results:
        print("✗ 警告: 未能成功对齐任何样本。")
        return

    df_final = pd.concat(aligned_results, ignore_index=True)
    df_final.to_csv(STAGE3_OUTPUT, index=False)

    print(f"\n{'='*80}")
    print(f"Stage 3 完成！(Kernel-based Hybrid VRM)")
    print(f"{'='*80}")
    print(f"  对齐模式: Mode {STAGE2_MODE}")
    print(f"  输入样本: {len(df_pool)} 对")
    print(f"  最终保留: {len(df_final)} 对 (保留比例: {SELECTION_RATIO*100}%)")
    print(f"  平均保真度: {df_final['fidelity_score'].mean():.3f}")

    # 分类别预估准确率 (对比 human_winner)
    if 'human_winner' in df_final.columns:
        print(f"\n  【对齐性能预估】")
        for cat in df_final['category'].unique():
            sub = df_final[df_final['category'] == cat]
            acc = (sub['synthetic_winner'] == sub['human_winner']).mean()
            print(f"    {cat:12s}: {len(sub):4d}对 | Accuracy: {acc*100:5.2f}%")

    # 数据隔离最后校验
    ref_ids = set(df_vrm_ref['left_id'].tolist() + df_vrm_ref['right_id'].tolist())
    pool_ids = set(df_final['left_id'].tolist() + df_final['right_id'].tolist())
    leakage = ref_ids & pool_ids
    if leakage:
        print(f"\n  ⚠️  告警: 发现 {len(leakage)} 个图片 ID 在标尺集和合成集中重叠，请检查隔离代码。")
    else:
        print(f"\n  ✓ 物理隔离验证通过：合成数据集与标尺集完全独立。")

    print(f"\n下一步: python abc_stage4_comprehensive_evaluation.py")

if __name__ == "__main__":
    run_kernel_vrm_alignment()