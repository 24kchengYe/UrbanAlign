"""
UrbanAlign 2.0 - Stage 4: Comprehensive Evaluation Module (完整评估模块)

功能：
1. 对比不同实验配置的性能(Mode 1 vs Mode 2, 不同α值等)
2. 评估指标：准确率、Kappa、F1-score
3. 分维度详细分析
4. 样本量影响分析
5. 生成对比报告和可视化

参考：02_stage2_rule_guided_synthesis-In-Context Alignment.py的评估模块
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. 导入配置
# ==============================================================================
from config import (
    OUTPUT_DIR, CATEGORIES,
    EXCLUDE_EQUAL_IN_EVAL, get_split_data,
    get_stage2_output, get_stage3_output, get_stage4_output, get_stage4_plot,
    get_stage7_output, get_stage2_sampled_pairs,
)

MODE_NAMES = {
    1: "单张直接", 2: "成对直接", 3: "单张多智能体", 4: "成对多智能体"
}

def _detect_experiments(category):
    """自动检测指定类别的所有实验文件"""
    experiments = []

    # 检测2.0的4种Mode（原始+对齐）
    for mode in [1, 2, 3, 4]:
        stage2_file = get_stage2_output(mode, category)
        if os.path.exists(stage2_file):
            experiments.append({
                'name': f'2.0-Mode{mode}-Raw',
                'file': stage2_file,
                'description': f'{MODE_NAMES[mode]} (原始)',
                'stage': 'Stage2',
                'mode': mode
            })

        stage3_file = get_stage3_output(mode, category)
        if os.path.exists(stage3_file):
            experiments.append({
                'name': f'2.0-Mode{mode}-Aligned',
                'file': stage3_file,
                'description': f'{MODE_NAMES[mode]} (VRM对齐)',
                'stage': 'Stage3',
                'mode': mode
            })

    # 检测Stage 7 traditional baselines
    for key in ['c0', 'c1', 'c2', 'c3']:
        baseline_file = get_stage7_output(key, category)
        if os.path.exists(baseline_file):
            baseline_names = {
                'c0': ('Baseline-C0-ResNet', '2016 Legacy孪生网络'),
                'c1': ('Baseline-C1-Siamese', '端到端孪生网络'),
                'c2': ('Baseline-C2-SegReg', '要素分割回归'),
                'c3': ('Baseline-C3-ZeroShot', '零样本VLM直接选择'),
            }
            name, desc = baseline_names[key]
            experiments.append({
                'name': name,
                'file': baseline_file,
                'description': desc,
                'stage': 'Baseline',
                'mode': None
            })

    return experiments

# ==============================================================================
# 2. 评估函数
# ==============================================================================
def load_experiment_data(file_path):
    """加载实验数据"""
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

def calculate_metrics(df, exclude_equal=False):
    """
    计算评估指标

    参数:
        df: DataFrame包含human_winner和synthetic_winner列
        exclude_equal: 是否排除'equal'标签
    """
    if df is None or len(df) == 0:
        return None

    df = df.copy()

    # 排除equal — 仅排除人类标注为equal的对, 模型预测equal在非equal对上算预测错误
    if exclude_equal:
        df = df[df['human_winner'] != 'equal']

    if len(df) < 2:
        return None

    try:
        # 基础指标
        acc = accuracy_score(df['human_winner'], df['synthetic_winner'])
        kappa = cohen_kappa_score(df['human_winner'], df['synthetic_winner'])

        # F1 score（需要准备标签编码）
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        all_labels = pd.concat([df['human_winner'], df['synthetic_winner']]).unique()
        le.fit(all_labels)

        y_true = le.transform(df['human_winner'])
        y_pred = le.transform(df['synthetic_winner'])

        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return {
            'n_samples': len(df),
            'accuracy': acc,
            'kappa': kappa,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    except Exception as e:
        print(f"[WARN] 计算指标失败: {e}")
        return None

def dimension_level_analysis(df_stage3):
    """
    维度级分析

    分析各维度的分数分布和判别力
    """
    if df_stage3 is None or 'image_a_scores' not in df_stage3.columns:
        return None

    # 解析维度分数
    def parse_scores(x):
        try:
            return json.loads(x) if isinstance(x, str) else x
        except:
            return {}

    df_stage3['image_a_scores'] = df_stage3['image_a_scores'].apply(parse_scores)
    df_stage3['image_b_scores'] = df_stage3['image_b_scores'].apply(parse_scores)

    dimension_stats = []

    for cat in df_stage3['category'].unique():
        df_cat = df_stage3[df_stage3['category'] == cat]

        # 获取所有维度名称
        all_dims = set()
        for scores in df_cat['image_a_scores']:
            all_dims.update(scores.keys())

        for dim in all_dims:
            # 收集该维度的所有分数差异
            deltas = []
            correct_predictions = []

            for _, row in df_cat.iterrows():
                score_a = row['image_a_scores'].get(dim, 5.0)
                score_b = row['image_b_scores'].get(dim, 5.0)
                delta = score_a - score_b

                deltas.append(delta)

                # 判断是否正确预测
                if row['human_winner'] == 'left' and delta > 0:
                    correct_predictions.append(1)
                elif row['human_winner'] == 'right' and delta < 0:
                    correct_predictions.append(1)
                elif row['human_winner'] == 'equal' and abs(delta) < 1.0:
                    correct_predictions.append(1)
                else:
                    correct_predictions.append(0)

            dimension_stats.append({
                'category': cat,
                'dimension': dim,
                'mean_delta': np.mean(deltas),
                'std_delta': np.std(deltas),
                'discriminative_power': np.mean(correct_predictions),
                'n_samples': len(deltas)
            })

    return pd.DataFrame(dimension_stats)

# ==============================================================================
# 3. 主评估函数
# ==============================================================================
def _evaluate_experiments(experiments, category):
    """评估指定类别的所有实验，返回结果列表"""
    all_results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"评估: {exp['name']} ({category})")
        print(f"{'='*60}")

        df = load_experiment_data(exp['file'])

        if df is None:
            print(f"  [SKIP] 文件不存在: {os.path.basename(exp['file'])}")
            continue

        print(f"  数据加载: {len(df)}对")

        # 根据配置决定评估方式
        if EXCLUDE_EQUAL_IN_EVAL:
            metrics = calculate_metrics(df, exclude_equal=True)
            if metrics:
                print(f"\n  【总体性能】(排除Equal)")
                print(f"    样本数: {metrics['n_samples']}")
                print(f"    准确率: {metrics['accuracy']*100:.2f}%")
                print(f"    Kappa: {metrics['kappa']:.3f}")
                print(f"    F1-Macro: {metrics['f1_macro']:.3f}")

                all_results.append({
                    'category': category,
                    'experiment': exp['name'],
                    'description': exp['description'],
                    'stage': exp.get('stage', 'Unknown'),
                    'mode': exp.get('mode', None),
                    'exclude_equal': True,
                    **metrics
                })
        else:
            metrics_all = calculate_metrics(df, exclude_equal=False)
            metrics_no_equal = calculate_metrics(df, exclude_equal=True)

            if metrics_all:
                print(f"\n  【总体性能】(包含Equal)")
                print(f"    样本数: {metrics_all['n_samples']}")
                print(f"    准确率: {metrics_all['accuracy']*100:.2f}%")
                print(f"    Kappa: {metrics_all['kappa']:.3f}")
                print(f"    F1-Macro: {metrics_all['f1_macro']:.3f}")

                all_results.append({
                    'category': category,
                    'experiment': exp['name'],
                    'description': exp['description'],
                    'stage': exp.get('stage', 'Unknown'),
                    'mode': exp.get('mode', None),
                    'exclude_equal': False,
                    **metrics_all
                })

            if metrics_no_equal:
                print(f"\n  【总体性能】(排除Equal)")
                print(f"    样本数: {metrics_no_equal['n_samples']}")
                print(f"    准确率: {metrics_no_equal['accuracy']*100:.2f}%")
                print(f"    Kappa: {metrics_no_equal['kappa']:.3f}")
                print(f"    F1-Macro: {metrics_no_equal['f1_macro']:.3f}")

                all_results.append({
                    'category': category,
                    'experiment': exp['name'],
                    'description': exp['description'],
                    'stage': exp.get('stage', 'Unknown'),
                    'mode': exp.get('mode', None),
                    'exclude_equal': True,
                    **metrics_no_equal
                })

    return all_results


def run_comprehensive_evaluation():
    """运行完整评估（按类别）"""
    print("\n" + "="*80)
    print("UrbanAlign 2.0 - Comprehensive Evaluation Module")
    print("="*80 + "\n")

    # 物理隔离协议说明
    print("【数据集物理隔离协议】")
    print("  ✓ Reference Set: 用于Stage 1维度定义 + Stage 3 VRM对齐")
    print("  ✓ Synthesis Pool: 用于Stage 2采样合成 + Stage 4评估")
    print("  ✓ 保证: Reference ∩ Pool = ∅ (完全不重叠)")
    print()

    # 显示各类别的划分情况
    for cat in CATEGORIES:
        df_ref, df_pool = get_split_data(category=cat)
        print(f"  {cat}: Ref={len(df_ref)}对, Pool={len(df_pool)}对")
    print()

    # 按类别评估
    for cat in CATEGORIES:
        evaluation_report = get_stage4_output('all_modes_comparison', cat)
        ablation_report = get_stage4_output('ablation_analysis', cat)
        dimension_analysis_file = get_stage4_output('dimension_discriminability', cat)
        evaluation_plot = get_stage4_plot(cat)

        # 跳过已完成的类别 (自动检测上游数据是否更新)
        if os.path.exists(evaluation_report):
            stage4_mtime = os.path.getmtime(evaluation_report)
            upstream_newer = False
            # 检查所有上游 Stage 2/3 输出和采样缓存
            for m in [1, 2, 3, 4]:
                for upstream_file in [get_stage2_output(m, cat), get_stage3_output(m, cat)]:
                    if os.path.exists(upstream_file) and os.path.getmtime(upstream_file) > stage4_mtime:
                        upstream_newer = True
                        break
                if upstream_newer:
                    break
            sampled_cache = get_stage2_sampled_pairs(cat)
            if os.path.exists(sampled_cache) and os.path.getmtime(sampled_cache) > stage4_mtime:
                upstream_newer = True
            if upstream_newer:
                print(f"\n[STALE] {cat}: 上游数据已更新, 删除旧评估结果重新计算...")
                for stale in [evaluation_report, ablation_report, dimension_analysis_file, evaluation_plot]:
                    if os.path.exists(stale):
                        os.remove(stale)
            else:
                print(f"\n[SKIP] {cat}: {os.path.basename(evaluation_report)} already exists")
                continue

        print(f"\n{'='*80}")
        print(f"评估类别: {cat.upper()}")
        print(f"{'='*80}")

        # 自动检测该类别的实验文件
        experiments = _detect_experiments(cat)
        if not experiments:
            print(f"  [SKIP] {cat}: 无可用实验文件")
            continue

        print(f"  检测到 {len(experiments)} 个实验配置")

        # 评估所有实验
        all_results = _evaluate_experiments(experiments, cat)

        # 保存对比结果
        if len(all_results) > 0:
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(evaluation_report, index=False)

            print(f"\n{'='*80}")
            print(f"评估报告已保存 ({cat})")
            print(f"{'='*80}")
            print(f"  文件: {os.path.basename(evaluation_report)}")

            # 生成对比表
            print(f"\n  实验性能对比:")
            print(df_results[['experiment', 'exclude_equal', 'n_samples',
                             'accuracy', 'kappa', 'f1_macro']].to_string(index=False))

            # 消融分析
            ablation_analysis(df_results, ablation_report)

            # 可视化
            generate_comparison_plot(df_results, evaluation_plot)

        # 维度级分析（仅对Stage 3数据）
        print(f"\n  维度级判别力分析 ({cat}):")

        all_dim_analyses = []
        for mode in [1, 2, 3, 4]:
            stage3_file = get_stage3_output(mode, cat)
            if os.path.exists(stage3_file):
                df_stage3 = pd.read_csv(stage3_file)
                dim_analysis = dimension_level_analysis(df_stage3)

                if dim_analysis is not None and len(dim_analysis) > 0:
                    dim_analysis['mode'] = mode
                    all_dim_analyses.append(dim_analysis)

                    print(f"\n    Mode {mode} 各维度判别力（Discriminative Power）:")
                    for _, row in dim_analysis.iterrows():
                        print(f"      {row['category']:12s} - {row['dimension']:25s}: "
                              f"{row['discriminative_power']*100:5.1f}% | "
                              f"Mean Δ: {row['mean_delta']:+5.2f} ± {row['std_delta']:.2f}")

        if all_dim_analyses:
            pd.concat(all_dim_analyses, ignore_index=True).to_csv(dimension_analysis_file, index=False)
            print(f"\n    维度分析已保存: {os.path.basename(dimension_analysis_file)}")

        print(f"\n  【{cat} 输出文件】")
        print(f"    - {os.path.basename(evaluation_report)}")
        print(f"    - {os.path.basename(ablation_report)}")
        print(f"    - {os.path.basename(dimension_analysis_file)}")
        print(f"    - {os.path.basename(evaluation_plot)}")

    print(f"\n{'='*80}")
    print("评估完成！")
    print(f"{'='*80}")
    print(f"\n  【数据隔离保证】")
    print(f"    ✓ 评估使用Pool集的真值（未被Reference污染）")
    print(f"    ✓ Stage 3对齐使用Reference集（未被Pool污染）")
    print(f"    ✓ 物理隔离确保实验严谨性")

def ablation_analysis(df_results, ablation_report):
    """
    消融分析：计算各因素的贡献

    维度1: 对比上下文 (成对 vs 单张)
    维度2: 多智能体 (多智能体 vs 单次)
    维度3: VRM对齐 (Aligned vs Raw)
    """
    ablation_results = []

    # 提取Mode结果（仅Stage3对齐后）
    mode_perf = {}
    for mode in [1, 2, 3, 4]:
        mode_data = df_results[(df_results['mode'] == mode) & (df_results['stage'] == 'Stage3')]
        if len(mode_data) > 0:
            mode_perf[mode] = mode_data.iloc[0]['accuracy']

    if len(mode_perf) >= 4:
        # 对比上下文贡献
        pairwise_gain_single = mode_perf.get(2, 0) - mode_perf.get(1, 0)  # Mode 2 vs 1（单次）
        pairwise_gain_multi = mode_perf.get(4, 0) - mode_perf.get(3, 0)   # Mode 4 vs 3（多智能体）

        ablation_results.append({
            'factor': '对比上下文 (Pairwise vs Single)',
            'comparison_single': 'Mode 2 vs 1',
            'gain_single': pairwise_gain_single,
            'comparison_multi': 'Mode 4 vs 3',
            'gain_multi': pairwise_gain_multi,
            'avg_gain': (pairwise_gain_single + pairwise_gain_multi) / 2
        })

        # 多智能体贡献
        multiagent_gain_single = mode_perf.get(3, 0) - mode_perf.get(1, 0)  # Mode 3 vs 1（单张）
        multiagent_gain_pair = mode_perf.get(4, 0) - mode_perf.get(2, 0)    # Mode 4 vs 2（成对）

        ablation_results.append({
            'factor': '多智能体 (Multi-Agent vs Single-Shot)',
            'comparison_single': 'Mode 3 vs 1',
            'gain_single': multiagent_gain_single,
            'comparison_multi': 'Mode 4 vs 2',
            'gain_multi': multiagent_gain_pair,
            'avg_gain': (multiagent_gain_single + multiagent_gain_pair) / 2
        })

    # VRM对齐贡献（每种Mode）
    for mode in [1, 2, 3, 4]:
        raw_data = df_results[(df_results['mode'] == mode) & (df_results['stage'] == 'Stage2')]
        aligned_data = df_results[(df_results['mode'] == mode) & (df_results['stage'] == 'Stage3')]

        if len(raw_data) > 0 and len(aligned_data) > 0:
            vrm_gain = aligned_data.iloc[0]['accuracy'] - raw_data.iloc[0]['accuracy']
            ablation_results.append({
                'factor': f'VRM对齐 (Mode {mode})',
                'comparison_single': 'Aligned vs Raw',
                'gain_single': vrm_gain,
                'comparison_multi': '',
                'gain_multi': 0,
                'avg_gain': vrm_gain
            })

    if len(ablation_results) > 0:
        df_ablation = pd.DataFrame(ablation_results)
        df_ablation.to_csv(ablation_report, index=False)

        print(f"\n  消融分析:")
        print(df_ablation.to_string(index=False))
        print(f"\n    已保存: {os.path.basename(ablation_report)}")

def generate_comparison_plot(df_results, evaluation_plot):
    """生成性能对比图"""
    if df_results is None or len(df_results) == 0:
        return

    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics = ['accuracy', 'kappa', 'f1_macro']
        titles = ['Accuracy', "Cohen's Kappa", 'F1-Score (Macro)']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]

            # 分组：包含Equal vs 排除Equal
            for exclude in [False, True]:
                df_sub = df_results[df_results['exclude_equal'] == exclude]

                if len(df_sub) > 0:
                    label = 'Exclude Equal' if exclude else 'Include Equal'
                    x_pos = np.arange(len(df_sub))
                    offset = 0.2 if exclude else -0.2

                    ax.bar(x_pos + offset, df_sub[metric], width=0.35,
                          label=label, alpha=0.8)

            ax.set_xlabel('Experiment')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticks(np.arange(len(df_results)//2))
            ax.set_xticklabels([name[:20] for name in df_results['experiment'].unique()],
                              rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(evaluation_plot, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n    可视化已保存: {os.path.basename(evaluation_plot)}")

    except Exception as e:
        print(f"[WARN] 生成图表失败: {e}")

if __name__ == "__main__":
    run_comprehensive_evaluation()
