"""
UrbanAlign Stage 4: Evaluation
评估UrbanAlign在不同阶段和基准方法上的性能
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import seaborn as sns

# ==============================================================================
# 配置区域
# ==============================================================================
# 导入全局配置
from config import (
    DATA_DIR, HUMAN_CHOICES_CSV, OUTPUT_DIR,
    STAGE2_SYNTHETIC_POOL, STAGE3_ALIGNED_DATA,
    STAGE4_RESULTS, STAGE4_PLOT,
    CATEGORIES,
    FONT_SIZE_TITLE, FONT_SIZE_AXIS_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TICK_LABEL,
    FIG_SIZE_EVALUATION
)

# 为了兼容性保留的别名
STAGE2_FILE = STAGE2_SYNTHETIC_POOL
STAGE3_FILE = STAGE3_ALIGNED_DATA
RESULTS_CSV = STAGE4_RESULTS
COMPARISON_PLOT = STAGE4_PLOT
FONT_SIZE_LABEL = FONT_SIZE_AXIS_LABEL
FONT_SIZE_TICK = FONT_SIZE_TICK_LABEL
FIG_SIZE = FIG_SIZE_EVALUATION

# ==============================================================================
# 评估函数
# ==============================================================================
def evaluate_predictions(df_synthetic, df_human):
    """
    评估合成标注与人类标注的一致性
    Returns: DataFrame with metrics per category
    """
    results = []

    for cat in CATEGORIES:
        df_syn_cat = df_synthetic[df_synthetic['category'] == cat]
        df_human_cat = df_human[df_human['category'] == cat]

        # 合并数据以匹配ID对
        merged = pd.merge(
            df_syn_cat,
            df_human_cat,
            left_on=['left_id', 'right_id'],
            right_on=['left_id', 'right_id'],
            suffixes=('_syn', '_human')
        )

        if len(merged) == 0:
            print(f"[WARN] {cat}: 无匹配数据")
            continue

        y_true = merged['winner'].values
        y_pred = merged['synthetic_winner'].values

        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, labels=['left', 'right', 'equal'])

        # F1计算(将'left'视为正类)
        y_true_binary = (y_true == 'left').astype(int)
        y_pred_binary = (y_pred == 'left').astype(int)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        results.append({
            'category': cat,
            'accuracy': accuracy,
            'kappa': kappa,
            'f1_score': f1,
            'n_samples': len(merged)
        })

        print(f"{cat:12s}: Acc={accuracy:.3f}, Kappa={kappa:.3f}, F1={f1:.3f}, N={len(merged)}")

    return pd.DataFrame(results)

# ==============================================================================
# 可视化
# ==============================================================================
def plot_comparison(df_stage2, df_stage3):
    """绘制Stage 2 vs Stage 3性能对比"""

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    plt.style.use('seaborn-v0_8-whitegrid')

    metrics = ['accuracy', 'kappa', 'f1_score']
    titles = ['Accuracy', "Cohen's Kappa", 'F1 Score']

    x = np.arange(len(CATEGORIES))
    width = 0.35

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        stage2_values = [df_stage2[df_stage2['category'] == cat][metric].values[0]
                         if len(df_stage2[df_stage2['category'] == cat]) > 0 else 0
                         for cat in CATEGORIES]

        stage3_values = [df_stage3[df_stage3['category'] == cat][metric].values[0]
                         if len(df_stage3[df_stage3['category'] == cat]) > 0 else 0
                         for cat in CATEGORIES]

        ax.bar(x - width/2, stage2_values, width, label='Stage 2 (Raw Synthesis)',
               color='#3498db', edgecolor='white')
        ax.bar(x + width/2, stage3_values, width, label='Stage 3 (VRM Aligned)',
               color='#e74c3c', edgecolor='white')

        ax.set_title(title, fontsize=FONT_SIZE_TITLE, weight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in CATEGORIES],
                          rotation=30, ha='right', fontsize=FONT_SIZE_TICK)
        ax.set_ylabel(title, fontsize=FONT_SIZE_LABEL)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='lower right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('UrbanAlign Performance: Stage 2 vs Stage 3',
                 fontsize=FONT_SIZE_TITLE + 4, weight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(COMPARISON_PLOT, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] 对比图已保存: {COMPARISON_PLOT}")

    plt.show()

# ==============================================================================
# 主函数
# ==============================================================================
def run_stage4_evaluation():
    """Stage 4主流程"""
    print("\n" + "="*80)
    print("UrbanAlign Stage 4: Evaluation")
    print("="*80 + "\n")

    # 1. 加载人类标注
    print("[STEP 1] 加载人类标注基准...")
    df_human = pd.read_csv(HUMAN_CHOICES_CSV)
    df_human['left_id'] = df_human['left_id'].astype(str)
    df_human['right_id'] = df_human['right_id'].astype(str)
    print(f"  人类标注: {len(df_human)} 条")

    # 2. 评估Stage 2
    print("\n[STEP 2] 评估Stage 2 (原始合成)...")
    if not os.path.exists(STAGE2_FILE):
        print(f"  错误: Stage 2文件不存在: {STAGE2_FILE}")
        return

    df_stage2 = pd.read_csv(STAGE2_FILE)
    df_stage2['left_id'] = df_stage2['left_id'].astype(str)
    df_stage2['right_id'] = df_stage2['right_id'].astype(str)

    print("\n=== Stage 2 Results ===")
    results_stage2 = evaluate_predictions(df_stage2, df_human)

    # 3. 评估Stage 3
    print("\n[STEP 3] 评估Stage 3 (VRM对齐)...")
    if not os.path.exists(STAGE3_FILE):
        print(f"  错误: Stage 3文件不存在: {STAGE3_FILE}")
        return

    df_stage3 = pd.read_csv(STAGE3_FILE)
    df_stage3['left_id'] = df_stage3['left_id'].astype(str)
    df_stage3['right_id'] = df_stage3['right_id'].astype(str)

    print("\n=== Stage 3 Results ===")
    results_stage3 = evaluate_predictions(df_stage3, df_human)

    # 4. 合并结果
    results_stage2['stage'] = 'Stage2_RawSynthesis'
    results_stage3['stage'] = 'Stage3_VRM_Aligned'

    df_all_results = pd.concat([results_stage2, results_stage3], ignore_index=True)
    df_all_results.to_csv(RESULTS_CSV, index=False)

    print(f"\n[INFO] 评估结果已保存: {RESULTS_CSV}")

    # 5. 绘制对比
    print("\n[STEP 4] 生成可视化对比...")
    plot_comparison(results_stage2, results_stage3)

    # 6. 打印汇总报告
    print("\n" + "="*80)
    print("汇总报告")
    print("="*80)

    print("\nStage 2 (Raw Synthesis)平均性能:")
    print(f"  Accuracy: {results_stage2['accuracy'].mean():.3f}")
    print(f"  Kappa:    {results_stage2['kappa'].mean():.3f}")
    print(f"  F1 Score: {results_stage2['f1_score'].mean():.3f}")

    print("\nStage 3 (VRM Aligned)平均性能:")
    print(f"  Accuracy: {results_stage3['accuracy'].mean():.3f}")
    print(f"  Kappa:    {results_stage3['kappa'].mean():.3f}")
    print(f"  F1 Score: {results_stage3['f1_score'].mean():.3f}")

    print("\n性能提升:")
    acc_gain = results_stage3['accuracy'].mean() - results_stage2['accuracy'].mean()
    kappa_gain = results_stage3['kappa'].mean() - results_stage2['kappa'].mean()
    f1_gain = results_stage3['f1_score'].mean() - results_stage2['f1_score'].mean()

    print(f"  Accuracy: +{acc_gain:.3f} ({acc_gain/results_stage2['accuracy'].mean()*100:+.1f}%)")
    print(f"  Kappa:    +{kappa_gain:.3f} ({kappa_gain/results_stage2['kappa'].mean()*100:+.1f}%)")
    print(f"  F1 Score: +{f1_gain:.3f} ({f1_gain/results_stage2['f1_score'].mean()*100:+.1f}%)")

    print("\n[完成] 评估流程结束!")

if __name__ == "__main__":
    run_stage4_evaluation()
