import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# 1. 全局样式与字体控制 (按要求设为 22)
# =================================================================
FONT_SIZE_TITLE = 22
FONT_SIZE_LABEL = 22
FONT_SIZE_TICK = 22
FONT_SIZE_LEGEND = 22
LINE_WIDTH = 2.5
MARKER_SIZE = 10
FIG_SIZE = (14, 10)

# =================================================================
# 2. 配置区域
# =================================================================
DATA_DIR = os.getenv(
    "PLACE_PULSE_DIR",
    r"H:\RawData13-全球街景\mit place pulse\01 Place Pluse2.0数据集\01 Place Pulse 2.0论文数据集"
)
HUMAN_CHOICES_CSV = os.path.join(DATA_DIR, "final_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urbanalign_outputs")

# 实验真值抽样阈值 (建议设为 3)
THRESHOLD_RELIABLE = 3

CATEGORIES = ['wealthy', 'safety', 'lively', 'beautiful', 'boring', 'depressing']
THRESHOLDS_PAIRS = [1, 2, 3, 4, 5]
THRESHOLDS_IMAGES = [1, 5, 10, 15, 20]


def run_comprehensive_analysis_and_export():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"正在读取原始数据: {HUMAN_CHOICES_CSV}...")
    df = pd.read_csv(HUMAN_CHOICES_CSV)

    # --- A. 图像对归一化 (处理镜像样本) ---
    print("正在进行归一化处理 (确保 ID_A < ID_B 且坐标同步交换)...")
    df_normalized = df.copy()
    cond = df_normalized['left_id'] > df_normalized['right_id']

    # 1. 交换 ID
    df_normalized.loc[cond, ['left_id', 'right_id']] = df_normalized.loc[cond, ['right_id', 'left_id']].values
    # 2. 交换 经纬度 (核心：坐标必须随图片ID一起走)
    df_normalized.loc[cond, ['left_lat', 'left_long', 'right_lat', 'right_long']] = \
        df_normalized.loc[cond, ['right_lat', 'right_long', 'left_lat', 'left_long']].values
    # 3. 翻转 winner 标签
    swap_map = {'left': 'right', 'right': 'left', 'equal': 'equal'}
    df_normalized.loc[cond, 'winner'] = df_normalized.loc[cond, 'winner'].map(swap_map)

    # 创建唯一对子 Key (包含维度)
    df_normalized['pair_key'] = df_normalized['category'] + "_" + df_normalized['left_id'].astype(str) + "_" + \
                                df_normalized['right_id'].astype(str)

    pair_results = []
    image_results = []

    # 用于存储符合条件的原始行和聚合行
    raw_reliable_list = []
    agg_reliable_list = []

    print("开始分维度统计与可靠样本筛选...")
    for cat in CATEGORIES:
        print(f"  维度: {cat}")
        df_cat = df_normalized[df_normalized['category'] == cat]

        # --- 1. 对子聚合分析 ---
        grouped = df_cat.groupby('pair_key')

        # 多数投票赢家函数
        def get_majority_winner(x):
            counts = x.value_counts()
            return counts.idxmax()

        # 统计每个对子的关键信息
        pair_stats = grouped.agg({
            'winner': ['count', get_majority_winner, lambda x: x.value_counts().max()],
            'left_id': 'first', 'right_id': 'first',
            'left_lat': 'first', 'left_long': 'first',
            'right_lat': 'first', 'right_long': 'first',
            'category': 'first'
        })
        pair_stats.columns = ['vote_count', 'winner', 'max_votes', 'left_id', 'right_id',
                              'left_lat', 'left_long', 'right_lat', 'right_long', 'category']
        pair_stats['consensus'] = pair_stats['max_votes'] / pair_stats['vote_count']

        # --- 2. 提取可靠样本集 ---
        # 识别符合阈值的 key
        reliable_keys = pair_stats[pair_stats['vote_count'] >= THRESHOLD_RELIABLE].index

        # A. 原始版 (Raw Reliable): 保留所有投票记录
        cat_raw_reliable = df_cat[df_cat['pair_key'].isin(reliable_keys)].copy()
        raw_reliable_list.append(cat_raw_reliable)

        # B. 聚合版 (Aggregated Reliable): 每对仅一行
        cat_agg_reliable = pair_stats[pair_stats['vote_count'] >= THRESHOLD_RELIABLE].copy()
        agg_reliable_list.append(cat_agg_reliable)

        # --- 3. 统计绘图数据 ---
        for t in THRESHOLDS_PAIRS:
            subset = pair_stats[pair_stats['vote_count'] >= t]
            avg_cons = subset['consensus'].mean() * 100 if t >= 2 and len(subset) > 0 else np.nan
            pair_results.append({'Category': cat, 'Threshold': t, 'Count': len(subset), 'Avg_Consensus': avg_cons})

        all_ids = pd.concat([df_cat['left_id'], df_cat['right_id']])
        img_counts = all_ids.value_counts()
        for t in THRESHOLDS_IMAGES:
            image_results.append({'Category': cat, 'Threshold': t, 'Count': (img_counts >= t).sum()})

    # --- B. 导出数据 (严格列名对齐) ---
    original_cols = ['left_id', 'right_id', 'winner', 'left_lat', 'left_long', 'right_lat', 'right_long', 'category']

    # 1. 导出 Raw 版 (用于 Stage 1 TrueSkill)
    df_raw_final = pd.concat(raw_reliable_list, ignore_index=True)[original_cols]
    raw_csv_path = os.path.join(DATA_DIR, f"final_data_reliable_raw_N{THRESHOLD_RELIABLE}.csv")
    df_raw_final.to_csv(raw_csv_path, index=False)

    # 2. 导出 Agg 版 (用于 Stage 2/4 抽样与评估)
    df_agg_final = pd.concat(agg_reliable_list, ignore_index=True)[original_cols]
    agg_csv_path = os.path.join(DATA_DIR, f"final_data_reliable_agg_N{THRESHOLD_RELIABLE}.csv")
    df_agg_final.to_csv(agg_csv_path, index=False)

    print(f"\n[成功] 已导出可靠数据集 (N>={THRESHOLD_RELIABLE}):")
    print(f"  - 原始投票流 (Raw): {raw_csv_path} ({len(df_raw_final)} 行)")
    print(f"  - 多数投票集 (Agg): {agg_csv_path} ({len(df_agg_final)} 对)")

    # 3. 按Category导出独立文件 (支持增量运行)
    print(f"\n  按维度导出独立文件:")
    for cat in CATEGORIES:
        df_raw_cat = df_raw_final[df_raw_final['category'] == cat]
        df_agg_cat = df_agg_final[df_agg_final['category'] == cat]

        raw_cat_path = os.path.join(DATA_DIR, f"final_data_reliable_raw_N{THRESHOLD_RELIABLE}_{cat}.csv")
        agg_cat_path = os.path.join(DATA_DIR, f"final_data_reliable_agg_N{THRESHOLD_RELIABLE}_{cat}.csv")

        df_raw_cat.to_csv(raw_cat_path, index=False)
        df_agg_cat.to_csv(agg_cat_path, index=False)

        print(f"    {cat:12s}: Raw={len(df_raw_cat):6d}行, Agg={len(df_agg_cat):6d}对")

    # --- C. 保存统计表与绘图 ---
    df_p_csv = pd.DataFrame(pair_results)
    df_i_csv = pd.DataFrame(image_results)
    df_p_csv.to_csv(os.path.join(OUTPUT_DIR, "human_detailed_pairs_consistency.csv"), index=False)
    df_i_csv.to_csv(os.path.join(OUTPUT_DIR, "human_detailed_images_frequency.csv"), index=False)

    create_plots(df_p_csv, df_i_csv)


ECCV_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ECCV_2026_Paper_Template")


def create_plots(df_p, df_i):
    sns.set_style("whitegrid")
    palette = sns.color_palette("bright", len(CATEGORIES))

    # 图 1: 图像对分布 (1-5)
    plt.figure(figsize=FIG_SIZE)
    for i, cat in enumerate(CATEGORIES):
        data = df_p[df_p['Category'] == cat]
        plt.plot(data['Threshold'], data['Count'], marker='o', markersize=MARKER_SIZE, lw=LINE_WIDTH, label=cat,
                 color=palette[i])
    plt.yscale('log')
    plt.title('Image Pairs Distribution (Threshold 1-5)', fontsize=FONT_SIZE_TITLE, pad=25)
    plt.xlabel('Vote Threshold (>= N votes)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Number of Pairs (Log Scale)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(THRESHOLDS_PAIRS, [f'>={t}' for t in THRESHOLDS_PAIRS], fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "human_pairs_distribution.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "human_pairs_distribution.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(ECCV_FIG_DIR, "human_pairs_distribution.pdf"), bbox_inches='tight')

    # 图 2: 单图频次分布
    plt.figure(figsize=FIG_SIZE)
    for i, cat in enumerate(CATEGORIES):
        data = df_i[df_i['Category'] == cat]
        plt.plot(data['Threshold'], data['Count'], marker='s', markersize=MARKER_SIZE, lw=LINE_WIDTH, label=cat,
                 color=palette[i])
    plt.yscale('log')
    plt.title('Individual Image Frequency Distribution', fontsize=FONT_SIZE_TITLE, pad=25)
    plt.xlabel('Appearance Threshold (>= N times)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Number of Images (Log Scale)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(THRESHOLDS_IMAGES, [f'>={t}' for t in THRESHOLDS_IMAGES], fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "human_images_distribution.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "human_images_distribution.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(ECCV_FIG_DIR, "human_images_distribution.pdf"), bbox_inches='tight')


if __name__ == "__main__":
    run_comprehensive_analysis_and_export()