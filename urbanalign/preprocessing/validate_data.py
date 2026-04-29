import pandas as pd
import numpy as np
import os

# =================================================================
# 配置区域 (请确保路径指向你电脑上的文件)
# =================================================================
DATA_DIR = os.getenv(
    "PLACE_PULSE_DIR",
    r"H:\RawData13-全球街景\mit place pulse\01 Place Pluse2.0数据集\01 Place Pulse 2.0论文数据集"
)
ORIGINAL_CSV = os.path.join(DATA_DIR, "final_data.csv")
FILTERED_CSV = os.path.join(DATA_DIR, "final_data_reliable_N3.csv")


def verify_filtered_data(n_checks=150):
    if not os.path.exists(FILTERED_CSV):
        print(f"错误: 找不到过滤后的文件 {FILTERED_CSV}")
        return

    print(f"--- 自动校验报告 ---")
    print(f"正在对比: \n原始总库 -> {ORIGINAL_CSV} \n实验子库 -> {FILTERED_CSV}\n")

    # 1. 加载数据
    df_orig = pd.read_csv(ORIGINAL_CSV)
    df_filt = pd.read_csv(FILTERED_CSV)

    # 2. 随机抽取样本进行深度检查
    samples = df_filt.sample(n=min(n_checks, len(df_filt)))

    for i, row in samples.iterrows():
        l_id = str(row['left_id'])
        r_id = str(row['right_id'])
        cat = row['category']
        filt_winner = row['winner']

        print(f"【检查项 #{i + 1}】")
        print(f"  对子 ID: {l_id} vs {r_id} ({cat})")

        # 3. 在原始数据中找回所有相关的投票记录 (考虑左右位置在原始数据中可能是反的)
        match_mask = (
                             ((df_orig['left_id'] == l_id) & (df_orig['right_id'] == r_id)) |
                             ((df_orig['left_id'] == r_id) & (df_orig['right_id'] == l_id))
                     ) & (df_orig['category'] == cat)

        orig_votes_df = df_orig[match_mask].copy()

        # 归一化原始投票结果：
        # 如果原始记录的 left_id 是我们的 r_id，说明那是镜像记录，需要翻转 winner 标签
        def normalize_orig_winner(r):
            if str(r['left_id']) == r_id:
                m = {'left': 'right', 'right': 'left', 'equal': 'equal'}
                return m.get(r['winner'], r['winner'])
            return r['winner']

        orig_votes_df['normalized_vote'] = orig_votes_df.apply(normalize_orig_winner, axis=1)

        vote_list = orig_votes_df['normalized_vote'].tolist()
        vote_counts = orig_votes_df['normalized_vote'].value_counts()
        expected_winner = vote_counts.idxmax()

        print(f"  -> 原始库共查到 {len(vote_list)} 票: {vote_list}")
        print(f"  -> 多数投票结果: {expected_winner} | 子库记录结果: {filt_winner}")

        # 验证结果
        status = "✅ 匹配正确" if expected_winner == filt_winner else "❌ 逻辑错误"
        print(f"  -> 判断校验: {status}")

        # 4. 坐标一致性检查
        # 随便找一行原始记录来比对经纬度
        sample_orig = orig_votes_df.iloc[0]
        # 如果原始左边是我们的左边
        if str(sample_orig['left_id']) == l_id:
            orig_l_lat, orig_l_long = sample_orig['left_lat'], sample_orig['left_long']
        else:
            orig_l_lat, orig_l_long = sample_orig['right_lat'], sample_orig['right_long']

        coord_status = "✅ 坐标锁定正确" if (np.isclose(row['left_lat'], orig_l_lat) and np.isclose(row['left_long'],
                                                                                                   orig_l_long)) else "❌ 坐标发生偏移"
        print(f"  -> 坐标校验: {coord_status}")
        print("-" * 60)


if __name__ == "__main__":
    verify_filtered_data(150)  # 默认随机抽查5组