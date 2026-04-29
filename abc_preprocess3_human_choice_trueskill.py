"""
UrbanAlign 2.0 - 辅助工具: TrueSkill 评分计算
将人类的成对选择 (Pairwise Choices) 转化为连续的评分 (Continuous Ratings)
"""
import pandas as pd
import numpy as np
import os
import trueskill
from tqdm import tqdm

# ==============================================================================
# 1. 导入配置
# ==============================================================================
from config import (
    CATEGORIES, get_id_mapping_csv,
    get_human_choices4trueskill_csv, get_trueskill_cache
)


def create_id_mapping(df_raw, category):
    """创建简化的ID映射（按类别保存）"""
    mapping_path = get_id_mapping_csv(category)
    if os.path.exists(mapping_path):
        print(f"[INFO] 加载已有ID映射: {mapping_path}")
        mapping_df = pd.read_csv(mapping_path)
        return dict(zip(mapping_df['original_id'].astype(str), mapping_df['alias_id']))

    print(f"[INFO] 创建新的ID映射 ({category})...")
    all_ids = pd.concat([df_raw['left_id'], df_raw['right_id']]).unique()
    mapping = {str(orig_id): f"IMG_{i+1:05d}" for i, orig_id in enumerate(all_ids)}

    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    pd.DataFrame(list(mapping.items()),
                 columns=['original_id', 'alias_id']).to_csv(mapping_path, index=False)
    print(f"[INFO] ID映射已保存: {mapping_path} (共 {len(mapping)} 个图片)")
    return mapping


def run_trueskill_calculation():
    print("\n" + "=" * 80)
    print("UrbanAlign 2.0 - Step 0: Human Choice to TrueSkill Rating")
    print("=" * 80)

    for cat in CATEGORIES:
        output_file = get_trueskill_cache(cat)
        input_file = get_human_choices4trueskill_csv(cat)

        # 跳过已完成的类别
        if os.path.exists(output_file):
            print(f"[SKIP] {cat}: {os.path.basename(output_file)} already exists")
            continue

        # 1. 加载该类别的人类标注原始数据
        if not os.path.exists(input_file):
            print(f"[ERROR] {cat}: 找不到人类标注文件: {input_file}")
            continue

        print(f"\n{'='*60}")
        print(f"[INFO] 正在处理类别: {cat}")
        print(f"[INFO] 正在加载原始数据: {os.path.basename(input_file)}...")
        df_raw = pd.read_csv(input_file)

        # 1.5 生成/加载 ID 映射（按类别）
        id_mapping = create_id_mapping(df_raw, cat)
        print(f"[INFO] ID映射就绪: {len(id_mapping)} 个图片")

        # 2. 初始化 TrueSkill 环境
        env = trueskill.TrueSkill(mu=25.0, sigma=8.333, draw_probability=0.10)
        ratings = {}

        # 3. 迭代计算评分
        print(f"[INFO] 正在为 {cat} 计算评分...")

        for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc=f"TrueSkill ({cat})"):
            row_cat = str(row['category']).lower()
            if row_cat != cat:
                continue

            l_id, r_id = str(row['left_id']), str(row['right_id'])
            winner = str(row['winner']).lower()

            if l_id not in ratings:
                ratings[l_id] = env.create_rating()
            if r_id not in ratings:
                ratings[r_id] = env.create_rating()

            r_l, r_r = ratings[l_id], ratings[r_id]
            if winner == 'left':
                ranks = [0, 1]
            elif winner == 'right':
                ranks = [1, 0]
            else:
                ranks = [0, 0]

            try:
                (new_r_l,), (new_r_r,) = env.rate([(r_l,), (r_r,)], ranks=ranks)
                ratings[l_id], ratings[r_id] = new_r_l, new_r_r
            except Exception as e:
                continue

        # 4. 汇总结果并保存
        records = []
        for img_id, rating in ratings.items():
            records.append({
                'category': cat,
                'image_id': img_id,
                'mu': rating.mu,
                'sigma': rating.sigma
            })

        df_ratings = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_ratings.to_csv(output_file, index=False)

        print(f"  输出: {os.path.basename(output_file)} ({len(df_ratings)} 张图片)")
        print(f"  平均 Mu: {df_ratings['mu'].mean():.2f}, 平均 Sigma: {df_ratings['sigma'].mean():.2f}")

    print(f"\n{'=' * 80}")
    print(f"TrueSkill 评分计算完成！")
    print(f"{'=' * 80}")
    print(f"提示: 此文件将被 Stage 1 用于筛选高/低共识样本以定义语义维度。")


if __name__ == "__main__":
    run_trueskill_calculation()