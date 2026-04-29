"""
测试数据集物理隔离功能
"""
from config import get_split_data, CATEGORIES

print("="*80)
print("测试数据集物理隔离")
print("="*80)
print()

# 测试单个类别
for cat in CATEGORIES:
    print(f"\n测试类别: {cat}")
    print("-"*60)

    df_ref, df_pool = get_split_data(category=cat)

    print(f"\n验证结果:")
    print(f"  Reference集: {len(df_ref)}对")
    print(f"  Pool集: {len(df_pool)}对")
    print(f"  总计: {len(df_ref) + len(df_pool)}对")

    # 验证不重叠
    ref_pairs = set(zip(df_ref['left_id'], df_ref['right_id']))
    pool_pairs = set(zip(df_pool['left_id'], df_pool['right_id']))
    overlap = ref_pairs & pool_pairs

    print(f"\n  重叠检查: {len(overlap)}对")
    if len(overlap) == 0:
        print(f"  ✓ 物理隔离成功！Reference和Pool完全不重叠")
    else:
        print(f"  ✗ 错误！发现{len(overlap)}对重叠数据")

    # 显示示例
    print(f"\n  Reference集样例（前3对）:")
    print(df_ref[['left_id', 'right_id', 'winner']].head(3).to_string(index=False))

    print(f"\n  Pool集样例（前3对）:")
    print(df_pool[['left_id', 'right_id', 'winner']].head(3).to_string(index=False))

print("\n" + "="*80)
print("测试完成")
print("="*80)
print("\n缓存文件:")
print("  - urbanalign_outputs/data_split_reference.csv")
print("  - urbanalign_outputs/data_split_pool.csv")
print("\n这两个文件将被所有Stage脚本使用，确保物理隔离")
