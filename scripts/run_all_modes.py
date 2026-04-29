"""
UrbanAlign 2.0 - Stage 2 Mode 5: 自动运行全部4种模式

功能：
- 自动运行Mode 1, 2, 3, 4
- 每个Mode完成后自动运行Stage 3对齐
- 最后运行Stage 4综合评估
- 适合放置过夜运行

运行时间估算（N_POOL_MULTIPLIER=0.01, ~150对/类别）：
- Mode 1: ~20分钟
- Mode 2: ~12分钟
- Mode 3: ~60分钟（最慢）
- Mode 4: ~40分钟
- Stage 3对齐: 4×5分钟 = ~20分钟
- 总计: ~2.5-3小时

优化选项：
- 跳过Mode 3（设置SKIP_MODE3=True）→ 节省1小时
- 减小样本量（N_POOL_MULTIPLIER=0.005）→ 节省50%时间
"""
import os
import sys
import time
import subprocess
from datetime import datetime

# ==============================================================================
# 配置
# ==============================================================================
from urbanalign.config import SKIP_MODE3, OUTPUT_DIR, CATEGORIES, get_stage1_dimensions

# 日志文件
LOG_FILE = os.path.join(OUTPUT_DIR, "mode5_auto_run_log.txt")

# ==============================================================================
# 辅助函数
# ==============================================================================
def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

def update_config_mode(mode):
    """修改config.py中的STAGE2_MODE"""
    config_path = 'config.py'

    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 找到STAGE2_MODE那一行并修改
    for i, line in enumerate(lines):
        if line.strip().startswith('STAGE2_MODE = '):
            lines[i] = f'STAGE2_MODE = {mode}  # 当前运行模式\n'
            break

    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    log(f"  Config更新: STAGE2_MODE = {mode}")

def run_script(script_name, description):
    """运行Python脚本（实时显示输出）"""
    log(f"开始运行: {description}")
    log(f"  脚本: {script_name}")
    print(f"\n{'='*80}")
    print(f"运行: {description}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # 不捕获输出，让进度直接显示到控制台
        result = subprocess.run(
            ['python', script_name],
            timeout=43200  # 12小时超时
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            log(f"  ✓ 完成: {description} (耗时: {elapsed/60:.1f}分钟)")
            print(f"\n✓ {description} 完成! (耗时: {elapsed/60:.1f}分钟)\n")
            return True
        else:
            log(f"  ✗ 失败: {description} (返回码: {result.returncode})")
            print(f"\n✗ {description} 失败!\n")
            return False

    except subprocess.TimeoutExpired:
        log(f"  ✗ 超时: {description} (>12小时)")
        print(f"\n✗ {description} 超时!\n")
        return False
    except Exception as e:
        log(f"  ✗ 异常: {description} - {str(e)}")
        print(f"\n✗ {description} 异常: {e}\n")
        return False

# ==============================================================================
# 主函数
# ==============================================================================
def run_all_modes():
    """自动运行全部4种模式的完整流程"""
    log("="*80)
    log("UrbanAlign 2.0 - Mode 5: 自动运行全部模式")
    log("="*80)
    log(f"跳过Mode 3: {SKIP_MODE3}")
    log("")

    start_time_total = time.time()

    # 确定要运行的模式
    modes_to_run = [1, 2, 4] if SKIP_MODE3 else [1, 2, 3, 4]
    mode_names = {
        1: "Mode 1 (单张直接)",
        2: "Mode 2 (成对直接)",
        3: "Mode 3 (单张多智能体)",
        4: "Mode 4 (成对多智能体)"
    }

    results_summary = []

    # 循环运行每个Mode
    for mode in modes_to_run:
        log("")
        log("="*60)
        log(f"处理 {mode_names[mode]}")
        log("="*60)

        mode_start = time.time()

        # 1. 更新config.py
        update_config_mode(mode)
        time.sleep(1)  # 确保文件写入完成

        # 2. 运行Stage 2
        success_stage2 = run_script(
            'abc_stage2_multi_mode_synthesis.py',
            f'{mode_names[mode]} - Stage 2合成'
        )

        if not success_stage2:
            log(f"  警告: {mode_names[mode]} Stage 2失败，跳过Stage 3")
            results_summary.append({
                'mode': mode,
                'stage2': 'Failed',
                'stage3': 'Skipped'
            })
            continue

        # 3. 运行Stage 3对齐
        success_stage3 = run_script(
            'abc_stage3_hybrid_vrm.py',
            f'{mode_names[mode]} - Stage 3对齐'
        )

        mode_elapsed = time.time() - mode_start
        log(f"  {mode_names[mode]} 总耗时: {mode_elapsed/60:.1f}分钟")

        results_summary.append({
            'mode': mode,
            'stage2': 'Success' if success_stage2 else 'Failed',
            'stage3': 'Success' if success_stage3 else 'Failed',
            'time_minutes': mode_elapsed / 60
        })

    # 4. 运行Stage 4综合评估
    log("")
    log("="*60)
    log("运行综合评估")
    log("="*60)

    run_script(
        'abc_stage4_comprehensive_evaluation.py',
        'Stage 4 - 综合评估所有Mode'
    )

    # 总结
    total_elapsed = time.time() - start_time_total

    log("")
    log("="*80)
    log("全部模式运行完成！")
    log("="*80)
    log(f"总耗时: {total_elapsed/60:.1f}分钟 ({total_elapsed/3600:.1f}小时)")
    log("")
    log("运行总结:")
    for item in results_summary:
        log(f"  Mode {item['mode']}: Stage2={item['stage2']}, Stage3={item['stage3']}, 耗时={item.get('time_minutes', 0):.1f}分钟")

    log("")
    log("输出文件:")
    log("  - urbanalign_outputs/stage2_mode*.csv (4个)")
    log("  - urbanalign_outputs/stage3_mode*.csv (4个)")
    log("  - urbanalign_outputs/stage4_all_modes_comparison.csv")
    log("  - urbanalign_outputs/stage4_ablation_analysis.csv")
    log("  - urbanalign_outputs/stage4_dimension_discriminability.csv")
    log("  - urbanalign_outputs/stage4_performance_comparison.png")
    log("")
    log("查看评估报告:")
    log("  python -c \"import pandas as pd; print(pd.read_csv('urbanalign_outputs/stage4_all_modes_comparison.csv'))\"")
    log("")
    log("="*80)

if __name__ == "__main__":
    print("\n⚠️  Mode 5: 自动运行全部4种模式")
    print("⚠️  预计耗时: 2-3小时（取决于N_POOL_MULTIPLIER配置）")
    print("⚠️  适合放置过夜运行")
    print("")

    # 检查前置条件：至少一个category有维度文件
    found_dims = [cat for cat in CATEGORIES if os.path.exists(get_stage1_dimensions(cat))]
    if not found_dims:
        print("❌ 错误: 缺少Stage 1维度定义")
        print("   请先运行: python abc_stage1_semantic_extractor.py")
        sys.exit(1)

    missing = [cat for cat in CATEGORIES if cat not in found_dims]
    if missing:
        print(f"⚠️  以下类别缺少维度文件（将跳过）: {', '.join(missing)}")
    print(f"✓ 前置条件检查通过 ({len(found_dims)}/{len(CATEGORIES)} 个类别有维度定义)")
    print("")

    # 确认运行
    user_input = input("确认开始运行全部4种模式？(yes/no): ")
    if user_input.lower() not in ['yes', 'y']:
        print("已取消")
        sys.exit(0)

    print("")
    print("开始运行...")
    print(f"日志文件: {LOG_FILE}")
    print("")

    run_all_modes()
