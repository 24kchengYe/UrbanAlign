"""
UrbanAlign 2.0 全局配置文件
清晰的分层结构：共用配置 → 1.0配置 → 2.0配置

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据流与参数对应关系图 (解耦架构)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HUMAN_CHOICES_CSV (人类标注原始数据)
         |
         ├──> Stage 1: 共识采样 (仅用TrueSkill高/低共识图片)
         |    参数: N_CONSENSUS_SAMPLES
         |    输出: stage1_semantic_dimensions.json
         |
         ├──> Stage 2: 统一打分 (对全量/采样数据统一评分，不区分Ref/Pool)
         |    参数: N_POOL_MULTIPLIER (采样比例，1.0=全量)
         |    公式: n_pairs = len(human_data) × N_POOL_MULTIPLIER
         |    输出: stage2_mode{1-4}_all_scored.csv (包含全部已打分数据)
         |
         └──> Stage 3: 动态拆分 + VRM对齐
              读取: stage2_mode{N}_all_scored.csv
              ├── get_split_data() 动态拆分为 Ref 和 Pool
              |   参数: LABELED_SET_SIZE_PER_CAT (Ref大小，可随时调整)
              |   Ref: 带TrueSkill的锚点，用于LWRR回归
              |   Pool: 待校准的合成数据
              └── LWRR局部加权岭回归 → 校准Pool
                  参数: SELECTION_RATIO (保留比例)
                  输出: stage3_mode{1-4}_aligned.csv

关键设计:
  - Stage 2 对所有采样数据统一打分，不做Ref/Pool划分
  - Stage 3 读取Stage 2的统一输出，动态拆分为Ref和Pool
  - 解耦优势：修改LABELED_SET_SIZE_PER_CAT无需重跑Stage 2 (API调用)
  - 随机种子固定: get_split_data()使用random_state=42保证可复现

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import numpy as np
import requests
import json as json_module

# Load .env if present (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# 1. API配置（1.0和2.0共用）
# ==============================================================================
API_KEY = os.getenv("URBANALIGN_API_KEY", "")
BASE_URL = os.getenv("URBANALIGN_BASE_URL", "https://api.mindcraft.com.cn/v1/chat/completions")
MODEL_NAME = os.getenv("URBANALIGN_MODEL_NAME", "qwen2.5-vl-72b-instruct")
STAGE2_MODE = 4  # 当前运行模式

# ==============================================================================
# 2. 数据集路径配置（1.0和2.0共用）
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ────────────────────────────────────────────────────────────
# Place Pulse 2.0数据集配置
# Set PLACE_PULSE_DIR env var to point to your local Place Pulse 2.0 dataset.
# Expected structure:
#   <PLACE_PULSE_DIR>/
#     ├── final_data_reliable_agg_N3.csv
#     ├── final_data_reliable_raw_N3.csv
#     └── final_photo_dataset/   (street-view images)
# ────────────────────────────────────────────────────────────
DATA_DIR = os.getenv(
    "PLACE_PULSE_DIR",
    r"H:\RawData13-全球街景\mit place pulse\01 Place Pluse2.0数据集\01 Place Pulse 2.0论文数据集"
)

# 人类标注数据（核心数据源）
# 此文件包含: (left_id, right_id, winner, category)
# 用途:
#   - Stage 1: 计算TrueSkill评分 → 采样共识样本
#   - Stage 2: 采样图片对用于维度分数提取
#   - Stage 3: 采样参考集用于VRM对齐
HUMAN_CHOICES4trueskill_CSV = os.path.join(DATA_DIR, "final_data_reliable_raw_N3.csv")
HUMAN_CHOICES_CSV = os.path.join(DATA_DIR, "final_data_reliable_agg_N3.csv")

# 数据集说明:
# - final_data_reliable_agg_N3.csv: 聚合版（每对多次标注的投票结果）
# - final_data_reliable_raw_N3.csv: 原始版（所有单次标注）
# - final_data.csv: 完整Place Pulse 2.0数据集
# 建议使用: final_data_reliable_agg_N3.csv (质量更高)

# 图片目录
IMAGE_DIR = os.path.join(DATA_DIR, "final_photo_dataset")

# 输出目录
OUTPUT_DIR = os.path.join(CURRENT_DIR, "urbanalign_outputs_qwen2_5_vl_72b")  # 开源VLM实验 (原: "urbanalign_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 3. 共用缓存文件（1.0和2.0都需要）
# ==============================================================================
CLIP_CACHE = os.path.join(OUTPUT_DIR, "clip_embeddings.npz")
TRUESKILL_CACHE = os.path.join(OUTPUT_DIR, "trueskill_ratings.csv")  # DEPRECATED: use get_trueskill_cache(category)
ID_MAPPING_CSV = os.path.join(OUTPUT_DIR, "id_mapping.csv")  # DEPRECATED: use get_id_mapping_csv(category)

# ==============================================================================
# 4. 实验参数配置（1.0和2.0共用）
# ==============================================================================
# 感知维度
CATEGORIES = ['safety', 'beautiful', 'lively', 'wealthy', 'boring', 'depressing']  # 当前研究维度'wealthy'
# 完整维度: ['safety', 'beautiful', 'lively', 'wealthy', 'boring', 'depressing']

# TrueSkill参数
TRUESKILL_MU = 25.0
TRUESKILL_SIGMA = 8.333
TRUESKILL_DRAW_PROB = 0.10

# 分层阈值
MU_HIGH_PERCENTILE = 0.75
MU_LOW_PERCENTILE = 0.25
SIGMA_THRESHOLD = 7.0
SIGMA_HIGH_PERCENTILE = 0.75

# CLIP降维
PCA_DIMS = 8

# VRM参数（1.0和2.0共用）
K_MIN = 5
K_MAX = 50
TAU_DENSE = 0.2
TAU_SPARSE = 0.5
EQUAL_PENALTY = 0.5

# ==============================================================================
# 5. UrbanAlign 1.0 专用配置
# ==============================================================================
# Stage 1: 规则学习
N_STRATIFIED_SAMPLES = 4  # 每层采样数

# Stage 1输出
STAGE1_RULES = os.path.join(OUTPUT_DIR, "stage1_evaluation_rules.json")
STAGE1_PREVIEW = os.path.join(OUTPUT_DIR, "stage1_stratified_samples.png")

# Stage 2: 规则引导合成
STAGE2_SYNTHETIC_POOL = os.path.join(OUTPUT_DIR, "stage2_synthetic_pool.csv")
STAGE2_SAMPLED_PAIRS = os.path.join(OUTPUT_DIR, "stage2_sampled_pairs.csv")
STAGE2_SYNTHETIC_POOL_RULE = os.path.join(OUTPUT_DIR, "stage2_synthetic_pool_with_rule.csv")
STAGE2_SYNTHETIC_POOL_NO_RULE = os.path.join(OUTPUT_DIR, "stage2_synthetic_pool_no_rule.csv")

CONFIDENCE_THRESHOLD = 4
SYNTHESIS_TEMPERATURE = 0
EXCLUDE_EQUAL_IN_EVAL = False

# ==============================================================================
# 6. UrbanAlign 2.0 专用配置
# ==============================================================================

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 1: 语义维度提取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE1_DIMENSIONS = os.path.join(OUTPUT_DIR, "stage1_semantic_dimensions.json")
N_CONSENSUS_SAMPLES = 5  # 每组（高/低）采样N张，当前设为1用于快速测试
N_DIMENSIONS_MIN = 5   # 语义维度数量下限
N_DIMENSIONS_MAX = 10  # 语义维度数量上限
DIMENSION_EXAMPLES_FROM_AI = False  # True=AI生成维度示例; False=硬编码
DIMENSION_EXAMPLES_LOG = os.path.join(OUTPUT_DIR, "stage1_dimension_examples_log.csv")  # AI生成维度示例的累积日志

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 2: 多模式合成（消融实验）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ────────────────────────────────────────────────────────────
# 模式选择（5种模式详解）
# ────────────────────────────────────────────────────────────


# ┌─────────────────────────────────────────────────────────────┐
# │ Mode 1: 单张图片直接打分                                    │
# ├─────────────────────────────────────────────────────────────┤
# │ 流程: 分别对A和B打维度分 → 比较总分 → 判断winner           │
# │ API: 2次/对 (A一次, B一次)                                  │
# │ 特点: A的分数固定（与对手无关）                             │
# │ 速度: 中等 (~8秒/对)                                        │
# │ 准确率: ~78-81%                                             │
# └─────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────┐
# │ Mode 2: 图片对直接打分 ★最快                                │
# ├─────────────────────────────────────────────────────────────┤
# │ 流程: 同时看A和B → 对比打分 → 判断winner                   │
# │ API: 1次/对                                                 │
# │ 特点: A的分数可能随对手变化（对比效应）                     │
# │ 速度: 最快 (~5秒/对)                                        │
# │ 准确率: ~84-87%                                             │
# └─────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────┐
# │ Mode 3: 单张图片多智能体 ★最慢                              │
# ├─────────────────────────────────────────────────────────────┤
# │ 流程: A: Obs→Deb→Judge + B: Obs→Deb→Judge → 比较          │
# │ API: 6次/对 (A三次, B三次)                                  │
# │ 特点: 自我验证，但缺少A vs B对比                            │
# │ 速度: 最慢 (~25秒/对)                                       │
# │ 准确率: ~83-85%                                             │
# └─────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────┐
# │ Mode 4: 图片对多智能体博弈 ★推荐★理论最优                  │
# ├─────────────────────────────────────────────────────────────┤
# │ 流程: Observer(A vs B差异) → Debater(为A和B辩护)           │
# │       → Judge(综合判断+打分)                                │
# │ API: 3次/对 (三个Agent都看A和B)                            │
# │ 特点: 对比上下文 + 自我验证 = 最优                          │
# │ 速度: 中等 (~15秒/对)                                       │
# │ 准确率: ~92-95% (理论最优)                                  │
# └─────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────┐
# │ Mode 5: 自动运行全部模式 ★放置过夜                          │
# ├─────────────────────────────────────────────────────────────┤
# │ 流程: 自动运行Mode 1→2→3→4                                 │
# │       每个Mode完成后运行Stage 3对齐                         │
# │       最后运行Stage 4综合评估                               │
# │ API: Mode 1+2+3+4的总和                                     │
# │ 特点: 完整消融实验，无需人工干预                            │
# │ 速度: ~2-3小时 (SKIP_MODE3=True时~1.5小时)                 │
# │ 输出: 完整性能对比+消融分析+维度判别力                      │
# └─────────────────────────────────────────────────────────────┘

# ────────────────────────────────────────────────────────────
# Stage 2数据来源与采样配置
# ────────────────────────────────────────────────────────────

# --- Stage 2 判定参数 ---
# ST2_INTENSITY_SIG_THRESH: 仅用于 Mode 1/3 (单张图片模式)
#   VLM对每张图片输出 overall_intensity (0-100分)
#   判定逻辑: if |intensity_A - intensity_B| < threshold → equal
#   即: A分 > B分 + threshold → A赢; B分 > A分 + threshold → B赢; 否则平局
#   值越大 → 越容易判定为平局; 值=0 → 永远不平局
#   注意: Mode 2/4 (成对模式) 由VLM直接输出winner，不使用此参数
#   建议范围: 2.0~10.0,  默认5.0
ST2_INTENSITY_SIG_THRESH = 3.0

EVAL_EXCLUDE_EQUAL = False         # 快速评估时是否排除平局样本

# --- Stage 3 局部加权岭回归 (LWRR) 核心参数 ---
#
# ┌───────────────────────────────────────────────────────────────────────┐
# │ LWRR 如何矫正 Pool 的 AI 判定？                                      │
# │                                                                       │
# │ 核心问题:                                                             │
# │   Stage 2 让 VLM 对 Pool 图片对打分并判定 winner (left/right/equal), │
# │   但 AI 判定有系统性偏差 (如: 偏好某些视觉模式, 对"平局"不敏感)。   │
# │   Ref 集有人类 TrueSkill 真值, Pool 没有。                            │
# │   如何利用 Ref 的真值来"校准"Pool 的 AI 判定？                        │
# │                                                                       │
# │ 核心洞察:                                                             │
# │   AI 输出的不只是 winner, 还有每个维度的分数向量 (如 Facade=7,        │
# │   Vegetation=4 等)。同样的"AI维度分差向量", 在不同区域对应不同的       │
# │   "人类感知分差"。局部回归可以学到这种非线性映射。                      │
# │                                                                       │
# │ 矫正流程 (对每一个 Pool 样本):                                        │
# │                                                                       │
# │   Step 1: 混合空间寻邻                                                │
# │     将每对图片编码为混合向量:                                          │
# │       v = [α * CLIP_L, α * CLIP_R, (1-α) * Semantic_Delta_norm]      │
# │     CLIP 部分捕捉"长什么样", Semantic 部分捕捉"AI觉得差异在哪"。     │
# │     在此空间中找 K 个最近的 Ref 邻居 (cosine similarity)。            │
# │     → 含义: 找到视觉+语义都相似的参考样本。                           │
# │                                                                       │
# │   Step 2: RBF 核加权                                                  │
# │     weight_j = exp(cosine_sim_j / TAU)                                │
# │     距离越近的邻居权重越大, TAU 控制衰减速度。                         │
# │     → 含义: 更相似的 Ref 对回归贡献更大。                             │
# │                                                                       │
# │   Step 3: 局部岭回归 (核心矫正步骤)                                   │
# │     在 K 个 Ref 邻居上拟合:                                           │
# │       AI_Semantic_Delta  →  Human_TrueSkill_Delta                     │
# │     即: 给定 AI 打的维度分差向量 s_diff = [Facade差, Vegetation差,...],│
# │     学习线性映射 f(s_diff) ≈ μ_L - μ_R (人类 TrueSkill 差异)        │
# │     结果: 一组局部维度权重 w = [w_1, w_2, ...], 使得                  │
# │       Σ(w_i * s_diff_i) ≈ Human_Delta                                │
# │     → 含义: AI 说"Facade 差 3 分", 但在这个局部区域, 人类可能         │
# │       只觉得差1分(w_Facade<1)或差5分(w_Facade>1)。                    │
# │       回归自动学到每个维度在人类眼中的真实权重。                       │
# │                                                                       │
# │   Step 4: 预测 Pool 样本的 fitted_delta                               │
# │     用 Step 3 拟合的模型对当前 Pool 样本的 AI 分差向量做预测:         │
# │       fitted_delta = f(pool_s_diff)                                   │
# │     → 含义: "如果人类来评判这对图片, TrueSkill 分差大约是多少"        │
# │                                                                       │
# │   Step 5: 重新判定 winner (覆盖 Stage 2 的 AI 原始判定)               │
# │     fitted_delta > +EPS  → left    (左图更强)                         │
# │     fitted_delta < -EPS  → right   (右图更强)                         │
# │     |fitted_delta| < EPS → equal   (差异太小, 平局)                   │
# │     额外: 如果 K 个邻居中人类判平局的比例 > CONSENSUS → 强制平局      │
# │     → 含义: 不信 AI 的原始判定, 而是用回归出的"人类视角"重新判定。    │
# │                                                                       │
# │   Step 6: 保真度筛选                                                  │
# │     计算回归的 R² (解释力/保真度 fidelity):                           │
# │     R² 高 = AI 的维度分差能很好拟合人类直觉 → 此样本的矫正可信       │
# │     R² 低 = 局部映射不稳定 → 矫正结果不可靠                          │
# │     按 SELECTION_RATIO 只保留 fidelity 最高的样本。                    │
# │                                                                       │
# │ 直觉总结:                                                             │
# │   LWRR 相当于在 Ref 真值的指导下, 为 Pool 的每个样本"翻译"一次:      │
# │   "AI 说的维度分差" → "人类会感知到的差异" → 重新判定胜负。           │
# │   这是一种 非参数的局部校准, 不假设全局线性关系,                      │
# │   而是在每个样本的邻域内单独学习校准函数。                             │
# └───────────────────────────────────────────────────────────────────────┘

# K_MAX_ST3: 局部邻域大小 — 每个Pool样本找多少个最近的Ref作为回归训练集
#   值越大 → 回归越稳定但越不"局部"; 值越小 → 越局部但可能过拟合
#   建议范围: 10~50,  需要 K << len(Ref)
K_MAX_ST3 = 20

# TAU_KERNEL_ST3: RBF核函数带宽 — 控制邻居权重随距离衰减的速度
#   weight = exp(cosine_similarity / TAU),  cosine_sim ∈ [-1, 1]
#   值越大 → 权重越均匀 (远近邻居权重差异小)
#   值越小 → 权重越集中在最近邻 (远邻几乎不贡献)
#   建议范围: 0.1~1.0
TAU_KERNEL_ST3 = 1

# RIDGE_ALPHA_ST3: 岭回归正则化强度 — 防止局部过拟合
#   值越大 → 权重越平滑 (各维度贡献接近); 值越小 → 允许某些维度主导
#   建议范围: 0.01~1.0
RIDGE_ALPHA_ST3 = 1

# EQUAL_EPS_ST3: 拟合分差判定平局的阈值
#   fitted_delta ∈ TrueSkill尺度 (mu差值约-10~+10)
#   if |fitted_delta| < EQUAL_EPS_ST3 → 判定为平局
#   值越大 → 越多判定为平局; 值=0 → 仅由EQUAL_CONSENSUS_MIN控制
#   与 EQUAL_CONSENSUS_MIN 是 OR 关系: 任一满足即判平局
#   建议范围: 0.3~2.0
EQUAL_EPS_ST3 = 0.8

# EQUAL_CONSENSUS_MIN: 邻域共识平局阈值
#   统计K个Ref邻居中，人类标注为 'equal' 的比例
#   if equal_ratio > EQUAL_CONSENSUS_MIN → 判定为平局 (即使fitted_delta不小)
#   值越大 → 需要更强共识才判平局; 值=1.0 → 禁用此判据
#   建议范围: 0.3~0.8
EQUAL_CONSENSUS_MIN = 0.6

# 数据集来源: HUMAN_CHOICES_CSV (Place Pulse 2.0人类标注)
# 采样策略: 从人类标注中随机采样一定比例的图片对
# 采样比例（控制Stage 2处理的图片对数量）
N_POOL_MULTIPLIER = 1  # 从人类标注中采样的比例，当前设为1用于全量运行
                          # 公式: n_pairs = len(human_data) × N_POOL_MULTIPLIER
                          #
                          # 示例（wealthy维度，假设有15万对人类标注）:
                          # 0.001 = 0.1%  → 150对   → ~10分钟
                          # 0.005 = 0.5%  → 750对   → ~50分钟
                          # 0.01  = 1%    → 1500对  → ~2小时
                          # 0.05  = 5%    → 7500对  → ~10小时
                          # 0.5   = 50%   → 75000对 → ~100小时
                          # 1.0   = 100%  → 全量     → 最长
                          #
                          # 建议: 0.001快速测试, 0.01论文实验, 1.0全量运行

# MAX_PAIRS_PER_CATEGORY: 每类别样本上限 (成本控制)
#   各类别原始对数差异大 (wealthy=288, 有的700+), 按比例采样会导致大类成本失控。
#   此参数设置绝对上限: 先按 N_POOL_MULTIPLIER 得到 n_sample,
#   再 cap 到 min(n_sample, MAX_PAIRS_PER_CATEGORY)。
#   0 = 不设上限 (仅受 N_POOL_MULTIPLIER 控制)
#   建议: 300~500 (论文实验); 0 (全量运行)
MAX_PAIRS_PER_CATEGORY = 288

# Mode 5配置
SKIP_MODE3 = False  # Mode 5运行时是否跳过Mode 3（最慢的模式）
                   # True: 运行Mode 1,2,4 (节省~1小时)
                   # False: 运行全部Mode 1,2,3,4

# Stage 2输出映射（4种Mode各自的统一打分输出文件）
# 解耦架构: 包含全部已打分数据（不区分Ref/Pool），Stage 3动态拆分
STAGE2_OUTPUT_MAP = {
    1: os.path.join(OUTPUT_DIR, "stage2_mode1_all_scored.csv"),
    2: os.path.join(OUTPUT_DIR, "stage2_mode2_all_scored.csv"),
    3: os.path.join(OUTPUT_DIR, "stage2_mode3_all_scored.csv"),
    4: os.path.join(OUTPUT_DIR, "stage2_mode4_all_scored.csv")
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 3: 混合空间VRM对齐
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LABELED_SET_RATIO: Reference集占每类别总数据的比例 (物理隔离)
#   Reference = 带TrueSkill真值的锚点集, 用于LWRR回归的训练目标
#   Pool = 待校准的数据, Stage 2的AI合成结果在此集上被Stage 3校准
#   值越大 → Ref越大(回归越稳), Pool越小(可校准的样本越少)
#   值越小 → Ref越小(回归可能欠拟合), Pool越大(更多可用校准数据)
#   解耦架构优势: 修改此值不需要重跑Stage 2的API调用
#   建议范围: 0.01~0.80
LABELED_SET_RATIO = 0.7

# SELECTION_RATIO: LWRR对齐后的置信度筛选比例
#   按 confidence_score (NAR+PM复合) 降序排列, 只保留前 SELECTION_RATIO 比例
#   confidence高 = 邻居共识强 + 预测边距大 → 高置信
#   1.0 = 全部保留 (不做筛选); 0.8 = 保留置信度最高的80%
#   建议: 敏感性分析时设为1.0, 论文最终结果可设0.8~1.0
SELECTION_RATIO = 1

# ALPHA_HYBRID: 混合空间中 CLIP视觉特征 vs 语义分差 的权重
#   混合坐标 = [α * CLIP_concat, (1-α) * Semantic_Delta_normalized]
#   α=1.0 → 纯CLIP空间 (忽略AI语义); α=0.0 → 纯语义空间 (忽略视觉)
#   0.3 = CLIP占30%, 语义占70% (语义为主导, CLIP做视觉锚定)
#   建议范围: 0.1~0.5
ALPHA_HYBRID = 0.3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 敏感性分析 (Grid Search) 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 运行 abc_stage5_sensitivity_analysis.py 时使用
# 所有参数搜索只涉及对齐阶段 (Stage 3), 不需要重跑Stage 2的API调用
SENSITIVITY_GRID = {
    'ST2_INTENSITY_SIG_THRESH': [2.0, 3.0, 5.0, 7.0, 10.0],
    'K_MAX_ST3': [10, 20, 30, 50],
    'TAU_KERNEL_ST3': [0.1, 0.3, 0.5, 0.8, 1.0],
    'RIDGE_ALPHA_ST3': [0.01, 0.05, 0.1, 0.5, 1.0],
    'EQUAL_EPS_ST3': [0.3, 0.5, 0.8, 1.2, 2.0],
    'EQUAL_CONSENSUS_MIN': [0.3, 0.4, 0.5, 0.6, 0.8],
    'ALPHA_HYBRID': [0.1, 0.2, 0.3, 0.5, 0.7],
    'SELECTION_RATIO': [0.6, 0.8, 0.9, 1.0],
    'LABELED_SET_RATIO': [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
}

# 组合参数随机搜索采样数（全组合空间可能50000+，用随机搜索近似）
# 200-500 通常足够覆盖参数空间；50000 相当于穷举，耗时极长
N_RANDOM_SEARCH = 1000

# 是否使用 Stage 5 优化后的 per-category 最优参数
# True: Stage 3 自动加载 stage5_best_params_{cat}.json (如果存在)
# False: 始终使用上面的全局默认参数
USE_OPTIMIZED_PARAMS = True


def get_optimal_lwrr_params(category):
    """
    获取指定类别的 LWRR 最优参数。
    如果 USE_OPTIMIZED_PARAMS=True 且 stage5_best_params_{cat}.json 存在, 使用优化后的参数;
    否则 fallback 到全局默认值。

    Returns:
        dict: {K_MAX_ST3, TAU_KERNEL_ST3, RIDGE_ALPHA_ST3, EQUAL_EPS_ST3,
               EQUAL_CONSENSUS_MIN, ALPHA_HYBRID, SELECTION_RATIO}
        str: 来源标识 ('optimized' 或 'default')
    """
    import json as _json

    defaults = {
        'K_MAX_ST3': K_MAX_ST3,
        'TAU_KERNEL_ST3': TAU_KERNEL_ST3,
        'RIDGE_ALPHA_ST3': RIDGE_ALPHA_ST3,
        'EQUAL_EPS_ST3': EQUAL_EPS_ST3,
        'EQUAL_CONSENSUS_MIN': EQUAL_CONSENSUS_MIN,
        'ALPHA_HYBRID': ALPHA_HYBRID,
        'SELECTION_RATIO': SELECTION_RATIO,
    }

    if not USE_OPTIMIZED_PARAMS:
        return defaults, 'default'

    best_params_file = os.path.join(OUTPUT_DIR, f'stage5_best_params_{category}.json')
    if not os.path.exists(best_params_file):
        return defaults, 'default'

    try:
        with open(best_params_file, 'r') as f:
            data = _json.load(f)
        params = data.get('params', {})
        # 合并: 优化参数覆盖默认值
        merged = defaults.copy()
        for k in defaults:
            if k in params:
                # K_MAX_ST3 必须为 int，其余为 float
                merged[k] = int(params[k]) if k == 'K_MAX_ST3' else float(params[k])
        return merged, 'optimized'
    except Exception as e:
        print(f"[WARN] 读取 {best_params_file} 失败: {e}, 使用默认参数")
        return defaults, 'default'


# Stage 3输出映射（4种Mode对齐后的输出）
STAGE3_OUTPUT_MAP = {
    1: os.path.join(OUTPUT_DIR, "stage3_mode1_aligned.csv"),
    2: os.path.join(OUTPUT_DIR, "stage3_mode2_aligned.csv"),
    3: os.path.join(OUTPUT_DIR, "stage3_mode3_aligned.csv"),
    4: os.path.join(OUTPUT_DIR, "stage3_mode4_aligned.csv")
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 7: 传统Baseline输出路径
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE7_OUTPUT_MAP = {
    'c0': os.path.join(OUTPUT_DIR, "stage7_baseline_c0_siamese_resnet.csv"), # 新增
    'c1': os.path.join(OUTPUT_DIR, "stage7_baseline_c1_siamese_clip.csv"),
    'c2': os.path.join(OUTPUT_DIR, "stage7_baseline_c2_segmentation_regression.csv"),
    'c3': os.path.join(OUTPUT_DIR, "stage7_baseline_c3_zeroshot_vlm.csv")
}

# 当前Mode的输入输出（根据STAGE2_MODE自动选择）
STAGE2_CURRENT_OUTPUT = STAGE2_OUTPUT_MAP.get(STAGE2_MODE, STAGE2_OUTPUT_MAP[4])
STAGE3_CURRENT_OUTPUT = STAGE3_OUTPUT_MAP.get(STAGE2_MODE, STAGE3_OUTPUT_MAP[4])

# 兼容旧代码（默认指向当前Mode）
STAGE3_ALIGNED_DATA = STAGE3_CURRENT_OUTPUT

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 6: 端到端维度优化 (E2E Dimension Optimization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N_DIMENSION_TRIALS = 15        # 维度集尝试次数 (每次用不同temperature生成新维度)
E2E_POOL_MULTIPLIER = 1   # E2E优化的采样比例 (~150对/类别, 控制API成本)
E2E_STAGE2_MODE = 4            # E2E使用Mode 2 (最快, 1 API/对)

# 两阶段收敛策略 (Explore → Converge)
#   阶段1 (Explore): 多样性搜索, 较高temperature, 独立生成
#     → 各类别独立跟踪最优维度集
#   阶段2 (Converge): 精细优化, 较低temperature, 基于最优变异
#     → 保留好的维度, 仅替换表现差的1-2个
#   早停: 连续 E2E_PATIENCE 次未超过当前最优 (按类别) 时停止
E2E_SAMPLE_RATIO = 1         # E2E评估的样本使用比例 (对已有采样缓存的二次采样)
                               #   1.0 = 使用全部已采样数据
                               #   0.2 = 使用20%已采样数据 (快速搜索)
                               #   注意: 作用于 Stage 2 采样缓存之上, 不影响缓存本身
E2E_PATIENCE = 5               # 早停耐心值: 连续N次无提升则停止 (0=不早停)
E2E_ELITE_SEED = True          # True: 用当前最优维度集引导后续生成 (收敛性)
                               # False: 每次独立生成 (纯随机搜索, 旧行为)
E2E_EXPLORE_RATIO = 0.4        # 探索阶段占比 (0.4 = 前40% trials为探索, 后60%为收敛)
                               #   探索阶段: temperature=0.9~1.0, 独立生成, 积累per-cat最优
                               #   收敛阶段: temperature=0.5~0.7, elite+mutation, 精细优化

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 4: 综合评估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE4_ALL_MODES_COMPARISON = os.path.join(OUTPUT_DIR, "stage4_all_modes_comparison.csv")
STAGE4_ABLATION_ANALYSIS = os.path.join(OUTPUT_DIR, "stage4_ablation_analysis.csv")
STAGE4_DIMENSION_ANALYSIS = os.path.join(OUTPUT_DIR, "stage4_dimension_discriminability.csv")
STAGE4_PLOT = os.path.join(OUTPUT_DIR, "stage4_performance_comparison.png")

# 兼容旧代码
STAGE4_RESULTS = STAGE4_ALL_MODES_COMPARISON

# ==============================================================================
# 多类别文件路径辅助函数
# ==============================================================================
# 所有 per-category 输出文件使用 _{category} 后缀，实现增量运行

def get_trueskill_cache(category):
    return os.path.join(OUTPUT_DIR, f"trueskill_ratings_{category}.csv")

def get_stage1_dimensions(category):
    return os.path.join(OUTPUT_DIR, f"stage1_semantic_dimensions_{category}.json")

def get_stage1_dimension_log(category):
    return os.path.join(OUTPUT_DIR, f"stage1_dimension_examples_log_{category}.csv")

def get_stage1_preview(category):
    return os.path.join(OUTPUT_DIR, f"stage1_stratified_samples_{category}.png")

def get_stage2_output(mode, category):
    return os.path.join(OUTPUT_DIR, f"stage2_mode{mode}_all_scored_{category}.csv")

def get_stage2_sampled_pairs(category):
    return os.path.join(OUTPUT_DIR, f"stage2_sampled_pairs_{category}.csv")

def get_stage3_output(mode, category):
    return os.path.join(OUTPUT_DIR, f"stage3_mode{mode}_aligned_{category}.csv")

def get_stage4_output(name, category):
    """name: 'all_modes_comparison', 'ablation_analysis', 'dimension_discriminability'"""
    return os.path.join(OUTPUT_DIR, f"stage4_{name}_{category}.csv")

def get_stage4_plot(category):
    return os.path.join(OUTPUT_DIR, f"stage4_performance_comparison_{category}.png")

def get_stage5_output(mode, category):
    return os.path.join(OUTPUT_DIR, f"stage5_sensitivity_mode{mode}_{category}.csv")

def get_stage6_trial_dims(trial, category):
    return os.path.join(OUTPUT_DIR, f"stage6_trial{trial}_dimensions_{category}.json")

def get_stage6_trial_scored(trial, category):
    return os.path.join(OUTPUT_DIR, f"stage6_trial{trial}_scored_{category}.csv")

def get_stage6_summary(category):
    return os.path.join(OUTPUT_DIR, f"stage6_e2e_summary_{category}.csv")

def get_stage7_output(baseline_key, category):
    baseline_names = {
        'c0': 'siamese_resnet', 'c1': 'siamese_clip',
        'c2': 'segmentation_regression', 'c3': 'zeroshot_vlm'
    }
    return os.path.join(OUTPUT_DIR, f"stage7_baseline_{baseline_key}_{baseline_names[baseline_key]}_{category}.csv")

def get_id_mapping_csv(category):
    return os.path.join(OUTPUT_DIR, f"id_mapping_{category}.csv")

def get_stage8_output(name):
    """Stage 8 汇总输出文件路径"""
    return os.path.join(OUTPUT_DIR, f'stage8_{name}.csv')

def get_split_cache_paths(category):
    return (os.path.join(OUTPUT_DIR, f"data_split_reference_{category}.csv"),
            os.path.join(OUTPUT_DIR, f"data_split_pool_{category}.csv"))

def get_human_choices_csv(category):
    """获取指定category的聚合版人类标注数据 (用于Stage 2/4)"""
    return os.path.join(DATA_DIR, f"final_data_reliable_agg_N3_{category}.csv")

def get_human_choices4trueskill_csv(category):
    """获取指定category的原始投票流数据 (用于Stage 1 TrueSkill)"""
    return os.path.join(DATA_DIR, f"final_data_reliable_raw_N3_{category}.csv")

# ==============================================================================
# 调试选项
# ==============================================================================
DEBUG_MODE = False
VERBOSE = True

if DEBUG_MODE:
    print("⚠️  DEBUG模式已开启")
    CATEGORIES = ['wealthy']
    N_CONSENSUS_SAMPLES = 3
    N_STRATIFIED_SAMPLES = 2
    N_POOL_MULTIPLIER = 0.001
    LABELED_SET_RATIO = 0.01
    SELECTION_RATIO = 1.0

# ==============================================================================
# 配置汇总显示
# ==============================================================================
def get_config_summary():
    """打印2.0配置摘要"""
    mode_names = {
        1: "单张直接", 2: "成对直接", 3: "单张多智能体",
        4: "成对多智能体", 5: "自动运行全部"
    }

    summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      UrbanAlign 2.0 Configuration                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 【基础配置】
║   API: {MODEL_NAME}
║   研究维度: {', '.join(CATEGORIES)}
║
║ 【解耦架构】
║   Stage 2: 对全量采样数据统一打分 (不区分Ref/Pool)
║   Stage 3: 动态拆分 Ref({LABELED_SET_RATIO*100:.1f}%/类别) / Pool(剩余)
║   采样比例: {N_POOL_MULTIPLIER*100:.1f}% (从全量人类标注采样)
║
║ 【Stage 2: 多模式合成】
║   当前模式: Mode {STAGE2_MODE} ({mode_names.get(STAGE2_MODE, 'Unknown')})
║   Skip Mode 3: {SKIP_MODE3}
║
║ 【Stage 3: 混合VRM对齐】
║   Reference比例: {LABELED_SET_RATIO*100:.1f}% (LWRR标尺, 动态从scored数据拆分)
║   保留比例: {SELECTION_RATIO*100:.0f}% (从对齐结果选择高保真)
║   混合权重: α={ALPHA_HYBRID} (CLIP {ALPHA_HYBRID*100:.0f}% + Semantic {(1-ALPHA_HYBRID)*100:.0f}%)
║
║ Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    return summary

# ==============================================================================
# API调用函数（1.0和2.0共用）
# ==============================================================================
def call_llm_api(messages, temperature=0, max_tokens=4096, response_format=None, timeout=120, max_retries=3):
    """统一的LLM API调用函数（带重试机制）"""
    import time

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    for retry in range(max_retries):
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=timeout, verify=False)

            try:
                result = response.json()
            except Exception:
                if retry < max_retries - 1:
                    time.sleep(1)
                    continue
                return None

            if response.status_code == 200:
                if 'error' in result:
                    error_info = result.get('error')
                    if isinstance(error_info, dict) and error_info.get('message') == '\n':
                        if retry < max_retries - 1:
                            wait_time = 2 ** (retry + 1)
                            print(f"[WARN] API空消息错误，等待{wait_time}秒重试")
                            time.sleep(wait_time)
                            continue
                        return None
                    else:
                        if retry < max_retries - 1:
                            time.sleep(3)
                            continue
                        return None

                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    if retry < max_retries - 1:
                        time.sleep(1)
                        continue
                    return None

            elif response.status_code == 429:
                wait_time = 2 ** retry
                time.sleep(wait_time)
                continue
            else:
                if retry < max_retries - 1:
                    time.sleep(1)
                    continue
                return None

        except requests.exceptions.Timeout:
            if retry < max_retries - 1:
                time.sleep(1)
                continue
            return None
        except Exception:
            return None

    return None

if __name__ == "__main__":
    print(get_config_summary())

# ==============================================================================
# 数据集物理隔离函数
# ==============================================================================
def get_split_data(category=None, force_reload=False):
    """
    将人类标注数据物理隔离为Reference和Pool两个互不相交的集合

    参数:
        category: 类别名称，None表示所有类别
        force_reload: 是否强制重新划分

    返回:
        df_ref: Reference Set (LABELED_SET_RATIO比例/类别)
        df_pool: Synthesis Pool (剩余数据)
    """
    import pandas as pd

    categories = [category] if category else CATEGORIES

    # 尝试从 per-category 缓存加载
    if not force_reload:
        ref_list_cached = []
        pool_list_cached = []
        all_cached = True

        for cat in categories:
            ref_cache_cat, pool_cache_cat = get_split_cache_paths(cat)
            if os.path.exists(ref_cache_cat) and os.path.exists(pool_cache_cat):
                df_ref_cat = pd.read_csv(ref_cache_cat)
                df_pool_cat = pd.read_csv(pool_cache_cat)

                # 验证缓存的 Ref 比例是否与当前配置一致
                cached_total = len(df_ref_cat) + len(df_pool_cat)
                if cached_total > 0:
                    expected_size = max(1, int(cached_total * LABELED_SET_RATIO))
                    if len(df_ref_cat) != expected_size:
                        print(f"[INFO] Ref缓存失效: {cat} 缓存={len(df_ref_cat)}对, "
                              f"配置比例{LABELED_SET_RATIO*100:.1f}%应为{expected_size}对, 重新划分...")
                        all_cached = False
                        break

                # 验证数据源是否变化 (Stage2采样缓存更新后, Ref/Pool也应重新划分)
                sampled_cache = get_stage2_sampled_pairs(cat)
                if os.path.exists(sampled_cache):
                    df_sampled = pd.read_csv(sampled_cache)
                    if cached_total != len(df_sampled):
                        print(f"[INFO] Split缓存失效: {cat} 缓存总量={cached_total}对, "
                              f"采样缓存={len(df_sampled)}对, 重新划分...")
                        all_cached = False
                        break

                ref_list_cached.append(df_ref_cat)
                pool_list_cached.append(df_pool_cat)
            else:
                all_cached = False
                break

        if all_cached and ref_list_cached:
            df_ref = pd.concat(ref_list_cached, ignore_index=True)
            df_pool = pd.concat(pool_list_cached, ignore_index=True)
            return df_ref, df_pool

    # 按类别划分
    # 优先从 Stage 2 采样缓存划分 (已经过 MAX_PAIRS_PER_CATEGORY cap)
    # 若采样缓存不存在 (Stage 1 阶段), 则从全量人类数据划分
    ref_list = []
    pool_list = []

    for cat in categories:
        # 优先使用 Stage 2 采样缓存 (已 cap 的数据)
        sampled_cache = get_stage2_sampled_pairs(cat)
        if os.path.exists(sampled_cache):
            df_cat = pd.read_csv(sampled_cache)
            data_source = f"Stage2采样缓存({len(df_cat)}对)"
        else:
            # Stage 1 阶段: 采样缓存还不存在, 用全量数据
            cat_csv = get_human_choices_csv(cat)
            if os.path.exists(cat_csv):
                df_cat = pd.read_csv(cat_csv)
            else:
                # 兼容旧数据：从全量文件中筛选
                df_cat = pd.read_csv(HUMAN_CHOICES_CSV)
                df_cat = df_cat[df_cat['category'] == cat]
            data_source = f"全量人类数据({len(df_cat)}对)"

        # 统一列名
        if 'winner' in df_cat.columns and 'human_winner' not in df_cat.columns:
            df_cat = df_cat.rename(columns={'winner': 'human_winner'})

        # 固定随机种子
        np.random.seed(42)

        # Reference: 按比例采样
        ref_size = max(1, min(int(len(df_cat) * LABELED_SET_RATIO), len(df_cat)))
        df_cat_ref = df_cat.sample(n=ref_size, random_state=42)

        # Pool: 剩余数据
        df_cat_pool = df_cat.drop(df_cat_ref.index)

        ref_list.append(df_cat_ref)
        pool_list.append(df_cat_pool)

        # 保存 per-category 缓存
        ref_cache_cat, pool_cache_cat = get_split_cache_paths(cat)
        df_cat_ref.to_csv(ref_cache_cat, index=False)
        df_cat_pool.to_csv(pool_cache_cat, index=False)
        print(f"  [split] {cat}: 源={data_source} → Ref={len(df_cat_ref)} / Pool={len(df_cat_pool)}")

    df_ref = pd.concat(ref_list, ignore_index=True)
    df_pool = pd.concat(pool_list, ignore_index=True)

    # 验证不重叠
    ref_pairs = set(zip(df_ref['left_id'].astype(str), df_ref['right_id'].astype(str), df_ref['category']))
    pool_pairs = set(zip(df_pool['left_id'].astype(str), df_pool['right_id'].astype(str), df_pool['category']))
    overlap = ref_pairs & pool_pairs
    assert len(overlap) == 0, f"错误: Reference和Pool重叠{len(overlap)}对"

    print(f"\n{'='*60}")
    print("数据集物理隔离完成")
    print(f"{'='*60}")
    for cat in categories:
        ref_cat = df_ref[df_ref['category'] == cat]
        pool_cat = df_pool[df_pool['category'] == cat]
        print(f"  {cat:12s}: Ref={len(ref_cat):5d}对 | Pool={len(pool_cat):5d}对")
    print(f"  重叠检查: {len(overlap)}对")
    for cat in categories:
        rc, pc = get_split_cache_paths(cat)
        print(f"  缓存: {os.path.basename(rc)}, {os.path.basename(pc)}")

    return df_ref, df_pool

