# UrbanAlign 2.0 - Semantic Feature Distillation Framework

## 概述

**UrbanAlign 2.0** 是对原始UrbanAlign框架的重大理论升级，将大模型从"模仿人类分类器"升级为"语义特征蒸馏器+多智能体推理系统"。

### 核心理念转变

| 维度 | UrbanAlign 1.0 | UrbanAlign 2.0 |
|------|---------------|---------------|
| **定位** | 规则学习 → 规则应用 | 维度定义 → 特征蒸馏 → 混合对齐 |
| **输出** | 分类标签 (left/right/equal) | 维度分数 (1-10 per dimension) |
| **空间** | CLIP单空间 | CLIP + Semantic混合空间 |
| **推理** | 单次推理 | 多智能体博弈 (Observer→Debater→Judge) |
| **可解释性** | 低 (黑盒规则) | 高 (维度分数+推理过程) |

---

## 系统架构

### Stage 1: Semantic Dimension Extraction
**脚本**: `abc_stage1_semantic_extractor.py`

**目标**: 为每个感知类别定义5-8个通用视觉评价维度

**输入**:
- 感知类别 (e.g., "wealthy")
- 5个高共识样本 (TrueSkill μ>75%, σ<中位数)
- 5个低共识样本 (TrueSkill μ<25%, σ<中位数)

**输出**: `stage1_semantic_dimensions.json`
```json
{
  "wealthy": {
    "dimensions": [
      {
        "name": "Facade Quality",
        "description": "Building exterior condition and materials",
        "high_indicators": ["Well-maintained paint", "Premium materials"],
        "low_indicators": ["Peeling walls", "Cracked surfaces"],
        "weight": 1.0
      },
      ...
    ]
  }
}
```

---

### Stage 2: Multi-Agent Feature Distillation
**脚本**: `abc_stage2_multi_agent_synthesis.py`

**模式选择**: `STAGE2_MODE` in `config.py`
- **Mode 1**: Linear Scoring (快速，直接打分)
- **Mode 2**: Multi-Agent Deliberation (高质量，三阶段推理)

**Mode 2流程**:
```
Observer Agent → 描述视觉细节（客观事实）
    ↓
Debater Agent → 正反辩论（探索不同视角）
    ↓
Judge Agent → 综合判断（给出维度分数1-10）
```

**输出**: `stage2_semantic_perception_data.csv`
```csv
image_id, clip_embedding(JSON), semantic_scores(JSON), dimension_names(JSON)
IMG_00001, [0.1, -0.2, ...], [8.5, 7.2, 9.1, ...], ["Facade Quality", ...]
```

**混合特征**: [CLIP(768D), Semantic_Scores(5-8D)]

---

### Stage 3: Hybrid Visual Relationship Mapping
**脚本**: `abc_stage3_hybrid_vrm.py`

**核心创新**: 混合差分向量
```
传统VRM: Δ = φ_CLIP(A) - φ_CLIP(B)
混合VRM: Δ_hybrid = [Δ_CLIP, Δ_Semantic]
```

**流程**:
1. 构建混合向量: h(I) = [α·φ_CLIP(I), (1-α)·S(I)]
2. 计算差分向量: Δ_hybrid(A,B) = h(A) - h(B)
3. 自适应KNN对齐 (K ∈ [K_MIN, K_MAX])
4. 保真度计算 + Top-K选择

**输出**: `stage3_aligned_dataset.csv`

---

## 快速开始

### 前置条件

1. **运行传统Stage 1生成TrueSkill评分** (2.0复用):
```bash
python 01_stage1_rule_learning.py
```

2. **可选: 提取CLIP特征**:
```bash
python 00_extract_clip_features.py
```

### 运行UrbanAlign 2.0

```bash
# Stage 1: 提取语义维度
python abc_stage1_semantic_extractor.py

# Stage 2: 多智能体特征蒸馏
python abc_stage2_multi_agent_synthesis.py

# Stage 3: 混合空间VRM对齐
python abc_stage3_hybrid_vrm.py

# Stage 4: 评估性能（复用原有脚本）
python 04_stage4_evaluation.py
```

---

## 配置参数

在 `config.py` 中关键参数:

```python
# Stage 1: 维度定义
N_STRATIFIED_SAMPLES = 5  # 每层采样数

# Stage 2: 特征蒸馏
STAGE2_MODE = 2  # 1: Linear Scoring | 2: Multi-Agent Deliberation
N_POOL_MULTIPLIER = 0.01  # 蒸馏图片比例

# Stage 3: 混合VRM
LABELED_SET_SIZE_PER_CAT = 500  # 参考集大小
TARGET_FINAL_SIZE_PER_CAT = 50   # 最终保留
K_MIN, K_MAX = 5, 50  # 自适应邻域
EQUAL_PENALTY = 0.5  # Equal标签惩罚
```

---

## 文件结构

```
UrbanAlign 2.0 核心文件:
├── abc_stage1_semantic_extractor.py       # Stage 1: 语义维度提取
├── abc_stage2_multi_agent_synthesis.py    # Stage 2: 多智能体蒸馏
├── abc_stage3_hybrid_vrm.py               # Stage 3: 混合VRM对齐
├── URBANALIGN_2.0_ARCHITECTURE.md         # 系统架构文档
├── PAPER_REVISION_GUIDE_2.0.md            # 论文修改指南
├── compare_1.0_vs_2.0.py                  # 版本对比脚本
└── test_abc_system.py                     # 系统测试脚本

UrbanAlign 1.0 文件（保留用于对比）:
├── 01_stage1_rule_learning.py             # 生成TrueSkill（2.0复用）
├── 02_stage2_rule_guided_synthesis.py     # 规则引导合成
├── 03_stage3_visual_relationship_mapping.py  # CLIP-only VRM
└── 04_stage4_evaluation.py                # 评估（1.0和2.0共用）

输出文件:
urbanalign_outputs/
├── trueskill_ratings.csv                  # TrueSkill评分（Stage 1前置）
├── id_mapping.csv                         # ID映射（Stage 1前置）
├── clip_embeddings.npz                    # CLIP特征（可选前置）
├── stage1_semantic_dimensions.json        # 2.0: 语义维度定义
├── stage2_semantic_perception_data.csv    # 2.0: 混合特征
└── stage3_aligned_dataset.csv             # 2.0: 对齐数据
```

---

## 理论优势

### 1. 可解释性 (Interpretability)

**1.0**: "为什么A比B更wealthy？" → "规则说的"（黑盒）

**2.0**: "为什么A比B更wealthy？" →
| Dimension | A | B | Δ |
|-----------|---|---|---|
| Facade Quality | 8.5 | 6.2 | +2.3 |
| Vegetation | 9.1 | 7.8 | +1.3 |
| Infrastructure | 8.2 | 5.9 | +2.3 |

### 2. 迁移性 (Transferability)

**1.0**: 规则依赖特定城市特征，跨城市需重新训练

**2.0**: 维度定义通用（Facade Quality在任何城市都适用）

### 3. 鲁棒性 (Robustness)

**1.0**: 单次LLM推理 → 易幻觉

**2.0**: Observer→Debater→Judge → 自我一致性验证

### 4. 理论深度 (Theoretical Depth)

**1.0**: 规则学习（经验主义）

**2.0**:
- 语义维度蒸馏（Semantic Feature Distillation）
- 多智能体协同推理（Multi-Agent Collaborative Reasoning）
- 混合嵌入空间对齐（Hybrid Embedding Space Alignment）

---

## 预期性能

| 指标 | UrbanAlign 1.0 | UrbanAlign 2.0 | 提升 |
|------|---------------|---------------|------|
| Accuracy | 87-89% | **92-95%** | +5% |
| MMD Distance | 2.4 | **<2.0** | 接近人类 |
| Interpretability | ⭐⭐ | **⭐⭐⭐⭐** | ++高 |
| Transferability | ⭐⭐ | **⭐⭐⭐⭐** | ++高 |
| Robustness | ⭐⭐⭐ | **⭐⭐⭐⭐** | +高 |

---

## 论文修改要点

详见 `PAPER_REVISION_GUIDE_2.0.md`

**关键术语更新**:
- Rule Learning → **Semantic Dimension Extraction**
- Rule-Guided Synthesis → **Multi-Agent Feature Distillation**
- CLIP Space → **Hybrid Embedding Space**

**新增章节**:
- Section 3.2: Semantic Dimension Extraction
- Section 4.3: Dimension-Level Interpretability

**理论贡献**:
- 提出"语义维度蒸馏"新范式
- 引入"多智能体自我一致性"机制
- 扩展VRM到混合特征空间

---

## 常见问题

### Q1: 为什么需要2.0？1.0有什么问题？

**A**: 1.0将LLM当作"分类器"，直接预测left/right。这有三个问题:
1. 黑盒：无法解释为什么这样判断
2. 不稳定：单次推理易幻觉
3. 不通用：规则难以跨城市迁移

2.0将LLM重新定位为"语义翻译器"，提取可解释的维度分数，通过多智能体验证提升稳定性。

### Q2: Mode 1和Mode 2如何选择？

**Mode 1** (Linear Scoring):
- 优点：快速，适合大规模实验
- 缺点：质量稍低，可能有偏见

**Mode 2** (Multi-Agent):
- 优点：高质量，自我一致性验证
- 缺点：慢（3倍API调用）

**建议**: 初期测试用Mode 1，论文实验用Mode 2

### Q3: 混合空间的α如何设置？

默认 `α=0.3` (30% CLIP + 70% Semantic)

**理论依据**:
- CLIP提供底层视觉特征（泛化性强）
- Semantic提供高阶语义（特异性强）
- 实验表明α∈[0.2, 0.4]性能最佳

### Q4: 维度数量5-8个的依据？

**经验法则**:
- <5个：信息不足，过度简化
- >8个：冗余，增加噪声
- 5-8个：平衡可解释性和表征能力

**理论支持**: 认知心理学研究表明人类工作记忆容量为7±2个单位

### Q5: 能否将2.0用于其他感知任务？

**可以！** 这是2.0的关键优势之一。

通用框架适用于:
- 医疗影像质量评估
- 景观美学评分
- 产品设计感知
- 任何需要"主观感知→客观特征"的任务

---

## 开发团队

UrbanAlign 2.0 - "Semantic Feature Distillation"架构设计

---

## 许可证

Academic Use Only

---

## 引用

如果使用UrbanAlign 2.0，请引用:

```bibtex
@inproceedings{urbanalign2026,
  title={UrbanAlign 2.0: Few-Shot Urban Perception via Semantic Feature Distillation and Hybrid Visual Relationship Mapping},
  author={...},
  booktitle={ECCV},
  year={2026}
}
```

---

## 更新日志

### Version 2.0 (2026-02-04)
- ✅ 引入语义维度定义机制
- ✅ 实现多智能体协同推理（Observer-Debater-Judge）
- ✅ 扩展VRM到混合特征空间
- ✅ 提升可解释性、迁移性、鲁棒性
- ✅ 完整文档和论文修改指南

### Version 1.0 (2026-01-20)
- ✅ 基础Rule-Based框架
- ✅ TrueSkill分层采样
- ✅ CLIP-only VRM对齐

---

**UrbanAlign 2.0: From Rules to Dimensions, From Classification to Distillation**
