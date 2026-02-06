# FluLens 评估报告 / Evaluation Report

日期 / Date: 2026-02-06

## 数据集 / Dataset
- 来源 / Source: IDR idr0128 (Georgi et al. Influenza A virus screen) — **IAV (甲型流感)**
- 切分策略 / Split strategy: **按 plate 切分**（避免数据泄漏）
  - 训练 / Train: 4 plates
  - 验证 / Val: 1 plate
  - 测试 / Test: 1 plate
- 标注口径 / Labeling:
  - 低感染 (label=0)：`numberOfInfectedNuclei` 位于训练集 **10% 分位及以下**
  - 高感染 (label=1)：`numberOfInfectedNuclei` 位于训练集 **90% 分位及以上**
- 当前样本规模 / Sample counts (class-balanced, for classroom demo):
  - 训练 / Train: 234
  - 验证 / Val: 18
  - 测试 / Test: 16

## 模型 / Model
- 主干 / Backbone: ResNeSt-50 (timm `resnest50d`)
- 注意力 / Attention: 通道 + 空间注意力 (CBAM 风格)
- 输入 / Input: 224x224，RGB（核染色通道作为 R，病毒通道作为 G/B）
- 训练策略 / Training: 预训练权重 + 端到端训练 + 轻量增强

## 指标 / Metrics (Test Split)
- 准确率 / Accuracy: **0.9375**
- 灵敏度 / Sensitivity (高感染召回): **0.8750**
- 特异性 / Specificity (低感染召回): **1.0000**

混淆矩阵 / Confusion Matrix:
- TP: 7
- TN: 8
- FP: 0
- FN: 1

## 结论与限制 / Notes & Limitations
- 本报告用于课堂 Demo，样本规模受流量限制，仅代表可复现流程，不作为论文结论。
- 标签为感染强度分层口径（10%/90% 极值区间），并非严格感染/未感染。
- 若用于论文，请扩大 plate 数量与样本量，并重新训练与评估。
