# FluLens 评估报告 / Evaluation Report

日期 / Date: 2026-02-06

## 数据集 / Dataset
- 来源 / Source: IDR idr0128 (Georgi et al. Influenza A virus screen)
- 标注口径 / Labeling:
  - 低感染 (label=0)：`numberOfInfectedNuclei` 位于训练集 20% 分位及以下
  - 高感染 (label=1)：`numberOfInfectedNuclei` 位于训练集 80% 分位及以上
- 样本规模 / Sample counts (当前下载可用):
  - 训练 / Train: 38
  - 验证 / Val: 10
  - 测试 / Test: 30

## 模型 / Model
- 主干 / Backbone: ResNeSt-50 (timm `resnest50d`)
- 注意力 / Attention: 通道 + 空间注意力 (CBAM 风格)
- 输入 / Input: 224x224，RGB（核染色通道作为 R，病毒通道作为 G/B）
- 训练策略 / Training: 预训练权重 + 冻结主干，只训练注意力与分类头

## 指标 / Metrics (Test Split)
- 准确率 / Accuracy: 0.5333
- 灵敏度 / Sensitivity (高感染召回): 0.5625
- 特异性 / Specificity (低感染召回): 0.5000

混淆矩阵 / Confusion Matrix:
- TP: 9
- TN: 7
- FP: 7
- FN: 7

## 结论与限制 / Notes & Limitations
- 数据量很小（受下载约束），结果**不可视为稳定最终指标**。
- 标签是“感染强度分层”替代口径，并非严格感染/未感染。
- 该报告用于验证流程、模型与网页 demo 的可用性；若扩大数据并执行 BYOL 预训练，指标应有明显提升。
