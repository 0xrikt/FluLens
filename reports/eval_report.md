# FluLens 评估报告 / Evaluation Report

日期 / Date: 2026-02-06

## 数据集 / Dataset
- 来源 / Source: IDR idr0128 (Georgi et al. Influenza A virus screen)
- 标注口径 / Labeling:
  - 低感染 (label=0)：`numberOfInfectedNuclei` 位于训练集 **10% 分位及以下**
  - 高感染 (label=1)：`numberOfInfectedNuclei` 位于训练集 **90% 分位及以上**
- 样本规模 / Sample counts (当前下载可用):
  - 训练 / Train: 45
  - 验证 / Val: 15
  - 测试 / Test: 21

## 模型 / Model
- 主干 / Backbone: ResNeSt-50 (timm `resnest50d`)
- 注意力 / Attention: 通道 + 空间注意力 (CBAM 风格)
- 输入 / Input: 224x224，RGB（核染色通道作为 R，病毒通道作为 G/B）
- 训练策略 / Training: 预训练权重，解冻主干端到端训练 + 轻量数据增强

## 指标 / Metrics (Test Split)
- 准确率 / Accuracy: **0.9524**
- 灵敏度 / Sensitivity (高感染召回): **0.9091**
- 特异性 / Specificity (低感染召回): **1.0000**

混淆矩阵 / Confusion Matrix:
- TP: 10
- TN: 10
- FP: 0
- FN: 1

## 结论与限制 / Notes & Limitations
- 数据量仍偏小，指标可能存在波动，但已达到 80% 以上要求。
- 标签为感染强度分层口径（10%/90% 极值区间），并非严格感染/未感染。
- 若引入真实阴性样本或扩大数据量，需重新训练并更新指标。
