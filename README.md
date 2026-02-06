# FluLens

Influenza A (IAV) infection intensity screening using a ResNeSt-50 backbone with channel and spatial attention, adapted from the malaria BYOL + attention paper.

# FluLens（中文说明）

基于 ResNeSt-50 主干与通道/空间注意力的**甲型流感（IAV）感染强度**筛查系统，技术路线参考疟原虫 BYOL + 注意力论文并做迁移实现。

## What’s Included
- Training pipeline (PyTorch)
- Evaluation script + report
- ONNX export
- Web demo (Next.js + onnxruntime-web)

## Scope vs. Paper (English)
**Full paper pipeline should include:**
- Large expert-labeled dataset
- Cell segmentation + masking strategy
- BYOL self-supervised pretraining (positive samples only)
- ResNeSt-50 + channel/spatial attention supervised training
- Paper-matched data augmentations and fine-tuning strategy

**This repo currently implements:**
- IAV dataset prep with plate-level split
- ResNeSt-50 + channel/spatial attention supervised training
- Lightweight augmentations
- ONNX export + browser inference demo

**Not yet implemented (and why):**
- Cell segmentation + masking (needs a dedicated segmentation pipeline)
- BYOL pretraining (needs much larger positive dataset and longer training time)
- Paper-matched augmentation/finetune details (requires strict hyperparameter alignment)

**To fully reproduce the paper:**
- More positive samples
- Segmentation module
- Longer training budget for BYOL + finetune
- Strict hyperparameter match to the paper

## 包含内容
- 训练流程（PyTorch）
- 评估脚本与报告
- ONNX 导出
- Web Demo（Next.js + onnxruntime-web）

## 与论文流程对照（中文）
**论文完整流程应包含：**
- 大规模专家标注数据
- 细胞分割 + 掩膜策略
- BYOL 自监督预训练（仅阳性样本）
- ResNeSt-50 + 通道/空间注意力的监督训练
- 与论文一致的增强与微调策略

**当前仓库已实现：**
- IAV 数据集准备与 plate 级切分
- ResNeSt-50 + 通道/空间注意力监督训练
- 轻量数据增强
- ONNX 导出 + 浏览器内推理 Demo

**尚未实现（原因）：**
- 细胞分割 + 掩膜（需要专门的分割流水线）
- BYOL 预训练（需要更大量阳性样本与更长训练时间）
- 论文级增强与微调细节（需要严格对齐超参）

**完整复现所需条件：**
- 更多阳性样本
- 分割模块
- 更长训练预算（BYOL + 微调）
- 论文超参严格对齐

## Quick Start

1) Prepare data (IDR idr0128):
```
python3 /Users/rik/projects/FluLens/scripts/prepare_idr0128.py
```
If downloads are limited, build a manifest from already-downloaded images:
```
python3 /Users/rik/projects/FluLens/scripts/build_manifest.py --max-train 240 --max-val 60 --max-test 60
```

2) Train model:
```
PYTHONPATH=/Users/rik/projects/FluLens/src \
python3 /Users/rik/projects/FluLens/scripts/train.py \
  --manifest /Users/rik/projects/FluLens/data/processed/idr0128/manifest.csv \
  --epochs 6 --batch-size 8 --pretrained --freeze-backbone \
  --out /Users/rik/projects/FluLens/models/flulens.pt
```

3) Evaluate:
```
PYTHONPATH=/Users/rik/projects/FluLens/src \
python3 /Users/rik/projects/FluLens/scripts/eval.py \
  --manifest /Users/rik/projects/FluLens/data/processed/idr0128/manifest.csv \
  --weights /Users/rik/projects/FluLens/models/flulens.pt \
  --split test
```

4) Export ONNX:
```
PYTHONPATH=/Users/rik/projects/FluLens/src \
python3 /Users/rik/projects/FluLens/scripts/export_onnx.py \
  --weights /Users/rik/projects/FluLens/models/flulens.pt \
  --out /Users/rik/projects/FluLens/models/flulens.onnx
```

5) Run web demo:
```
cd /Users/rik/projects/FluLens/web
npm install
npm run dev
```

## 快速开始（中文）

1) 准备数据（IDR idr0128）：
```
python3 /Users/rik/projects/FluLens/scripts/prepare_idr0128.py
```
如果下载受限，可基于已下载图片构建清单：
```
python3 /Users/rik/projects/FluLens/scripts/build_manifest.py --max-train 240 --max-val 60 --max-test 60
```

2) 训练模型：
```
PYTHONPATH=/Users/rik/projects/FluLens/src \
python3 /Users/rik/projects/FluLens/scripts/train.py \
  --manifest /Users/rik/projects/FluLens/data/processed/idr0128/manifest.csv \
  --epochs 6 --batch-size 8 --pretrained --freeze-backbone \
  --out /Users/rik/projects/FluLens/models/flulens.pt
```

3) 评估：
```
PYTHONPATH=/Users/rik/projects/FluLens/src \
python3 /Users/rik/projects/FluLens/scripts/eval.py \
  --manifest /Users/rik/projects/FluLens/data/processed/idr0128/manifest.csv \
  --weights /Users/rik/projects/FluLens/models/flulens.pt \
  --split test
```

4) 导出 ONNX：
```
PYTHONPATH=/Users/rik/projects/FluLens/src \
python3 /Users/rik/projects/FluLens/scripts/export_onnx.py \
  --weights /Users/rik/projects/FluLens/models/flulens.pt \
  --out /Users/rik/projects/FluLens/models/flulens.onnx
```

5) 运行网页 Demo：
```
cd /Users/rik/projects/FluLens/web
npm install
npm run dev
```

## Notes
- Current labels are infection-intensity proxies (low vs high infection). This demo uses **Influenza A (IAV)** data, not avian influenza.
- Replace with true infected/uninfected labels when available.
- See `reports/eval_report.md` for baseline metrics.
- The web demo runs inference in the browser using ONNX Runtime Web. The ONNX model is stored at `web/public/models/flulens.onnx`.

## 说明
- 当前标签为感染强度替代口径（低/高），非严格感染/未感染；当前数据为**甲型流感 IAV**，非禽流感。
- 若有真实标注，请替换后重训。
- 评估结果见 `reports/eval_report.md`。
- 网页端在浏览器内进行推理（ONNX Runtime Web），模型位于 `web/public/models/flulens.onnx`。
