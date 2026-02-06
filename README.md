# FluLens

Influenza infection intensity screening using a ResNeSt-50 backbone with channel and spatial attention, adapted from the malaria BYOL + attention paper.

# FluLens（中文说明）

基于 ResNeSt-50 主干与通道/空间注意力的流感感染强度筛查系统，技术路线参考疟原虫 BYOL + 注意力论文并做迁移实现。

## What’s Included
- Training pipeline (PyTorch)
- Evaluation script + report
- ONNX export
- Web demo (Next.js + onnxruntime-web)

## 包含内容
- 训练流程（PyTorch）
- 评估脚本与报告
- ONNX 导出
- Web Demo（Next.js + onnxruntime-web）

## Quick Start

1) Prepare data (IDR idr0128):
```
python3 /Users/rik/projects/FluLens/scripts/prepare_idr0128.py
```
If downloads are limited, build a manifest from already-downloaded images:
```
python3 /Users/rik/projects/FluLens/scripts/build_manifest.py --max-train 120 --max-val 30 --max-test 120
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
python3 /Users/rik/projects/FluLens/scripts/build_manifest.py --max-train 120 --max-val 30 --max-test 120
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
- Current labels are infection-intensity proxies (low vs high infection). Replace with true infected/uninfected labels when available.
- See `reports/eval_report.md` for baseline metrics.
- The web demo runs inference in the browser using ONNX Runtime Web. The ONNX model is stored at `web/public/models/flulens.onnx`.

## 说明
- 当前标签为感染强度替代口径（低/高），非严格感染/未感染。若有真实标注，请替换后重训。
- 评估结果见 `reports/eval_report.md`。
- 网页端在浏览器内进行推理（ONNX Runtime Web），模型位于 `web/public/models/flulens.onnx`。
