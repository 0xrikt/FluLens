# FluLens

Influenza infection intensity screening using a ResNeSt-50 backbone with channel and spatial attention, adapted from the malaria BYOL + attention paper.

## What’s Included
- Training pipeline (PyTorch)
- Evaluation script + report
- ONNX export
- Web demo (Next.js + onnxruntime-node)

## Quick Start

1) Prepare data (IDR idr0128):
```
python3 /Users/rik/projects/FluLens/scripts/prepare_idr0128.py
```
If downloads are limited, build a manifest from already-downloaded images:
```
python3 /Users/rik/projects/FluLens/scripts/build_manifest.py --max-train 40 --max-val 10 --max-test 40
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

## Notes
- Current labels are infection-intensity proxies (low vs high infection). Replace with true infected/uninfected labels when available.
- See `reports/eval_report.md` for baseline metrics.
- The web demo runs inference **in the browser** using ONNX Runtime Web. The ONNX model is stored at `web/public/models/flulens.onnx`.
