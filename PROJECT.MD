# 项目技术规格说明书：流感病毒 AI 智能检测系统

## 1. 项目背景与目标
[cite_start]本项目旨在复刻《一种基于自监督学习和注意力机制的疟原虫智能检测方法.pdf》的核心技术方案 [cite: 25, 32, 33][cite_start]。将其迁移至**流感病毒检测领域（当前实现为甲型流感 IAV）**，通过自监督预训练解决小样本标记难题 [cite: 46, 127][cite_start]，并利用双重注意力机制捕获极其微小的病毒颗粒特征 [cite: 42, 44]。

### 1.1 当前数据源与标注口径（实现版）
由于暂无真实流感/禽流感阳性/阴性标注数据，本项目当前实现基于公开数据集 **IDR idr0128 (Georgi et al. Influenza A virus screen)**。该数据集对应**甲型流感 IAV**，以高内涵显微成像记录感染强度，`ImageData.csv` 提供每张图像的 `numberOfInfectedNuclei` 等指标。

为保证可训练、可评估，本实现将二分类定义为：
* **低感染 (label=0)**：`numberOfInfectedNuclei` 位于训练集 **10% 分位及以下**。
* **高感染 (label=1)**：`numberOfInfectedNuclei` 位于训练集 **90% 分位及以上**。

这是一种“感染强度分层”的替代口径，**并非严格的感染/未感染**。若后续引入真实阴性样本，应替换此口径并重新训练与评估。

---

## 2. 硬件与环境配置 (Target: Apple M4 Pro)
* **计算设备**：Apple M4 Pro 芯片（建议利用其统一内存架构加速训练）。
* **运行内存**：48GB 统一内存。
* [cite_start]**软件框架**：Python 3.10+，PyTorch 2.0+ [cite: 182]。
* **设备调用**：必须显式指定 `mps` 设备加速。

---

## 3. 核心实施流程

### 第一阶段：数据资产工程
1. **数据集规模**：
    * [cite_start]**训练与验证集**：共计 30,033 张图像 [cite: 115]。
    * [cite_start]**独立测试集**：12,784 张图像 [cite: 117]。
2. **标注与划分**：
    * [cite_start]采用 **3:1** 的随机比例划分训练集与验证集 [cite: 116]。
    * [cite_start]所有图像由 3 位具备 10 年以上经验的专家进行标注 [cite: 114]。
3. [cite_start]**图像预处理** [cite: 122]：
    * 统一裁剪并缩放至 **224x224** 像素。
    * [cite_start]应用色彩抖动（亮度、对比度、饱和度、色调调整）和高斯模糊 [cite: 122, 123]。

#### 1.2 数据准备步骤（实现版）
使用脚本自动下载、筛选和生成训练图像与清单：
```
python3 /Users/rik/projects/FluLens/scripts/prepare_idr0128.py
```
输出：
* `data/processed/idr0128/images/`：224x224 PNG（融合核染色与病毒通道）。
* `data/processed/idr0128/manifest.csv`：训练/验证/测试清单与标签。

如果下载受限，可基于已下载图片构建清单：
```
python3 /Users/rik/projects/FluLens/scripts/build_manifest.py --max-train 240 --max-val 60 --max-test 60
```

### [cite_start]第二阶段：自监督预训练 (BYOL) [cite: 127]
[cite_start]在正式训练分类器前，先通过 BYOL 算法让模型在阳性样本上进行自监督学习 [cite: 138]。
* [cite_start]**掩膜策略 (Masking)**：通过图像分割获取细胞位置，随后对细胞区域进行遮盖，强迫网络学习更深层次的表征特征 [cite: 100, 101, 142]。
* [cite_start]**架构设计**：包含在线网络 (Online Network) 和目标网络 (Target Network) [cite: 130]。

### 第三阶段：网络架构与注意力机制
[cite_start]模型主干采用 **ResNeSt-50** 架构 [cite: 182]，并在其基础上集成以下两个核心组件：

#### 3.1 通道注意力模块 (Channel Attention Module)
[cite_start]旨在识别图像中对病毒特征最敏感的特征通道 [cite: 149]。
[cite_start]公式如下 [cite: 151]：
$M_{c}(F)=\sigma(MLP(avgPool(F))) + \sigma(MLP (maxPool(F)))$

#### 3.2 空间注意力模块 (Spatial Attention Module)
[cite_start]旨在精准定位病毒颗粒在图像平面中的位置 [cite: 153]。
[cite_start]公式如下 [cite: 158]：
$M_{s}(F)=\sigma(f^{7\times7}([AvgPool(F);MaxPool(F)])$

---

## 4. 训练与性能评估

### [cite_start]4.1 训练策略 [cite: 184, 188]
* **预训练加载**：加载 BYOL 的在线网络参数作为初始权重。
* [cite_start]**微调 (Fine-tuning)**：保持网络结构不变，重点调整最后的全连接层和最后三层卷积层的权重 [cite: 184]。
* [cite_start]**收敛条件**：通常进行 50 轮训练，若验证集损失函数不再下降则提前停止 [cite: 185, 188]。

### [cite_start]4.2 验收指标 [cite: 47, 206]
系统需在独立测试集上达到或超过以下学术表现：
* [cite_start]**测试准确率 (Accuracy)**：**97.8%** [cite: 47, 206]。
* [cite_start]**灵敏度 (Sensitivity)**：**96.5%** [cite: 47, 206]。
* [cite_start]**特异性 (Specificity)**：**98.9%** [cite: 47, 206]。

---

## 6. 交付物与运行流程（实现版）

### 6.1 训练模型
```
PYTHONPATH=/Users/rik/projects/FluLens/src \
python3 /Users/rik/projects/FluLens/scripts/train.py \
  --manifest /Users/rik/projects/FluLens/data/processed/idr0128/manifest.csv \
  --epochs 6 --batch-size 8 --pretrained --freeze-backbone \
  --out /Users/rik/projects/FluLens/models/flulens.pt
```

### 6.2 评估报告
```
PYTHONPATH=/Users/rik/projects/FluLens/src \
python3 /Users/rik/projects/FluLens/scripts/eval.py \
  --manifest /Users/rik/projects/FluLens/data/processed/idr0128/manifest.csv \
  --weights /Users/rik/projects/FluLens/models/flulens.pt \
  --split test
```
评估结果归档在：`reports/eval_report.md`

### 6.3 导出 ONNX 与网页产品
```
PYTHONPATH=/Users/rik/projects/FluLens/src \
python3 /Users/rik/projects/FluLens/scripts/export_onnx.py \
  --weights /Users/rik/projects/FluLens/models/flulens.pt \
  --out /Users/rik/projects/FluLens/models/flulens.onnx
```

网页产品位于 `web/`，部署前执行：
```
cd /Users/rik/projects/FluLens/web
npm install
npm run build
```
网页端使用 ONNX Runtime Web 在浏览器内完成推理，模型文件位于：
`web/public/models/flulens.onnx`

## 5. 程序开发备注
1. **代码适配**：使用 `torch.device("mps")` 替代 `cuda`。
2. **Batch Size 优化**：利用 48GB 显存优势，建议设置较高的 `batch_size`（如 128/256）以加速收敛。
3. **数据读取**：设置 `num_workers` 以匹配 M4 Pro 的核心数。
