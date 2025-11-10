
# Model Training & Deployment Guide (MMDetection)

This guide explains how to train **object detection** and **instance segmentation** models using **MMDetection**, and how to run inference on images and videos. It includes **explicit base config paths** and **dataset paths** so you can copy‑paste and run without guesswork.

---

## 0) Prerequisites

Install the following (GPU recommended for training):

- **Python** ≥ 3.8 (3.8/3.9 are well-tested)
- **CUDA** ≥ 11.0 (match your PyTorch build)
- **PyTorch** ≥ 1.8
- **Conda** (recommended)
- **MMEngine, MMCV, MMDetection**

Official install guide:  
https://mmdetection.readthedocs.io/en/latest/get_started.html

> **Tip:** Always match the **PyTorch + CUDA** versions between your environment and the wheels you install. Check PyTorch’s official “Previous Versions” page for the exact `pip` command that matches your CUDA.

---

## 1) Initial Setup (Conda)

```bash
# Create & activate environment
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# (Example) Install PyTorch with CUDA 11.8 (adjust if needed)
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

# Install MMEngine + MMCV and MMDetection
pip install -U openmim
mim install "mmengine>=0.10.0"
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
```

> If you cloned MMDetection from source (optional for development):
>
> ```bash
> git clone https://github.com/open-mmlab/mmdetection.git
> cd mmdetection
> pip install -e .  # editable install
> ```

---

## 2) Dataset Preparation (COCO format)

**Dataset root** (replace with your actual path):

```
/ABS/PATH/TO/data_cocoformat/football-players-detection.v19-yolo11m.coco/
├── train/
│   ├── *.jpg
│   └── _annotations.coco.json
├── valid/
│   ├── *.jpg
│   └── _annotations.coco.json
└── test/
    ├── *.jpg
    └── _annotations.coco.json
```

> Your repository/tree may look like this:
>
> ```
> soccer_pitch_identification_model_training/
> ├── README.md
> ├── model/
> │   └── epoch_12.pth
> ├── src/
> │   └── mask-rcnn_r50-caffe_football.py
> └── data_cocoformat/
>     └── football-players-detection.v19-yolo11m.coco/   # <- DATASET_ROOT
>         ├── train/  ├── valid/  └── test/
> ```

Define two **absolute** path variables you’ll reuse below:

- `MMDET_DIR`: absolute path to your **mmdetection** repo (if installed in editable mode)  
  e.g., `/home/you/Downloads/mmdetection`
- `DATASET_ROOT`: absolute path to the dataset folder above  
  e.g., `/home/you/yourrepo/data_cocoformat/football-players-detection.v19-yolo11m.coco/`

---

## 3) Tasks & Base Configs

Two common tasks:

| Task | Output |
|------|--------|
| **Object Detection** | Bounding boxes (no masks) |
| **Instance Segmentation** | Bounding boxes + masks |

You will **inherit a base config** from the official MMDetection configs and **override paths, classes, and evaluators**.

### ✅ Base config **paths** (explicit)

When you store your custom config at, say, `configs/Soccer/mask-rcnn_r50-caffe_football.py`, you can inherit like this:

```python
# For Detection (Faster R-CNN)
_base_ = f'{MMDET_DIR}/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# For Instance Segmentation (Mask R-CNN)
_base_ = f'{MMDET_DIR}/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
```

> If you prefer relative paths and your custom config lives inside the `mmdetection/configs/` tree, you can also do:
> ```python
> _base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
> ```

---

## 4) Complete Example: **Mask R‑CNN** (Instance Segmentation)

Save as: `configs/Soccer/mask-rcnn_r50-caffe_football.py`

```python
# ==== User paths you MUST set ====
MMDET_DIR = '/ABS/PATH/TO/mmdetection'  # e.g., '/home/you/Downloads/mmdetection'
DATASET_ROOT = '/ABS/PATH/TO/data_cocoformat/football-players-detection.v19-yolo11m.coco/'

# Inherit the official Mask R-CNN R50-FPN COCO schedule
_base_ = f'{MMDET_DIR}/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

# ---- Classes & palette ----
metainfo = dict(
    classes=('ball', 'goalkeeper', 'player', 'referee'),
    palette=[(220,20,60), (0,128,255), (0,200,0), (255,160,0)],
)

# ---- Model tweaks ----
model = dict(
    data_preprocessor=dict(
        mean=[103.530, 116.280, 123.675],  # Caffe-style mean
        std=[1.0, 1.0, 1.0],               # keep std 1.0 when using Caffe init
        bgr_to_rgb=False                   # Caffe weights expect BGR
    ),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://detectron2/resnet50_caffe')
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=4),
        mask_head=dict(num_classes=4),
    ),
)

# ---- DataLoader & Dataset (COCO format) ----
train_batch_size = 2
workers_per_gpu = 2

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=workers_per_gpu,
    dataset=dict(
        type='CocoDataset',
        data_root=DATASET_ROOT,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=workers_per_gpu,
    dataset=dict(
        type='CocoDataset',
        data_root=DATASET_ROOT,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=metainfo
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=workers_per_gpu,
    dataset=dict(
        type='CocoDataset',
        data_root=DATASET_ROOT,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo
    )
)

# ---- Evaluators (bbox + segm) ----
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{DATASET_ROOT}valid/_annotations.coco.json',
    metric=['bbox', 'segm'],
    format_only=False,
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{DATASET_ROOT}test/_annotations.coco.json',
    metric=['bbox', 'segm'],
    format_only=False,
)

# ---- Schedules & runtime ----
train_cfg = dict(max_epochs=12)  # increase to 24/36 for better results
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50),
)

# Where to save work
work_dir = f'{MMDET_DIR}/work_dirs/mask-rcnn_r50-caffe_football'
```

---

## 5) Optional: **Faster R‑CNN** (Object Detection only)

Key change is **no masks** and metrics set to `['bbox']`. Minimal diff:

```python
_base_ = f'{MMDET_DIR}/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
model = dict(
  roi_head=dict(bbox_head=dict(num_classes=4))
)
val_evaluator = dict(type='CocoMetric', metric=['bbox'], ...)
test_evaluator = dict(type='CocoMetric', metric=['bbox'], ...)
```

---

## 6) Training

Run from **anywhere** (paths are absolute in the config):

```bash
python {MMDET_DIR}/tools/train.py \
  configs/Soccer/mask-rcnn_r50-caffe_football.py
```

### Checkpoints & logs

Saved to:

```
{MMDET_DIR}/work_dirs/mask-rcnn_r50-caffe_football/
├── epoch_1.pth
├── epoch_2.pth
├── last_checkpoint
└── vis_data/
```

> Increase `max_epochs` (e.g., 24/36) and consider a cosine schedule for better accuracy on small datasets.

---

## 7) Evaluation (mAP)

Set via `val_evaluator` / `test_evaluator`. Common outputs:

- `mAP@[0.50:0.95]`
- `mAP50`, `mAP75`
- `AP_small`, `AP_medium`, `AP_large`
- Class-wise AP

Example (bbox):  
| Category   | mAP   |
|------------|-------|
| ball       | 0.000 |
| goalkeeper | 0.505 |
| player     | 0.605 |
| referee    | 0.454 |

> For instance segmentation, you’ll also see `segm` AP metrics.

---

## 8) Inference — Images

```bash
python {MMDET_DIR}/demo/image_demo.py \
  /ABS/PATH/TO/image.jpg \
  configs/Soccer/mask-rcnn_r50-caffe_football.py \
  --weights {MMDET_DIR}/work_dirs/mask-rcnn_r50-caffe_football/epoch_12.pth \
  --device cuda:0 \
  --out-dir outputs/images \
  --show
```

---

## 9) Inference — Videos

```bash
python {MMDET_DIR}/demo/video_demo.py \
  /ABS/PATH/TO/video.mp4 \
  configs/Soccer/mask-rcnn_r50-caffe_football.py \
  {MMDET_DIR}/work_dirs/mask-rcnn_r50-caffe_football/epoch_12.pth \
  --device cuda:0 \
  --out outputs/video_out.mp4 \
  --show \
  --score-thr 0.5
```

---

## 10) Quick Tips

- **Choose model**:  
  - Bounding boxes only → **Faster R-CNN**  
  - Boxes + masks → **Mask R-CNN**
- **Epochs**: More epochs generally improve results; watch for overfitting.
- **Weights**: Using `detectron2` Caffe weights requires **BGR** and Caffe-style mean/std.
- **Small datasets**: Try stronger augmentations or a longer schedule.
- **Reproducibility**: Set seeds (`--cfg-options randomness.seed=42` + deterministic settings) if needed.

---

## 11) One-Page Summary

| Step | Action |
|------|--------|
| 1 | Create & activate Conda env |
| 2 | Install PyTorch, MMEngine, MMCV, MMDetection |
| 3 | Prepare COCO dataset & set `DATASET_ROOT` |
| 4 | Copy example config & set `MMDET_DIR` and base path |
| 5 | Train with `tools/train.py` |
| 6 | Evaluate with CocoMetric (bbox/segm) |
| 7 | Run image/video demos for inference |

---


