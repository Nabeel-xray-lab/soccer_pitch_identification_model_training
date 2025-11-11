# MMDetection — Training & Inference Guide (Detection & Segmentation)

A **copy‑paste ready** guide you can reuse across projects for **object detection** (bbox) and **instance segmentation** (bbox + masks). Includes explicit, absolute paths and minimal config templates for both tasks.

---

## 0) Prerequisites

Install the following (GPU recommended):

* **Python** ≥ 3.8 (3.8/3.9 widely used)
* **CUDA** that matches your PyTorch build (e.g., 11.8)
* **PyTorch** ≥ 1.8
* **Conda** (recommended)
* **MMEngine, MMCV, MMDetection**: https://mmdetection.readthedocs.io/en/latest/get_started.html

> Use PyTorch’s “Previous Versions” page to match **Torch + CUDA** exactly.

---

## 1) Environment Setup (Conda)

Activate using the documentation in MMDetection: https://mmdetection.readthedocs.io/en/latest/get_started.html#installation

---

## 2) Paths & Project Layout (Absolute Paths)

Define two **absolute** variables used through this guide:

* `MMDET_DIR` → absolute path to your **mmdetection** repo (if editable install)
  e.g., `/home/you/code/mmdetection`
* `DATASET_ROOT` → absolute path to your COCO dataset folder
  e.g., `/home/you/data/coco_dataset/`

**COCO-style dataset layout:**

```
/ABS/PATH/TO/data/your_dataset/
├── train/  ├── valid/  └── test/
    ├── *.jpg
    └── _annotations.coco.json
```

Recommended repo layout:

```
project_root/
├── configs/
│   └── yourrepo/
│       ├── faster-rcnn_r50_fpns_customconfig.py     # Detection
│       └── mask-rcnn_r50_fpns_customconfig.py       # Segmentation
├── src/
└── data/  -> DATASET_ROOT
```

---

## 3) Task Overview

| Task                      | What you get           | Typical base config                                  |
| ------------------------- | ---------------------- | ---------------------------------------------------- |
| **Object Detection**      | Bounding boxes only    | `configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py` |
| **Instance Segmentation** | Bounding boxes + masks | `configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py`     |

> You can swap to RetinaNet, YOLOX, Cascade R-CNN, or Mask2Former later. Start simple.

---

## 4) Minimal, Generic Configs (Copy–Paste)

Below are **two small, generic templates** you can reuse across datasets by just changing `metainfo.classes`, `DATASET_ROOT`, and `work_dir`. Both use absolute paths and the same dataloader/evaluator blocks.

### 4.1) **Detection** (Faster R-CNN, R50-FPN)

**Save as:** `configs/Soccer/faster-rcnn_r50_fpns_customconfig.py`

```python
# ==== User paths ====
MMDET_DIR = '/ABS/PATH/TO/mmdetection'
DATASET_ROOT = '/ABS/PATH/TO/data/your_dataset/'

# ==== Classes ====
CLASSES = ('class1', 'class2', 'class3')  # <-- EDIT
NUM_CLASSES = len(CLASSES)

# ==== Inherit base ====
_base_ = f'{MMDET_DIR}/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# ---- Metainfo ----
metainfo = dict(classes=CLASSES)

# ---- Model (num classes) ----
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=NUM_CLASSES)
    )
)

# ---- Datasets ----
train_batch = 2
workers = 2

train_dataloader = dict(
    batch_size=train_batch,
    num_workers=workers,
    dataset=dict(
        type='CocoDataset',
        data_root=DATASET_ROOT,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=CLASSES)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=workers,
    dataset=dict(
        type='CocoDataset',
        data_root=DATASET_ROOT,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=dict(classes=CLASSES)
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=workers,
    dataset=dict(
        type='CocoDataset',
        data_root=DATASET_ROOT,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=CLASSES)
    )
)

# ---- Evaluators ----
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{DATASET_ROOT}valid/_annotations.coco.json',
    metric=['bbox'],
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{DATASET_ROOT}test/_annotations.coco.json',
    metric=['bbox'],
)

# ---- Schedule & runtime ----
train_cfg = dict(max_epochs=12)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5),
)

work_dir = f'{MMDET_DIR}/work_dirs/faster-rcnn_r50_fpns_customconfig'
```

### 4.2) **Segmentation** (Mask R-CNN, R50-FPN)

**Save as:** `configs/Soccer/mask-rcnn_r50_fpns_customconfig.py`

```python
# ==== User paths ====
MMDET_DIR = '/ABS/PATH/TO/mmdetection'
DATASET_ROOT = '/ABS/PATH/TO/data/your_dataset/'

# ==== Classes ====
CLASSES = ('class1', 'class2', 'class3')  # <-- EDIT
NUM_CLASSES = len(CLASSES)

# ==== Inherit base ====
_base_ = f'{MMDET_DIR}/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

# ---- Metainfo ----
metainfo = dict(classes=CLASSES)

# ---- Model (num classes) ----
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=NUM_CLASSES),
        mask_head=dict(num_classes=NUM_CLASSES),
    )
)

# ---- Datasets ----
train_batch = 2
workers = 2

train_dataloader = dict(
    batch_size=train_batch,
    num_workers=workers,
    dataset=dict(
        type='CocoDataset',
        data_root=DATASET_ROOT,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=CLASSES)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=workers,
    dataset=dict(
        type='CocoDataset',
        data_root=DATASET_ROOT,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=dict(classes=CLASSES)
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=workers,
    dataset=dict(
        type='CocoDataset',
        data_root=DATASET_ROOT,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=CLASSES)
    )
)

# ---- Evaluators ----
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{DATASET_ROOT}valid/_annotations.coco.json',
    metric=['bbox', 'segm'],
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{DATASET_ROOT}test/_annotations.coco.json',
    metric=['bbox', 'segm'],
)

# ---- Schedule & runtime ----
train_cfg = dict(max_epochs=12)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5),
)

work_dir = f'{MMDET_DIR}/work_dirs/mask-rcnn_r50_fpns_customconfig'
```


---

## 5) Training

Run from anywhere (absolute paths inside configs):

```bash
# Detection
python ${MMDET_DIR}/tools/train.py \
  configs/yourrepo/faster-rcnn_r50_fpns_customconfig.py

# Segmentation
python ${MMDET_DIR}/tools/train.py \
  configs/yourrepo/mask-rcnn_r50_fpns_generic.py
```

**Outputs** saved to `work_dir`:

```
work_dirs/<exp_name>/
├── epoch_1.pth
├── last_checkpoint
└── vis_data/
```

> Scale epochs to 24/36 for stronger results. Use cosine or multi‑step LR as needed.

---

## 6) Evaluation (COCO mAP)

Metrics configured via `val_evaluator` / `test_evaluator`:

* **Detection:** `metric=['bbox']`
* **Segmentation:** `metric=['bbox', 'segm']`

```bash
# Validation during/after training happens automatically if configured.
# To run a separate test:
python ${MMDET_DIR}/tools/test.py \
  configs/yourrepo/mask-rcnn_r50_fpns_customconfig.py \
  ${MMDET_DIR}/work_dirs/mask-rcnn_r50_fpns_customconfig/latest.pth \
  --eval bbox segm
```

---

## 7) Inference — Images & Videos

### 7.1) Images

```bash
python ${MMDET_DIR}/demo/image_demo.py \
  /ABS/PATH/TO/image.jpg \
  configs/Soccer/mask-rcnn_r50_fpns_customconfig.py \
  --weights ${MMDET_DIR}/work_dirs/mask-rcnn_r50_fpns_customconfig/epoch_12.pth \
  --device cuda:0 \
  --out-dir outputs/images \
  --show
```

### 7.2) Videos

```bash
python ${MMDET_DIR}/demo/video_demo.py \
  /ABS/PATH/TO/video.mp4 \
  configs/Soccer/faster-rcnn_r50_fpns_customconfig.py \
  ${MMDET_DIR}/work_dirs/faster-rcnn_r50_fpns_customconfig/epoch_12.pth \
  --device cuda:0 \
  --out outputs/video_out.mp4 \
  --show \
  --score-thr 0.5
```

---


### Optional: Quick Backbone Base Paths

* Faster R-CNN R50-FPN: `{MMDET_DIR}/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`
* RetinaNet R50-FPN: `{MMDET_DIR}/configs/retinanet/retinanet_r50_fpn_1x_coco.py`
* YOLOX-S: `{MMDET_DIR}/configs/yolox/yolox_s_8xb8-300e_coco.py`
* Cascade R-CNN R50: `{MMDET_DIR}/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py`
* Mask R-CNN R50-FPN: `{MMDET_DIR}/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py`
* Mask2Former R50: `{MMDET_DIR}/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py`

