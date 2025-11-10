# path to model in MMdetection
_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

metainfo = dict(
    classes=('ball', 'goalkeeper', 'player', 'referee'),
    palette=[(220, 20, 60), (0, 128, 255), (0, 200, 0), (255, 160, 0)]
)

model = dict(
    data_preprocessor=dict(
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(
        # Faster R-CNN has only bbox head; no mask head here
        bbox_head=dict(num_classes=4)
    )
)
datasetpath = '/home/nsk/Downloads/mmdetection/configs/Soccer/'
data_root = datasetpath + 'football-players-detection.v19-yolo11m.coco/'

train_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo
    )
)
val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',   # change to 'val/' if your folder is named 'val'
        data_prefix=dict(img='valid/'),
        metainfo=metainfo
    )
)
test_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo
    )
)

# bbox-only metrics (no 'segm')
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric=['bbox'],
    classwise=True
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric=['bbox'],
    classwise=True
)

train_cfg = dict(val_interval=1)

