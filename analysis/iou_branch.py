from mmdet.datasets import build_dataset
from mmdet.core import bbox_overlaps
import mmcv

import torch

# dataset settings
dataset_type = 'WIDERFaceDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomSquareCrop', crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Albu',
         transforms=[
             dict(type='Rotate',
                  limit=10)],
         update_pad_shape=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2150, 1600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/WIDER_train/train.txt',
        img_prefix='data/WIDERFace/WIDER_train/',
        min_size=1,
        offset=0,
        pipeline=train_pipeline),
    val=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/WIDER_val/val.txt',
        img_prefix='data/WIDERFace/WIDER_val/',
        min_size=1,
        offset=0,
        pipeline=test_pipeline),
    # test=dict(
    #     type='WIDERFaceDataset',
    #     ann_file='data/quar_train.txt',
    #     img_prefix='data/WIDERFace/WIDER_train/',
    #     min_size=1,
    #     offset=0,
    #     pipeline=test_pipeline),
    test=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/WIDER_val/val.txt',
        img_prefix='data/WIDERFace/WIDER_val/',
        min_size=1,
        offset=0,
        pipeline=test_pipeline)
)

dataset = build_dataset(data['test'])
results = mmcv.load('retina_full_photo_bfp_biupsample_ssh_dcn_sgdr_gn_rot_601_iou_aware.pkl')

annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]

for i, result in enumerate(results):

    dts = torch.from_numpy(result[0][:, :4])
    scores = result[0][:, 4]

    gts = torch.from_numpy(annotations[i]['bboxes'])

    if len(dts) == 0 or len(gts) == 0:
        continue

    overlaps = bbox_overlaps(dts, gts).numpy()

    max_ious = overlaps.max(axis=1)

    scores = max_ious

    results[i][0][:, 4] = scores

dataset.evaluate(
    results,
    'mAP',
    None,
    scale_ranges=[(0, 8), (8, 16), (16, 24), (24, 32), (32, 96), (96, 5000), (0, 5000)],
    iou_thr=0.5)
