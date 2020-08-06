
####
# dataset settings
dataset_type = 'WIDERFaceDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomSquareCrop', crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]),
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(9108, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, keep_height=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='WIDERFaceDataset',
            ann_file='data/quar_train.txt',
            img_prefix='data/WIDERFace/WIDER_train/',
            min_size=1,
            pipeline=train_pipeline)),
    val=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/WIDER_val/val.txt',
        img_prefix='data/WIDERFace/WIDER_val/',
        min_size=1,
        pipeline=test_pipeline),
    # test=dict(
    #     type='WIDERFaceDataset',
    #     ann_file='data/quar_train.txt',
    #     img_prefix='data/WIDERFace/WIDER_train/',
    #     min_size=9,
    #     pipeline=test_pipeline),
    test=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/WIDER_val/val.txt',
        img_prefix='data/WIDERFace/WIDER_val/',
        min_size=1,
        pipeline=test_pipeline)
)

from mmdet.datasets import build_dataset

dataset = build_dataset(data['test'])

data1 = {}
for i in range(len(dataset)):
    t = dataset.get_ann_info(i)
    n = dataset.data_infos[i]['id']
    data1[n] = t

####
import os
import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import loadmat

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

ann_path = 'data/WIDERFace/WIDER_val/Annotations/'
gt_path = '/DATA/data/public/WiderFace/eval_tools/ground_truth/'

(facebox_list, event_list, file_list, hard_gt_list,
    medium_gt_list, easy_gt_list) = get_gt_boxes(gt_path)

data2 = {}
count_face = 0
for i in range(len(event_list)):
    img_list = file_list[i][0]
    sub_gt_list = hard_gt_list[i][0]
    # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
    gt_bbx_list = facebox_list[i][0]

    for j in range(len(img_list)):

        gt_boxes = gt_bbx_list[j][0].astype('float')

        keep_index = sub_gt_list[j][0]
        count_face += len(keep_index)

        data2[str(img_list[j][0][0])] = keep_index

