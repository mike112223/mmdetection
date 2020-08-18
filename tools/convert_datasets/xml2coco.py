
import json
from mmdet.datasets import build_dataset

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
            ann_file='data/WIDERFace/WIDER_train/train.txt',
            img_prefix='data/WIDERFace/WIDER_train/',
            min_size=9,
            offset=0,
            pipeline=train_pipeline)),
    val=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/WIDER_val/val.txt',
        img_prefix='data/WIDERFace/WIDER_val/',
        min_size=1,
        offset=0,
        pipeline=test_pipeline),
    test=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/WIDER_train/train.txt',
        img_prefix='data/WIDERFace/WIDER_train/',
        min_size=9,
        offset=0,
        pipeline=test_pipeline),
    # test=dict(
    #     type='WIDERFaceDataset',
    #     ann_file='data/WIDERFace/WIDER_val/val.txt',
    #     img_prefix='data/WIDERFace/WIDER_val/',
    #     min_size=1,
    #     offset=0,
    #     pipeline=test_pipeline)
)


dataset = build_dataset(data['test'])

data = []
for i in range(len(dataset)):
    _data = {}
    t = dataset.get_ann_info(i)
    n = dataset.data_infos[i]
    _data['img_info'] = n
    _data['ann_info'] = t
    data.append(_data)

json_data = {'images': [], 'annotations': []}

json_data['licenses'] = [{'id': 1, 'name': '', 'url': ''}]
json_data['categories'] = [{'id': 1, 'name': 'face', 'supercategory': 'human body part'}]
json_data['inf0'] = {'year': 2016,
                     'version': 'v1.0',
                     'description': 'WIDER FACE: A Face Detection Benchmark',
                     'contributor': '',
                     'url': 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/',
                     'date_created': '2020-07-09T08:14:07.999019'}


img_id = 0
ann_id = 0
for d in data:

    img_info = d['img_info']
    ann_info = d['ann_info']

    json_data['images'].append({
        'coco_url': '',
        'data_captured': '',
        'file_name': img_info['filename'],
        'flickr_url': '',
        'id': img_id,
        'height': img_info['height'],
        'width': img_info['width'],
        'license': 1
    })

    img_id += 1

    for i in range(len(ann_info['bboxes'])):

        bbox = ann_info['bboxes'][i].tolist()
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        json_data['annotations'].append({
            'id': ann_id,
            'image_id': img_id,
            'category_id': 1,
            'iscrowd': 0,
            'segmentation': [],
            'area': w * h,
            'bbox': bbox,
        })

        ann_id += 1

with open('widerface_fulltrain.json', 'w') as f:
    json.dump(json_data, f)
