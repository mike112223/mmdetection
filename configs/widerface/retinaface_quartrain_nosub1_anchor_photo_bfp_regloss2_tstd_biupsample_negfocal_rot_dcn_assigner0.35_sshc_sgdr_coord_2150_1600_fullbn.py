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
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='WIDERFaceDataset',
            ann_file='data/quar_train.txt',
            img_prefix='data/WIDERFace/WIDER_train/',
            min_size=1,
            offset=0,
            pipeline=train_pipeline)),
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


# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        style='pytorch'),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=6,
            norm_cfg=dict(type='BN', requires_grad=True),
            coord_cfg=dict(with_r=False),
            upsample_cfg=dict(mode='bilinear')),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=6,
            refine_level=2,
            refine_type='non_local',
            norm_cfg=dict(type='BN', requires_grad=True),
            upsample_cfg=dict(mode='bilinear')),
        dict(
            type='SSHC',
            in_channel=256,
            num_levels=6,
            norm_cfg=dict(type='BN', requires_grad=True),
            share=True)
    ],
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        coord_cfg=dict(with_r=False),
        norm_cfg=dict(type='BN', requires_grad=True),
        #norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=2**(4 / 3),
            scales_per_octave=3,
            ratios=[1.3],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.135, 0.135, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            no_focal_pos=True,
            bg_id=1,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.35,
        neg_iou_thr=0.35,
        min_pos_iou=0.35,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=3000,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_threshold=0.4),
    max_per_img=800)


# optimizer
optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()#grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineRestart',
    periods=[30, 30, 30, 30, 30, 30],
    restart_weights=[1, 1, 1, 1, 1, 1],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-1,
    min_lr_ratio=1e-2)
# runtime settings
total_epochs = 180
log_config = dict(interval=100)


checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
