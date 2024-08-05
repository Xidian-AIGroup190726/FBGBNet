# dataset settings
dataset_type = 'CocoDataset'
#data_root = '/media/ExtDisk/yxt/sardataset/'
#data_root = '/media/ExtDisk/nwpu/'
data_root = '/media/ExtDisk/yxt/ssdd_coco-20221019/ssdd_coco/'#ssdd_coco-20221019/ssdd_coco/train1/train.json
#data_root = 'HRSID/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(500, 500), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(500, 500),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train1/train.json',
        #ann_file=data_root + 'train.json',
        #img_prefix=data_root + 'images',
        # ann_file=data_root + 'annotations/train2017.json',
        # img_prefix=data_root + 'train_image',
        #img_prefix=data_root + 'after_train_image70',
        #img_prefix=data_root + 'train1/after_trainCR94',
        img_prefix=data_root + 'train1/train_image',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root +'test1/test.json',
        #ann_file=data_root + 'test.json',
        #img_prefix=data_root + 'images',
        # ann_file=data_root + 'annotations/test2017.json',
        # img_prefix=data_root + 'test_image',
        #img_prefix=data_root + 'val1/after_valCR94',
        #img_prefix=data_root + 'after_test_image70',
        img_prefix=data_root + 'test1/test_image',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        #ann_file=data_root + 'annotations/test2017.json',##
        ann_file=data_root + 'test1/test.json',
        #ann_file=data_root + 'annotations/val.json',
        #img_prefix=data_root + 'images',
        #img_prefix=data_root + 'test_image',
        img_prefix=data_root + 'test1/test_image',
        #img_prefix=data_root + 'test1/after_testCR94',
        #img_prefix=data_root + 'after_test_image79',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox',save_best='auto')
