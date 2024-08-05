# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',#'linear',
    warmup_iters=500,
    warmup_ratio=0.1,#0.001,
    step=[40, 60])
runner = dict(type='EpochBasedRunner', max_epochs=60)
