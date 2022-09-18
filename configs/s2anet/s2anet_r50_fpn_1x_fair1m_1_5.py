# {DATASET_PATH}
#dataset_root = '/mnt/disk3/flowey/dataset/fair1m_1_5_a'
#dataset_root = '/run/media/guangzhi/MLDATA/FAIR1M1-5'
dataset_root = '/root/autodl-tmp/dataset/'
# model settings
model = dict(
    type='S2ANet',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages=["layer1","layer2","layer3","layer4"],
        pretrained= True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5),
    bbox_head=dict(
        type='S2ANetHead',
        num_classes=11,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        with_orconv=True,
        #anchor_ratios=[1.0],
        anchor_ratios=[3.0,],  # needs to put 1.0 at the 1st
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_scales=[4],
        #anchor_scales=[2,4,6],
        anchor_angles=[0],  # needs to put 0 at the 1st
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            #gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_fam_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        #type='PolyCIoULoss', iou_thr=0.2, loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            #gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(
            #type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            type='PolyCIoULoss', iou_thr=0.2, loss_weight=1.0),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            #score_thr=0.20,
            score_thr=0.10,
            nms=dict(type='nms_rotated', iou_thr=0.05),
            max_per_img=2000),
        train_cfg=dict(
            fam_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.,
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    gt_max_assign_all=True,
                    iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                    #assigner=dict(type='MinCenterDiffAssigner'),
                bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                                target_means=(0., 0., 0., 0., 0.),
                                target_stds=(1., 1., 1., 1., 1.),
                                clip_border=True),
                reg_decoded_bbox=False,
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            odm_cfg=dict(
                assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                match_low_quality=True,
                gt_max_assign_all=True,
                iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                #assigner=dict(type='MinCenterDiffAssigner'),
                bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                                target_means=(0., 0., 0., 0., 0.),
                                target_stds=(1., 1., 1., 1., 1.),
                                clip_border=True),
                allowed_border=-1,
                reg_decoded_bbox=True,
                pos_weight=-1,
                debug=False))
        )
    )
dataset = dict(
    train=dict(
        type="FAIR1M_1_5_Dataset",
        dataset_dir=f'{dataset_root}/preprocessed/train_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(type='RotatedRandomFlip', prob=0.5),
            dict(
            type="RandomRotateAug",
            random_rotate_on=True,
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)

        ],
        batch_size=2,
        num_workers=4,
        shuffle=True,
        #balance_category={
        #"Airplane": 1.0,
        #"Ship": 1,
        #"Vehicle": 1,
        #"Basketball_Court": 1,
        #"Tennis_Court": 0.8,
        #"Football_Field": 1,
        #"Baseball_Field": 0.8,
        #"Intersection": 1,
        #"Roundabout": 1,
        #"Bridge": 1,
        #},
        #balance_category={
            #"Airplane": 0.1,
            #"Ship": 1,
            #"Vehicle": 1,
            #"Basketball_Court": 0.3,
            #"Tennis_Court": 0.1,
            #"Football_Field": 0.1,
            #"Baseball_Field": 0.1,
            #"Intersection": 1,
            #"Roundabout": 0.1,
            #"Bridge": 1,
            #},
        filter_empty_gt=False
    ),
    val=dict(
        type="FAIR1M_1_5_Dataset",
        dataset_dir=f'{dataset_root}/preprocessed/train_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        balance_category=False,
        #balance_category={
        #"Airplane": 0.3,
        #"Ship": 0.3,
        #"Vehicle": 0.3,
        #"Basketball_Court": 0.3,
        #"Tennis_Court": 0.3,
        #"Football_Field": 0.3,
        #"Baseball_Field": 0.3,
        #"Intersection": 0.3,
        #"Roundabout": 0.3,
        #"Bridge": 0.3,
        #},
        batch_size=4,
        num_workers=8,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        images_dir=f'{dataset_root}/preprocessed/test_1024_200_1.0/images',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,),
        ],
        dataset_type="FAIR1M_1_5",
        num_workers=2,
        batch_size=1,
    )
)

optimizer = dict(
    type='SGD',
    #lr=0.01/4., #0.0,#0.01*(1/8.),
    lr=0.01/6., #0.0,#0.01*(1/8.),
    momentum=0.95,
    weight_decay=0.0001,
    grad_clip=dict(
        max_norm=30,
        norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    gamma=0.2,
    milestones=[7, 13])


logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
max_epoch = 6
eval_interval = 6
checkpoint_interval = 1
log_interval = 50