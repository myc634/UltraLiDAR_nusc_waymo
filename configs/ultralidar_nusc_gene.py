
model_type = "transformer_training"
# model_type = "conditional_generation"
# model_type = "unconditional_generation"
# If you want training, switch model_type to unconditional_generation or conditional_generation
_base_ = ["./_base_/default_runtime.py"]
batch_size = 2
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = [0.15625, 0.15625, 0.2]
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
plugin = True
plugin_dir = "plugin/"


dataset_type = 'NuscDataset'
data_root = "./datasets/nuScenes/"
file_client_args = dict(backend="disk")


model = dict(
    type='UltraLiDAR',
    model_type=model_type,
    dataset_type=dataset_type,
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    voxelizer=dict(
        type="Voxelizer",
        x_min=point_cloud_range[0],
        x_max=point_cloud_range[3],
        y_min=point_cloud_range[1],
        y_max=point_cloud_range[4],
        z_min=point_cloud_range[2],
        z_max=point_cloud_range[5],
        step=voxel_size[0],
        z_step=voxel_size[2],
    ),
    vector_quantizer=dict(
        type='VectorQuantizer',
        n_e=1024,
        e_dim=1024,
        beta=0.25,
        cosine_similarity=False,
    ),
    maskgit_transformer=dict(
        type='BidirectionalTransformer',
        n_e=1024, 
        e_dim=1024, 
        img_size=80,
    ),
    lidar_encoder=dict(
        type='VQEncoder',
        img_size=640,
        codebook_dim=1024,
    ),
    lidar_decoder=dict(
        type='VQDecoder',
        img_size=(640, 640),
        num_patches=6400,
        codebook_dim=1024,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=point_cloud_range,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,

            # Scale-NMS
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])))

# Data
# Data


bda_aug_conf = dict(rot_lim=(-22.5, 22.5), scale_lim=(0.95, 1.05), flip_dx_ratio=0.5, flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

input_modality = dict(use_lidar=True, use_camera=False, use_radar=False, use_map=False, use_external=False)


data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))


optimizer = dict(
    type="AdamW",
    lr=0.00024,
    betas=(0.9, 0.95),  # the momentum is change during training
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "embedding": dict(decay_mult=0.0),
            "img_backbone": dict(lr_mult=0.1, decay_mult=0.001),
        }
    ),

    weight_decay=0.0001,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=4, grad_clip=dict(max_norm=140, norm_type=2))
lr_config = dict(policy="CosineAnnealing", warmup="linear", warmup_iters=2000, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
runner = dict(type="EpochBasedRunner", max_epochs=160)


find_unused_parameters = False

checkpoint = "./work_dirs/nusc_stage1/epoch_80.pth"
# Remember to switch the dir into the best result in stage1 training

work_dir = "./work_dirs/nusc_stage2"
