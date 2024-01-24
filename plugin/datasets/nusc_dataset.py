import os
import os.path as osp
import time
import random
import mmcv
import numpy as np
from IPython import embed
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets import Custom3DDataset
import pickle
from .evaluation.jsd_mmd import compute_mmd, gaussian, jsd_2d
import torch


@DATASETS.register_module()
class NuscDataset(Custom3DDataset):
    def __init__(self,
                 ann_file,
                 data_root,
                 classes,
                 modality=dict(
                     use_camera=True,
                     use_lidar=False,
                     use_radar=False,
                     use_map=True,
                     use_external=False,
                 ),
                 pipeline=None,
                 box_type_3d='LiDAR',
                 coord_dim=3,
                 interval=1,
                 work_dir=None,
                 use_valid_flag=True,
                 eval_cfg: dict = dict(),
                 test_mode=False,
                 **kwargs,
                 ):
        self.interval = interval
        super().__init__(
            classes=classes,
            data_root=data_root,
            ann_file=ann_file,
            modality=modality,
            pipeline=pipeline,
            test_mode=test_mode,
        )

        self.coord_dim = coord_dim
        self.eval_cfg = eval_cfg
        self.use_valid_flag = use_valid_flag
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        # dummy flag to fit with mmdet
        self.flag = np.zeros(len(self), dtype=np.uint8)
        self.data_infos = self.load_annotations(self.ann_file)
        # print(self.test_mode)
        self.work_dir = work_dir
        if self.test_mode:
            random.seed(42)
            self.data_infos = random.sample(self.data_infos, 2000)
            
        

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        self.data_infos = {}
        print('collecting samples...')
        start_time = time.time()
        ann = mmcv.load(self.ann_file)
        samples = ann[::self.interval]
        print(
            f'collected {len(samples)} samples in {(time.time() - start_time):.2f}s')
        self.loda_flag = True
        return samples

    
        
    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def get_data_info(self, idx):


        sample = self.data_infos[idx]
        location = sample['location']

        # breakpoint()
        input_dict = {
            # for nuscenes, the order is
            # 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            # 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            'sample_idx': sample['token'],
            'location': location,
            'ego2global_translation': sample['e2g_translation'],
            'ego2global_rotation': sample['e2g_rotation'],
            'lidar2ego_translation': sample['lidar2ego_translation'],
            'lidar2ego_rotation': sample['lidar2ego_rotation'],
            'bbox3d_fields':[],
            'pts_seg_fields':[],
        }

        if self.modality['use_lidar']:
            input_dict.update(
                dict(
                    pts_filename=sample['lidar_path'],
                )
            )
        annos = self.get_ann_info(idx)
        input_dict['ann_info'] = annos


        return input_dict
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """

        info = self.data_infos[index]

        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)


        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            # pts_semantic_mask_path=pts_semantic_mask_path,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    
    