import numpy as np
import torch
import os
import os.path as osp
import mmengine
from glob import glob
from kitti360scripts.helpers import data, annotation, project
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid
from pyquaternion import Quaternion
# from kitti360scripts.devkits.convertOxtsPose.python.data import loadTimestamps
PERCEPTION_RANGE = 100 # we look for objects within 100 meters of each frame
CLASSES = ['bicycle', 'box', 'bridge', 'building', 'bus', 'car',
           'caravan', 'garage', 'lamp', 'motorcycle', 'person', 
           'pole', 'rider', 'smallpole', 'stop', 'traffic light', 
           'traffic sign', 'trailer', 'train', 'trash bin', 'truck', 
           'tunnel', 'unknown construction', 'unknown object', 
           'unknown vehicle', 'vending machine']
SELECTED_CLASS = ['bicycle', 'bus', 'car',
           'caravan', 'motorcycle', 'person', 
           'rider', 'trailer', 'train', 'truck', 
           'unknown vehicle']   # static objects are too many, about 2/3 of objects
categories = {}
for i, name in enumerate(CLASSES):
    categories[name] = i
METAINFO = {
    'categories': categories,
    'dataset': 'kitti-360',
    'info_version': '1.1',
    'version': 'v0.0-trainval'}

class CustomCameraPerspective(project.CameraPerspective):
    def projectCenter(self, obj3d, frameId):
        vertices = obj3d.vertices.copy()
        vertices = vertices.mean(0,keepdims=True)
        uv, depth = self.project_vertices(vertices, frameId)
        obj3d.vertices_proj = uv
        obj3d.vertices_depth = depth 

# TODO: filter out occluded objects via tracing lidar ray to bbox points
class kitti360Dataloader(object):
    def __init__(self, data_root, gt_in_cam_only = True, out_dir = None):
        self.data_root = data_root
        self.gt_in_cam_only = gt_in_cam_only
        if out_dir is None:
            out_dir = data_root
        self.out_dir = data_root
        self.scenes = os.listdir(self.data_root+'/data_2d_raw')
        self.loaders = []
        # breakpoint()
        self.val_split = ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync']
        self.train_split = ['2013_05_28_drive_0003_sync', 
                            '2013_05_28_drive_0004_sync', '2013_05_28_drive_0005_sync', 
                            '2013_05_28_drive_0006_sync', '2013_05_28_drive_0007_sync', 
                            '2013_05_28_drive_0009_sync', '2013_05_28_drive_0010_sync']
        self.val_loaders = []
        self.train_loaders = []
        for scene in self.scenes:
            Loader = kitti360Dataloader_OneScene(self.data_root, scene, gt_in_cam_only = self.gt_in_cam_only)
            self.loaders.append(Loader)
        for scene in self.val_split:
            Loader = kitti360Dataloader_OneScene(self.data_root, scene, gt_in_cam_only = self.gt_in_cam_only)
            self.val_loaders.append(Loader)
        for scene in self.train_split:
            Loader = kitti360Dataloader_OneScene(self.data_root, scene, gt_in_cam_only = self.gt_in_cam_only)
            self.train_loaders.append(Loader)

    def get_classes(self):
        self.class_names = {}
        for Loader in self.loaders:
            self.class_names.update(Loader.class_names)
        self.class_names = sorted(list(self.class_names.keys()))
        return self.class_names

    def make_data_list(self):
                
        self.train_data_list = []
        cnt=0
        for Loader in self.train_loaders:

            Loader.make_data_list()
            Loader.save()
            for i in Loader.data_list:
                i['sample_idx'] = cnt
                cnt += 1
                self.train_data_list.append(i)
                
        self.val_data_list = []
        cnt=0
        for Loader in self.val_loaders:

            Loader.make_data_list()
            Loader.save()
            for i in Loader.data_list:
                i['sample_idx'] = cnt
                cnt += 1
                self.val_data_list.append(i)

    def save(self):
        
        infos_train = {
            'data_list': self.train_data_list,
            'metainfo': METAINFO,
            }
        out_name = osp.join(self.out_dir, 'kitti360_infos_new_train.pkl')
        mmengine.dump(infos_train, out_name)
        print('save to',out_name)
        infos_val = {
            'data_list': self.val_data_list,
            'metainfo': METAINFO,
            }
        out_name = osp.join(self.out_dir, 'kitti360_infos_new_val.pkl')
        mmengine.dump(infos_val, out_name)
        print('save to',out_name)

class kitti360Dataloader_OneScene(object):
    def __init__(self, data_root, scene, out_dir = None, cam_ids = [0,1], gt_in_cam_only = True, calc_num_lidar_pts=True):# cam_ids must be ascending, must have '0'
        self.kitti360Path = data_root
        self.data_root = data_root
        self.gt_in_cam_only = gt_in_cam_only
        if out_dir is None:
            out_dir = data_root
        self.out_dir = data_root
        self.sequence = scene
        self.cam_ids = cam_ids # no fisheye now
        self.cameras = []
        self.class_names = {}
        self.calc_num_lidar_pts = calc_num_lidar_pts
        for cam_id in cam_ids:
            self.cameras.append(CustomCameraPerspective(self.kitti360Path, 
                                                          self.sequence, cam_id))
        self.object_per_frame = { i:[] for i in self.cameras[0].frames }
        self.load_lidar()

    def loadTimestamps(self, ts_path):
        with open(os.path.join(ts_path, 'timestamps.txt')) as f:
            data=f.read().splitlines()
        ts = [l.replace(' ','_') for l in data]
        return ts

    def load_lidar(self):
        '''load velodyne as lidar'''
        # calibration/calib_cam_to_velo.txt
        # breakpoint()
        self.cam0_to_velo = loadCalibrationRigid(osp.join(self.kitti360Path, 
                                'calibration/calib_cam_to_velo.txt'))
        # self.cam0_to_velo[:,[1,2]] = self.cam0_to_velo[:,[2,1]]
        self.cam0_to_pose = self.cameras[0].camToPose
        # pt_cam0 = pose_to_cam0 @ cami_to_pose @ pt_cami
        self.cam_to_cam0 = [np.linalg.inv(self.cam0_to_pose) @ i.camToPose
                                for i in self.cameras]
        # cam_to_velo = cam0_to_velo @cam_to_cam0
        self.cam2lidar = [
            self.cam0_to_velo @ cami_to_cam0
                for cami_to_cam0 in self.cam_to_cam0
        ]
        # velo2pose = cam0_to_pose @ velo_to_cam0
        self.lidar2ego =self.cam0_to_pose @ np.linalg.inv(self.cam0_to_velo)
        
        self.lidar_dir = osp.join(self.kitti360Path,'data_3d_raw',
                                  self.sequence,'velodyne_points')
        # May miss out some files, but no timestamps are missed
        # self.lidar_timestamps = self.loadTimestamps(self.lidar_dir)
        self.lidar_dir = osp.join(self.lidar_dir,'data')
        lidar_files = sorted(os.listdir(self.lidar_dir))
        self.lidar_files = {}
        for file in lidar_files:
            self.lidar_files[int(file.split('.')[0])] = file
        print('there are {} pointclouds'.format(len(self.lidar_files)))

    

    def make_data_list(self):
        print('\nmaking MMDet3D data_list of scene', self.sequence)
        self.data_list = []
        prog_bar = mmengine.ProgressBar(len(self.lidar_files))
        i=0
        for frameID in self.lidar_files:
            ID = int(frameID)

            sample_idx = i
            i+=1
            log_id = self.sequence
            timestamp = frameID

            images = {}
        
            lidar_points = {}
            lidar_points['num_pts_feats'] = 4

            lidar_path = osp.join(self.lidar_dir, self.lidar_files[ID])
            lidar_points['lidar_path'] = osp.relpath(lidar_path, self.data_root)
            lidar_points['lidar2ego'] = self.lidar2ego
            
            info = {
                'sample_idx': sample_idx,
                'log_id': log_id,
                'lidar_points': lidar_points,
            }
            self.data_list.append(info)
            prog_bar.update()

    def save(self):
        infos_all = {
            'data_list': self.data_list,
            'metainfo': METAINFO,
            }
        out_name = osp.join(self.out_dir, 'kitti360_infos_test_{}.pkl'.format(self.sequence))
        mmengine.dump(infos_all, out_name)
        print('saved to', out_name)

if __name__ == '__main__':
    loader = kitti360Dataloader('datasets/kitti-360')
    loader.make_data_list()
    loader.save()
    exit(0)