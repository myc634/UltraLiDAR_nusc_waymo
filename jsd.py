import torch
import numpy as np

import open3d as o3d
import matplotlib.pyplot as plt
import glob
import random
import os
from plugin.models.necks.voxelizer import Voxelizer


def jsd_2d(p, q):
    p = p / p.sum()
    q = q / q.sum()
    from scipy.spatial.distance import jensenshannon

    return jensenshannon(p.flatten(), q.flatten())


voxelizer = Voxelizer(-50, 50, -50, 50, 0.15625, -3.73, 2.27, 0.15)


def load_kitti(count, seed):
    full_list = glob.glob(
        os.environ.get("KITTI360_DATASET") + "/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/*"
    )
    full_list.extend(
        glob.glob(os.environ.get("KITTI360_DATASET") + "/data_3d_raw/2013_05_28_drive_0002_sync/velodyne_points/data/*")
    )
    random.Random(seed).shuffle(full_list)
    full_list = full_list[0:count]

    all_arrays = []
    for file in full_list:
        if file.endswith(".bin"):
            pcd = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:, :3]
            # pcd[:, 2] += 1.73
            xyz = pcd
            all_arrays.append(xyz)

    return all_arrays


def load_range_images(files, suffix, is_range=True):
    all_arrays = []
    counter = 0
    for idx, file in enumerate(files):
        print(idx, len(files))
        if suffix in file:
            if is_range:
                sample = torch.load(file).numpy()
                all_arrays.append(sample[0])
            else:
                pcd = np.array(o3d.io.read_point_cloud(file).points)
                # pcd[:, 1] = -pcd[:, 1]
                xyz = torch.from_numpy(pcd)
                bev = (voxelizer([[torch.cat([xyz, torch.zeros_like((xyz[:, [0]]))], dim=1)]]) != 0).float()
                if bev[:, :, 350:370, 310:330].sum() < 200:
                    counter += 1
                    xyz[:, 1] = -xyz[:, 1]

                all_arrays.append(xyz.numpy())
    print(counter/len(files))
    return all_arrays


def point_cloud_to_histogram(field_size, bins, point_cloud):
    point_cloud_flat = point_cloud[:, 0:2]  # .cpu().detach().numpy()

    square_size = field_size / bins

    halfway_offset = 0
    if bins % 2 == 0:
        halfway_offset = (bins / 2) * square_size
    else:
        print("ERROR")

    histogram = np.histogramdd(
        point_cloud_flat, bins=bins, range=([-halfway_offset, halfway_offset], [-halfway_offset, halfway_offset])
    )

    return histogram


def array_to_histograms(samples, src):
    hist = []
    for sample in samples:
        if src == "gt":
            voxels = voxelizer([[torch.from_numpy(sample)]])
            non_zero_indices = torch.nonzero(voxels)

            xy = (non_zero_indices[:, 2:] * voxelizer.step) + voxelizer.y_min
            z = (non_zero_indices[:, 1] * voxelizer.z_step) + voxelizer.z_min
            point_cloud = torch.cat([xy, z.unsqueeze(1)], dim=1).detach().cpu().numpy()
        else:
            point_cloud = sample

        histogram = point_cloud_to_histogram(160, 100, point_cloud)[0]
        hist.append(histogram)
    return hist


def calculate_jsd(sample_folder):
    kitti_samples = load_kitti(2000, 0)

    kitti_histograms = array_to_histograms(kitti_samples, src="gt")

    model_samples = load_range_images(sorted(glob.glob(f"{sample_folder}/*.ply"))[:2000], suffix=".ply", is_range=False)
    print(len(model_samples))

    model_histograms = array_to_histograms(model_samples, src="model")

    model_p = np.stack(model_histograms, axis=0)
    model_p = np.sum(model_p, axis=0)

    kitti_p = np.stack(kitti_histograms, axis=0)
    kitti_p = np.sum(kitti_p, axis=0)

    jsd_score = jsd_2d(kitti_p, model_p)

    return jsd_score


if __name__ == "__main__":
    print(calculate_jsd("ultralidar_samples"))