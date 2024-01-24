import torch
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import glob
import random
import os
from tqdm import tqdm
from plugin.models.necks.voxelizer import Voxelizer
import concurrent.futures
from functools import partial
from plyfile import PlyData, PlyElement
import open3d as o3d


def dump_ply(save_path, points):
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(save_path, point_cloud)

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
                bev = voxelizer([[torch.cat([xyz, torch.zeros_like((xyz[:, [0]]))], dim=1)]]).float()
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


def gaussian(x, y, sigma=0.5):
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(x)
    # plt.subplot(1, 2, 2)
    # plt.imshow(y)
    # plt.savefig("hist.png")

    # TODO: Calculate empirical sigma by fitting dist to gaussian
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=False, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0

    if not is_parallel:
        for s1 in tqdm(samples1):
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
            for dist in executor.map(
                kernel_parallel_worker, [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]
            ):
                d += dist

    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """MMD between two samples"""
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    # print('===============================')
    # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
    # print('--------------------------')
    cross = disc(samples1, samples2, kernel, *args, **kwargs)
    print("cross: ", cross)
    print("===============================")
    return (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * cross
    )


def calculate_mmd(sample_folder):
    kitti_samples = load_kitti(2000, 0)

    kitti_histograms = array_to_histograms(kitti_samples, src="gt")

    model_samples = load_range_images(sorted(glob.glob(f"{sample_folder}/*.ply"))[:2000], suffix=".ply", is_range=False)
    print(len(model_samples))

    model_histograms = array_to_histograms(model_samples, src="model")

    kitti_model_distance = compute_mmd(kitti_histograms, model_histograms, gaussian, is_hist=True)

    return kitti_model_distance


if __name__ == "__main__":
    print(calculate_mmd("ultralidar_samples"))