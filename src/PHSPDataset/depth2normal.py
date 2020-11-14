import sys
sys.path.append('../')
import cv2
import pickle
import os
from src.PHSPDataset.utils import convert_param2tranform, uvd2xyz, projection, depth2pts
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import multiprocessing


def connect_component(img, uvd):
    # stats [x0, y0, width, height, area] N*5
    _, _, stats, _ = cv2.connectedComponentsWithStats((img > 0).astype(np.uint8))
    stat = stats[np.argmax(stats[1:, 4]) + 1, :]
    u_min = stat[0]
    v_min = stat[1]
    u_max = stat[0] + stat[2]
    v_max = stat[1] + stat[3]

    idx = (u_min < uvd[:, 0]) & (uvd[:, 0] < u_max) & (v_min < uvd[:, 1]) & (uvd[:, 1] < v_max)
    uvd = uvd[idx, :]
    return uvd


def obtain_normal_single_processor(filenames, root_dir, params0906, params0909, cpu_i, visualize=False):
    H, W = 1024, 1224  # height and width of polarization images
    for idx, filename in enumerate(filenames):
        # load correpsonding camera extrinsic params
        # convert depth points to polarization camera coordinate
        name = filename.split('_')[0]
        if name in ['subject06', 'subject09', 'subject11', 'subject05', 'subject12', 'subject04']:
            param_p = params0906['param_p']
            param_d1 = params0906['param_d1']
            param_d2 = params0906['param_d2']
            param_d3 = params0906['param_d3']
            T_d1p = convert_param2tranform(params0906['d1p'])
            T_pd1 = T_d1p.inv()
            T_d2p = convert_param2tranform(params0906['d2p'])
            T_pd2 = T_d2p.inv()
            T_d3p = convert_param2tranform(params0906['d3p'])
            T_pd3 = T_d3p.inv()
        else:
            param_p = params0909['param_p']
            param_d1 = params0909['param_d1']
            param_d2 = params0909['param_d2']
            param_d3 = params0909['param_d3']
            T_d1p = convert_param2tranform(params0909['d1p'])
            T_pd1 = T_d1p.inv()
            T_d2p = convert_param2tranform(params0909['d2p'])
            T_pd2 = T_d2p.inv()
            T_d3p = convert_param2tranform(params0909['d3p'])
            T_pd3 = T_d3p.inv()

        # get filenames of three-view depth images
        N = len(glob.glob('%s/depth/view1/%s/depth_*.png' % (root_dir, filename)))
        print('working on %s (%i), %i examples' % (filename, idx, N))
        d1_files = ['%s/depth/view1/%s/depth_%i.png' % (root_dir, filename, i) for i in range(N)]
        d2_files = ['%s/depth/view2/%s/depth_%i.png' % (root_dir, filename, i) for i in range(N)]
        d3_files = ['%s/depth/view3/%s/depth_%i.png' % (root_dir, filename, i) for i in range(N)]

        # get filenames of three-view depth segmentation
        seg1_files = ['%s/depth_segmentation/view1/%s/seg_depth_%i.mat' %
                      (root_dir, filename, i) for i in range(N)]
        seg2_files = ['%s/depth_segmentation/view2/%s/seg_depth_%i.mat' %
                      (root_dir, filename, i) for i in range(N)]
        seg3_files = ['%s/depth_segmentation/view3/%s/seg_depth_%i.mat' %
                      (root_dir, filename, i) for i in range(N)]

        # check the folder to save normal result
        save_dir = '%s/polar_normal/%s' % (root_dir, filename)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # log used time
        start_time = time.time()
        time_idx = 0
        for i in range(N):
            time_idx += 1
            if i % 300 == 1:
                end_time = time.time()
                print('[cpu %2d, %i-th file] %s (%i / %i), %.2f s/frame'
                      % (cpu_i, idx, filename, i, N, (end_time - start_time) / time_idx))

            # if the normal exists, skip to next one
            if os.path.exists('%s/normal_%i.pkl' % (save_dir, i)):
                continue

            try:
                # get the point cloud from three-view depth images and convert to polarization camera coordinate
                img_d1 = cv2.imread(d1_files[i], -1)[:, ::-1]
                img_d2 = cv2.imread(d2_files[i], -1)[:, ::-1]
                img_d3 = cv2.imread(d3_files[i], -1)[:, ::-1]

                seg_uv1 = loadmat(seg1_files[i])['uv'] - 1
                seg_uv2 = loadmat(seg2_files[i])['uv'] - 1
                seg_uv3 = loadmat(seg3_files[i])['uv'] - 1

                uvd1 = np.stack([seg_uv1[:, 0], seg_uv1[:, 1], img_d1[seg_uv1[:, 1], seg_uv1[:, 0]]], axis=1)
                uvd2 = np.stack([seg_uv2[:, 0], seg_uv2[:, 1], img_d2[seg_uv2[:, 1], seg_uv2[:, 0]]], axis=1)
                uvd3 = np.stack([seg_uv3[:, 0], seg_uv3[:, 1], img_d3[seg_uv3[:, 1], seg_uv3[:, 0]]], axis=1)

                img1 = np.zeros_like(img_d1)
                img1[uvd1[:, 1], uvd1[:, 0]] = uvd1[:, 2]
                img2 = np.zeros_like(img_d2)
                img2[uvd2[:, 1], uvd2[:, 0]] = uvd2[:, 2]
                img3 = np.zeros_like(img_d3)
                img3[uvd3[:, 1], uvd3[:, 0]] = uvd3[:, 2]

                uvd1 = connect_component(img1, uvd1)
                uvd2 = connect_component(img2, uvd2)
                uvd3 = connect_component(img3, uvd3)

                xyz_p1 = T_pd1.transform(uvd2xyz(uvd1, param_d1))
                xyz_p2 = T_pd2.transform(uvd2xyz(uvd2, param_d2))
                xyz_p3 = T_pd3.transform(uvd2xyz(uvd3, param_d3))
            except:
                print('[warning] failed %s' % seg1_files[i])
                continue

            # (1) we project each view point cloud to the polarization image plane as one view polarization depth image
            # (2) for each view polarization depth image, we crop the valid the region (non-zero depth)
            # (3) calculate the normal for each view polarization depth image
            # (4) synthesize three-vew normal and smooth the normal map
            # (5) convert xyz normal to angle normal and save it
            results = []
            for point_cloud in [xyz_p1, xyz_p2, xyz_p3]:
                if point_cloud.shape[0] < 2000:
                    print('[warning] less than 2000 vertex, %s' % seg1_files[i])
                    continue

                # project each view point cloud to the polarization image plane as one view polarization depth image
                uv = projection(point_cloud, param_p).astype(np.int32)
                point_idx = (0 <= uv[:, 0]) & (uv[:, 0] < W) & (0 <= uv[:, 1]) & (uv[:, 1] < H)
                uv = uv[point_idx, :]
                d = point_cloud[point_idx, 2]

                # crop the valid the region (non-zero depth)
                img = np.zeros([H, W])
                img[uv[:, 1], uv[:, 0]] = d
                tmp = np.where(img > 0)
                u_min = max(np.min(tmp[1]) - 10, 0)
                u_max = min(np.max(tmp[1]) + 10, W)
                v_min = max(np.min(tmp[0]) - 10, 0)
                v_max = min(np.max(tmp[0]) + 10, H)
                img_crop = img[v_min: v_max, u_min: u_max]
                # print(u_max-u_min, v_max-v_min)

                # revise cam intrinsic parameter according to the bounding box
                fx, fy, cx, cy, k1, k2, k3 = param_p
                cx = cx - u_min
                cy = cy - v_min
                new_param_p = (fx, fy, cx, cy, k1, k2, k3)

                lut_xyz = depth2pts(img_crop, new_param_p)
                # pixel (v, u), get its nearby patch [v-4, v+4], [u-4, u+4]
                k = 4
                threshold = 30  # threshold to remove outlier depth point
                cart_normal = np.zeros([v_max - v_min, u_max - u_min, 3])
                for u in range(u_max - u_min):
                    for v in range(v_max - v_min):
                        u_idx_min = max(0, u - k)
                        u_idx_max = min(u + k + 1, W)
                        v_idx_min = max(0, v - k)
                        v_idx_max = min(v + k + 1, H)
                        points = np.reshape(lut_xyz[v_idx_min:v_idx_max, u_idx_min:u_idx_max], [-1, 3])
                        # if the num of non-zero depth points is larger than 5, we will calculate the normal
                        # otherwise, we think we will not get the accurate normal for pixel (v, u)
                        if np.sum(points[:, 2] > 0) > 5:
                            points = points[points[:, 2] > 0]  # get rid of zero-depth points
                            # calculate distances between the center and points
                            distances = np.sqrt(np.sum((points - np.mean(points, axis=0)) ** 2, axis=1))
                            # the distance should lie between 0 and 30mm
                            # the num of valid points should be larger than 5
                            if np.sum(np.logical_and(0 < distances, distances < threshold)) > 5:
                                pointsNN = points[np.logical_and(0 < distances, distances < threshold), :]
                                # calculate the normal
                                query_normal = np.sum(np.dot(np.linalg.pinv(np.dot(pointsNN.T, pointsNN)), pointsNN.T),
                                                      axis=1)
                                query_normal = query_normal / np.linalg.norm(query_normal)
                                if query_normal[2] < 0:
                                    query_normal = - query_normal
                                cart_normal[v, u, :] = np.clip(query_normal, -1, 1)

                img_result = np.zeros([H, W, 3])
                img_result[v_min: v_max, u_min: u_max] = cart_normal
                results.append(img_result)

                # tmp = (cart_normal + 1) / 2
                # plt.figure(figsize=(12, 8))
                # plt.subplot(121)
                # plt.imshow(img_crop, cmap='gray')
                # plt.axis('off')
                # plt.subplot(122)
                # plt.imshow(tmp)
                # plt.axis('off')
                # plt.show()

            # synthesize three-vew normal
            img_normal = None
            for result in results:
                if img_normal is None:
                    img_normal = result
                else:
                    img_normal = img_normal + result
            img_tmp = np.linalg.norm(img_normal, axis=2, keepdims=True)
            img_normal = img_normal / (img_tmp + (img_tmp == 0).astype(np.float32))
            tmp = np.where(img_tmp > 0)
            u_min = max(np.min(tmp[1]) - 10, 0)
            u_max = min(np.max(tmp[1]) + 10, W)
            v_min = max(np.min(tmp[0]) - 10, 0)
            v_max = min(np.max(tmp[0]) + 10, H)
            img_crop = img_normal[v_min: v_max, u_min: u_max, :]
            # print(u_max - u_min, v_max - v_min)

            # smooth the normal map
            k = 1
            img_smooth = img_crop.copy()
            for u in range(u_max - u_min):
                for v in range(v_max - v_min):
                    if img_crop[v, u, 0] == 0:
                        u_idx_min = max(0, u - k)
                        u_idx_max = min(u + k + 1, u_max - u_min)
                        v_idx_min = max(0, v - k)
                        v_idx_max = min(v + k + 1, v_max - v_min)
                        normal = np.sum(np.reshape(img_crop[v_idx_min:v_idx_max, u_idx_min:u_idx_max, :], [-1, 3]),
                                        axis=0)
                        if np.linalg.norm(normal) > 0:
                            img_smooth[v, u] = normal / np.linalg.norm(normal)

            # convert xyz normal to angle normal
            img_angle = np.zeros([v_max - v_min, u_max - u_min, 2])  # theta, phi
            img_angle[:, :, 0] = np.arccos(img_smooth[:, :, 2]) * (img_smooth[:, :, 2] != 0).astype(np.float32)
            img_angle[:, :, 1] = np.arctan2(img_smooth[:, :, 0], img_smooth[:, :, 1])
            img_angle = (img_angle * 1e4).astype(np.int16)  # minimize the requirement of disk space, float32 to int16

            # save files
            with open('%s/normal_%i.pkl' % (save_dir, i), 'wb') as f:
                pickle.dump([img_angle, v_min, v_max, u_min, u_max], f)

            # visualize
            if visualize:
                img_angle = img_angle.astype(np.float32) / 1e4
                img_recover = np.zeros([v_max - v_min, u_max - u_min, 3])
                img_recover[:, :, 2] = np.cos(img_angle[:, :, 0]) * (img_angle[:, :, 0] != 0).astype(np.float32)
                img_recover[:, :, 0] = np.sin(img_angle[:, :, 0]) * np.sin(img_angle[:, :, 1])
                img_recover[:, :, 1] = np.sin(img_angle[:, :, 0]) * np.cos(img_angle[:, :, 1])
                print(np.max(np.abs(img_smooth-img_recover)))

                plt.figure()
                plt.imshow((img_recover + 1) / 2)
                plt.axis('off')
                plt.show()


def obtain_normal_multi_processor(root_dir, num_cpus=18):
    # load extrinsic params
    with open('%s/CamParams0906.pkl' % root_dir, 'rb') as f:
        params0906 = pickle.load(f)
    with open('%s/CamParams0909.pkl' % root_dir, 'rb') as f:
        params0909 = pickle.load(f)

    # for the entire dataset, we have filenames such as ['subject01_group1_time1', 'subject01_group1_time2', ...]
    filenames = sorted(os.listdir('%s/depth/view1' % root_dir))
    N = len(filenames)
    n_files_cpu = N // num_cpus

    # # for debugging
    # obtain_normal_single_processor(filenames, root_dir, params0906, params0909)

    results = []
    pool = multiprocessing.Pool(num_cpus)
    for i in range(num_cpus):
        # split filenames to partitions and feed the partition to corresponding cpu to process
        idx1 = i * n_files_cpu
        idx2 = min((i + 1) * n_files_cpu, N)
        results.append(pool.apply_async(obtain_normal_single_processor,
                                        (filenames[idx1: idx2], root_dir, params0906, params0909, i)))
    pool.close()
    pool.join()
    pool.terminate()

    for result in results:
        tmp = result.get()
        if tmp is not None:
            print(tmp)
    print('Multi-cpu pre-processing ends.')


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    # obtain_normal_multi_processor(root_dir, num_cpus=6)

    root_dir = '../../data/samples'
    # for the entire dataset, we have filenames such as ['subject01_group1_time1', 'subject01_group1_time2', ...]
    filenames = ['subject04_demo']
    # load extrinsic params
    with open('%s/CamParams0906.pkl' % root_dir, 'rb') as f:
        params0906 = pickle.load(f)
    with open('%s/CamParams0909.pkl' % root_dir, 'rb') as f:
        params0909 = pickle.load(f)
    obtain_normal_single_processor(filenames, root_dir, params0906, params0909, cpu_i=0, visualize=False)


