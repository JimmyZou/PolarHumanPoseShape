import cv2
import numpy as np
from src.PHSPDataset.utils import load_pose_and_shape, render_model, projection
from src.PHSPDataset.camera_parameters import CameraParams
from src.PHSPDataset.SMPL import SMPL
import torch
import pickle


def shape_render_demo(data_dir, model_dir, cam_name, subject_no, subfile_name, frame_idx, annotations):
    # camera parameter
    cam_param = CameraParams(data_dir)

    # SMPL model directions
    model_male_path = '%s/basicModel_m_lbs_10_207_0_v1.0.0.pkl' % model_dir
    model_female_path = '%s/basicModel_f_lbs_10_207_0_v1.0.0.pkl' % model_dir
    model_path = [model_male_path, model_female_path]  # 0 male, 1 female

    # render shape and pose
    if cam_name is 'p':
        # read polarization image
        img_file = '%s/polarization_images/%s_%s/polar0-45_%s.jpg' % (data_dir, subject_no, subfile_name, frame_idx)
        tmp_img = cv2.imread(img_file)
        img0 = (tmp_img[:, :, 0]).astype(np.float32) / 255
        img45 = (tmp_img[:, :, 1]).astype(np.float32) / 255
        tmp_img = cv2.imread(img_file.replace('polar0-45', 'polar90-135'))
        img90 = (tmp_img[:, :, 0]).astype(np.float32) / 255
        img = np.stack([img0, img45, img90], axis=2)  # only use three channels for visualization
        # get annotated pose and shape
        pose, shape = annotations['%s_%s-%s' % (subject_no, subfile_name, frame_idx)]
        # get cam intrinsic parameter
        intrinsic_param = np.asarray(cam_param.get_intrinsic(cam_name, subject_no))
        # get gender
        gender = cam_param.get_gender(subject_no)

        # load SMPL model and get shape
        device = torch.device('cpu')
        SMPL_model = SMPL(model_path[gender], 1, obj_saveable=True).to(device)
        _shape = torch.from_numpy(shape).float().to(device).unsqueeze(dim=0)
        verts, joint, _ = SMPL_model(_shape[:, 0:10], _shape[:, 13:85], get_skin=True)
        verts = verts[0].cpu().numpy() + np.expand_dims(shape[10:13], axis=0)

        # render
        h, w = 1024, 1224
        im = render_model(verts, SMPL_model.faces, w, h, intrinsic_param,
                          np.zeros([3]), np.zeros([3]), img=img, far=100)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(im)
        plt.show()

        # plot 2D pose
        kinematic_tree = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8],
                          [6, 9], [7, 10], [8, 11], [9, 12], [9, 14], [9, 13], [12, 15],
                          [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]
        uvd = projection(pose, intrinsic_param)
        plt.figure()
        plt.imshow(img)
        plt.scatter(uvd[:, 0], uvd[:, 1], color='r', marker='h', s=15)
        for idx1, idx2 in kinematic_tree:
            plt.plot([uvd[idx1, 0], uvd[idx2, 0]], [uvd[idx1, 1], uvd[idx2, 1]], color='b', linewidth=1.5)
        plt.show()

    else:
        color_filename = '%s/color/view%s/%s_%s/color_%s.jpg' % \
                         (data_dir, cam_name[1], subject_no, subfile_name, frame_idx)
        # flip the color image along the width
        img = (cv2.cvtColor(cv2.imread(color_filename), cv2.COLOR_BGR2RGB)[:, ::-1, :]).astype(np.uint8)

        # get cam intrinsic parameter
        intrinsic_param = np.asarray(cam_param.get_intrinsic(cam_name, subject_no))
        T_cp = cam_param.get_extrinsic(cam_name, subject_no)
        # get annotated pose and shape and transform to color camera coordinate
        pose, shape = annotations['%s_%s-%s' % (subject_no, subfile_name, frame_idx)]
        # get gender
        gender = cam_param.get_gender(subject_no)

        # load SMPL model and get shape
        device = torch.device('cpu')
        SMPL_model = SMPL(model_path[gender], 1, obj_saveable=True).to(device)
        _shape = torch.from_numpy(shape).float().to(device).unsqueeze(dim=0)
        verts, joint, _ = SMPL_model(_shape[:, 0:10], _shape[:, 13:85], get_skin=True)
        verts = verts[0].cpu().numpy() + np.expand_dims(shape[10:13], axis=0)
        verts = T_cp.transform(verts * 1000) / 1000  # transform shape

        # render
        h, w = 1080, 1920
        im = render_model(verts, SMPL_model.faces, w, h, intrinsic_param,
                          np.zeros([3]), np.zeros([3]), img=img, far=100)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(im)
        plt.show()

        # plot 2D pose
        kinematic_tree = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8],
                          [6, 9], [7, 10], [8, 11], [9, 12], [9, 14], [9, 13], [12, 15],
                          [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]
        pose = T_cp.transform(pose * 1000) / 1000  # transform pose
        uvd = projection(pose, intrinsic_param)
        plt.figure()
        plt.imshow(img)
        plt.scatter(uvd[:, 0], uvd[:, 1], color='r', marker='h', s=15)
        for idx1, idx2 in kinematic_tree:
            plt.plot([uvd[idx1, 0], uvd[idx2, 0]], [uvd[idx1, 1], uvd[idx2, 1]], color='b', linewidth=1.5)
        plt.show()


def mask_demo(data_dir, subject_no, cam_name, subfile_name, frame_idx):
    # polarization image
    if cam_name is 'p':
        # read polarization image
        img_file = '%s/polarization_images/%s_%s/polar0-45_%s.jpg' % \
                   (data_dir, subject_no, subfile_name, frame_idx)
        tmp_img = cv2.imread(img_file)
        img0 = (tmp_img[:, :, 0]).astype(np.float32) / 255
        img45 = (tmp_img[:, :, 1]).astype(np.float32) / 255
        tmp_img = cv2.imread(img_file.replace('polar0-45', 'polar90-135'))
        img90 = (tmp_img[:, :, 0]).astype(np.float32) / 255
        img = np.stack([img0, img45, img90], axis=2)  # only use three channels for visualization

        mask_file = '%s/mask/polar_mask/%s_%s/mask_%s.pkl' % (data_dir, subject_no, subfile_name, frame_idx)
        h, w = 1024, 1224
        with open(mask_file, 'rb') as f:
            # bounding box: _u_min, _u_max, _v_min, _v_max
            # mask_crop: only store the image patch of the bounding box to save disk space
            mask_crop, _u_min, _u_max, _v_min, _v_max = pickle.load(f)
            mask = np.zeros([h, w, 1], dtype=mask_crop.dtype)
            mask[_v_min:_v_max, _u_min:_u_max, 0] = mask_crop

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(img * mask)
        plt.show()

    else:
        color_filename = '%s/color/view%s/%s_%s/color_%s.jpg' % \
                         (data_dir, cam_name[1], subject_no, subfile_name, frame_idx)
        # flip the color image along the width
        img = (cv2.cvtColor(cv2.imread(color_filename), cv2.COLOR_BGR2RGB)[:, ::-1, :]).astype(np.uint8)

        mask_file = '%s/mask/color%s_mask/%s_%s/mask_%s.pkl' % (data_dir, cam_name[1], subject_no, subfile_name, frame_idx)
        h, w = 1080, 1920
        with open(mask_file, 'rb') as f:
            # bounding box: _u_min, _u_max, _v_min, _v_max
            # mask_crop: only store the image patch of the bounding box to save disk space
            mask_crop, _u_min, _u_max, _v_min, _v_max = pickle.load(f)
            mask = np.zeros([h, w, 1], dtype=mask_crop.dtype)
            mask[_v_min:_v_max, _u_min:_u_max, 0] = mask_crop

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(img * mask)
        plt.show()


if __name__ == '__main__':
    # revise the corresponding path
    model_dir = '/home/shihao/MultiCameraDataProcessing/fitting/models'  # model of SMPL data
    data_dir = '../../data/samples'  # root path of PHSPDataset
    subject_no = 'subject04'
    subfile_name = 'demo'
    frame_idx = 1000
    # load pose shape annotation
    annotations = load_pose_and_shape(data_dir, subject_no)
    if '%s_%s-%s' % (subject_no, subfile_name, frame_idx) not in annotations.keys():
        print('no annotation for this frame')
    else:
        # render multi-view shape and pose, plot the figure
        shape_render_demo(data_dir, model_dir, subject_no=subject_no, cam_name='p', subfile_name=subfile_name,
                          frame_idx=frame_idx, annotations=annotations)
        shape_render_demo(data_dir, model_dir, subject_no=subject_no, cam_name='c1', subfile_name=subfile_name,
                          frame_idx=frame_idx, annotations=annotations)
        shape_render_demo(data_dir, model_dir, subject_no=subject_no, cam_name='c2', subfile_name=subfile_name,
                          frame_idx=frame_idx, annotations=annotations)
        shape_render_demo(data_dir, model_dir, subject_no=subject_no, cam_name='c3', subfile_name=subfile_name,
                          frame_idx=frame_idx, annotations=annotations)

        # load mask and plot the figure
        mask_demo(data_dir, subject_no, cam_name='p', subfile_name=subfile_name, frame_idx=frame_idx)
        mask_demo(data_dir, subject_no, cam_name='c1', subfile_name=subfile_name, frame_idx=frame_idx)
        mask_demo(data_dir, subject_no, cam_name='c2', subfile_name=subfile_name, frame_idx=frame_idx)
