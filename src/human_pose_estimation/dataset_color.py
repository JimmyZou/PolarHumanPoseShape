import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import Dataset
import pickle
import glob
import joblib
import torch
import torch.nn.functional as F


class PHSPDatasetColor(Dataset):
    def __init__(
        self,
        data_dir1="/home/shihao/data_polar",
        data_dir2="/home/shihao/data_polar",
        dataset="dataset1",
        mode="train",
        task="img2normal",
        normal_mode="2_stages",
        shape_mode="normal_img",
        img_size=512,
        test_action=None,
        test_frame_idx=None,
    ):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        assert dataset in ["dataset1", "dataset2", "all"]
        self.dataset = dataset
        self.img_size = img_size
        self.mode = mode
        assert task in ["img2normal", "img2shape", "all"]
        self.task = task
        assert normal_mode in ["color"]
        self.normal_mode = normal_mode
        assert shape_mode in ["normal_color", "color", "mask_color"]
        self.shape_mode = shape_mode
        self.test_action = test_action
        self.test_frame_idx = test_frame_idx

        if self.dataset == "dataset1" or self.dataset == "all":
            self.bbx1 = (200, 1080, 400, 1280)

            # load intrisic/extrinsic params
            self.cam_params1 = []
            with open("%s/dataset1/CamParams0906.pkl" % self.data_dir1, "rb") as f:
                self.cam_params1.append(pickle.load(f)["param_c2"][0:4])
            with open("%s/dataset1/CamParams0909.pkl" % self.data_dir1, "rb") as f:
                self.cam_params1.append(pickle.load(f)["param_c2"][0:4])

            # corresponding cam params to each subject
            self.subject_cam_params1 = {}  # {"name": 0 or 1}
            for name in [
                "subject06",
                "subject09",
                "subject11",
                "subject05",
                "subject12",
                "subject04",
            ]:
                self.subject_cam_params1[name] = 0
            for name in [
                "subject03",
                "subject01",
                "subject02",
                "subject10",
                "subject07",
                "subject08",
            ]:
                self.subject_cam_params1[name] = 1

            # split train and test sets
            self.train_name_list = [
                "subject06",
                "subject09",
                "subject08",
                "subject05",
                "subject03",
                "subject01",
                "subject02",
                "subject10",
                "subject12",
            ]
            self.test_name_list = ["subject07", "subject04", "subject11"]

            if os.path.exists(
                "%s/dataset1/color_%s_files.pkl" % (self.data_dir1, self.mode)
            ):
                with open(
                    "%s/dataset1/color_%s_files.pkl" % (self.data_dir1, self.mode), "rb"
                ) as f:
                    all_files1 = pickle.load(f)
            else:
                all_files1 = self.obtain_all_filenames(self.data_dir1, "dataset1")
            print("[%s: dataset1, %i examples]" % (self.mode, len(all_files1)))
        else:
            all_files1 = []

        if self.dataset == "dataset2" or self.dataset == "all":
            self.bbx2 = (200, 1300, 500, 1600)

            # load intrisic/extrinsic params
            self.cam_params2 = []
            with open("%s/dataset2/intrinsic1024.pkl" % self.data_dir2, "rb") as f:
                self.cam_params2.append(pickle.load(f)["azure_kinect_0_color"][0:4])
            with open("%s/dataset2/intrinsic1028.pkl" % self.data_dir2, "rb") as f:
                self.cam_params2.append(pickle.load(f)["azure_kinect_0_color"][0:4])
            with open("%s/dataset2/intrinsic1101.pkl" % self.data_dir2, "rb") as f:
                self.cam_params2.append(pickle.load(f)["azure_kinect_0_color"][0:4])

            # corresponding cam params to each subject
            self.subject_cam_params2 = {}  # {"name": 0 or 1}
            for name in ["subject01", "subject02", "subject03"]:
                self.subject_cam_params2[name] = 0
            for name in [
                "subject04",
                "subject05",
                "subject06",
                "subject07",
                "subject08",
                "subject09",
                "subject10",
            ]:
                self.subject_cam_params2[name] = 1
            for name in [
                "subject11",
                "subject12",
                "subject13",
                "subject14",
                "subject15",
            ]:
                self.subject_cam_params2[name] = 2

            # split train and test sets
            self.train_name_list = [
                "subject03",
                "subject04",
                "subject05",
                "subject06",
                "subject08",
                "subject09",
                "subject10",
                "subject11",
                "subject12",
                "subject13",
                "subject14",
                "subject15",
            ]
            self.test_name_list = ["subject07", "subject01", "subject02"]

            if os.path.exists(
                "%s/dataset2/color_%s_files.pkl" % (self.data_dir2, self.mode)
            ):
                with open(
                    "%s/dataset2/color_%s_files.pkl" % (self.data_dir2, self.mode), "rb"
                ) as f:
                    all_files2 = pickle.load(f)
            else:
                all_files2 = self.obtain_all_filenames(self.data_dir2, "dataset2")
            print("[%s: dataset2, %i examples]" % (self.mode, len(all_files2)))
        else:
            all_files2 = []

        if self.test_action is not None:
            self.all_files = []
            all_files = all_files1 + all_files2
            for sample in all_files:
                if sample[1] == self.test_action:
                    if self.test_frame_idx is not None:
                        if sample[2] in self.test_frame_idx:
                            self.all_files.append(sample)
                    else:
                        self.all_files.append(sample)
        else:
            self.all_files = all_files1 + all_files2
        print("[%s: %i examples]" % (self.mode, len(self.all_files)))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        dataset, action, frame_idx = self.all_files[idx]
        # print(dataset, action, frame_idx)

        if dataset == "dataset1":
            data_dir = self.data_dir1
        elif dataset == "dataset2":
            data_dir = self.data_dir2
        else:
            raise ValueError("dataset [%s] errors..." % dataset)

        sample = {}
        sample["info"] = "%s-%s-%04i" % (dataset, action, frame_idx)
        sample["ambiguity_normal"] = np.array([0])
        sample["category"] = np.array([0])
        # read polarization image
        fname = "%s/%s/color_images/%s/color_%04i.jpg" % (
            data_dir,
            dataset,
            action,
            frame_idx,
        )
        color_img = (
            cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        )

        if self.task == "img2normal" or self.task == "all":
            # read normal and mask
            fname = "%s/%s/color_normal/%s/normal_%04i.pkl" % (
                data_dir,
                dataset,
                action,
                frame_idx,
            )
            normal = self.get_normal(fname, normal_img_size=512)
            fname = "%s/%s/color_mask/%s/mask_%04i.png" % (
                data_dir,
                dataset,
                action,
                frame_idx,
            )
            mask = cv2.imread(fname, -1)  # [H, W]
            sample["mask4loss"] = np.expand_dims(mask, axis=0)  # size 512
            if self.img_size == 256:
                # resize polar img
                color_img = cv2.resize(color_img, (self.img_size, self.img_size))
                # resize mask
                mask = cv2.resize(
                    mask.astype(np.float32), (self.img_size, self.img_size)
                )
                mask = mask == 1
                # # resize normal
                # normal = cv2.resize(normal, (self.img_size, self.img_size))
                # norm = np.linalg.norm(normal, axis=2, keepdims=True)
                # normal = normal / (norm + (norm == 0))
            sample["img"] = np.transpose(color_img, [2, 0, 1])
            sample["normal"] = np.transpose(normal, [2, 0, 1])
            sample["mask"] = np.expand_dims(mask, axis=0)

        if self.task == "img2shape" or self.task == "all":
            if self.img_size == 256:
                # resize polar img
                color_img = cv2.resize(color_img, (self.img_size, self.img_size))
            sample["img"] = np.transpose(color_img, [2, 0, 1])

            if self.shape_mode == "mask_color" or self.shape_mode == "normal_color":
                fname = "%s/%s/color_mask/%s/mask_%04i.png" % (
                    data_dir,
                    dataset,
                    action,
                    frame_idx,
                )
                mask = cv2.imread(fname, -1)  # [H, W]
                if self.img_size == 256:
                    # resize mask
                    mask = cv2.resize(
                        mask.astype(np.float32), (self.img_size, self.img_size)
                    )
                    mask = mask == 1
                sample["mask"] = np.expand_dims(mask, axis=0)
            else:
                sample["mask"] = np.array([0])
                # pass

            # get cam intr
            if dataset == "dataset1":
                cam_intr = self.cam_params1[
                    self.subject_cam_params1[action.split("_")[0]]
                ]
                fx, fy, cx, cy = cam_intr[0:4]
                ratio = (self.bbx1[1] - self.bbx1[0]) / self.img_size
                cx = (cx - self.bbx1[2]) / ratio
                fx = fx / ratio
                cy = (cy - self.bbx1[0]) / ratio
                fy = fy / ratio
                cam_intr = np.array([fx, fy, cx, cy])
            elif dataset == "dataset2":
                cam_intr = self.cam_params2[
                    self.subject_cam_params2[action.split("_")[0]]
                ]
                fx, fy, cx, cy = cam_intr[0:4]
                ratio = (self.bbx2[1] - self.bbx2[0]) / self.img_size
                cx = (cx - self.bbx2[2]) / ratio
                fx = fx / ratio
                cy = (cy - self.bbx2[0]) / ratio
                fy = fy / ratio
                cam_intr = np.array([fx, fy, cx, cy])
            else:
                raise ValueError("dataset [%s] errors..." % dataset)
            sample["cam_intr"] = cam_intr

            fname = "%s/%s/color_pose/%s/pose_%04i.pkl" % (
                data_dir,
                dataset,
                action,
                frame_idx,
            )
            beta, theta, trans, joints3d, joints2d = joblib.load(fname)
            # print(beta.shape, theta.shape, trans.shape, joints3d.shape, joints2d.shape)
            # smpl_param = np.concatenate([beta[0], theta[0], trans[0]], axis=0)
            sample["beta"] = beta[0]
            sample["theta"] = theta[0]
            sample["trans"] = trans[0]
            sample["joints3d"] = joints3d
            sample["joints2d"] = self.project(
                joints3d, cam_intr, self.img_size, self.img_size
            )  # normalize to 0-1

        for key, item in sample.items():
            if key != "info":
                sample[key] = torch.from_numpy(item).float()
        return sample

    def get_normal(self, fname, normal_img_size=512):
        img_crop, v_min, v_max, u_min, u_max = joblib.load(fname)
        img_angle = np.zeros([normal_img_size, normal_img_size, 2])
        img_angle[v_min:v_max, u_min:u_max, :] = img_crop
        mask = img_angle[:, :, 0] > 0

        # angles are stored as "int16" by multiplying 1e4 to save disk space
        img_angle = img_angle.astype(np.float32) / 1e4
        _theta, _phi = img_angle[:, :, 0], img_angle[:, :, 1]
        z = np.cos(_theta) * mask
        x = np.sin(_theta) * np.sin(_phi)
        y = np.sin(_theta) * np.cos(_phi)
        normal = np.stack([x, y, z], axis=2)
        return normal

    def obtain_all_filenames(self, data_dir, dataset):
        if self.mode == "train":
            name_list = self.train_name_list
        elif self.mode == "test":
            name_list = self.test_name_list
        else:
            raise ValueError("Unkonwn mode %s" % self.mode)

        print("[Obtain all filenames for %s]" % self.mode)
        all_files = []
        actions = sorted(os.listdir("%s/%s/color_pose/" % (data_dir, dataset)))
        for idx, action in enumerate(actions):
            name = action.split("_")[0]
            if name in name_list:
                print("process %s" % action)
                filenames = sorted(
                    glob.glob("%s/%s/color_pose/%s/*.pkl" % (data_dir, dataset, action))
                )
                for filename in filenames:
                    frame_idx = int(filename.split("/")[-1].split(".")[0].split("_")[1])
                    # check polar image exists
                    img_exist = os.path.exists(
                        "%s/%s/color_images/%s/color_%04i.jpg"
                        % (data_dir, dataset, action, frame_idx)
                    )
                    # check normal exists
                    normal_exist = os.path.exists(
                        "%s/%s/color_normal/%s/normal_%04i.pkl"
                        % (data_dir, dataset, action, frame_idx)
                    )
                    # check mask exists
                    mask_exist = os.path.exists(
                        "%s/%s/color_mask/%s/mask_%04i.png"
                        % (data_dir, dataset, action, frame_idx)
                    )

                    if img_exist and normal_exist and mask_exist:
                        all_files.append((dataset, action, frame_idx))

        with open(
            "%s/%s/color_%s_files.pkl" % (data_dir, dataset, self.mode), "wb"
        ) as f:
            pickle.dump(all_files, f)
        return all_files

    @staticmethod
    def project(points, params, H=1.0, W=1.0):
        # points: [N, 3]
        fx, fy, cx, cy = params
        U = (cx + fx * points[:, 0] / points[:, 2]) / W
        V = (cy + fy * points[:, 1] / points[:, 2]) / H
        return np.stack([U, V], axis=-1)  # N x 2

    def visualize(self, idx):
        sample = self.__getitem__(idx)
        for k, v in sample.items():
            print(k, v.size())

        # [H, W, C]
        color_img = np.transpose(sample["img"].numpy(), [1, 2, 0])
        normal = np.transpose(sample["normal"].numpy(), [1, 2, 0])
        mask = np.transpose(sample["mask"].numpy(), [1, 2, 0])
        joints2d = sample["joints2d"].numpy() * self.img_size
        joints3d_proj = self.project(
            sample["joints3d"].numpy(), sample["cam_intr"].numpy()
        )

        plt.figure(figsize=(15, 15))
        plt.subplot(331)
        plt.imshow((normal + 1) / 2)
        plt.axis("off")

        plt.subplot(332)
        plt.imshow(color_img)
        plt.axis("off")

        plt.subplot(333)
        plt.imshow(mask * color_img)
        plt.axis("off")

        plt.subplot(334)
        plt.imshow(mask[:, :, 0].astype(np.float32), cmap="gray")
        plt.axis("off")

        kinematic_tree = [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 7],
            [5, 8],
            [6, 9],
            [7, 10],
            [8, 11],
            [9, 12],
            [9, 14],
            [9, 13],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
        plt.subplot(335)
        plt.imshow(color_img)
        plt.scatter(joints2d[:, 0], joints2d[:, 1], color="r", marker="h", s=15)
        for idx1, idx2 in kinematic_tree:
            plt.plot(
                [joints2d[idx1, 0], joints2d[idx2, 0]],
                [joints2d[idx1, 1], joints2d[idx2, 1]],
                color="r",
                linewidth=1.5,
            )

        plt.subplot(336)
        plt.imshow(color_img)
        plt.scatter(
            joints3d_proj[:, 0], joints3d_proj[:, 1], color="r", marker="h", s=15
        )
        for idx1, idx2 in kinematic_tree:
            plt.plot(
                [joints3d_proj[idx1, 0], joints3d_proj[idx2, 0]],
                [joints3d_proj[idx1, 1], joints3d_proj[idx2, 1]],
                color="r",
                linewidth=1.5,
            )

        plt.show()


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    dataset_train = PHSPDatasetColor(
        data_dir1="/data_shihao",
        data_dir2="/home/datassd/",
        dataset="dataset2",
        mode="train",
        task="all",
        normal_mode="color",
        shape_mode="normal_color",
        img_size=512,
        test_action=None,
    )

    # sample = dataset_train[1000]
    # for k, v in sample.items():
    #     print(k, v.size())

    dataset_train.visualize(1000)
