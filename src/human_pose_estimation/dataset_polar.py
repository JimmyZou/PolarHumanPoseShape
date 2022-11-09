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


class PHSPDatasetPolar(Dataset):
    def __init__(
        self,
        data_dir1="/data_shihao",
        data_dir2="/home/datassd/",
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
        assert normal_mode in ["no_prior", "physics", "2_stages", "eccv2020"]
        self.normal_mode = normal_mode
        assert shape_mode in ["normal_polar", "polar", "mask_polar"]
        self.shape_mode = shape_mode
        self.test_action = test_action
        self.test_frame_idx = test_frame_idx

        if self.dataset == "dataset1" or self.dataset == "all":
            self.bbx1 = (50, 924, 200, 1074)

            # load intrisic/extrinsic params
            self.cam_params1 = []
            with open("%s/dataset1/CamParams0906.pkl" % self.data_dir1, "rb") as f:
                self.cam_params1.append(pickle.load(f)["param_p"][0:4])
            with open("%s/dataset1/CamParams0909.pkl" % self.data_dir1, "rb") as f:
                self.cam_params1.append(pickle.load(f)["param_p"][0:4])

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
                "%s/dataset1/polar_%s_files.pkl" % (self.data_dir1, self.mode)
            ):
                with open(
                    "%s/dataset1/polar_%s_files.pkl" % (self.data_dir1, self.mode), "rb"
                ) as f:
                    all_files1 = pickle.load(f)
            else:
                all_files1 = self.obtain_all_filenames(self.data_dir1, "dataset1")
            print("[%s: dataset1, %i examples]" % (self.mode, len(all_files1)))
        else:
            all_files1 = []

        if self.dataset == "dataset2" or self.dataset == "all":
            self.bbx2 = (100, 1000, 200, 1100)

            # load intrisic/extrinsic params
            self.cam_params2 = []
            with open("%s/dataset2/intrinsic1024.pkl" % self.data_dir2, "rb") as f:
                self.cam_params2.append(pickle.load(f)["polar"][0:4])
            with open("%s/dataset2/intrinsic1028.pkl" % self.data_dir2, "rb") as f:
                self.cam_params2.append(pickle.load(f)["polar"][0:4])
            with open("%s/dataset2/intrinsic1101.pkl" % self.data_dir2, "rb") as f:
                self.cam_params2.append(pickle.load(f)["polar"][0:4])

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
                "%s/dataset2/polar_%s_files.pkl" % (self.data_dir2, self.mode)
            ):
                with open(
                    "%s/dataset2/polar_%s_files.pkl" % (self.data_dir2, self.mode), "rb"
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
        # read polarization image
        fname = "%s/%s/polar_images/%s/polar0-45_%04i.jpg" % (
            data_dir,
            dataset,
            action,
            frame_idx,
        )
        tmp_img = cv2.imread(fname)
        img0 = (tmp_img[:, :, 0]).astype(np.float32) / 255
        img45 = (tmp_img[:, :, 1]).astype(np.float32) / 255
        tmp_img = cv2.imread(fname.replace("polar0-45", "polar90-135"))
        img90 = (tmp_img[:, :, 0]).astype(np.float32) / 255
        img135 = (tmp_img[:, :, 1]).astype(np.float32) / 255
        polar_img = np.stack([img0, img45, img90, img135], axis=2)

        if self.task == "img2normal" or self.task == "all":
            # read normal and mask
            fname = "%s/%s/polar_normal/%s/normal_%04i.pkl" % (
                data_dir,
                dataset,
                action,
                frame_idx,
            )
            normal = self.get_normal(fname, normal_img_size=512)
            fname = "%s/%s/polar_mask/%s/mask_%04i.png" % (
                data_dir,
                dataset,
                action,
                frame_idx,
            )
            mask = cv2.imread(fname, -1)  # [H, W]
            sample["mask4loss"] = np.expand_dims(mask, axis=0)  # size 512
            if self.img_size == 256:
                # resize polar img
                polar_img = cv2.resize(polar_img, (self.img_size, self.img_size))
                # resize mask
                mask = cv2.resize(
                    mask.astype(np.float32), (self.img_size, self.img_size)
                )
                mask = mask == 1

            # collect other data in different cases
            if self.normal_mode == "no_prior":
                # no classification and no ab normal
                sample["ambiguity_normal"] = np.array([0])
                sample["category"] = np.array([0])
            elif self.normal_mode == "physics":
                # no classification and with ab normal, physics
                ambiguity_normal = self.get_ambiguity_normal(polar_img)
                sample["ambiguity_normal"] = np.transpose(ambiguity_normal, [2, 0, 1])
                sample["category"] = np.array([0])
            else:
                # 2_stages, eccv2020
                ambiguity_normal = self.get_ambiguity_normal(polar_img)
                sample["ambiguity_normal"] = np.transpose(ambiguity_normal, [2, 0, 1])
                sample["category"] = self.get_normal_category(
                    sample["ambiguity_normal"], normal, mask
                )

            sample["img"] = np.transpose(polar_img, [2, 0, 1])  # size 256 or 512
            sample["normal"] = np.transpose(normal, [2, 0, 1])  # size 512
            sample["mask"] = np.expand_dims(mask, axis=0)  # size 256 or 512

        if self.task == "img2shape" or self.task == "all":
            if self.img_size == 256:
                # resize polar img
                polar_img = cv2.resize(polar_img, (self.img_size, self.img_size))
            sample["img"] = np.transpose(polar_img, [2, 0, 1])

            if self.shape_mode == "mask_polar" or self.shape_mode == "normal_polar":
                fname = "%s/%s/polar_mask/%s/mask_%04i.png" % (
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

                ambiguity_normal = self.get_ambiguity_normal(polar_img)
                sample["ambiguity_normal"] = np.transpose(ambiguity_normal, [2, 0, 1])
            else:
                sample["mask"] = np.array([0])
                sample["ambiguity_normal"] = np.array([0])
                # pass
            sample["category"] = np.array([0])

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

            fname = "%s/%s/polar_pose/%s/pose_%04i.pkl" % (
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

    def get_ambiguity_normal(self, img_polar):
        polar_img_0 = img_polar[:, :, 0]
        polar_img_45 = img_polar[:, :, 1]
        polar_img_90 = img_polar[:, :, 2]
        polar_img_135 = img_polar[:, :, 3]
        _phi1 = np.arctan2(polar_img_45 - polar_img_135, polar_img_0 - polar_img_90) / 2
        rho_den = (polar_img_0 + polar_img_90) * np.cos(2 * _phi1)
        rho = (polar_img_0 - polar_img_90) / (
            rho_den + (rho_den == 0).astype(np.float32)
        )
        _theta = self.equation(rho)
        _phi2 = _phi1 + np.pi  # pi radius ambiguity

        # from angle normal to xyz normal
        z = np.cos(_theta)
        x = np.sin(_theta) * np.sin(_phi1)
        y = np.sin(_theta) * np.cos(_phi1)
        xyz_ambiguity1 = np.stack([x, y, z], axis=2)
        z = np.cos(_theta)
        x = np.sin(_theta) * np.sin(_phi2)
        y = np.sin(_theta) * np.cos(_phi2)
        xyz_ambiguity2 = np.stack([x, y, z], axis=2)
        ambiguity_normal = np.concatenate([xyz_ambiguity1, xyz_ambiguity2], axis=2)
        return ambiguity_normal

    def get_normal_category(self, ambiguous_normal, normal, mask):
        if self.img_size == 256:
            # resize normal
            normal = cv2.resize(normal, (self.img_size, self.img_size))
            norm = np.linalg.norm(normal, axis=2, keepdims=True)
            normal = normal / (norm + (norm == 0))

        normal = np.transpose(normal, [2, 0, 1])
        mask = np.expand_dims(mask, axis=0)
        ambiguous_normal = torch.from_numpy(ambiguous_normal).float()  # [6, H, W]
        normal = torch.from_numpy(normal).float()  # [3, H, W]
        mask = torch.from_numpy(mask).float()  # [1, H, W]

        category = torch.zeros([mask.size(1), mask.size(2)]).long()
        cosin_sim_1 = F.cosine_similarity(
            ambiguous_normal[0:3, :, :], normal, dim=0, eps=1e-8
        )
        cosin_sim_2 = F.cosine_similarity(
            ambiguous_normal[3:6, :, :], normal, dim=0, eps=1e-8
        )
        category[cosin_sim_1 > cosin_sim_2] = 1
        category[cosin_sim_1 <= cosin_sim_2] = 2
        category = mask * category.unsqueeze(dim=0).float()
        return category.numpy()

    def obtain_all_filenames(self, data_dir, dataset):
        if self.mode == "train":
            name_list = self.train_name_list
        elif self.mode == "test":
            name_list = self.test_name_list
        else:
            raise ValueError("Unkonwn mode %s" % self.mode)

        print("[Obtain all filenames for %s]" % self.mode)
        all_files = []
        actions = sorted(os.listdir("%s/%s/polar_pose/" % (data_dir, dataset)))
        for idx, action in enumerate(actions):
            name = action.split("_")[0]
            if name in name_list:
                print("process %s" % action)
                filenames = sorted(
                    glob.glob("%s/%s/polar_pose/%s/*.pkl" % (data_dir, dataset, action))
                )
                for filename in filenames:
                    frame_idx = int(filename.split("/")[-1].split(".")[0].split("_")[1])
                    # check polar image exists
                    img_exist = os.path.exists(
                        "%s/%s/polar_images/%s/polar0-45_%04i.jpg"
                        % (data_dir, dataset, action, frame_idx)
                    )
                    # check normal exists
                    normal_exist = os.path.exists(
                        "%s/%s/polar_normal/%s/normal_%04i.pkl"
                        % (data_dir, dataset, action, frame_idx)
                    )
                    # check mask exists
                    mask_exist = os.path.exists(
                        "%s/%s/polar_mask/%s/mask_%04i.png"
                        % (data_dir, dataset, action, frame_idx)
                    )

                    if img_exist and normal_exist and mask_exist:
                        all_files.append((dataset, action, frame_idx))

        with open(
            "%s/%s/polar_%s_files.pkl" % (data_dir, dataset, self.mode), "wb"
        ) as f:
            pickle.dump(all_files, f)
        return all_files

    @staticmethod
    def equation(rho, n=1.5):
        alpha = 2 + 2 * n**2
        beta = -((n + 1 / n) ** 2)
        gamma = (n - 1 / n) ** 2

        a = (gamma - rho * (alpha + beta)) ** 2
        b = 16 * (1 - n**2) * rho**2 - 2 * rho * alpha * (
            gamma - rho * beta - rho * alpha
        )
        c = rho**2 * alpha**2 - 16 * rho**2 * n**2

        mask = a == 0  # avoid invalid value error
        delta = b**2 - 4 * a * c
        delta[delta < 0] = (
            b[delta < 0]
        ) ** 2  # invalid points due to noise, set its theta to be 0
        tan_theta = np.sqrt((1 - mask) * (np.sqrt(delta) - b) / (2 * a + mask))
        # # tan_theta = np.sqrt((1 - mask) * (- np.sqrt(delta) - b) / (2 * a + mask))
        theta = np.arctan(tan_theta)
        return theta

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
        polar_img = np.transpose(sample["img"].numpy(), [1, 2, 0])
        ambiguity_normal = np.transpose(sample["ambiguity_normal"].numpy(), [1, 2, 0])
        normal = np.transpose(sample["normal"].numpy(), [1, 2, 0])
        mask = np.transpose(sample["mask"].numpy(), [1, 2, 0])
        joints2d = sample["joints2d"].numpy() * self.img_size
        joints3d_proj = self.project(
            sample["joints3d"].numpy(), sample["cam_intr"].numpy()
        )

        category = sample["category"][0].numpy()
        fused_image = np.zeros([self.img_size, self.img_size, 3])
        fused_image[category == 1, :] = ambiguity_normal[category == 1][:, 0:3]
        fused_image[category == 2, :] = ambiguity_normal[category == 2][:, 3:6]
        fused_image = mask * fused_image

        plt.figure(figsize=(15, 15))
        plt.subplot(331)
        plt.imshow((normal + 1) / 2)
        # plt.axis('off')

        plt.subplot(332)
        plt.imshow((ambiguity_normal[:, :, 0:3] + 1) / 2)
        # plt.axis('off')

        plt.subplot(333)
        plt.imshow((ambiguity_normal[:, :, 3:6] + 1) / 2)
        # plt.axis('off')

        plt.subplot(334)
        plt.imshow((fused_image + 1) / 2)
        # plt.axis('off')

        plt.subplot(335)
        plt.imshow(polar_img[:, :, 0:3])
        # plt.axis('off')

        plt.subplot(336)
        plt.imshow(mask * polar_img[:, :, 0:3])
        # plt.axis('off')

        plt.subplot(337)
        plt.imshow(mask[:, :, 0].astype(np.float32), cmap="gray")
        # plt.axis('off')

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
        plt.subplot(338)
        plt.imshow(polar_img[:, :, 0:3])
        plt.scatter(joints2d[:, 0], joints2d[:, 1], color="r", marker="h", s=15)
        for idx1, idx2 in kinematic_tree:
            plt.plot(
                [joints2d[idx1, 0], joints2d[idx2, 0]],
                [joints2d[idx1, 1], joints2d[idx2, 1]],
                color="r",
                linewidth=1.5,
            )

        plt.subplot(339)
        plt.imshow(polar_img[:, :, 0:3])
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

    dataset_train = PHSPDatasetPolar(
        data_dir1="/home/data/shihao/data_polar",
        data_dir2="/home/data/shihao/data_polar",
        dataset="dataset1",
        mode="test",
        task="all",
        normal_mode="2_stages",
        shape_mode="normal_polar",
        img_size=512,
        test_action=None,
    )

    sample = dataset_train[1000]
    for k, v in sample.items():
        if k == "cam_intr":
            print(v)

    # dataset_train.visualize(1000)
