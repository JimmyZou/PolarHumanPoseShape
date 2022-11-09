import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import copy
import trimesh
import joblib
import cv2
import sys

sys.path.append("../")
from human_pose_estimation.models import Model
from human_pose_estimation.utils import render_model


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


def get_ambiguity_normal(img_polar):
    polar_img_0 = img_polar[:, :, 0]
    polar_img_45 = img_polar[:, :, 1]
    polar_img_90 = img_polar[:, :, 2]
    polar_img_135 = img_polar[:, :, 3]
    _phi1 = np.arctan2(polar_img_45 - polar_img_135, polar_img_0 - polar_img_90) / 2
    rho_den = (polar_img_0 + polar_img_90) * np.cos(2 * _phi1)
    rho = (polar_img_0 - polar_img_90) / (rho_den + (rho_den == 0).astype(np.float32))
    _theta = equation(rho)
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


def load_inference_data(args):
    data_dir = args.data_dir

    samples = []

    for i in range(1):
        data = {}
        # read polarization image
        fname = "%s/polar0-45_%04d.jpg" % (data_dir, i + 1)
        print("load %s" % fname)
        tmp_img = cv2.imread(fname)
        img0 = (tmp_img[:, :, 0]).astype(np.float32) / 255
        img45 = (tmp_img[:, :, 1]).astype(np.float32) / 255
        tmp_img = cv2.imread(fname.replace("polar0-45", "polar90-135"))
        img90 = (tmp_img[:, :, 0]).astype(np.float32) / 255
        img135 = (tmp_img[:, :, 1]).astype(np.float32) / 255
        polar_img = np.stack([img0, img45, img90, img135], axis=2)

        ambiguity_normal = get_ambiguity_normal(polar_img)
        data["info"] = fname.split("/")[-1].replace("polar0-45", "")
        # print('here', data['info'])
        data["ambiguity_normal"] = np.transpose(ambiguity_normal, [2, 0, 1])
        data["img"] = np.transpose(polar_img, [2, 0, 1])  # size 256 or 512
        for key, item in data.items():
            if key != "info":
                data[key] = torch.from_numpy(item).float().unsqueeze(dim=0)
        samples.append(data)
    return samples


def inference(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set model
    model = Model(
        normal_mode=args.normal_mode,
        shape_mode=args.shape_mode,
        temperature=args.temperature,
        img_size=args.img_size,
        use_6drotation=True,
        smpl_dir=args.smpl_dir,
        batch_size=args.train_batch_size,
        task="img2shape",
        iter_num=args.iter_num,
    )
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    smpl_model_faces = model.smpl.faces

    # load trained model
    if args.model_dir is not None:
        print("[model dir] model loaded from %s" % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Cannot find the model... (%s)" % args.model_dir)

    model.eval()  # dropout layers will not work in eval mode
    samples = load_inference_data(args)
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for data in samples:
            img = data["img"].to(device=device, dtype=dtype)
            ab_normal = data["ambiguity_normal"].to(device=device, dtype=dtype)
            out = model(img, ab_normal, mask=None, cam_intr=None)
            # print(out.keys())

            normal = out["normal_stage2"]
            norm = torch.norm(normal, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
            normal = normal / norm

            normal = np.transpose(normal.detach().cpu().numpy()[0], [1, 2, 0])

            normal_rgb = ((normal + 1) / 2 * 255).astype(np.uint8)[:, :, ::-1]
            fname = data["info"]
            print("{}/normal{}".format(args.save_dir, fname))
            cv2.imwrite("{}/normal{}".format(args.save_dir, fname), normal_rgb)

            verts = out["verts"].cpu().numpy()[0]  # [6890, 3]
            img = data["img"].permute(0, 2, 3, 1).cpu().numpy()[0]  # [H, W, 4]
            # # if demo sample is from dataset2, using corresponding cam_intr
            # cam_intr = np.array([[1109.2906, 1108.8344,  260.3571,  274.5648]])
            # if demo sample is from dataset2, using corresponding cam_intr
            cam_intr = np.array([1062.1536, 1058.6014, 166.3433, 257.9308])

            render_img = render_model(
                verts,
                smpl_model_faces,
                args.img_size,
                args.img_size,
                cam_intr,
                np.zeros([3]),
                np.zeros([3]),
                img=img[:, :, 0:3],
            )
            render_img = (render_img[:, :, 0:3] * 255).astype(np.uint8)
            print("{}/shape{}".format(args.save_dir, fname))
            cv2.imwrite(
                "{}/shape{}".format(args.save_dir, fname), render_img[:, :, ::-1]
            )

            # mesh_smpl = trimesh.Trimesh(vertices=verts[i], faces=smpl_model_faces)
            # mesh_smpl.export(
            #     "%s/%s/our_results/%s/raw_smpl%s.ply"
            #     % (args.save_dir, _dataset, action, frame_idx)
            # )


def get_args():
    def print_args(args):
        """Prints the argparse argmuments applied
        Args:
          args = parser.parse_args()
        """
        _args = vars(args)
        max_length = max([len(k) for k, _ in _args.items()])
        for k, v in _args.items():
            print(" " * (max_length - len(k)) + k + ": " + str(v))

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="3")
    parser.add_argument("--data_dir", type=str, default="../../data/inference_demo")
    parser.add_argument("--save_dir", type=str, default="../../data/inference_demo")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../../data/model/img2shape_2_stages_normal_polar.pkl",
    )
    parser.add_argument(
        "--smpl_dir",
        type=str,
        default="../smpl_model/SMPL_MALE.pkl",
    )
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=0)

    # ['no_prior', 'physics', '2_stages', 'eccv2020']
    parser.add_argument("--normal_mode", type=str, default="2_stages")
    # ['normal_polar', 'polar', 'mask_polar']
    parser.add_argument("--shape_mode", type=str, default="normal_polar")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--provide_mask", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=10)
    parser.add_argument("--iter_num", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--train_batch_size", type=int, default=32
    )  # required when loading model
    args = parser.parse_args()
    print_args(args)
    return args


def main():
    args = get_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    inference(args)


if __name__ == "__main__":
    main()
