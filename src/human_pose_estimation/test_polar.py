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
import collections
from human_pose_estimation.losses import (
    compute_normal_losses,
    compute_shape_losses,
    compute_mpjpe,
    compute_pa_mpjpe,
    compute_pelvis_mpjpe,
    compute_pck,
)
from human_pose_estimation.models import Model
from human_pose_estimation.dataset_polar import PHSPDatasetPolar
from human_pose_estimation.dataset_color import PHSPDatasetColor
from human_pose_estimation.utils import load_trained_model, render_model, render_depth_v


def test_normal(args):
    assert args.task == "img2normal"
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    if args.normal_mode == "color":
        dataset_test = PHSPDatasetColor(
            data_dir1=args.data_dir1,
            data_dir2=args.data_dir2,
            dataset=args.test_dataset,
            mode="test",
            task=args.task,
            normal_mode=args.normal_mode,
            shape_mode=args.shape_mode,
            img_size=args.img_size,
        )
    else:
        dataset_test = PHSPDatasetPolar(
            data_dir1=args.data_dir1,
            data_dir2=args.data_dir2,
            dataset=args.test_dataset,
            mode="test",
            task=args.task,
            normal_mode=args.normal_mode,
            shape_mode=args.shape_mode,
            img_size=args.img_size,
        )
    test_generator = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=args.pin_memory,
    )
    total_iters = len(dataset_test) // args.batch_size + 1

    # set model
    print(
        "[task: %s] normal mode: %s, shape mode: %s"
        % (args.task, args.normal_mode, args.shape_mode)
    )
    model = Model(
        normal_mode=args.normal_mode,
        shape_mode=args.shape_mode,
        temperature=args.temperature,
        img_size=args.img_size,
        use_6drotation=args.use_6drotation,
        smpl_dir=args.smpl_dir,
        batch_size=args.batch_size,
        task=args.task,
        iter_num=args.iter_num,
    )
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # load trained model
    if args.model_dir is not None:
        print("[model dir] model loaded from %s" % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Cannot find the model... (%s)" % args.model_dir)

    # test
    print(
        "------------------------------------- Test ------------------------------------"
    )
    start_time = time.time()
    model.eval()  # dropout layers will not work in eval mode
    results = collections.defaultdict(list)
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for iter, data in enumerate(test_generator):
            # data: {img, ambiguity_normal, normal, mask, category, joints3d, joints2d, smpl_param, cam_intr}
            for k in data.keys():
                if k == "category":
                    data[k] = data[k].to(device=device, dtype=torch.long)
                else:
                    if k != "info":
                        data[k] = data[k].to(device=device, dtype=dtype)

            if args.provide_mask:
                out = model(
                    data["img"],
                    data["ambiguity_normal"],
                    mask=data["mask"],
                    cam_intr=None,
                )
            else:
                out = model(
                    data["img"], data["ambiguity_normal"], mask=None, cam_intr=None
                )

            loss_dict = compute_normal_losses(out, data, args, device)
            loss = torch.mean(
                args.normal1_loss * loss_dict["normal1"]
                + args.normal2_loss * loss_dict["normal2"]
                + args.category_loss * loss_dict["category"]
            )

            # collect results
            results["scalar/loss"].append(loss.detach())
            results["scalar/normal1_loss"].append(
                torch.mean(loss_dict["normal1"].detach())
            )
            results["scalar/normal2_loss"].append(
                torch.mean(loss_dict["normal2"].detach())
            )
            results["scalar/category_loss"].append(
                torch.mean(loss_dict["category"].detach())
            )
            results["scalar/mae1"].append(torch.mean(loss_dict["mae1"]).detach())
            results["scalar/mae2"].append(torch.mean(loss_dict["mae2"]).detach())

            # if iter > 10:
            #     break
            # if iter % 3 == 0:
            if iter % 500 == 0:
                print(iter, total_iters)
            #     results['image/pred_normal2'].append(out['normal_stage2'][0].detach())
            #     results['image/target_normal'].append(data['normal'][0])
            #     results['image/target_mask'].append(data['mask'][0])
            #     results['image/img'].append(data['img'][0])
            #     if out['normal_stage1'] is not None:
            #         results['image/pred_normal1'].append(out['normal_stage1'][0].detach())

        results["normal1_loss"] = torch.mean(
            torch.stack(results["scalar/normal1_loss"], dim=0)
        )
        results["normal2_loss"] = torch.mean(
            torch.stack(results["scalar/normal2_loss"], dim=0)
        )
        results["category_loss"] = torch.mean(
            torch.stack(results["scalar/category_loss"], dim=0)
        )
        results["loss"] = torch.mean(torch.stack(results["scalar/loss"], dim=0))
        results["mae1"] = torch.mean(torch.stack(results["scalar/mae1"], dim=0))
        results["mae2"] = torch.mean(torch.stack(results["scalar/mae2"], dim=0))

        end_time = time.time()
        print(
            ">>> Test loss: {:.4f}\n"
            "         normal1 loss: {:.4f}\n"
            "         normal2 loss: {:.4f}\n"
            "         category loss: {:.4f}\n"
            "         mae1: {:.4f}\n"
            "         mae2: {:.4f}\n"
            "         time used: {:.2f} mins".format(
                results["loss"],
                results["normal1_loss"],
                results["normal2_loss"],
                results["category_loss"],
                results["mae1"],
                results["mae2"],
                (end_time - start_time) / 60.0,
            )
        )
        # break


def test_shape(args):
    assert args.task == "img2shape"
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set dataset
    if "color" in args.shape_mode:
        dataset_test = PHSPDatasetColor(
            data_dir1=args.data_dir1,
            data_dir2=args.data_dir2,
            dataset=args.test_dataset,
            mode="test",
            task=args.task,
            normal_mode=args.normal_mode,
            shape_mode=args.shape_mode,
            img_size=args.img_size,
        )
    else:
        dataset_test = PHSPDatasetPolar(
            data_dir1=args.data_dir1,
            data_dir2=args.data_dir2,
            dataset=args.test_dataset,
            mode="test",
            task=args.task,
            normal_mode=args.normal_mode,
            shape_mode=args.shape_mode,
            img_size=args.img_size,
        )
    test_generator = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=args.pin_memory,
    )
    total_iters = len(dataset_test) // args.batch_size + 1

    # set model
    print(
        "[task: %s] normal mode: %s, shape mode: %s"
        % (args.task, args.normal_mode, args.shape_mode)
    )
    model = Model(
        normal_mode=args.normal_mode,
        shape_mode=args.shape_mode,
        temperature=args.temperature,
        img_size=args.img_size,
        use_6drotation=args.use_6drotation,
        smpl_dir=args.smpl_dir,
        batch_size=args.train_batch_size,
        task=args.task,
    )
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    mse_func = torch.nn.MSELoss()
    smpl_model_faces = model.smpl.faces

    if args.model_dir is not None:
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[model] img2shape, model is loaded from %s" % args.model_dir)
    else:
        raise ValueError("model not found...")

    # test
    print(
        "------------------------------------- Test ------------------------------------"
    )
    start_time = time.time()
    model.eval()  # dropout layers will not work in eval mode
    results = collections.defaultdict(list)
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for iter, data in enumerate(test_generator):
            # data: {img, ambiguity_normal, normal, mask, category, joints3d, joints2d, smpl_param, cam_intr}
            for k in data.keys():
                if k != "info":
                    data[k] = data[k].to(device=device, dtype=dtype)

            if args.provide_mask:
                out = model(
                    data["img"],
                    data["ambiguity_normal"],
                    mask=data["mask"],
                    cam_intr=data["cam_intr"],
                )
            else:
                out = model(
                    data["img"],
                    data["ambiguity_normal"],
                    mask=None,
                    cam_intr=data["cam_intr"],
                )

            loss_dict = compute_shape_losses(out, data, args, device, mse_func)

            loss_dict["mpjpe"] = compute_mpjpe(
                out["joints3d"].detach(), data["joints3d"]
            )
            loss_dict["pa_mpjpe"], scale, R, t = compute_pa_mpjpe(
                out["joints3d"].detach(), data["joints3d"], return_transform=True
            )
            loss_dict["pel_mpjpe"] = compute_pelvis_mpjpe(
                out["joints3d"].detach(), data["joints3d"]
            )
            loss_dict["pck"] = compute_pck(out["joints3d"].detach(), data["joints3d"])

            # write data
            masks = data["mask"].permute(0, 2, 3, 1).cpu().numpy()
            normals = masks * out["normal_stage2"].permute(0, 2, 3, 1).cpu().numpy()
            verts = out["verts"].cpu().numpy()
            cam_intrs = data["cam_intr"].cpu().numpy()
            # imgs = data['img'].permute(0, 2, 3, 1).cpu().numpy()
            for i, info in enumerate(data["info"]):
                _dataset, action, frame_idx = info.split("-")

                # if not os.path.exists(
                #     "%s/%s/our_results/%s" % (args.save_dir, _dataset, action)
                # ):
                #     os.mkdir("%s/%s/our_results/%s" % (args.save_dir, _dataset, action))

                # mask = masks[i]
                # normal = normals[i]
                # norm = np.linalg.norm(normal, axis=2, keepdims=True)
                # normal = normal / (norm + (norm == 0))

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow((normal + 1) / 2)
                # plt.show()

                # # save disk space
                # img_angle = np.zeros([args.img_size, args.img_size, 2])  # theta, phi
                # img_angle[:, :, 0] = np.arccos(normal[:, :, 2]) * (mask[:, :, 0] > 0)
                # img_angle[:, :, 1] = np.arctan2(normal[:, :, 0], normal[:, :, 1])
                # img_angle = (img_angle * 1e4).astype(np.int16)

                # tmp = np.where(mask > 0)
                # u_min = max(np.min(tmp[1]), 0)
                # u_max = min(np.max(tmp[1]), args.img_size)
                # v_min = max(np.min(tmp[0]), 0)
                # v_max = min(np.max(tmp[0]), args.img_size)
                # img_crop = img_angle[v_min:v_max, u_min:u_max, :]

                # joblib.dump(
                #     [img_crop, v_min, v_max, u_min, u_max, cam_intrs[i]],
                #     "%s/%s/our_results/%s/pred_normal%s.pkl"
                #     % (args.save_dir, _dataset, action, frame_idx),
                #     compress=3,
                # )

                # render_img = render_model(verts[i], smpl_model_faces, args.img_size, args.img_size, cam_intrs[i],
                #                           np.zeros([3]), np.zeros([3]), img=imgs[i, :, :, 0:3])

                # depth_im = render_depth_v(
                #     verts[i],
                #     smpl_model_faces,
                #     t=np.zeros([3]),
                #     img_size=[args.img_size, args.img_size],
                #     f=cam_intrs[i, 0:2],
                #     c=cam_intrs[i, 2:4],
                # )
                # depth_im = copy.copy(depth_im)
                # depth_im[depth_im > 10] = 0.0
                # cv2.imwrite(
                #     "%s/%s/our_results/%s/raw_depth%s.png"
                #     % (args.save_dir, _dataset, action, frame_idx),
                #     (depth_im * 1000).astype(np.uint16),
                # )

                # plt.figure()
                # plt.imshow(depth_im, cmap='gray')
                # plt.show()

                # mesh_smpl = trimesh.Trimesh(vertices=verts[i], faces=smpl_model_faces)
                # mesh_smpl.export(
                #     "%s/%s/our_results/%s/raw_smpl%s.ply"
                #     % (args.save_dir, _dataset, action, frame_idx)
                # )

            # print(data['info'])
            # # normal
            # normal = out['normal_stage2']
            # print(normal.size())
            # verts = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(out['verts'].permute(0, 2, 1)) + t
            # verts = verts.permute(0, 2, 1)
            # print(verts.size())

            # collect results
            results["scalar/trans"].append(loss_dict["trans"].detach())
            results["scalar/beta"].append(loss_dict["beta"].detach())
            results["scalar/theta"].append(loss_dict["theta"].detach())
            results["scalar/joints3d"].append(loss_dict["joints3d"].detach())
            results["scalar/joints2d"].append(loss_dict["joints2d"].detach())
            results["scalar/mpjpe"].append(torch.mean(loss_dict["mpjpe"]))
            results["scalar/pa_mpjpe"].append(torch.mean(loss_dict["pa_mpjpe"]))
            results["scalar/pel_mpjpe"].append(torch.mean(loss_dict["pel_mpjpe"]))
            results["scalar/pck"].append(torch.mean(loss_dict["pck"]))

            # if iter > 2:
            #     break
            # if iter % 2 == 0:
            if (iter + 1) % 600 == 0:
                print(iter, total_iters)
                # results['image/verts'].append(out['verts'][0].detach())
                # results['image/cam_param'].append(out['cam_intr'][0])
                # results['image/img'].append(data['img'][0])
                # if args.shape_mode == 'normal_polar':
                #     results['image/mask'].append(data['mask'][0].detach())
                #     results['image/pred_normal'].append(out['normal_stage2'][0].detach())

        results["beta"] = torch.mean(torch.stack(results["scalar/beta"], dim=0))
        results["theta"] = torch.mean(torch.stack(results["scalar/theta"], dim=0))
        results["trans"] = torch.mean(torch.stack(results["scalar/trans"], dim=0))
        results["joints3d"] = torch.mean(torch.stack(results["scalar/joints3d"], dim=0))
        results["joints2d"] = torch.mean(torch.stack(results["scalar/joints2d"], dim=0))
        results["mpjpe"] = (
            torch.mean(torch.stack(results["scalar/mpjpe"], dim=0)) * 1000
        )
        results["pa_mpjpe"] = (
            torch.mean(torch.stack(results["scalar/pa_mpjpe"], dim=0)) * 1000
        )
        results["pel_mpjpe"] = (
            torch.mean(torch.stack(results["scalar/pel_mpjpe"], dim=0)) * 1000
        )
        results["pck"] = torch.mean(torch.stack(results["scalar/pck"], dim=0))

        end_time = time.time()
        print(
            ">>> Test mpjpe: {:.4f}\n"
            "         pa_mpjpe: {:.4f} mm\n"
            "         pel_mpjpe: {:.4f} mm\n"
            "         pck: {:.2f} \n"
            "         beta loss: {:.4f}\n"
            "         theta loss: {:.4f}\n"
            "         trans loss: {:.4f}\n"
            "         time used: {:.2f} min".format(
                results["mpjpe"],
                results["pa_mpjpe"],
                results["pel_mpjpe"],
                results["pck"],
                results["beta"],
                results["theta"],
                results["trans"],
                (end_time - start_time) / 60,
            )
        )


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
    parser.add_argument("--gpu_id", type=str, default="1")
    parser.add_argument("--data_dir1", type=str, default="/home/shihao/data_polar")
    parser.add_argument("--data_dir2", type=str, default="/home/shihao/data_polar")
    parser.add_argument("--test_dataset", type=str, default="all")
    parser.add_argument(
        "--save_dir", type=str, default="/home/shihao/data_polar/data2sen"
    )
    parser.add_argument(
        "--smpl_dir",
        type=str,
        default="../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl",
    )
    parser.add_argument("--log_dir", type=str, default="log")
    # parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/shihao/exp_polar/model_shape_no_mask/img2shape_2_stages_normal_polar.pkl",
    )
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=0)

    # ['img2normal', 'img2shape', 'all']
    parser.add_argument("--task", type=str, default="img2shape")
    # ['no_prior', 'physics', '2_stages', 'eccv2020']
    parser.add_argument("--normal_mode", type=str, default="2_stages")
    # ['normal_polar', 'polar', 'mask_polar']
    parser.add_argument("--shape_mode", type=str, default="normal_polar")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--provide_mask", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=10)
    parser.add_argument("--use_6drotation", type=int, default=1)
    parser.add_argument("--iter_num", type=int, default=1)

    parser.add_argument("--norm_weight", type=float, default=0.2)
    parser.add_argument("--normal1_loss", type=float, default=1)
    parser.add_argument("--normal2_loss", type=float, default=1)
    parser.add_argument("--category_loss", type=float, default=1)
    parser.add_argument("--normal_huber_weight", type=float, default=0.5)

    parser.add_argument("--trans_loss", type=float, default=0.1)
    parser.add_argument("--theta_loss", type=float, default=1)
    parser.add_argument("--beta_loss", type=float, default=0.1)
    parser.add_argument("--joints3d_loss", type=float, default=1)
    parser.add_argument("--joints2d_loss", type=float, default=1)
    parser.add_argument("--use_geodesic_loss", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    args = parser.parse_args()
    print_args(args)
    return args


def main():
    args = get_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if args.task == "img2normal":
        test_normal(args)
    elif args.task == "img2shape":
        test_shape(args)
    else:
        pass


if __name__ == "__main__":
    main()
