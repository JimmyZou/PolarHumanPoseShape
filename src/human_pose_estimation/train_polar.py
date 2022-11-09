import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
from tensorboardX import SummaryWriter
import numpy as np
import sys

sys.path.append("../")
import collections
from human_pose_estimation.losses import (
    compute_normal_losses,
    compute_shape_losses,
    compute_pck,
    compute_mpjpe,
    compute_pelvis_mpjpe,
    compute_pa_mpjpe,
)
from human_pose_estimation.models import Model
from human_pose_estimation.dataset_polar import PHSPDatasetPolar
from human_pose_estimation.dataset_color import PHSPDatasetColor
from human_pose_estimation.utils import load_trained_model, render_model


def write_tensorboard(writer, results, epoch, args, mode, smpl_model_faces=None):
    task = args.task
    if task == "img2normal":
        target_mask = torch.stack(results["image/target_mask"], dim=0)  # [N, 1, H, W]
        polar_img = torch.stack(results["image/img"], dim=0)[
            :, 0:3, :, :
        ]  # [N, C, H, W]
        target_normal = target_mask * torch.stack(
            results["image/target_normal"], dim=0
        )  # [N, 3, H, W]

        if len(results["image/pred_normal1"]) > 0:
            pred_normal1 = torch.stack(
                results["image/pred_normal1"], dim=0
            )  # [N, 3, H, W]
            # norm1 = torch.norm(pred_normal1, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
            # pred_normal1 = target_mask * pred_normal1 / norm1
            pred_normal1 = target_mask * pred_normal1
            writer.add_images(
                "%s/%s/pred_normal1" % (task, mode), (pred_normal1 + 1) / 2, epoch
            )

        pred_normal2 = torch.stack(results["image/pred_normal2"], dim=0)  # [N, 3, H, W]
        norm2 = torch.norm(pred_normal2, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
        pred_normal2 = target_mask * pred_normal2 / norm2
        # pred_normal2 = target_mask * pred_normal2

        writer.add_images("%s/%s/img" % (task, mode), polar_img, epoch)
        writer.add_images(
            "%s/%s/target_normal" % (task, mode), (target_normal + 1) / 2, epoch
        )
        writer.add_images(
            "%s/%s/%s/pred_normal2" % (task, mode, args.normal_mode),
            (pred_normal2 + 1) / 2,
            epoch,
        )

    if task == "img2shape":
        if args.shape_mode == "normal_polar":
            mask = torch.stack(results["image/mask"], dim=0)  # [N, 1, H, W]
            pred_normal = torch.stack(
                results["image/pred_normal"], dim=0
            )  # [N, 3, H, W]
            norm = torch.norm(pred_normal, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
            pred_normal = mask * pred_normal / norm
            writer.add_images(
                "%s/%s/pred_normal" % (task, mode), (pred_normal + 1) / 2, epoch
            )

        verts = torch.stack(results["image/verts"], dim=0).cpu().numpy()
        cam_param = torch.stack(results["image/cam_param"], dim=0).cpu().numpy()
        img = (torch.stack(results["image/img"], dim=0)).cpu().numpy()
        img = np.transpose(img, [0, 2, 3, 1])[:, :, :, 0:3]
        h = img.shape[1]
        w = img.shape[2]

        render_imgs = []
        for i in range(img.shape[0]):
            render_img = render_model(
                verts[i],
                smpl_model_faces,
                w,
                h,
                cam_param[i],
                np.zeros([3]),
                np.zeros([3]),
                img=img[i],
            )
            render_imgs.append(render_img)
        render_imgs = np.transpose(np.stack(render_imgs, axis=0), [0, 3, 1, 2])
        writer.add_images(
            "%s/%s/%s/render_shape" % (task, mode, args.shape_mode), render_imgs, epoch
        )


def train_normal(args):
    assert args.task == "img2normal"
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set dataset
    if args.normal_mode == "color":
        dataset_train = PHSPDatasetColor(
            data_dir1=args.data_dir1,
            data_dir2=args.data_dir2,
            dataset=args.train_dataset,
            mode="train",
            task=args.task,
            normal_mode=args.normal_mode,
            shape_mode=args.shape_mode,
            img_size=args.img_size,
        )
    else:
        dataset_train = PHSPDatasetPolar(
            data_dir1=args.data_dir1,
            data_dir2=args.data_dir2,
            dataset=args.train_dataset,
            mode="train",
            task=args.task,
            normal_mode=args.normal_mode,
            shape_mode=args.shape_mode,
            img_size=args.img_size,
        )
    train_generator = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=args.pin_memory,
    )
    total_iters = len(dataset_train) // args.batch_size + 1

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

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_start)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.lr_decay_step], gamma=args.lr_decay_rate
    )
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # set tensorboard
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter("%s/%s/%s" % (args.result_dir, args.log_dir, start_time))
    print("[tensorboard] %s/%s/%s" % (args.result_dir, args.log_dir, start_time))

    # load trained model
    if args.model_dir is not None:
        print("[model dir] model loaded from %s" % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    save_dir = "%s/%s/%s" % (args.result_dir, args.log_dir, start_time)
    model_dir = "%s/best_model_%s_%s.pkl" % (save_dir, args.task, args.normal_mode)

    # set trainable parameters
    print("[status] gradients of only img2normal parameters are calculated...")
    for name, param in model.named_parameters():
        if "img2normal." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # set scaler
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # training
    best_loss = 1e5
    for epoch in range(args.epochs):
        print(
            "====================================== Epoch %i ========================================"
            % (epoch + 1)
        )
        # train
        print(
            "------------------------------------- Training ------------------------------------"
        )
        model.train()
        results = collections.defaultdict(list)
        start_time = time.time()
        for iter, data in enumerate(train_generator):
            # data: {img, ambiguity_normal, normal, mask, category, joints3d, joints2d, smpl_param, cam_intr}
            for k in data.keys():
                if k == "category":
                    data[k] = data[k].to(device=device, dtype=torch.long)
                else:
                    if k != "info":
                        data[k] = data[k].to(device=device, dtype=dtype)

            optimizer.zero_grad()
            if args.use_amp:
                # cast operations to mixed precision
                with torch.cuda.amp.autocast():
                    if args.provide_mask:
                        out = model(
                            data["img"],
                            data["ambiguity_normal"],
                            mask=data["mask"],
                            cam_intr=None,
                        )
                    else:
                        out = model(
                            data["img"],
                            data["ambiguity_normal"],
                            mask=None,
                            cam_intr=None,
                        )

                    loss_dict = compute_normal_losses(out, data, args, device)
                    loss = torch.mean(
                        args.normal1_loss * loss_dict["normal1"]
                        + args.normal2_loss * loss_dict["normal2"]
                        + args.category_loss * loss_dict["category"]
                    )

                # scale the loss and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                loss.backward()
                optimizer.step()

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

            display = 10
            # if iter > 10:
            #     break
            # if iter % 3 == 0:
            if iter % (total_iters // display) == 0:
                results["image/pred_normal2"].append(
                    out["normal_stage2"][-1][0].detach()
                )
                results["image/target_normal"].append(data["normal"][0])
                results["image/target_mask"].append(data["mask4loss"][0])
                results["image/img"].append(data["img"][0])
                if out["normal_stage1"] is not None:
                    results["image/pred_normal1"].append(
                        out["normal_stage1"][0].detach()
                    )

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

                progress = (100 // display) * iter // (total_iters // display) + 1
                end_time = time.time()
                time_used = (end_time - start_time) / 60.0
                print(
                    ">>> [epoch {:2d}/ iter {:6d}]\n"
                    "    loss: {:.4f}, normal1 loss: {:.4f}, normal2 loss: {:.4f}, category loss: {:.4f}\n"
                    "    mae1: {:.4f}, mae2: {:.4f}\n"
                    "    lr: {:.6f}, time used: {:.2f} mins, still need time for this epoch: {:.2f} mins.".format(
                        epoch,
                        iter + 1,
                        results["loss"],
                        results["normal1_loss"],
                        results["normal2_loss"],
                        results["category_loss"],
                        results["mae1"],
                        results["mae2"],
                        scheduler.get_last_lr()[0],
                        time_used,
                        (100 / progress - 1) * time_used,
                    )
                )
        write_tensorboard(writer, results, epoch, args, "train")

        # test
        print(
            "------------------------------------- Test ------------------------------------"
        )
        start_time = time.time()
        model.eval()  # dropout layers will not work in eval mode
        results = collections.defaultdict(list)
        with torch.set_grad_enabled(
            False
        ):  # deactivate autograd to reduce memory usage
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
                if (iter + 1) % 600 == 0:
                    results["image/pred_normal2"].append(
                        out["normal_stage2"][-1][0].detach()
                    )
                    results["image/target_normal"].append(data["normal"][0])
                    results["image/target_mask"].append(data["mask4loss"][0])
                    results["image/img"].append(data["img"][0])
                    if out["normal_stage1"] is not None:
                        results["image/pred_normal1"].append(
                            out["normal_stage1"][0].detach()
                        )

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
            write_tensorboard(writer, results, epoch, args, "val")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "%s/model_%s.pkl" % (save_dir, args.task),
            )

            if best_loss > results["mae2"]:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    model_dir,
                )
                # torch.save(model.state_dict(), model_dir)
                best_loss = results["mae2"]
                print(
                    ">>> Model saved as {}... best loss {:.4f}".format(
                        model_dir, best_loss
                    )
                )

            end_time = time.time()
            print(
                ">>> Test loss: {:.4f}\n"
                "         normal1 loss: {:.4f}\n"
                "         normal2 loss: {:.4f}\n"
                "         category loss: {:.4f}\n"
                "         mae1: {:.4f}\n"
                "         mae2: {:.4f} (best mae {:.4f})\n"
                "         time used: {:.2f} mins".format(
                    results["loss"],
                    results["normal1_loss"],
                    results["normal2_loss"],
                    results["category_loss"],
                    results["mae1"],
                    results["mae2"],
                    best_loss,
                    (end_time - start_time) / 60.0,
                )
            )
        # break
        scheduler.step(None)
    writer.close()


def train_shape(args):
    assert args.task == "img2shape"
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set dataset
    if "color" in args.shape_mode:
        dataset_train = PHSPDatasetColor(
            data_dir1=args.data_dir1,
            data_dir2=args.data_dir2,
            dataset=args.train_dataset,
            mode="train",
            task=args.task,
            normal_mode=args.normal_mode,
            shape_mode=args.shape_mode,
            img_size=args.img_size,
        )
    else:
        dataset_train = PHSPDatasetPolar(
            data_dir1=args.data_dir1,
            data_dir2=args.data_dir2,
            dataset=args.train_dataset,
            mode="train",
            task=args.task,
            normal_mode=args.normal_mode,
            shape_mode=args.shape_mode,
            img_size=args.img_size,
        )
    train_generator = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=args.pin_memory,
    )
    total_iters = len(dataset_train) // args.batch_size + 1

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
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=args.pin_memory,
    )

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
    )

    # set optimizer
    mse_func = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr_start)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.lr_decay_step], gamma=args.lr_decay_rate
    )
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # set tensorboard
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter("%s/%s/%s" % (args.result_dir, args.log_dir, start_time))
    print("[tensorboard] %s/%s/%s" % (args.result_dir, args.log_dir, start_time))

    # load parameters after training polar2normal
    # checkpoint = torch.load(args.model_dir, map_location=device)
    # model = model.load_state_dict(checkpoint['model_state_dict'])
    if args.model_dir is not None:
        model = load_trained_model(model, args.model_dir, device)
        print("[model] img2shape, model is loaded from %s" % args.model_dir)
    else:
        print("[model] img2shape, model is initilized.")
    # print(model)

    save_dir = "%s/%s/%s" % (args.result_dir, args.log_dir, start_time)
    model_dir = "%s/best_model_%s_%s.pkl" % (
        save_dir,
        args.normal_mode,
        args.shape_mode,
    )

    # set trainable parameters
    for name, param in model.named_parameters():
        if "img2shape." in name:
            param.requires_grad = True
        else:
            # img2normal
            param.requires_grad = False

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_loss = 1e5
    for epoch in range(args.epochs + 1):
        print(
            "====================================== Epoch %i ========================================"
            % (epoch + 1)
        )
        # '''
        # train
        print(
            "------------------------------------- Training ------------------------------------"
        )
        model.train()
        results = collections.defaultdict(list)
        start_time = time.time()
        for iter, data in enumerate(train_generator):
            # data: {img, ambiguity_normal, normal, mask, category, joints3d, joints2d, smpl_param, cam_intr}
            for k in data.keys():
                if k != "info":
                    data[k] = data[k].to(device=device, dtype=dtype)

            optimizer.zero_grad()
            if args.use_amp:
                # cast operations to mixed precision
                with torch.cuda.amp.autocast():
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
                    loss = (
                        loss_dict["trans"]
                        + loss_dict["beta"]
                        + loss_dict["theta"]
                        + loss_dict["joints3d"]
                        + loss_dict["joints2d"]
                    )

                # scale the loss and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                loss = (
                    loss_dict["trans"]
                    + loss_dict["beta"]
                    + loss_dict["theta"]
                    + loss_dict["joints3d"]
                    + loss_dict["joints2d"]
                )

                loss.backward()
                optimizer.step()

            loss_dict["mpjpe"] = compute_mpjpe(
                out["joints3d"].detach(), data["joints3d"]
            )
            loss_dict["pa_mpjpe"] = compute_pa_mpjpe(
                out["joints3d"].detach(), data["joints3d"]
            )
            loss_dict["pel_mpjpe"] = compute_pelvis_mpjpe(
                out["joints3d"].detach(), data["joints3d"]
            )
            loss_dict["pck"] = compute_pck(out["joints3d"].detach(), data["joints3d"])

            # collect results
            results["scalar/trans"].append(loss_dict["trans"].detach())
            results["scalar/beta"].append(loss_dict["beta"].detach())
            results["scalar/theta"].append(loss_dict["theta"].detach())
            results["scalar/joints3d"].append(loss_dict["joints3d"].detach())
            results["scalar/joints2d"].append(loss_dict["joints2d"].detach())
            results["scalar/loss"].append(loss.detach())
            results["scalar/mpjpe"].append(torch.mean(loss_dict["mpjpe"]))
            results["scalar/pa_mpjpe"].append(torch.mean(loss_dict["pa_mpjpe"]))
            results["scalar/pel_mpjpe"].append(torch.mean(loss_dict["pel_mpjpe"]))
            results["scalar/pck"].append(torch.mean(loss_dict["pck"]))

            display = 10
            # if iter > 10:
            #     break
            # if iter % 2 == 0:
            if iter % (total_iters // display) == 0:
                results["image/verts"].append(out["verts"][0].detach())
                results["image/cam_param"].append(out["cam_intr"][0])
                results["image/img"].append(data["img"][0])
                if args.shape_mode == "normal_polar":
                    results["image/mask"].append(data["mask"][0].detach())
                    results["image/pred_normal"].append(
                        out["normal_stage2"][0].detach()
                    )

                results["beta"] = torch.mean(torch.stack(results["scalar/beta"], dim=0))
                results["theta"] = torch.mean(
                    torch.stack(results["scalar/theta"], dim=0)
                )
                results["trans"] = torch.mean(
                    torch.stack(results["scalar/trans"], dim=0)
                )
                results["joints3d"] = torch.mean(
                    torch.stack(results["scalar/joints3d"], dim=0)
                )
                results["joints2d"] = torch.mean(
                    torch.stack(results["scalar/joints2d"], dim=0)
                )
                results["loss"] = torch.mean(torch.stack(results["scalar/loss"], dim=0))
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
                progress = (100 // display) * iter // (total_iters // display) + 1

                end_time = time.time()
                time_used = (end_time - start_time) / 60.0
                print(
                    ">>> [epoch {:2d}/ iter {:6d}]\n"
                    "    loss: {:.4f}, beta loss: {:.4f}, theta loss: {:.4f}, trans loss: {:.4f}, "
                    "joint3d loss: {:.4f}, joint2d loss: {:.4f}\n"
                    "    mpjpe: {:.2f} mm, pa_mpjpe: {:.2f} mm, pel_mpjpe: {:.2f} mm, pck: {:.2f}\n"
                    "    lr: {:.6f}, time used: {:.2f} mins, still need time for this epoch: {:.2f} mins.".format(
                        epoch,
                        iter + 1,
                        results["loss"],
                        results["beta"],
                        results["theta"],
                        results["trans"],
                        results["joints3d"],
                        results["joints2d"],
                        results["mpjpe"],
                        results["pa_mpjpe"],
                        results["pel_mpjpe"],
                        results["pck"],
                        scheduler.get_last_lr()[0],
                        time_used,
                        (100 / progress - 1) * time_used,
                    )
                )

        write_tensorboard(writer, results, epoch, args, "train", model.smpl.faces)
        # '''

        # test
        print(
            "------------------------------------- Test ------------------------------------"
        )
        start_time = time.time()
        model.eval()  # dropout layers will not work in eval mode
        results = collections.defaultdict(list)
        with torch.set_grad_enabled(
            False
        ):  # deactivate autograd to reduce memory usage
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
                loss = (
                    loss_dict["trans"]
                    + loss_dict["beta"]
                    + loss_dict["theta"]
                    + loss_dict["joints3d"]
                    + loss_dict["joints2d"]
                )

                loss_dict["mpjpe"] = compute_mpjpe(
                    out["joints3d"].detach(), data["joints3d"]
                )
                loss_dict["pa_mpjpe"] = compute_pa_mpjpe(
                    out["joints3d"].detach(), data["joints3d"]
                )
                loss_dict["pel_mpjpe"] = compute_pelvis_mpjpe(
                    out["joints3d"].detach(), data["joints3d"]
                )
                loss_dict["pck"] = compute_pck(
                    out["joints3d"].detach(), data["joints3d"]
                )

                # collect results
                results["scalar/trans"].append(loss_dict["trans"].detach())
                results["scalar/beta"].append(loss_dict["beta"].detach())
                results["scalar/theta"].append(loss_dict["theta"].detach())
                results["scalar/joints3d"].append(loss_dict["joints3d"].detach())
                results["scalar/joints2d"].append(loss_dict["joints2d"].detach())
                results["scalar/loss"].append(loss.detach())
                results["scalar/mpjpe"].append(torch.mean(loss_dict["mpjpe"]))
                results["scalar/pa_mpjpe"].append(torch.mean(loss_dict["pa_mpjpe"]))
                results["scalar/pel_mpjpe"].append(torch.mean(loss_dict["pel_mpjpe"]))
                results["scalar/pck"].append(torch.mean(loss_dict["pck"]))

                # if iter > 10:
                #     break
                # if iter % 2 == 0:
                if (iter + 1) % 600 == 0:
                    results["image/verts"].append(out["verts"][0].detach())
                    results["image/cam_param"].append(out["cam_intr"][0])
                    results["image/img"].append(data["img"][0])
                    if args.shape_mode == "normal_polar":
                        results["image/mask"].append(data["mask"][0].detach())
                        results["image/pred_normal"].append(
                            out["normal_stage2"][0].detach()
                        )

            results["beta"] = torch.mean(torch.stack(results["scalar/beta"], dim=0))
            results["theta"] = torch.mean(torch.stack(results["scalar/theta"], dim=0))
            results["trans"] = torch.mean(torch.stack(results["scalar/trans"], dim=0))
            results["joints3d"] = torch.mean(
                torch.stack(results["scalar/joints3d"], dim=0)
            )
            results["joints2d"] = torch.mean(
                torch.stack(results["scalar/joints2d"], dim=0)
            )
            results["loss"] = torch.mean(torch.stack(results["scalar/loss"], dim=0))
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

            write_tensorboard(writer, results, epoch, args, "test", model.smpl.faces)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "%s/model_%s.pkl" % (save_dir, args.task),
            )

            if best_loss > results["mpjpe"]:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    model_dir,
                )
                # torch.save(model.state_dict(), model_dir)
                best_loss = results["mpjpe"]
                print(
                    ">>> Model saved as {}... best loss {:.4f}".format(
                        model_dir, best_loss
                    )
                )

            end_time = time.time()
            print(
                ">>> Test mpjpe: {:.4f} (best mpjpe {:.4f})\n"
                "         pa_mpjpe: {:.4f} mm\n"
                "         pel_mpjpe: {:.4f} mm\n"
                "         pck: {:.2f} \n"
                "         beta loss: {:.4f}\n"
                "         theta loss: {:.4f}\n"
                "         trans loss: {:.4f}\n"
                "         time used: {:.2f} min".format(
                    results["mpjpe"],
                    best_loss,
                    results["pa_mpjpe"],
                    results["pel_mpjpe"],
                    results["pck"],
                    results["beta"],
                    results["theta"],
                    results["trans"],
                    (end_time - start_time) / 60,
                )
            )
        # break
        scheduler.step(None)
    writer.close()


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
    parser.add_argument("--data_dir1", type=str, default="/home/datassd")
    parser.add_argument("--data_dir2", type=str, default="/home/datassd")
    parser.add_argument("--train_dataset", type=str, default="dataset1")
    parser.add_argument("--test_dataset", type=str, default="all")
    parser.add_argument("--result_dir", type=str, default="/home/shihao/exp_polar")
    parser.add_argument(
        "--smpl_dir",
        type=str,
        default="../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl",
    )
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--model_dir", type=str, default=None)
    # parser.add_argument('--model_dir', type=str,
    #                     default='/home/shihao/exp_polar/log/2021-04-13-18-58/model_img2normal.pkl')
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--use_amp", type=int, default=1)

    # ['img2normal', 'img2shape']
    parser.add_argument("--task", type=str, default="img2normal")
    # ['no_prior', 'physics', '2_stages', 'eccv2020']
    parser.add_argument("--normal_mode", type=str, default="2_stages")
    # ['normal_polar', 'polar', 'mask_polar']
    parser.add_argument("--shape_mode", type=str, default="normal_polar")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--provide_mask", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=10)
    parser.add_argument("--iter_num", type=int, default=1)

    parser.add_argument("--norm_weight", type=float, default=0.2)
    parser.add_argument("--normal1_loss", type=float, default=1)
    parser.add_argument("--normal2_loss", type=float, default=1)
    parser.add_argument("--category_loss", type=float, default=0.1)
    parser.add_argument("--normal_huber_weight", type=float, default=0.5)

    parser.add_argument("--trans_loss", type=float, default=0.1)
    parser.add_argument("--theta_loss", type=float, default=1)
    parser.add_argument("--beta_loss", type=float, default=0.1)
    parser.add_argument("--joints3d_loss", type=float, default=10)
    parser.add_argument("--joints2d_loss", type=float, default=1)
    parser.add_argument("--use_6drotation", type=int, default=1)
    parser.add_argument("--use_geodesic_loss", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr_start", type=float, default=0.001)
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--lr_decay_step", type=int, default=5)
    args = parser.parse_args()
    print_args(args)
    return args


def main():
    args = get_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if args.task == "img2normal":
        train_normal(args)
    elif args.task == "img2shape":
        train_shape(args)
    else:
        pass


if __name__ == "__main__":
    main()
