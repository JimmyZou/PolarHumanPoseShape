import torch
import torch.nn.functional as F
from human_pose_estimation.SMPL import batch_rodrigues
from human_pose_estimation.utils import batch_compute_similarity_transform_torch


def compute_normal_losses(out, target, args, device):
    losses = {}
    eps = 1e-8

    # normal 1 loss
    if out["normal_stage1"] is None:
        losses["normal1"] = torch.tensor([0], device=device).float()
        losses["mae1"] = torch.tensor([0], device=device).float()
    else:
        norm = torch.norm(out["normal_stage1"], p=2, dim=1, keepdim=False)  # [N, H, W]
        cosin_sim = F.cosine_similarity(
            out["normal_stage1"], target["normal"], dim=1, eps=1e-8
        )
        tmp = huber_loss_func(
            cosin_sim, 1, args.normal_huber_weight
        ) + args.norm_weight * (1 - norm).pow(2)
        #
        # cosin_sim = F.cosine_similarity(out['normal_stage1'], target['normal'], dim=1, eps=1e-8)
        # tmp = huber_loss_func(cosin_sim, 1, args.normal_huber_weight)

        losses["normal1"] = torch.sum(
            target["mask4loss"][:, 0, :, :] * tmp, dim=(1, 2)
        ) / (torch.sum(target["mask4loss"], dim=(1, 2, 3)) + eps)
        losses["mae1"] = compute_angle_error(
            out["normal_stage1"], target["normal"], target["mask4loss"]
        )

    # normal 2 loss
    normal2_losses = []
    for i in range(len(out["normal_stage2"])):
        norm = torch.norm(
            out["normal_stage2"][i], p=2, dim=1, keepdim=False
        )  # [N, H, W]
        cosin_sim = F.cosine_similarity(
            out["normal_stage2"][i], target["normal"], dim=1, eps=1e-8
        )
        tmp = torch.abs(1 - cosin_sim) + args.norm_weight * (1 - norm).pow(2)
        #
        # cosin_sim = F.cosine_similarity(out['normal_stage2'][i], target['normal'], dim=1, eps=1e-8)
        # tmp = torch.abs(1 - cosin_sim)

        normal2_loss = torch.sum(target["mask4loss"][:, 0, :, :] * tmp, dim=(1, 2)) / (
            torch.sum(target["mask4loss"], dim=(1, 2, 3)) + eps
        )
        normal2_losses.append(normal2_loss)  # [B]
    losses["normal2"] = torch.mean(torch.stack(normal2_losses, dim=1), dim=1)

    # norm = torch.norm(out['normal_stage2'][-1], p=2, dim=1, keepdim=False)  # [N, H, W]
    # cosin_sim = F.cosine_similarity(out['normal_stage2'][-1], target['normal'], dim=1, eps=1e-8)
    # tmp = torch.abs(1 - cosin_sim) + args.norm_weight * (1 - norm).pow(2)
    # #
    # # cosin_sim = F.cosine_similarity(out['normal_stage2'][-1], target['normal'], dim=1, eps=1e-8)
    # # tmp = torch.abs(1 - cosin_sim)
    #
    # losses['normal2'] = torch.sum(target['mask4loss'][:, 0, :, :] * tmp, dim=(1, 2)) / \
    #                     (torch.sum(target['mask4loss'], dim=(1, 2, 3)) + eps)

    losses["mae2"] = compute_angle_error(
        out["normal_stage2"][-1], target["normal"], target["mask4loss"]
    )

    # category loss
    if out["category"] is None:
        losses["category"] = torch.tensor([0], device=device).float()
    else:
        # three categories: background + ambiguous normal 1 + ambiguous normal 2
        tmp3 = F.cross_entropy(
            out["category"], target["category"][:, 0, :, :], reduction="none"
        )
        # losses['category'] = torch.sum(target['mask4loss'][:, 0, :, :] * tmp3, (1, 2)) / \
        #                      (torch.sum(target['mask4loss'], (1, 2, 3)) + eps)
        losses["category"] = torch.mean(tmp3, (1, 2))

    return losses


def huber_loss_func(pred, target, weight):
    # t < weight, loss = (pred - target).pow(2)
    # t > weight, loss = weight * (t - 0.5 * weight)
    t = torch.abs(pred - target)
    huber_loss = torch.where(
        t < weight, 0.5 * (pred - target).pow(2), weight * (t - 0.5 * weight)
    )
    return huber_loss


def compute_angle_error(pred_normal, target_normal, gt_mask, eps=1e-8):
    cosin_sim = F.cosine_similarity(pred_normal, target_normal, dim=1, eps=1e-8)

    angular = gt_mask[:, 0, :, :] * torch.acos(
        torch.clamp(cosin_sim, -1, 1)
    )  # [N, H, W]
    mae = torch.sum(angular, dim=(1, 2)) / (
        torch.sum(gt_mask[:, 0, :, :], dim=(1, 2)) + eps
    )
    mae = 180 * mae / 3.1415  # radius to degree
    return mae


def compute_shape_losses(out, target, args, device, mse_func):
    losses = {}

    if args.trans_loss > 0:
        tran_loss = mse_func(out["trans"], target["trans"])
        losses["trans"] = args.trans_loss * tran_loss
    else:
        losses["trans"] = torch.tensor([0], device=device).float()

    if args.beta_loss > 0:
        beta_loss = mse_func(out["beta"], target["beta"])
        losses["beta"] = args.beta_loss * beta_loss
    else:
        losses["beta"] = torch.tensor([0], device=device).float()

    if args.theta_loss > 0:
        if args.use_geodesic_loss:
            eps = 1e-6
            target_rotmats = batch_rodrigues(target["theta"].view(-1, 3)).view(
                -1, 24, 3, 3
            )
            # square geodesic loss arccos[(Tr(R1R2^T) -1 )/2]
            trace_rrt = torch.sum(
                out["rotmats"] * target_rotmats, dim=(-2, -1)
            )  # [B, 24]
            degree_dif = torch.abs(
                torch.acos(torch.clamp(0.5 * (trace_rrt - 1), -1 + eps, 1 - eps))
            )
            theta_loss = torch.mean(degree_dif)
            # _pred = out['pred_rotmats'].view(-1, 3, 3)
            # _target = target_rotmats.view(-1, 3, 3)
            # m = torch.bmm(_pred, _target.transpose(1, 2))  # batch*3*3
            # cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
            # cos = torch.min(cos, torch.ones([_target.size(0)], requires_grad=True, device=_pred.device))
            # cos = torch.max(cos, - torch.ones([_target.size(0)], requires_grad=True, device=_pred.device))
            # theta = torch.acos(cos)
            # theta_loss = torch.mean(theta)
        else:
            theta_loss = mse_func(out["theta"], target["theta"])
        losses["theta"] = args.theta_loss * theta_loss
    else:
        losses["theta"] = torch.tensor([0], device=device).float()

    if args.joints3d_loss > 0:
        joints3d_loss = mse_func(out["joints3d"], target["joints3d"])
        losses["joints3d"] = args.joints3d_loss * joints3d_loss
    else:
        losses["joints3d"] = torch.tensor([0], device=device).float()

    if args.joints2d_loss > 0 and out["joints2d"] is not None:
        # print(torch.max(out['joints2d'].detach()), torch.max(target['joints2d'].detach()))
        pred_joints2d = torch.clamp(out["joints2d"], 0, 1)
        joints2d_loss = mse_func(pred_joints2d, target["joints2d"])
        losses["joints2d"] = args.joints2d_loss * joints2d_loss
    else:
        losses["joints2d"] = torch.tensor([0], device=device).float()

    return losses


def compute_mpjpe(pred, target):
    # [B, 24, 3]
    mpjpe = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    return mpjpe


def compute_pa_mpjpe(pred, target, return_transform=False):
    # pred, target: [B, 24, 3]
    pred_hat, scale, R, t = batch_compute_similarity_transform_torch(pred, target, True)
    pa_mpjpe = torch.sqrt(torch.sum((pred_hat - target) ** 2, dim=-1))
    if return_transform:
        return pa_mpjpe, scale, R, t
    return pa_mpjpe


def compute_pelvis_mpjpe(pred, target):
    # [B, 24, 3]
    left_heap_idx = 1
    right_heap_idx = 2
    pred_pel = (
        pred[:, left_heap_idx : left_heap_idx + 1, :]
        + pred[:, right_heap_idx : right_heap_idx + 1, :]
    ) / 2
    pred = pred - pred_pel
    target_pel = (
        target[:, left_heap_idx : left_heap_idx + 1, :]
        + target[:, right_heap_idx : right_heap_idx + 1, :]
    ) / 2
    target = target - target_pel
    pel_mpjpe = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    return pel_mpjpe


def compute_pck(pred, target):
    pel_mpjpe = compute_pelvis_mpjpe(pred, target)
    pck = (pel_mpjpe < 0.1).float()
    return pck
