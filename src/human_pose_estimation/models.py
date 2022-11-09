import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from human_pose_estimation.SMPL import SMPL, batch_rodrigues
from human_pose_estimation.utils import rot6d_to_rotmat, projection_torch

"""
our encoder-decoder model
"""


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.conv_block = nn.Sequential(
            torch.nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=1, stride=2, bias=False
            ),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False
            ),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False
            ),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
        )

    def forward(self, _input):
        out = self.conv_block(_input)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_last_layer=False):
        super(DecoderBlock, self).__init__()
        if is_last_layer:
            self.conv_block = nn.Sequential(
                torch.nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    output_padding=1,
                    stride=2,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    out_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    out_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(out_channel),
            )
        else:
            self.conv_block = nn.Sequential(
                torch.nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    output_padding=1,
                    stride=2,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    out_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    out_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(out_channel),
                torch.nn.ReLU(),
            )

    def forward(self, _input):
        out = self.conv_block(_input)
        return out


class Polar2NormalNew(nn.Module):
    def __init__(self, mode, temperature, iter_num=1):
        super(Polar2NormalNew, self).__init__()
        assert mode in ["2_stages"]
        self.mode = mode
        self.temperature = temperature
        self.iter_num = iter_num

        self.input_channel1 = 4  # polarization image
        self.input_channel2 = 6  # ambiguous normal
        self.input_channel3 = 3 + 1 + 3  # fused normal + normal residual
        self.out_channel1 = 3  # 3 categories: mask, ab normal case 1, ab normal case 2
        self.out_channel2 = 3  # normal stage 1
        self.out_channel3 = 3  # normal stage 2

        self.encoder1_1 = EncoderBlock(self.input_channel1, 8)
        self.encoder1_2 = EncoderBlock(8, 16)
        self.encoder1_3 = EncoderBlock(16, 32)
        self.encoder1_4 = EncoderBlock(32, 64)
        self.encoder1_5 = EncoderBlock(64, 128)
        self.encoder1_6 = EncoderBlock(128, 256)

        self.encoder2_1 = EncoderBlock(self.input_channel2, 8)
        self.encoder2_2 = EncoderBlock(8, 16)
        self.encoder2_3 = EncoderBlock(16, 32)
        self.encoder2_4 = EncoderBlock(32, 64)
        self.encoder2_5 = EncoderBlock(64, 128)
        self.encoder2_6 = EncoderBlock(128, 256)

        self.encoder3_1 = EncoderBlock(self.input_channel3, 8)
        self.encoder3_2 = EncoderBlock(8, 16)
        self.encoder3_3 = EncoderBlock(16, 32)
        self.encoder3_4 = EncoderBlock(32, 64)
        self.encoder3_5 = EncoderBlock(64, 128)
        self.encoder3_6 = EncoderBlock(128, 256)

        self.decoder1_1 = DecoderBlock(256, 128)
        self.decoder1_2 = DecoderBlock(128, 64)
        self.decoder1_3 = DecoderBlock(64, 32)
        self.decoder1_4 = DecoderBlock(32, 16)
        self.decoder1_5 = DecoderBlock(16, 8)
        self.decoder1_6 = DecoderBlock(8, self.out_channel1, True)

        self.decoder2_1 = DecoderBlock(256, 128)
        self.decoder2_2 = DecoderBlock(128, 64)
        self.decoder2_3 = DecoderBlock(64, 32)
        self.decoder2_4 = DecoderBlock(32, 16)
        self.decoder2_5 = DecoderBlock(16, 8)
        self.decoder2_6 = DecoderBlock(8, self.out_channel2, True)

        self.decoder3_1 = DecoderBlock(256, 128)
        self.decoder3_2 = DecoderBlock(128, 64)
        self.decoder3_3 = DecoderBlock(64, 32)
        self.decoder3_4 = DecoderBlock(32, 16)
        self.decoder3_5 = DecoderBlock(16, 8)
        self.decoder3_6 = DecoderBlock(8, self.out_channel3, True)

    def forward(self, img, ab_normal, mask=None):
        # encoder1 polarization image
        encoder1_1 = self.encoder1_1(img)
        encoder1_2 = self.encoder1_2(encoder1_1)
        encoder1_3 = self.encoder1_3(encoder1_2)
        encoder1_4 = self.encoder1_4(encoder1_3)
        encoder1_5 = self.encoder1_5(encoder1_4)
        encoder1_out = self.encoder1_6(encoder1_5)

        # encoder2 ambiguous normal
        encoder2_1 = self.encoder2_1(ab_normal)
        encoder2_2 = self.encoder2_2(encoder2_1)
        encoder2_3 = self.encoder2_3(encoder2_2)
        encoder2_4 = self.encoder2_4(encoder2_3)
        encoder2_5 = self.encoder2_5(encoder2_4)
        encoder2_out = self.encoder2_6(encoder2_5)

        # decoder1 category
        input_decoder1 = encoder2_out
        decoder1_1 = self.decoder1_1(input_decoder1) + encoder2_5
        decoder1_2 = self.decoder1_2(decoder1_1) + encoder2_4
        decoder1_3 = self.decoder1_3(decoder1_2) + encoder2_3
        decoder1_4 = self.decoder1_4(decoder1_3) + encoder2_2
        decoder1_5 = self.decoder1_5(decoder1_4) + encoder2_1
        out_category = self.temperature * self.decoder1_6(
            decoder1_5
        )  # [N, category, H, W]

        # get fused normal
        category = F.softmax(out_category, dim=1)
        if mask is None:
            mask = 1 - category[:, 0:1, :, :]  # [N, 1, H, W] soft mask is used
            mask = mask.detach()

        normal1 = category[:, 1:2, :, :]  # [N, 1, H, W]
        normal2 = category[:, 2:3, :, :]  # [N, 1, H, W]
        fused_normal = (
            normal1 * ab_normal[:, 0:3, :, :] + normal2 * ab_normal[:, 3:6, :, :]
        )
        norm = torch.norm(fused_normal, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
        fused_normal = fused_normal / (norm + 1e-8)  # [N, 3, H, W]

        # decoder2 normal stage 1
        input_decoder2 = encoder1_out + encoder2_out
        decoder2_1 = self.decoder2_1(input_decoder2) + encoder1_5 + encoder2_5
        decoder2_2 = self.decoder2_2(decoder2_1) + encoder1_4 + encoder2_4
        decoder2_3 = self.decoder2_3(decoder2_2) + encoder1_3 + encoder2_3
        decoder2_4 = self.decoder2_4(decoder2_3) + encoder1_2 + encoder2_2
        decoder2_5 = self.decoder2_5(decoder2_4) + encoder1_1 + encoder2_1
        pred_normal1 = self.decoder2_6(decoder2_5)  # [N, 3, H, W]
        # normalize
        # norm = torch.norm(pred_normal1, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
        # pred_normal1 = pred_normal1 / (norm + 1e-8)  # [N, 3, H, W]

        # coarse_normal = torch.clone(pred_normal1.detach())
        coarse_normal = torch.clone(pred_normal1)
        pred_normal2, normal_residual = [], None
        for i in range(self.iter_num):
            # decode fused normal and residual normal
            normal_residual = (
                1 - F.cosine_similarity(coarse_normal, fused_normal, dim=1, eps=1e-8)
            ).unsqueeze(dim=1)
            _input3 = mask * torch.cat(
                [fused_normal, coarse_normal, normal_residual], dim=1
            )
            encoder3_1 = self.encoder3_1(_input3)
            encoder3_2 = self.encoder3_2(encoder3_1)
            encoder3_3 = self.encoder3_3(encoder3_2)
            encoder3_4 = self.encoder3_4(encoder3_3)
            encoder3_5 = self.encoder3_5(encoder3_4)
            encoder3_out = self.encoder3_6(encoder3_5)

            # decoder3 normal stage 2
            input_decoder3 = encoder1_out + encoder3_out
            decoder3_1 = self.decoder3_1(input_decoder3) + encoder1_5 + encoder3_5
            decoder3_2 = self.decoder3_2(decoder3_1) + encoder1_4 + encoder3_4
            decoder3_3 = self.decoder3_3(decoder3_2) + encoder1_3 + encoder3_3
            decoder3_4 = self.decoder3_4(decoder3_3) + encoder1_2 + encoder3_2
            decoder3_5 = self.decoder3_5(decoder3_4) + encoder1_1 + encoder3_1
            normal_iter = self.decoder3_6(decoder3_5)

            # normalize
            # norm = torch.norm(normal_iter, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
            # coarse_normal = normal_iter / (norm + 1e-8)  # [N, 3, H, W]
            coarse_normal = normal_iter
            pred_normal2.append(coarse_normal)

        return (
            out_category,
            pred_normal1,
            pred_normal2,
            mask,
            normal_residual,
            fused_normal,
        )


class Polar2NormalECCV(nn.Module):
    def __init__(self, mode, temperature):
        super(Polar2NormalECCV, self).__init__()
        assert mode in ["eccv2020"]
        self.mode = mode
        self.temperature = temperature
        # input_channel1, input_channel2, out_channel1, out_channel2
        self.input_channel1 = 4 + 6  # polar + ambiguous normal
        self.out_channel1 = 3  # category
        self.input_channel2 = 4  # polar
        self.input_channel3 = 6 + 3  # ambiguous normal + fused normal
        self.out_channel2 = 3  # normal

        self.encoder1_1 = EncoderBlock(self.input_channel1, 8)
        self.encoder1_2 = EncoderBlock(8, 16)
        self.encoder1_3 = EncoderBlock(16, 32)
        self.encoder1_4 = EncoderBlock(32, 64)
        self.encoder1_5 = EncoderBlock(64, 128)
        self.encoder1_6 = EncoderBlock(128, 256)

        self.encoder2_1 = EncoderBlock(self.input_channel2, 8)
        self.encoder2_2 = EncoderBlock(8, 16)
        self.encoder2_3 = EncoderBlock(16, 32)
        self.encoder2_4 = EncoderBlock(32, 64)
        self.encoder2_5 = EncoderBlock(64, 128)
        self.encoder2_6 = EncoderBlock(128, 256)

        self.encoder3_1 = EncoderBlock(self.input_channel3, 8)
        self.encoder3_2 = EncoderBlock(8, 16)
        self.encoder3_3 = EncoderBlock(16, 32)
        self.encoder3_4 = EncoderBlock(32, 64)
        self.encoder3_5 = EncoderBlock(64, 128)
        self.encoder3_6 = EncoderBlock(128, 256)

        self.decoder1_1 = DecoderBlock(256, 128)
        self.decoder1_2 = DecoderBlock(128, 64)
        self.decoder1_3 = DecoderBlock(64, 32)
        self.decoder1_4 = DecoderBlock(32, 16)
        self.decoder1_5 = DecoderBlock(16, 8)
        self.decoder1_6 = DecoderBlock(8, self.out_channel1, True)

        self.decoder2_1 = DecoderBlock(256, 128)
        self.decoder2_2 = DecoderBlock(128, 64)
        self.decoder2_3 = DecoderBlock(64, 32)
        self.decoder2_4 = DecoderBlock(32, 16)
        self.decoder2_5 = DecoderBlock(16, 8)
        self.decoder2_6 = DecoderBlock(8, self.out_channel2, True)

    def forward(self, img, ab_normal, mask=None):
        input1 = torch.cat([img, ab_normal], dim=1)
        encoder1_1 = self.encoder1_1(input1)
        encoder1_2 = self.encoder1_2(encoder1_1)
        encoder1_3 = self.encoder1_3(encoder1_2)
        encoder1_4 = self.encoder1_4(encoder1_3)
        encoder1_5 = self.encoder1_5(encoder1_4)
        encoder1_out = self.encoder1_6(encoder1_5)

        input_decoder1 = encoder1_out
        decoder1_1 = self.decoder1_1(input_decoder1) + encoder1_5
        decoder1_2 = self.decoder1_2(decoder1_1) + encoder1_4
        decoder1_3 = self.decoder1_3(decoder1_2) + encoder1_3
        decoder1_4 = self.decoder1_4(decoder1_3) + encoder1_2
        decoder1_5 = self.decoder1_5(decoder1_4) + encoder1_1
        out_category = self.temperature * self.decoder1_6(
            decoder1_5
        )  # [N, category, H, W]

        # get fused normal
        category = F.softmax(out_category, dim=1)
        if mask is None:
            mask = 1 - category[:, 0:1, :, :]  # [N, 1, H, W] soft mask is used
        normal1 = category[:, 1:2, :, :]  # [N, 1, H, W]
        normal2 = category[:, 2:3, :, :]  # [N, 1, H, W]
        fused_normal = (
            normal1 * ab_normal[:, 0:3, :, :] + normal2 * ab_normal[:, 3:6, :, :]
        )
        norm = torch.norm(fused_normal, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
        fused_normal = mask * fused_normal / (norm + 1e-8)  # [N, 3, H, W]

        # encode polarization img
        encoder2_1 = self.encoder2_1(img)
        encoder2_2 = self.encoder2_2(encoder2_1)
        encoder2_3 = self.encoder2_3(encoder2_2)
        encoder2_4 = self.encoder2_4(encoder2_3)
        encoder2_5 = self.encoder2_5(encoder2_4)
        encoder2_out = self.encoder2_6(encoder2_5)

        input3 = torch.cat([ab_normal, fused_normal], dim=1)
        encoder3_1 = self.encoder3_1(input3)
        encoder3_2 = self.encoder3_2(encoder3_1)
        encoder3_3 = self.encoder3_3(encoder3_2)
        encoder3_4 = self.encoder3_4(encoder3_3)
        encoder3_5 = self.encoder3_5(encoder3_4)
        encoder3_out = self.encoder3_6(encoder3_5)

        # decoder3 normal stage 2
        input_decoder2 = encoder2_out + encoder3_out
        decoder2_1 = self.decoder2_1(input_decoder2) + encoder3_5 + encoder2_5
        decoder2_2 = self.decoder2_2(decoder2_1) + encoder3_4 + encoder2_4
        decoder2_3 = self.decoder2_3(decoder2_2) + encoder3_3 + encoder2_3
        decoder2_4 = self.decoder2_4(decoder2_3) + encoder3_2 + encoder2_2
        decoder2_5 = self.decoder2_5(decoder2_4) + encoder3_1 + encoder2_1
        pred_normal2 = self.decoder2_6(decoder2_5)

        # norm = torch.norm(pred_normal2, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
        # pred_normal2 = pred_normal2 / (norm + 1e-8)  # [N, 3, H, W]
        return out_category, [pred_normal2], mask


class Polar2NormalPhysics(nn.Module):
    def __init__(self, mode):
        super(Polar2NormalPhysics, self).__init__()
        assert mode in ["physics"]
        # input_channel1, input_channel2, out_channel1, out_channel2
        self.input_channel2 = 6
        self.input_channel1 = 4
        self.out_channel1 = 3

        self.encoder1_1 = EncoderBlock(self.input_channel1, 8)
        self.encoder1_2 = EncoderBlock(8, 16)
        self.encoder1_3 = EncoderBlock(16, 32)
        self.encoder1_4 = EncoderBlock(32, 64)
        self.encoder1_5 = EncoderBlock(64, 128)
        self.encoder1_6 = EncoderBlock(128, 256)

        self.encoder2_1 = EncoderBlock(self.input_channel2, 8)
        self.encoder2_2 = EncoderBlock(8, 16)
        self.encoder2_3 = EncoderBlock(16, 32)
        self.encoder2_4 = EncoderBlock(32, 64)
        self.encoder2_5 = EncoderBlock(64, 128)
        self.encoder2_6 = EncoderBlock(128, 256)

        self.decoder1_1 = DecoderBlock(256, 128)
        self.decoder1_2 = DecoderBlock(128, 64)
        self.decoder1_3 = DecoderBlock(64, 32)
        self.decoder1_4 = DecoderBlock(32, 16)
        self.decoder1_5 = DecoderBlock(16, 8)
        self.decoder1_6 = DecoderBlock(8, self.out_channel1, True)

    def forward(self, img, ab_normal):
        # encoder1 polarization image
        encoder1_1 = self.encoder1_1(img)
        encoder1_2 = self.encoder1_2(encoder1_1)
        encoder1_3 = self.encoder1_3(encoder1_2)
        encoder1_4 = self.encoder1_4(encoder1_3)
        encoder1_5 = self.encoder1_5(encoder1_4)
        encoder1_out = self.encoder1_6(encoder1_5)

        # encoder2 ambiguous normal
        encoder2_1 = self.encoder2_1(ab_normal)
        encoder2_2 = self.encoder2_2(encoder2_1)
        encoder2_3 = self.encoder2_3(encoder2_2)
        encoder2_4 = self.encoder2_4(encoder2_3)
        encoder2_5 = self.encoder2_5(encoder2_4)
        encoder2_out = self.encoder2_6(encoder2_5)

        # decoder2 normal stage 1
        input_decoder2 = encoder1_out + encoder2_out
        decoder1_1 = self.decoder1_1(input_decoder2) + encoder1_5 + encoder2_5
        decoder1_2 = self.decoder1_2(decoder1_1) + encoder1_4 + encoder2_4
        decoder1_3 = self.decoder1_3(decoder1_2) + encoder1_3 + encoder2_3
        decoder1_4 = self.decoder1_4(decoder1_3) + encoder1_2 + encoder2_2
        decoder1_5 = self.decoder1_5(decoder1_4) + encoder1_1 + encoder2_1
        pred_normal = self.decoder1_6(decoder1_5)  # [N, 3, H, W]

        # norm = torch.norm(pred_normal, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
        # pred_normal = pred_normal / (norm + 1e-8)  # [N, 3, H, W]
        return [pred_normal]


class Img2NormalNoPrior(nn.Module):
    def __init__(self, mode, iter_num=1):
        super(Img2NormalNoPrior, self).__init__()
        assert mode in ["no_prior", "color"]
        # input_channel1, input_channel2, out_channel1, out_channel2
        if mode == "no_prior":
            self.input_channel1 = 4
        else:
            # color
            self.input_channel1 = 3
        self.iter_num = iter_num
        self.input_channel2 = 3
        self.out_channel1 = 3
        self.out_channel2 = 3

        self.encoder1_1 = EncoderBlock(self.input_channel1, 8)
        self.encoder1_2 = EncoderBlock(8, 16)
        self.encoder1_3 = EncoderBlock(16, 32)
        self.encoder1_4 = EncoderBlock(32, 64)
        self.encoder1_5 = EncoderBlock(64, 128)
        self.encoder1_6 = EncoderBlock(128, 256)

        self.encoder2_1 = EncoderBlock(self.input_channel2, 8)
        self.encoder2_2 = EncoderBlock(8, 16)
        self.encoder2_3 = EncoderBlock(16, 32)
        self.encoder2_4 = EncoderBlock(32, 64)
        self.encoder2_5 = EncoderBlock(64, 128)
        self.encoder2_6 = EncoderBlock(128, 256)

        self.decoder1_1 = DecoderBlock(256, 128)
        self.decoder1_2 = DecoderBlock(128, 64)
        self.decoder1_3 = DecoderBlock(64, 32)
        self.decoder1_4 = DecoderBlock(32, 16)
        self.decoder1_5 = DecoderBlock(16, 8)
        self.decoder1_6 = DecoderBlock(8, self.out_channel1, True)

        self.decoder2_1 = DecoderBlock(256, 128)
        self.decoder2_2 = DecoderBlock(128, 64)
        self.decoder2_3 = DecoderBlock(64, 32)
        self.decoder2_4 = DecoderBlock(32, 16)
        self.decoder2_5 = DecoderBlock(16, 8)
        self.decoder2_6 = DecoderBlock(8, self.out_channel2, True)

    def forward(self, img):
        # encoder1 polarization/color image
        encoder1_1 = self.encoder1_1(img)
        encoder1_2 = self.encoder1_2(encoder1_1)
        encoder1_3 = self.encoder1_3(encoder1_2)
        encoder1_4 = self.encoder1_4(encoder1_3)
        encoder1_5 = self.encoder1_5(encoder1_4)
        encoder1_out = self.encoder1_6(encoder1_5)

        input_decoder1 = encoder1_out
        decoder1_1 = self.decoder1_1(input_decoder1) + encoder1_5
        decoder1_2 = self.decoder1_2(decoder1_1) + encoder1_4
        decoder1_3 = self.decoder1_3(decoder1_2) + encoder1_3
        decoder1_4 = self.decoder1_4(decoder1_3) + encoder1_2
        decoder1_5 = self.decoder1_5(decoder1_4) + encoder1_1
        pred_normal1 = self.decoder1_6(decoder1_5)

        # norm = torch.norm(pred_normal1, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
        # pred_normal1 = pred_normal1 / (norm + 1e-8)  # [N, 3, H, W]
        coarse_normal = torch.clone(pred_normal1)
        pred_normal2 = []
        for i in range(self.iter_num):
            # encoder2 ambiguous normal
            encoder2_1 = self.encoder2_1(coarse_normal)
            encoder2_2 = self.encoder2_2(encoder2_1)
            encoder2_3 = self.encoder2_3(encoder2_2)
            encoder2_4 = self.encoder2_4(encoder2_3)
            encoder2_5 = self.encoder2_5(encoder2_4)
            encoder2_out = self.encoder2_6(encoder2_5)

            input_decoder2 = encoder2_out
            decoder2_1 = self.decoder2_1(input_decoder2) + encoder2_5 + encoder1_5
            decoder2_2 = self.decoder2_2(decoder2_1) + encoder2_4 + encoder1_4
            decoder2_3 = self.decoder2_3(decoder2_2) + encoder2_3 + encoder1_3
            decoder2_4 = self.decoder2_4(decoder2_3) + encoder2_2 + encoder1_2
            decoder2_5 = self.decoder2_5(decoder2_4) + encoder2_1 + encoder1_1
            normal_iter = self.decoder1_6(decoder2_5)  # [N, 3, H, W]

            # normalize
            # norm = torch.norm(normal_iter, p=2, dim=1, keepdim=True)  # [N, 1, H, W]
            # coarse_normal = normal_iter / (norm + 1e-8)  # [N, 3, H, W]
            coarse_normal = normal_iter
            pred_normal2.append(coarse_normal)

        return pred_normal1, pred_normal2


class Polar2Shape(nn.Module):
    def __init__(self, mode, use_6drotation):
        super(Polar2Shape, self).__init__()
        assert mode in [
            "normal_polar",
            "polar",
            "mask_polar",
            "normal_color",
            "color",
            "mask_color",
        ]
        self.mode = mode
        if self.mode == "normal_polar":
            input_channel = 4 + 3  # polar + normal
        elif self.mode == "polar":
            input_channel = 4  # polar
        elif self.mode == "mask_polar":
            input_channel = 4 + 1  # polar, mask
        elif self.mode == "normal_color":
            input_channel = 3 + 3  # color + normal
        elif self.mode == "mask_color":
            input_channel = 3 + 1  # color, mask
        else:
            # color
            input_channel = 3  # color
        beta_dim = 10
        if use_6drotation:
            theta_dim = 24 * 6
        else:
            theta_dim = 24 * 3
        trans_dim = 3

        self.encoder = self.resnet50_encoder(input_channel)
        self.beta_fc = torch.nn.Linear(2048, beta_dim)
        self.theta_fc = torch.nn.Linear(2048, theta_dim)
        self.trans_fc = torch.nn.Linear(2048, trans_dim)

    def resnet50_encoder(self, input_channel):
        model = resnet50(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Sequential()
        return model

    def forward(self, _input):
        x = self.encoder(_input)
        beta = self.beta_fc(x)
        theta = self.theta_fc(x)
        trans = self.trans_fc(x)
        return beta, theta, trans


class Model(nn.Module):
    def __init__(
        self,
        normal_mode="2_stages",
        shape_mode="normal_polar",
        temperature=10,
        img_size=512,
        use_6drotation=True,
        smpl_dir="../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl",
        batch_size=16,
        task="img2normal",
        iter_num=1,
    ):
        super(Model, self).__init__()
        self.normal_mode = normal_mode
        self.shape_mode = shape_mode
        self.temperature = temperature
        self.img_size = img_size
        self.use_6drotation = use_6drotation
        self.task = task
        self.iter_num = iter_num

        if self.normal_mode == "2_stages":
            self.img2normal = Polar2NormalNew(
                mode=self.normal_mode,
                temperature=self.temperature,
                iter_num=self.iter_num,
            )
        elif self.normal_mode == "physics":
            self.img2normal = Polar2NormalPhysics(mode=self.normal_mode)
        elif self.normal_mode == "no_prior":
            self.img2normal = Img2NormalNoPrior(
                mode=self.normal_mode, iter_num=self.iter_num
            )
        elif self.normal_mode == "color":
            self.img2normal = Img2NormalNoPrior(
                mode=self.normal_mode, iter_num=self.iter_num
            )
        elif self.normal_mode == "eccv2020":
            self.img2normal = Polar2NormalECCV(
                mode=self.normal_mode, temperature=self.temperature
            )
        else:
            raise ValueError("normal_mode errors %s." % normal_mode)

        # shape part
        self.img2shape = Polar2Shape(
            mode=self.shape_mode, use_6drotation=use_6drotation
        )
        self.img_scaler_half = torch.nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.img_scaler_double = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.smpl = SMPL(smpl_dir, batch_size)

    def forward(
        self, img, ambiguous_normal=None, mask=None, cam_intr=None, gt_smpl=None
    ):
        out = {}

        if self.task == "img2normal":
            # polar to normal and mask
            if self.normal_mode == "2_stages":
                (
                    pred_category,
                    pred_normal1,
                    pred_normal2,
                    pred_mask,
                    normal_residual,
                    fused_normal,
                ) = self.img2normal(img, ambiguous_normal, mask)
            elif self.normal_mode == "eccv2020":
                pred_category, pred_normal2, pred_mask = self.img2normal(
                    img, ambiguous_normal, mask
                )
                pred_normal1, normal_residual, fused_normal = None, None, None
            elif self.normal_mode == "physics":
                pred_normal2 = self.img2normal(img, ambiguous_normal)
                (
                    pred_category,
                    pred_normal1,
                    pred_mask,
                    normal_residual,
                    fused_normal,
                ) = (None, None, None, None, None)
            elif self.normal_mode == "no_prior" or self.normal_mode == "color":
                pred_normal1, pred_normal2 = self.img2normal(img)
                pred_category, pred_mask, normal_residual, fused_normal = (
                    None,
                    None,
                    None,
                    None,
                )
            else:
                (
                    pred_category,
                    pred_normal2,
                    pred_normal1,
                    pred_mask,
                    normal_residual,
                    fused_normal,
                ) = (None, None, None, None, None, None)

            out["category"] = pred_category
            out["normal_residual"] = normal_residual
            out["fused_normal"] = fused_normal
            if self.img_size == 512:
                out["normal_stage1"] = pred_normal1
                out["normal_stage2"] = pred_normal2
            else:
                if pred_normal1 is not None:
                    out["normal_stage1"] = self.img_scaler_double(pred_normal1)
                else:
                    out["normal_stage1"] = None
                out["normal_stage2"] = []
                for normal in pred_normal2:
                    out["normal_stage2"].append(self.img_scaler_double(normal))

        elif self.task == "img2shape":
            # assert self.normal_mode == '2_stages'
            # img to pose
            if self.shape_mode == "polar" or self.shape_mode == "color":
                # only images
                shape_input_img = img
            elif self.shape_mode == "mask_polar" or self.shape_mode == "mask_color":
                # mask with images
                if self.normal_mode == "2_stages":
                    _, _, pred_normal2, pred_mask, _, _ = self.img2normal(
                        img, ambiguous_normal, mask
                    )
                elif self.normal_mode == "eccv2020":
                    _, pred_normal2, pred_mask = self.img2normal(
                        img, ambiguous_normal, mask
                    )
                else:
                    _, pred_normal2 = self.img2normal(img)
                    pred_mask = mask

                shape_input_img = torch.cat([img, pred_mask.detach()], dim=1)
            else:
                # normal with images
                if self.normal_mode == "2_stages":
                    _, _, pred_normal2, pred_mask, _, _ = self.img2normal(
                        img, ambiguous_normal, mask
                    )
                elif self.normal_mode == "eccv2020":
                    _, pred_normal2, pred_mask = self.img2normal(
                        img, ambiguous_normal, mask
                    )
                else:
                    _, pred_normal2 = self.img2normal(img)
                    pred_mask = mask

                shape_input_img = torch.cat(
                    [img, pred_mask.detach() * pred_normal2[-1].detach()], dim=1
                )

                out["normal_stage2"] = pred_normal2[-1]
                out["mask"] = pred_mask

            if self.img_size == 512:
                shape_input_img = self.img_scaler_half(shape_input_img)

            beta, theta, trans = self.img2shape(shape_input_img)
            if gt_smpl is not None:
                beta = gt_smpl

            if self.use_6drotation:
                rotmats = rot6d_to_rotmat(theta).view(-1, 24, 3, 3)
                pred_verts, pred_joints3d, _ = self.smpl(
                    beta=beta, theta=None, get_skin=True, rotmats=rotmats
                )
            else:
                rotmats = None
                pred_verts, pred_joints3d, _ = self.smpl(
                    beta=beta, theta=theta, get_skin=True, rotmats=None
                )

            out["beta"] = beta
            out["theta"] = theta
            out["rotmats"] = rotmats
            out["trans"] = trans
            out["verts"] = pred_verts + trans.unsqueeze(1)
            out["joints3d"] = pred_joints3d + trans.unsqueeze(1)
            if cam_intr is not None:
                out["joints2d"] = projection_torch(
                    out["joints3d"], cam_intr, self.img_size, self.img_size
                )
                out["cam_intr"] = cam_intr
            else:
                out["joints2d"] = None
                out["cam_intr"] = None

        return out


class ModelHMR(nn.Module):
    """re-train HMR on our dataset, the input image can be a color image or a polarization image"""

    def __init__(
        self,
        image_mode,
        smpl_mean,
        batch_size=16,
        smpl_dir="../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl",
        iterations=3,
        img_size=256,
    ):
        super(ModelHMR, self).__init__()
        assert image_mode in ["color", "polar"]
        if image_mode == "color":
            input_channel = 3
        else:
            input_channel = 4
        self.iterations = iterations
        self.img_size = img_size
        self.smpl = SMPL(smpl_dir, batch_size)

        # smpl_mean should be [85]
        self.register_buffer("smpl_mean", smpl_mean.repeat([batch_size, 1]))
        self.encoder = self.resnet50_encoder(input_channel)
        self.regressor = nn.Sequential(
            nn.Linear(2048 + 85, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 85),
        )

    def resnet50_encoder(self, input_channel):
        model = resnet50(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Sequential()
        return model

    def forward(self, img, cam_intr=None):
        x = self.encoder(img)  # [N, 2048]
        n = x.size(0)
        x_smpl = self.smpl_mean[:n, :]

        out = {}
        out["trans"], out["beta"], out["theta"] = [], [], []
        for _ in range(self.iterations):
            fc_input = torch.cat([x, x_smpl], dim=1)
            x_smpl = x_smpl + self.regressor(fc_input)
            out["trans"].append(x_smpl[:, 0:3])
            out["beta"].append(x_smpl[:, 3:13])
            out["theta"].append(x_smpl[:, 13:85])

        pred_verts, pred_joints3d, _ = self.smpl(
            beta=torch.clone(out["beta"][-1]),
            theta=torch.clone(out["theta"][-1]),
            get_skin=True,
            rotmats=None,
        )
        trans = out["trans"][-1]
        out["verts"] = pred_verts + trans.unsqueeze(1)
        out["joints3d"] = pred_joints3d + trans.unsqueeze(1)
        if cam_intr is not None:
            out["joints2d"] = projection_torch(
                out["joints3d"], cam_intr, self.img_size, self.img_size
            )
            out["cam_intr"] = cam_intr
        else:
            out["joints2d"] = None
            out["cam_intr"] = None
        return out


if __name__ == "__main__":
    device = torch.device("cuda:1")
    model = Model(
        normal_mode="2_stages",
        shape_mode="normal_polar",
        temperature=10,
        img_size=256,
        use_6drotation=True,
        smpl_dir="../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl",
        batch_size=16,
    )
    # model = Polar2NormalCategory(3)
    model = model.to(device=device)
    tmp_input1 = torch.rand([2, 4, 256, 256])
    tmp_input2 = torch.rand([2, 6, 256, 256])
    tmp_input3 = torch.rand([2, 1, 256, 256])
    tmp_input4 = torch.rand([2, 4])
    tmp = {
        "polar": tmp_input1,
        "ambiguous_normal": tmp_input2,
        "mask": tmp_input3,
        "cam_intr": tmp_input4,
    }
    for k, v in tmp.items():
        tmp[k] = v.to(device=device, dtype=torch.float32)

    out = model(tmp["polar"], tmp["ambiguous_normal"], None, tmp["cam_intr"])
    for k, v in out.items():
        if v is not None:
            print(k, v.size())
