"""
    file:   SMPL.py
    date:   2018_05_03
    author: zhangxiong(1025679612@qq.com)
    mark:   the algorithm is cited from original SMPL
"""
import torch
import pickle
import numpy as np
from src.PHSPDataset.utils import batch_global_rigid_transformation, batch_rodrigues
import torch.nn as nn
from plyfile import PlyData, PlyElement


class SMPL(nn.Module):
    def __init__(self, model_path, max_batch_size, obj_saveable=False):
        super(SMPL, self).__init__()

        self.model_path = model_path
        model = pickle.load(open(model_path, 'rb'), encoding='iso-8859-1')

        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None

        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = np.array(model['J_regressor'].todense().T, dtype=np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_weights = np.array(model['weights'], dtype=np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        # batch_size = max(args.batch_size + args.batch_3d_size, args.eval_batch_size)
        np_weights = np.tile(np_weights, (max_batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))
        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        vertex = np.array([tuple(i) for i in verts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        face = np.array([(tuple(i), 255, 255, 255) for i in self.faces],
                        dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        el2 = PlyElement.describe(face, 'face')
        plydata = PlyData([el, el2])
        plydata.write(obj_mesh_name)

    def forward(self, beta, theta, get_skin=False):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.size(0)

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents,
                                                                  rotate_base=False, device=self.cur_device)

        weight = self.weight[:num_batch]
        W = weight.view(num_batch, -1, 24)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joints = self.J_transformed

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

