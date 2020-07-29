import numpy as np
import os
"""camera parameters"""


def quat2R(quat):
    """
    Description
    ===========
    convert vector q to matrix R

    Parameters
    ==========
    :param quat: (4,) array

    Returns
    =======
    :return: (3,3) array
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    n = w * w + x * x + y * y + z * z
    s = 2. / np.clip(n, 1e-7, 1e7)

    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z

    R = np.stack([1 - (yy + zz), xy - wz, xz + wy,
                  xy + wz, 1 - (xx + zz), yz - wx,
                  xz - wy, yz + wx, 1 - (xx + yy)])

    return R.reshape((3, 3))


def convert_param2tranform(param, scale=1):
    R = quat2R(param[0:4])
    t = param[4:7]
    s = scale * np.ones(3, 'float')
    return Transform(R, t, s)


class Transform:
    def __init__(self, R=np.eye(3, dtype='float'), t=np.zeros(3, 'float'), s=np.ones(3, 'float')):
        self.R = R.copy()  # rotation
        self.t = t.reshape(-1).copy()  # translation
        self.s = s.copy()  # scale

    def __mul__(self, other):
        # combine two transformation together
        R = np.dot(self.R, other.R)
        t = np.dot(self.R, other.t * self.s) + self.t
        if not hasattr(other, 's'):
            other.s = np.ones(3, 'float').copy()
        s = other.s.copy()
        return Transform(R, t, s)

    def inv(self):
        # inverse the rigid tansformation
        R = self.R.T
        t = -np.dot(self.R.T, self.t)
        return Transform(R, t)

    def transform(self, xyz):
        # transform 3D point
        if not hasattr(self, 's'):
            self.s = np.ones(3, 'float').copy()
        assert xyz.shape[-1] == 3
        assert len(self.s) == 3
        return np.dot(xyz * self.s, self.R.T) + self.t

    def getmat4(self):
        # homogeneous transformation matrix
        M = np.eye(4)
        M[:3, :3] = self.R * self.s
        M[:3, 3] = self.t
        return M


def load_pose_and_shape(data_dir, subject_no='all'):
    annotations = {}
    filenames = sorted(os.listdir('%s/pose/' % data_dir))
    for idx, filename in enumerate(filenames):
        if subject_no not in filename and subject_no is not 'all':
            continue
        else:
            print('processed %s %i / %i' % (filename, idx, len(filenames)))
            # read smpl shape parameters as dict
            shape_params = {}
            with open('%s/pose/%s/shape_smpl.txt' % (data_dir, filename), 'r') as f:
                for line in f.readlines():
                    tmp = line.split(' ')  # frame_idx
                    smpl_param = np.asarray([float(i) for i in tmp[1:]])
                    # [beta, translation, theta]
                    smpl_param = np.concatenate([smpl_param[3:13], smpl_param[0:3], smpl_param[13:85]], axis=0)
                    shape_params[tmp[0]] = smpl_param

            # read 3D joint pose
            with open('%s/pose/%s/pose.txt' % (data_dir, filename), 'r') as f:
                for line in f.readlines():
                    tmp = line.split(' ')  # frame_idx
                    pose = np.asarray([float(i) for i in tmp[1:]]).reshape([-1, 3])
                    annotations['%s-%s' % (filename, tmp[0])] = (pose, shape_params[tmp[0]])
    return annotations


def projection(points, params):
    # points: [N, 3]
    fx, fy, cx, cy = params
    u = cx + fx * points[:, 0] / points[:, 2]
    v = cy + fy * points[:, 1] / points[:, 2]
    d = points[:, 2]
    return np.stack([u, v, d], axis=-1)


"""SMPL pytorch implementation"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False, device='cpu'):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = Variable(torch.from_numpy(np_rot_x).float()).cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1, device=device))], dim=1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1, device=device))], dim=2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A


def batch_rodrigues(theta):
    # theta N x 3
    # batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


"""render model"""
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
import cv2

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.01,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


def simple_renderer(rn, verts, faces, yrot=np.radians(120)):

    # Rendered model color
    color = colors['pink']

    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge(
        (b_channel, g_channel, r_channel, alpha.astype(imtmp.dtype)))
    return im_RGBA


def render_model(verts, faces, w, h, cam_param, cam_t, cam_rt, near=0.5, far=25, img=None):
    f = cam_param[0:2]
    c = cam_param[2:4]
    rn = _create_renderer(w=w, h=h, near=near, far=far, rt=cam_rt, t=cam_t, f=f, c=c)
    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    imtmp = simple_renderer(rn, verts, faces)

    # If white bg, make transparent.
    if img is None:
        imtmp = get_alpha(imtmp)

    return imtmp


def render_depth_v(verts, faces, require_visi = False,
                   t = [0.,0.,0.], img_size=[448, 448], f=[400.0,400.0], c=[224.,224.]):
    from opendr.renderer import DepthRenderer
    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt = np.zeros(3),
                              t = t,
                              f = f,
                              c = c,
                              k = np.zeros(5)
                             )
    rn.frustum = {'near': .01, 'far': 10000.,
                  'width': img_size[1], 'height': img_size[0]}
    rn.v = verts
    rn.f = faces
    rn.bgcolor = np.zeros(3)
    if require_visi is True:
        return rn.r, rn.visibility_image
    else:
        return rn.r


def projection(points, params):
    # points: [N, 3]
    fx, fy, cx, cy = params
    u = cx + fx * points[:, 0] / points[:, 2]
    v = cy + fy * points[:, 1] / points[:, 2]
    d = points[:, 2]
    return np.stack([u, v, d], axis=-1)
