import pickle
import src.PHSPDataset.utils as utils


class CameraParams:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir

        # load camera params, save intrinsic and extrinsic camera parameters as a dictionary
        # intrinsic ['param_p', 'param_c1', 'param_d1', 'param_c2', 'param_d2', 'param_c3', 'param_d3']
        # extrinsic ['d1p', 'd2p', 'd3p', 'cd1', 'cd2', 'cd3']
        self.cam_params = []
        with open('%s/CamParams0906.pkl' % self.data_dir, 'rb') as f:
            self.cam_params.append(pickle.load(f))
        with open('%s/CamParams0909.pkl' % self.data_dir, 'rb') as f:
            self.cam_params.append(pickle.load(f))

        # corresponding cam params to each subject
        self.name_cam_params = {}  # {"name": 0 or 1}
        for name in ['subject06', 'subject09', 'subject11', 'subject05', 'subject12', 'subject04']:
            self.name_cam_params[name] = 0
        for name in ['subject03', 'subject01', 'subject02', 'subject10', 'subject07', 'subject08']:
            self.name_cam_params[name] = 1

        # corresponding cam params to each subject
        self.name_gender = {}  # {"name": 0 or 1}
        for name in ['subject02', 'subject03', 'subject04', 'subject05', 'subject06',
                     'subject08', 'subject09', 'subject11', 'subject12']:
            self.name_gender[name] = 0  # male
        for name in ['subject01', 'subject07', 'subject10']:
            self.name_gender[name] = 1  # female

    def get_intrinsic(self, cam_name, subject_no):
        """
        'p': polarization camera, color
        'c1': color camera for the 1st Kinect
        'd1': depth (ToF) camera for the 1st Kinect
        ...
        return
            (fx, fy, cx, cy)
        """
        assert cam_name in ['p', 'c1', 'd1', 'c2', 'd2', 'c3', 'd3']
        assert subject_no in ['subject06', 'subject09', 'subject11', 'subject05', 'subject12', 'subject04',
                              'subject03', 'subject01', 'subject02', 'subject10', 'subject07', 'subject08']
        fx, fy, cx, cy, _, _, _ = self.cam_params[self.name_cam_params[subject_no]]['param_%s' % cam_name]
        intrinsic = (fx, fy, cx, cy)
        return intrinsic

    def get_extrinsic(self, cams_name, subject_no):
        """
        The annotated poses and shapes are saved in polarization camera coordinate.
        'd1p': transform from polarization camera to 1st Kinect depth image
        'c1p': transform from polarization camera to 1st Kinect color image
        ...
        return
            transform class
        """
        assert cams_name in ['d1', 'd2', 'd3', 'c1', 'c2', 'c3']
        assert subject_no in ['subject06', 'subject09', 'subject11', 'subject05', 'subject12', 'subject04',
                              'subject03', 'subject01', 'subject02', 'subject10', 'subject07', 'subject08']

        if cams_name in ['d1p', 'd2p', 'd3p']:
            T = utils.convert_param2tranform(self.cam_params[self.name_cam_params[subject_no]][cams_name])
        else:
            i = cams_name[1]
            T_dp = utils.convert_param2tranform(self.cam_params[self.name_cam_params[subject_no]]['d%sp' % i])
            T_cd = utils.convert_param2tranform(self.cam_params[self.name_cam_params[subject_no]]['cd%s' % i])
            T = T_cd * T_dp
        return T

    def get_gender(self, subject_no):
        return self.name_gender[subject_no]


if __name__ == '__main__':
    # test
    camera_params = CameraParams(data_dir='/media/data/data_shihao')

