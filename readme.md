## Human Pose and Shape Estimation from Single Polarization Images

Shihao Zou, Xinxin Zuo, Sen Wang, Yiming Qian, Chuan Guo, and Li Cheng. In IEEE Transaction on Multimedia (TMM) 2022.

### [inference]
Please download the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1-0gvpgVXiXZfJUtxIupgZMuFXBMIFljf?usp=sharing) or [Microsoft OneDrive](https://ualbertaca-my.sharepoint.com/:f:/g/personal/szou2_ualberta_ca/EvDseToaRM1Hon2qq-x91XoBlySF3TaZqcIAyjtN4VnTyg?e=dfhIjd). Then place it in the folder ```data/model```. 


To download the SMPL model go to [this](https://smpl.is.tue.mpg.de/) project website and register to get access to the downloads section. The model version used in our project is
```basicModel_m_lbs_10_207_0_v1.0.0.pkl``` or ```SMPLE_MALE.pkl``` (renamed for SMPLX).


Finally, ```cd src/human_pose_estimation``` and ```python inference.py``` to run inference on the demo sample in the folder ```data/inference_demo```. ```data/inference_demo/normal_0001.jpg``` and ```data/inference_demo/shape_0001.jpg``` are predicted normal map and SMPL shape. 

Opendr is used to render SMPL shape on the image. Opendr was installed successfully according to [link](https://github.com/mattloper/opendr/issues/19#issuecomment-532509726).

Requirements
```
python >= 3.7
torch >= 1.1.0
opendr = 0.78
cv2 >= 4.1.1
```

### [PHSPDataset v2]
Our PHSPDataset v2 provides:
- one view polarization image
- five-view Kinects (five-view RGBD images)
- 15 subjects (11 males and 4 females)
- each subject to do 3 different groups of actions (21 different actions in total) for 4 times.
- annotations of SMPL shape and pose parameters

We also captured PHSPDataset v2, which has the similar amount of data with PHSPDataset v1. If you are interested in using our dataset, please contact ```szou2@ualberta.ca```. We will response as soon as possible.

## 3D Human Shape Reconstruction from a Polarization Image

Shihao Zou, Xinxin Zuo, Yiming Qian, Sen Wang, Chi Xu, Minglun Gong and Li Cheng. In Proceedings of the 16th European Conference on Computer Vision (ECCV) 2020.

### [Project page](https://jimmyzou.github.io/publication/2020-polarization-clothed-human-shape)
<center><img src="demo_detailed_shape.gif" width=“500”/></center>

### [PHSPDataset v1](https://jimmyzou.github.io/publication/2020-PHSPDataset)
Our PHSPDataset v1 provides:
- one view polarization image
- three-view Kinects v2 (three-view RGBD images)
- 12 subjects (9 males and 3 females)
- each subject to do 3 different groups of actions (18 different actions in total) for 4 times plus one free-style group. (around 22K frames of each subject with about 15 fps)
- annotations of SMPL shape and pose parameters
- annotated actions of 34 video clips

<center><img src="demo_annotation_shape.gif" width=“500”/></center>

## The code for the usage of PHSPDataset v1.
PHSPDataset v1 consists of
- **samples.tar.gz** (a small subset of our dataset that you can have a snap of our dataset)
- color/color_view*.tar.gz.* (three-view color images, first * means three cameras and second * means the partition of the packed file.)
  - color/color_view*/subject\*\*\_group\*\_time\*/color\_\*.jpg
- depth/depth.tar.gz (three-view depth images)
  - depth/view*/subject\*\*\_group\*\_time\*/depth\_\*.png
- bbx.tar.gz (bounding box, format: frame_idx, bbx for polarization image, bbx for color_view2 images)
  - bbx/subject\*\*\_group\*\_time\*/bbx.txt
- mask (human mask for polarization images and color images)
  - mask/polar_mask.tar.gz
  - mask/color2_mask.tar.gz
  - mask/color1_mask.tar.gz
- depth_segmentation.tar.gz (2D image uv index of human body in depth images)
  - depth_segmentation/view1/subject\*\*\_group\*\_time\*/seg_depth\_\*.mat
- HumanAct12.zip (human action types annotations)
  - HumanAct12/*npy
  - AboutHumanAct12.txt

[//]: # (tar -zcf color_view2.tar.gz ./color/view2/ | split -b 70000m -d -a 1 color_view2.tar.gz color_view2.tar.gz.)

After downloading the dataset, for split files (upload size limitation of OneDrive), you can cat and unpack the files using the command like
```
cat color_view2.tar.gz.* | tar -zx
```
for other files, you can directly unpack using the command like
```
tar -zx samples.tar.gz
```

To download the SMPL model go to [this](https://smpl.is.tue.mpg.de/) (male and female models) project website and register to get access to the downloads section. The model version used in our project is
```
basicModel_m_lbs_10_207_0_v1.0.0.pkl
basicModel_f_lbs_10_207_0_v1.0.0.pkl
```

Requirements
```
python 3.7.5
torch 1.1.0
opendr 0.78
cv2 4.1.1
```

After moving the data in the direction ./data, please revise the direction settings and run to see the demo figures
```
python PHSPDataset/multi_view_shape_and_pose.py
```

### Citation
If you would like to use our code or dataset, please cite either
```
@article{zou2022human,
  title={Human Pose and Shape Estimation from Single Polarization Images},
  author={Zou, Shihao and Zuo, Xinxin and Wang, Sen and Qian, Yiming and Guo, Chuan and Cheng, Li},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```
or
```
@inproceedings{zou2020detailed,  
  title={3D Human Shape Reconstruction from a Polarization Image},  
  author={Zou, Shihao and Zuo, Xinxin and Qian, Yiming and Wang, Sen and Xu, Chi and Gong, Minglun and Cheng, Li},  
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
  year={2020}  
} 
```

