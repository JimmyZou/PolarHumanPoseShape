## 3D Human Shape Reconstruction from a Polarization Image

Shihao Zou, Xinxin Zuo, Yiming Qian, Sen Wang, Chi Xu, Minglun Gong and Li Cheng. In Proceedings of the 16th European Conference on Computer Vision (ECCV) 2020.

### [Project page](https://jimmyzou.github.io/publication/2020-polarization-clothed-human-shape)
<center><img src="demo_detailed_shape.gif" width=“500”/></center>

### [PHSPDataset page](https://jimmyzou.github.io/publication/2020-PHSPDataset)
Our PHSPDataset provides:
- one view polarization image
- three-view Kinects v2 (three-view ToF depth and color images)
- 12 subjects (9 males and 3 females)
- each subject to do 3 different groups of actions (18 different actions in total) for 4 times plus one free-style group. (around 22K frames of each subject with about 13 fps)
- annotations of SMPL shape, pose and actions of 34 video clips

<center><img src="demo_annotation_shape.gif" width=“500”/></center>

## The code for our 3D human shape reconstruction method
coming soon...


## The code for the usage of Polarization Human Pose and Shape Dataset.
### **This dataset can only be used for academic purpose. Commercial use is strictly prohibited without permission.**

You can download the data from [Google Drive]() or [Microsoft OneDrive](), which consists of
- **samples.tar.gz** (a small subset of our dataset that you can have a snap of our dataset)
- color/color_view*.tar.gz.* (three-view color images, first * means three cameras and second * means the partition of the packed file.)
  - color/color_view*/subject\*\*\_group\*\_time\*/color\_\*.jpg
- depth.tar.gz (three-view depth images)
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

[//]: # (tar -zcf color_view2.tar.gz ./color/view2/ | split -b 70000m -d -a 1 - color_view2.tar.gz.)

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
opendr 0.78 (for render SMPL shape, installed successfully only under ubuntu 18.04)
cv2 4.1.1
```

After moving the data in the direction ./data, please revise the direction settings and run to see the demo figures
```
python PHSPDataset/multi_view_shape_and_pose.py
```

### Citation
If you would like to use our code or dataset, please cite either
```
@inproceedings{zou2020detailed,  
  title={3D Human Shape Reconstruction from a Polarization Image},  
  author={Zou, Shihao and Zuo, Xinxin and Qian, Yiming and Wang, Sen and Xu, Chi and Gong, Minglun and Cheng, Li},  
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
  year={2020}  
} 
```
or
```
@article{zou2020polarization,  
  title={Polarization Human Shape and Pose Dataset},  
  author={Zou, Shihao and Zuo, Xinxin and Qian, Yiming and Wang, Sen and Xu, Chi and Gong, Minglun and Cheng, Li},  
  journal={arXiv preprint arXiv:2004.14899},  
  year={2020}  
}  
```
