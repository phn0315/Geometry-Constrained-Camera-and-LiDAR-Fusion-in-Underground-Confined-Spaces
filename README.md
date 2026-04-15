# Geometry-Constrained-Camera-and-LiDAR-Fusion-in-Underground-Confined-Spaces
ROS implementation for underground point cloud fusion, denoising, and 3D reconstruction with a depth camera and a single-line LiDAR. The related dataset is publicly available on IEEE DataPort (Author: Haonan Pang; Submitted by: Hongtao Yang; DOI: 10.21227/7ba7-tv69).
# Geometry Constrained Camera and LiDAR Fusion in Underground Confined Spaces

> ROS-based offline implementation for heterogeneous point cloud fusion and denoising in underground tunnel environments using a depth camera and a single-line LiDAR.

## Highlights

- Heterogeneous fusion of **depth camera** and **single-line LiDAR**
- LiDAR-referenced **centerline-based structured representation**
- **Unfolded-domain** correspondence construction and residual correction
- **Ensemble regression** with entropy-based prediction filtering
- Inverse mapping for **denoised 3D point cloud reconstruction**
- Export of both **intermediate** and **final** point cloud results

## Overview

This repository provides an offline ROS1 implementation of a geometry-constrained fusion-denoising framework for underground confined spaces. The method takes two registered point clouds as input:

- a **dense but noise-sensitive** depth-camera point cloud
- a **sparse but geometrically stable** single-line LiDAR point cloud

The workflow first estimates a LiDAR-referenced centerline, then unfolds both point clouds into a common parameter domain, performs residual correction using ensemble regression and confidence screening, and finally maps the corrected results back into three-dimensional space.

## Method Pipeline

The main processing pipeline is:

1. Read registered camera and LiDAR point clouds
2. Estimate a straight centerline from LiDAR
3. Build local frames along the centerline
4. Project both modalities onto the centerline
5. Unfold point clouds into a shared parameter domain
6. Construct cross-modal correspondences
7. Build features from geometry and residuals
8. Predict residuals using ensemble regression
9. Filter unreliable predictions by entropy
10. Refine corrected points with geometric constraints
11. Inversely map corrected results back to 3D
12. Save intermediate and final point clouds

## Repository Structure

```text
ros_pkg_ws/
├── src/
│   └── tunnel_fusion_denoise/
│       ├── launch/
│       │   └── offline_denoise.launch
│       ├── config/
│       │   └── denoise_params.yaml
│       ├── scripts/
│       │   └── tunnel_fusion_denoise_node.py
│       ├── datasets/
│       │   ├── CA.pcd
│       │   └── LD.pcd
│       ├── package.xml
│       ├── CMakeLists.txt
│       ├── requirements.txt
│       ├── README.md
│       └── LICENSE

Environment
System
Ubuntu 20.04
ROS Noetic
Python 3.8+
Python Dependencies
numpy
scipy
scikit-learn
open3d

Install dependencies:
pip install -r requirements.txt

Build
cd ros_pkg_ws
catkin_make
source devel/setup.bash

Input

The code expects two registered point clouds:

camera_pcd: depth-camera point cloud
lidar_pcd: single-line LiDAR point cloud

Supported formats:

.pcd
.ply

Example file placement:
ros_pkg_ws/src/tunnel_fusion_denoise/datasets/
├── CA.pcd
└── LD.pcd

Quick Start
cd ros_pkg_ws
source devel/setup.bash
roslaunch tunnel_fusion_denoise offline_denoise.launch \
  camera_pcd:=$(pwd)/src/tunnel_fusion_denoise/datasets/CA.pcd \
  lidar_pcd:=$(pwd)/src/tunnel_fusion_denoise/datasets/LD.pcd \
  output_dir:=$(pwd)/src/tunnel_fusion_denoise/datasets/results

 You may also use absolute paths:
 roslaunch tunnel_fusion_denoise offline_denoise.launch \
  camera_pcd:=/absolute/path/to/CA.pcd \
  lidar_pcd:=/absolute/path/to/LD.pcd \
  output_dir:=/absolute/path/to/results

 Main Parameters

Key configurable parameters include:

voxel_size: point cloud downsampling size
n_dense: number of centerline samples
extend_ratio: centerline extension ratio
y_mode: unfolded coordinate mode (arc or theta)
k_geom: neighborhood size for local geometry estimation
k_cov: neighborhood size for covariance estimation
max_s_diff: longitudinal matching threshold
max_theta_diff: angular matching threshold
max_3d_dist: spatial matching threshold
lambda_res: residual confidence weight
beta_curv: curvature-related confidence weight
n_estimators: number of trees in the ensemble regressor
max_depth: maximum tree depth
min_samples_leaf: minimum samples per leaf
entropy_quantile: entropy-based filtering threshold
lambda_n: normal-direction refinement weight
lambda_t: tangential refinement weight
keep_unmatched: whether to keep unmatched original camera points
Output

The following files are generated:

lidar_centerline.pcd — estimated LiDAR centerline
camera_unrolled.pcd — unfolded depth-camera point cloud
lidar_unrolled.pcd — unfolded LiDAR point cloud
camera_matched_input_rgb.pcd — matched camera input points
lidar_matched_reference.pcd — matched LiDAR reference points
camera_denoised_matched_rgb.pcd — denoised matched camera points
camera_denoised_final_rgb.pcd — final denoised 3D point cloud
Example Results

You can place representative figures here, for example:

raw depth-camera point cloud
unfolded dual-source point clouds
matched point pairs
denoised unfolded point cloud
final remapped denoised point cloud

Example section:

## Visualization

| Raw Input | Unfolded Domain | Final Output |
|----------|------------------|--------------|
| ![](docs/raw.png) | ![](docs/unfolded.png) | ![](docs/final.png) |

##Dataset

The point cloud datasets associated with this work include:

real tunnel point clouds
low-light scenes
strong-light scenes
dust-fog interference scenes

The dataset contains both before-processing and after-processing point clouds, including intermediate results and final denoised outputs.

More complete datasets can be found in the files submitted to IEEE DataPort. The corresponding link is provided in the paper.

Dataset DOI: 10.21227/7ba7-tv69
Dataset URL: https://ieee-dataport.org/documents/geometry-constrained-camera-and-lidar-fusion-underground-confined-spaces-1

##Notes
The input point clouds must already be aligned in a common coordinate system.
This repository currently provides offline file-based processing, not real-time ROS topic subscription.
The current implementation uses a straight LiDAR centerline model.
For robust matching, reasonable registration quality and overlap between the two modalities are recommended.
Citation

If you use this repository in your research, please cite:(waiting)


##License

This project is released under the MIT License unless otherwise specified.

##Contact

Hongtao Yang
School of Mechatronics Engineering
Anhui University of Science and Technology
Huainan 232001, China
Email: lloid@163.com

