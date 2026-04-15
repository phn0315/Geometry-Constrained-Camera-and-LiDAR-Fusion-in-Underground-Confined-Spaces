# tunnel_fusion_denoise

ROS package for geometry-constrained fusion denoising of a registered depth-camera point cloud using a registered single-line LiDAR point cloud.

## Features
- Straight centerline estimation from LiDAR points
- Dual-source point cloud unrolling into a common parameter domain
- Cross-modal correspondence construction
- Ensemble residual regression with entropy filtering
- 3D inverse mapping and PCD export of intermediate and final results

## Package structure
```text
ros_pkg_ws/
└── src/
    └── tunnel_fusion_denoise/
        ├── CMakeLists.txt
        ├── package.xml
        ├── README.md
        ├── LICENSE
        ├── requirements.txt
        ├── config/
        │   └── default.yaml
        ├── launch/
        │   └── offline_denoise.launch
        └── scripts/
            └── tunnel_fusion_denoise_node.py
```

## Dependencies
### ROS
- ROS Noetic (recommended)
- catkin
- rospy

### Python
```bash
pip install -r requirements.txt
```

## Build
```bash
cd ros_pkg_ws
catkin_make
source devel/setup.bash
```

## Run
```bash
roslaunch tunnel_fusion_denoise offline_denoise.launch \
  camera_pcd:=/absolute/path/to/CA.pcd \
  lidar_pcd:=/absolute/path/to/LD.pcd \
  output_dir:=/absolute/path/to/results
```

## Output files
The node exports the following files:
- `lidar_centerline.pcd`
- `camera_unrolled.pcd`
- `lidar_unrolled.pcd`
- `camera_matched_input_rgb.pcd`
- `lidar_matched_reference.pcd`
- `camera_denoised_matched_rgb.pcd`
- `camera_denoised_final_rgb.pcd`

## Input requirement
The camera and LiDAR point clouds must already be registered into a common coordinate frame before running this node.

## Notes for open-source release
- Replace all private paths with generic paths.
- Provide at least one small public sample dataset.
- Add a citation section if the code corresponds to a paper.
- Include a version tag when publishing on GitHub or Zenodo.
