# C<sup>3</sup>P-VoxelMap

This repository implements **C<sup>3</sup>P-VoxelMap**, a compact, cumulative, and coalescible probabilistic voxel mapping method to enhance performance, accuracy, and memory efficiency in LiDAR odometry. Based on [VoxelMap](https://github.com/hku-mars/VoxelMap), our work reduces memory consumption by two strategies: 
1. Compact point-free representation for probabilistic voxels and cumulative update of the planar uncertainty without caching original point clouds.
2. On-demand voxel merging by taking advantage of the geometric features in the real world, accumulating voxels in a locality-sensitive hash and triggers merging lazily.
  
The paper is available on **arxiv**: 
[C<sup>3</sup>P-VoxelMap: Compact, Cumulative and Coalescible Probabilistic Voxel Mapping](https://arxiv.org/abs/2406.01195)

## Dependency

ROS (tested on Noetic)

PCL (>= 1.8)

Eigen (>= 3.3.4)

livox_ros_driver, follow [livox_ros_driver Installation](https://github.com/Livox-SDK/livox_ros_driver).

## Compilation

Clone the repository and compile it by catkin_make:
```
    mkdir -P ~/catkin_ws/src & cd ~/catkin_ws/src
    git clone https://github.com/deptrum/c3p-voxelmap.git
    cd ..
    catkin_make
    source devel/setup.bash
```

## Running on Dataset

To run on dataset (KITTI dataset for example), firstly edit configuration file ``` config/velodyne.yaml ```. 

Besides general parameters such as point cloud topic name, there are some extra configutations concerning on-demand voxel merging. To enable voxel merging, set ```voxel_merging/enable_voxel_merging``` to ```true```. The other parameters in ```voxel_merging``` can also be configured to adjust the effectiveness of on-demand voxel merging. 

Optionally, set ```visualization/pub_merged_voxel``` to ```true``` to visualize voxel merging results. Merged voxel planes are shown with the same color in visualization.

After setting parameters, run the ROS package:
```
    cd ~/catkin_ws
    source devel/setup.bash
    roslaunch c3p_voxelmap mapping_velodyne.launch
```

In the meanwhile, play rosbag and the visualization results will be shown in RViz window.

## Acknowledgments
Thanks for [VoxelMap](https://github.com/hku-mars/VoxelMap).