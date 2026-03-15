# ROS Warehouse Robot

基于ROS的智能仓库物流机器人系统，实现包裹自动识别、抓取、分类运输功能。

## 项目概述

本项目是一个完整的仓库物流机器人仿真系统，机器人在Gazebo仿真环境中自主导航，识别不同颜色的包裹并将其运送到指定区域。系统集成了视觉处理、路径规划、运动控制和机械臂操作等功能模块。作为非机器人专业的智能机器人系统的课程设计存在。

### 主要功能

- **自主导航**: 基于move\_base的路径规划与导航
- **车道线跟随**: 视觉引导的车道线检测与跟随
- **包裹识别**: 基于颜色特征的包裹检测与分类
- **机械臂操作**: 自动抓取与放置包裹
- **多区域配送**: 支持红、蓝、绿、紫四个配送区域
- **状态监控**: 实时监控GUI界面

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Controller                          │
│                   (状态机主控制器)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Navigation   │ │ Manipulation  │ │    Vision     │
│   Manager     │ │   Manager     │ │   Processors  │
│  (导航管理)   │ │  (操作管理)   │ │  (视觉处理)   │
└───────────────┘ └───────────────┘ └───────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Gazebo Simulation                         │
│                   (仿真环境与机器人)                          │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
ros_warehouse_robot/
├── src/
│   ├── config/              # 配置文件
│   │   ├── amcl_params.yaml           # AMCL定位参数
│   │   ├── costmap_common_params.yaml # 代价地图通用参数
│   │   ├── global_costmap_params.yaml # 全局代价地图参数
│   │   ├── local_costmap_params.yaml  # 局部代价地图参数
│   │   ├── move_base.yaml             # move_base参数
│   │   ├── zone_positions.yaml        # 区域位置配置
│   │   ├── lane_config.yaml           # 车道线配置
│   │   └── front_cam_config.yaml      # 前置摄像头配置
│   │
│   ├── launch/              # 启动文件
│   │   ├── warehouse_simulation.launch # 完整仿真启动
│   │   ├── main_controller.launch      # 主控制器启动
│   │   ├── navigation.launch           # 导航系统启动
│   │   └── mapping_checker.launch      # 建图检查
│   │
│   ├── scripts/             # Python脚本
│   │   ├── main_controller.py          # 主控制器(状态机)
│   │   ├── navigation_manager.py       # 导航管理器
│   │   ├── manipulation_manager.py     # 操作管理器
│   │   ├── package_detector.py         # 包裹检测器
│   │   ├── lane_detector.py            # 车道线检测器
│   │   ├── front_camera_processor.py   # 前置摄像头处理
│   │   ├── overhead_camera_processor.py # 俯视摄像头处理
│   │   ├── sensor_fusion.py            # 传感器融合
│   │   ├── location_checker.py         # 位置检查器
│   │   ├── monitor_gui.py              # 监控界面
│   │   └── test/                       # 测试脚本
│   │
│   ├── urdf/                # 机器人模型
│   │   ├── warehouse_robot_with_arm.urdf    # 机器人URDF
│   │   └── warehouse_robot_with_arm.gazebo  # Gazebo配置
│   │
│   ├── worlds/              # Gazebo世界
│   │   └── warehouse.world             # 仓库环境
│   │
│   ├── maps/                # 地图文件
│   │   ├── warehouse_map.yaml          # 地图配置
│   │   └── warehouse_map.pgm           # 地图图像
│   │
│   └── models/              # Gazebo模型
│       ├── package_box/                # 包裹模型
│       └── shelf/                      # 货架模型
│
├── build/                   # 构建目录
├── devel/                   # 开发目录
├── CMakeLists.txt
└── package.xml
```

## 环境要求

### 软件依赖

- **操作系统**: Ubuntu 18.04 / Ubuntu 20.04
- **ROS版本**: ROS Melodic / ROS Noetic
- **Python**: Python 2.7

### ROS功能包依赖

```bash
sudo apt-get install ros-${ROS_DISTRO}-gazebo-ros
sudo apt-get install ros-${ROS_DISTRO}-gazebo-ros-control
sudo apt-get install ros-${ROS_DISTRO}-move-base
sudo apt-get install ros-${ROS_DISTRO}-amcl
sudo apt-get install ros-${ROS_DISTRO}-map-server
sudo apt-get install ros-${ROS_DISTRO}-gmapping
sudo apt-get install ros-${ROS_DISTRO}-robot-state-publisher
sudo apt-get install ros-${ROS_DISTRO}-joint-state-publisher
sudo apt-get install ros-${ROS_DISTRO}-tf
sudo apt-get install ros-${ROS_DISTRO}-cv-bridge
sudo apt-get install ros-${ROS_DISTRO}-image-transport
sudo apt-get install ros-${ROS_DISTRO}-diagnostic-aggregator
```

### Python依赖

```bash
pip install opencv-python numpy
```

## 编译与运行

### 1. 编译项目

```bash
cd ros_warehouse_robot
catkin_make
source devel/setup.bash
```

### 2. 启动完整仿真

```bash
roslaunch ros_warehouse_robot warehouse_simulation.launch
```

### 3. 仅启动主控制器

```bash
roslaunch ros_warehouse_robot main_controller.launch
```

### 4. 仅启动导航系统

```bash
roslaunch ros_warehouse_robot navigation.launch
```

## 工作流程

机器人执行以下状态循环：

```
INIT → NAVIGATE_TO_PICKUP → DETECT_PACKAGE_COLOR → PICK_PACKAGE 
    → EXIT_PICKUP_ZONE → NAVIGATE_TO_DROP → PLACE_PACKAGE 
    → RETURN_TO_PICKUP → (循环) → COMPLETED
```

### 状态说明

| 状态                     | 描述           |
| ---------------------- | ------------ |
| INIT                   | 初始化状态，等待系统就绪 |
| NAVIGATE\_TO\_PICKUP   | 导航到取货区域      |
| DETECT\_PACKAGE\_COLOR | 检测包裹颜色       |
| PICK\_PACKAGE          | 抓取包裹         |
| EXIT\_PICKUP\_ZONE     | 安全退出取货区域     |
| NAVIGATE\_TO\_DROP     | 导航到对应颜色配送区域  |
| PLACE\_PACKAGE         | 放置包裹         |
| RETURN\_TO\_PICKUP     | 返回取货区域继续工作   |
| COMPLETED              | 任务完成         |
| ERROR                  | 错误状态，自动恢复    |

## 配置说明

### 区域配置

在 `config/zone_positions.yaml` 中定义：

```yaml
pickup_zone:
  center: [0.0, -1.5]
  radius: 1.5

drop_zones:
  red:
    center: [1.0, 5.0]
    radius: 0.9
  blue:
    center: [3.0, 5.0]
    radius: 0.9
  green:
    center: [-5.0, 1.0]
    radius: 0.9
  purple:
    center: [5.0, 1.0]
    radius: 0.9
```

### 导航参数

在 `config/move_base.yaml` 和 `config/costmap_*.yaml` 中配置：

- 代价地图分辨率
- 障碍物膨胀半径
- 路径规划算法参数

## 话题与服务

### 发布话题

| 话题                       | 消息类型                 | 描述      |
| ------------------------ | -------------------- | ------- |
| `/robot_state`           | std\_msgs/String     | 机器人当前状态 |
| `/current_package_color` | std\_msgs/String     | 当前包裹颜色  |
| `/cmd_vel`               | geometry\_msgs/Twist | 速度命令    |
| `/navigation/command`    | std\_msgs/String     | 导航命令    |
| `/manipulation/command`  | std\_msgs/String     | 操作命令    |

### 订阅话题

| 话题                     | 消息类型                                     | 描述       |
| ---------------------- | ---------------------------------------- | -------- |
| `/current_zone`        | std\_msgs/String                         | 当前所在区域   |
| `/package_color`       | std\_msgs/String                         | 检测到的包裹颜色 |
| `/navigation/arrived`  | std\_msgs/String                         | 导航到达状态   |
| `/manipulation/status` | std\_msgs/String                         | 操作状态     |
| `/amcl_pose`           | geometry\_msgs/PoseWithCovarianceStamped | AMCL位姿   |

## 测试与诊断

### 运行诊断工具

```bash
rosrun ros_warehouse_robot NavigationDiagnostics.py
rosrun ros_warehouse_robot CostmapAnalyzer.py
rosrun ros_warehouse_robot MoveBaseStressTest.py
```

### 查看日志

```bash
rostopic echo /robot_state
rostopic echo /diagnostics_agg
```

## 许可证

MIT License

