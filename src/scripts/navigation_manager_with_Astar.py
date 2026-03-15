#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import math
import cv2
import numpy as np
import time
import heapq
from std_msgs.msg import String, Float32, Bool
from geometry_msgs.msg import Twist, PoseStamped, Point, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import GetModelState
from collections import deque

class IntegratedNavigationController:
    def __init__(self):
        rospy.init_node('integrated_navigation_controller', anonymous=True)
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 等待Gazebo服务获取准确坐标
        rospy.loginfo("Waiting for Gazebo get_model_state service...")
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=10.0)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.gazebo_available = True
            rospy.loginfo("Gazebo service connected")
        except:
            rospy.logwarn("Gazebo service not available, using AMCL coordinates")
            self.gazebo_available = False
            self.get_model_state = None
        
        # 加载控制参数
        self.max_linear_vel = rospy.get_param('/robot_control/max_linear_vel', 0.8)
        self.max_angular_vel = rospy.get_param('/robot_control/max_angular_vel', 1.2)
        self.position_tolerance = rospy.get_param('/robot_control/position_tolerance', 0.25)
        self.angle_tolerance = rospy.get_param('/robot_control/angle_tolerance', 0.15)
        
        # 车道线检测参数
        self.yellow_lower = np.array([15, 80, 80])
        self.yellow_upper = np.array([35, 255, 255])
        
        # 车道约束参数
        self.max_lane_deviation = 0.12
        self.lane_correction_gain = 2.0
        self.emergency_lane_correction_gain = 3.0
        self.lane_following_weight = 0.8
        self.waypoint_guidance_weight = 0.2
        
        # 车道线检测状态
        self.min_lane_confidence = 0.4
        self.lane_lost_timeout = 2.5
        self.last_lane_detection_time = rospy.Time.now()
        
        # 安全距离参数
        self.safe_distance_to_packages = 0.8
        self.emergency_stop_distance = 0.25
        self.slow_down_distance = 1.0
        
        # Pickup区域参数
        self.pickup_zone_center = [0.0, -1.5]
        self.pickup_zone_radius = 3.5
        self.pickup_safe_stop_distance = 0.35
        self.pickup_creep_speed = 0.15
        self.pickup_mode_active = False
        
        # 预定义车道线路径
        self.lane_waypoints = {
            'pickup_to_red': [
                [0.0, 1.0, 0.0],     # 从pickup沿主车道线到安全点
                [0.0, 3.0, 0.0],     # 沿主车道线北上到y=3
                [0.0, 5.0, 0.0],     # 沿主车道线北上到y=5
                [1.0, 5.0, 0.0]      # 直接到red区域(1,5)
            ],
            'pickup_to_blue': [
                [0.0, 1.0, 0.0],     
                [0.0, 3.0, 0.0],     
                [0.0, 5.0, 0.0],     
                [3.0, 5.0, 0.0]      # 直接到blue区域(3,5)
            ],
            'pickup_to_green': [
                [0.0, 1.0, 0.0],     
                [-5.0, 1.0, -1.5708] # 沿y=1水平线到green区域(-5,1)
            ],
            'pickup_to_purple': [
                [0.0, 1.0, 0.0],     
                [5.0, 1.0, 1.5708]   # 沿y=1水平线到purple区域(5,1)
            ],
            # 快速EXIT路径
            'pickup_to_safety': [
                [0.0, 0.0, 0.0],    # pickup边缘
                [0.0, 0.5, 0.0],    # 中间过渡点
                [0.0, 1.0, 0.0]     # 安全检查点[0, 1.0]
            ],
            # 返回路径
            'red_to_pickup': [
                [1.0, 5.0, 3.14159],    # 从red区域开始
                [0.0, 5.0, 3.14159],    # 到主车道线y=5
                [0.0, 3.0, 3.14159],    # 沿主车道线南下到y=3
                [0.0, 1.0, 3.14159],    # 到达安全点[0, 1.0]
                [0.0, 0.5, 3.14159],    # 继续向南
                [0.0, -0.5, 3.14159]    # 到达pickup接近点
            ],
            'blue_to_pickup': [
                [3.0, 5.0, 3.14159],    
                [0.0, 5.0, 3.14159],    
                [0.0, 3.0, 3.14159],    
                [0.0, 1.0, 3.14159],    # 到达安全点[0, 1.0]
                [0.0, 0.5, 3.14159],    
                [0.0, -0.5, 3.14159]    
            ],
            'green_to_pickup': [
                [-5.0, 1.0, 1.5708],    
                [0.0, 1.0, 1.5708],     # 沿y=1水平线到主车道线
                [0.0, 0.5, 3.14159],    # 转向南下
                [0.0, -0.5, 3.14159]    # 到达pickup接近点
            ],
            'purple_to_pickup': [
                [5.0, 1.0, -1.5708],    
                [0.0, 1.0, -1.5708],    # 沿y=1水平线到主车道线
                [0.0, 0.5, 3.14159],    # 转向南下
                [0.0, -0.5, 3.14159]    # 到达pickup接近点
            ]
        }
        
        # 备用返回路径
        self.backup_return_paths = {
            'red_to_pickup': [
                [0.0, 4.0, 3.14159],    # 从red区域快速到主车道线
                [0.0, 1.0, 3.14159],    # 安全点[0, 1.0]
                [0.0, -0.5, 3.14159]    # 到pickup
            ],
            'blue_to_pickup': [
                [0.0, 4.0, 3.14159],    
                [0.0, 1.0, 3.14159],    
                [0.0, -0.5, 3.14159]    
            ],
            'green_to_pickup': [
                [-3.0, 1.0, 1.5708],    # 中间点
                [0.0, 1.0, 1.5708],     # 安全点[0, 1.0]
                [0.0, -0.5, 3.14159]    # 到pickup
            ],
            'purple_to_pickup': [
                [3.0, 1.0, -1.5708],    # 中间点
                [0.0, 1.0, -1.5708],    # 安全点[0, 1.0]
                [0.0, -0.5, 3.14159]    # 到pickup
            ]
        }
        
        # 当前状态
        self.current_position = None
        self.current_orientation = None
        self.current_goal = None
        self.path = []
        self.path_index = 0
        self.obstacle_detected = False
        self.package_nearby = False
        self.in_pickup_area = False
        self.navigation_active = False
        
        # 导航状态
        self.navigation_timeout = 45.0
        self.navigation_start_time = None
        self.robot_state = "UNKNOWN"
        self.goal_type = "NONE"
        
        # 车道跟随状态
        self.lane_following_active = True
        self.strict_lane_following = True
        
        # 包裹检测和避障
        self.package_positions = []
        
        # 激光雷达数据处理
        self.laser_data = None
        self.obstacle_zones = {
            'front': {'min_angle': -0.3, 'max_angle': 0.3, 'min_distance': float('inf')},
            'front_left': {'min_angle': 0.2, 'max_angle': 0.8, 'min_distance': float('inf')},
            'front_right': {'min_angle': -0.8, 'max_angle': -0.2, 'min_distance': float('inf')}
        }
        
        # 统一坐标系统
        self.robot_amcl_pose = None
        self.robot_odom_pose = None
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, yaw]
        self.coordinate_source = "none"
        
        # 车道线检测状态
        self.lane_center_x = None
        self.image_center_x = 320  # 640/2
        self.lane_detection_active = False
        self.last_lane_time = 0
        self.lane_timeout = 2.0
        
        # 车道线跟随控制
        self.lane_guidance_active = True
        self.gentle_correction_enabled = True
        
        # PID参数
        self.kp = 0.4
        self.ki = 0.05
        self.kd = 0.1
        
        # PID状态变量
        self.last_lane_error = 0.0
        self.integral_error = 0.0
        self.lane_history = []
        self.max_correction_angular = 0.4
        
        # 恢复系统
        self.recovery_active = False
        self.recovery_state = "NONE"
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.stuck_threshold = 0.08
        self.stuck_timer = 0
        self.max_stuck_time = 15.0
        
        # 位置历史
        self.position_history = deque(maxlen=20)
        
        # 返回导航控制
        self.force_return_mode = False
        self.return_waypoint_index = 0
        self.max_return_time = 60.0
        self.waypoint_reached_threshold = 0.4
        
        # 返回尝试控制
        self.return_attempt_count = 0
        self.max_return_attempts = 2
        self.using_backup_path = False
        self.return_start_time = None
        
        # 安全点检查
        self.safety_checkpoint_reached = False
        self.safety_checkpoint_position = [0.0, 1.0]
        self.safety_checkpoint_tolerance = 0.5
        
        # A*算法参数
        self.map_width = 15   # -7.5 to 7.5
        self.map_height = 12  # -4 to 8
        self.grid_resolution = 0.2  # 20cm分辨率
        self.map_origin_x = -7.5
        self.map_origin_y = -4.0
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.goal_reached_pub = rospy.Publisher('/goal_reached', Bool, queue_size=1)
        self.current_waypoint_pub = rospy.Publisher('/current_waypoint', String, queue_size=1)
        self.navigation_status_pub = rospy.Publisher('/navigation_status', String, queue_size=1)
        self.lane_error_pub = rospy.Publisher('/lane_error', Float32, queue_size=1)
        self.lane_detected_pub = rospy.Publisher('/lane_detected', Bool, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/lane_debug_image', Image, queue_size=1)
        
        # Subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.goal_sub = rospy.Subscriber('/move_to_goal', String, self.goal_callback)
        self.package_detected_sub = rospy.Subscriber('/package_detected', Bool, self.package_detected_callback)
        self.package_position_sub = rospy.Subscriber('/package_position', Point, self.package_position_callback)
        self.robot_state_sub = rospy.Subscriber('/robot_state', String, self.robot_state_callback)
        self.amcl_pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        self.camera_sub = rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        
        # TF listener
        self.tf_listener = tf.TransformListener()
        
        # 控制变量
        self.lane_error = 0.0
        self.lane_detected = False
        self.lane_confidence = 1.0
        self.last_lane_error = 0.0
        self.lane_error_history = deque(maxlen=8)
        
        # 等待初始位置
        rospy.loginfo("Waiting for robot position...")
        while self.current_position is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        rospy.loginfo("Integrated Navigation Controller initialized")
        rospy.loginfo("Robot position: (%.2f, %.2f)", self.current_position[0], self.current_position[1])
        
        # 控制定时器
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_callback)
        self.lane_guidance_timer = rospy.Timer(rospy.Duration(0.1), self.lane_guidance_loop)
        
    def pose_callback(self, msg):
        """处理位姿信息，优先使用Gazebo坐标"""
        # 优先使用Gazebo准确坐标
        if self.gazebo_available:
            try:
                response = self.get_model_state("warehouse_robot", "")
                if response.success:
                    pos = response.pose.position
                    orient = response.pose.orientation
                    yaw = self.get_yaw_from_quaternion(orient)
                    self.current_position = [pos.x, pos.y, yaw]
                    self.coordinate_source = "gazebo"
                    return
            except Exception as e:
                rospy.logdebug("Gazebo coordinate access failed: %s", str(e))
        
        # 备用：使用AMCL坐标
        self.robot_amcl_pose = msg.pose.pose
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        yaw = self.get_yaw_from_quaternion(orient)
        self.current_position = [pos.x, pos.y, yaw]
        self.coordinate_source = "amcl"
        
    def odom_callback(self, msg):
        """处理里程计数据"""
        self.robot_odom_pose = msg.pose.pose
        
        # 如果amcl_pose不可用，使用odom作为备用
        if self.robot_amcl_pose is None and not self.gazebo_available:
            pos = msg.pose.pose.position
            orient = msg.pose.pose.orientation
            yaw = self.get_yaw_from_quaternion(orient)
            self.current_position = [pos.x, pos.y, yaw]
            self.coordinate_source = "odom"
        
        # 更新位置历史
        if self.current_position:
            self.position_history.append({
                'position': self.current_position[:],
                'timestamp': rospy.Time.now()
            })
        
        # 检查是否在pickup区域
        if self.current_position:
            distance_to_pickup = self.calculate_distance(self.current_position, self.pickup_zone_center)
            was_in_pickup = self.in_pickup_area
            self.in_pickup_area = distance_to_pickup < self.pickup_zone_radius
            
            if self.in_pickup_area and not was_in_pickup:
                rospy.loginfo("Navigation: Entered pickup area")
            elif not self.in_pickup_area and was_in_pickup:
                rospy.loginfo("Navigation: Left pickup area")
        
        # 运动检测
        self.update_motion_detection()
        
    def camera_callback(self, msg):
        """处理前置摄像头图像进行车道线检测"""
        if not self.lane_guidance_active:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detect_lane_for_guidance(cv_image)
        except Exception as e:
            rospy.logwarn("Lane detection error: %s", str(e))
            
    def detect_lane_for_guidance(self, image):
        """检测车道线用于导航引导"""
        height, width = image.shape[:2]
        
        # 检测图像下半部分
        roi_image = image[height//2:, :]
        
        # 转换到HSV
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # 创建黄色车道线掩码
        mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓 - OpenCV 3.2兼容
        try:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算车道线中心
        lane_center_x = self.calculate_lane_center(contours, roi_image.shape[1])
        
        if lane_center_x is not None:
            self.lane_center_x = lane_center_x
            self.last_lane_time = rospy.Time.now().to_sec()
            self.lane_detection_active = True
            
            # 计算车道线误差
            lane_offset = self.lane_center_x - self.image_center_x
            offset_ratio = lane_offset / float(self.image_center_x)
            self.lane_error = offset_ratio
            
            # 发布车道线检测状态
            lane_detected_msg = Bool()
            lane_detected_msg.data = True
            self.lane_detected_pub.publish(lane_detected_msg)
            
            lane_error_msg = Float32()
            lane_error_msg.data = self.lane_error
            self.lane_error_pub.publish(lane_error_msg)
            
        else:
            # 检查车道线检测超时
            if rospy.Time.now().to_sec() - self.last_lane_time > self.lane_timeout:
                self.lane_detection_active = False
                lane_detected_msg = Bool()
                lane_detected_msg.data = False
                self.lane_detected_pub.publish(lane_detected_msg)
                
        # 创建调试图像
        debug_image = self.create_debug_image(image, mask, lane_center_x)
        self.publish_debug_image(debug_image)
        
    def calculate_lane_center(self, contours, image_width):
        """计算车道线中心"""
        if not contours:
            return None
            
        total_area = 0
        weighted_x_sum = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                # 计算重心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    weighted_x_sum += cx * area
                    total_area += area
        
        if total_area > 0:
            return weighted_x_sum / total_area
        else:
            return None
            
    def lane_guidance_loop(self, event):
        """车道线引导主循环 - PID控制"""
        if not self.navigation_active or not self.gentle_correction_enabled:
            return
            
        if not self.lane_detection_active or self.lane_center_x is None:
            return
            
        # 计算车道线偏移
        lane_offset = self.lane_center_x - self.image_center_x
        offset_ratio = lane_offset / float(self.image_center_x)
        
        # 只在偏移较大时才进行修正
        if abs(offset_ratio) < 0.30:
            return
            
        # PID计算
        current_error = offset_ratio
        
        # 积分项
        self.integral_error += current_error
        # 限制积分饱和
        self.integral_error = max(-1.0, min(1.0, self.integral_error))
        
        # 微分项
        derivative_error = current_error - self.last_lane_error
        self.last_lane_error = current_error
        
        # PID输出
        pid_output = (self.kp * current_error + 
                     self.ki * self.integral_error + 
                     self.kd * derivative_error)
        
        # 角度修正
        angular_correction = -pid_output
        angular_correction = max(-self.max_correction_angular, 
                                min(self.max_correction_angular, angular_correction))
        
        # 发布修正命令
        if abs(angular_correction) > 0.05:
            rospy.logdebug("Lane guidance: offset=%.3f, correction=%.3f", 
                         offset_ratio, angular_correction)
    
    def robot_state_callback(self, msg):
        """处理机器人状态回调"""
        self.robot_state = msg.data
        
        # 根据机器人状态调整参数
        if self.robot_state in ["SEARCHING", "APPROACHING", "PICKING", "DROPPING"]:
            self.pickup_mode_active = True
        else:
            self.pickup_mode_active = False
            
        # 根据状态启用/禁用车道线引导
        navigation_states = [
            "NAVIGATE_TO_PICKUP", "NAVIGATE_TO_DROP", 
            "RETURN_TO_PICKUP", "EXIT_PICKUP_ZONE"
        ]
        self.lane_guidance_active = (self.robot_state in navigation_states)
        
    def laser_callback(self, msg):
        """处理激光雷达数据"""
        self.laser_data = msg
        
        # 重置障碍物区域
        for zone in self.obstacle_zones.values():
            zone['min_distance'] = float('inf')
        
        # 处理激光数据
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        
        package_obstacles = []
        
        for i, (angle, distance) in enumerate(zip(angles, msg.ranges)):
            if msg.range_min < distance < msg.range_max:
                # 分配到不同区域
                for zone_name, zone in self.obstacle_zones.items():
                    if zone['min_angle'] <= angle <= zone['max_angle']:
                        zone['min_distance'] = min(zone['min_distance'], distance)
                
                # 检测包裹障碍物
                if 0.15 < distance < 2.5 and abs(angle) < math.pi/4:
                    if self.is_potential_package_obstacle(msg.ranges, i, distance):
                        package_obstacles.append({
                            'distance': distance,
                            'angle': angle
                        })
        
        # 更新包裹障碍物
        self.update_package_obstacles(package_obstacles)
        
        # 检测前方障碍物
        front_distance = self.obstacle_zones['front']['min_distance']
        
        # 障碍物威胁评估
        self.obstacle_detected = self.evaluate_obstacle_threat(front_distance, package_obstacles)
        
        # 检查是否有包裹在附近
        self.check_packages_nearby()
        
    def is_potential_package_obstacle(self, ranges, index, distance):
        """判断是否可能是包裹障碍物"""
        window_size = 3
        start = max(0, index - window_size)
        end = min(len(ranges), index + window_size + 1)
        
        local_ranges = [r for r in ranges[start:end] if 0.1 < r < 10.0]
        
        if len(local_ranges) < 2:
            return False
            
        std_dev = np.std(local_ranges)
        mean_distance = np.mean(local_ranges)
        
        is_package_like = (
            std_dev < 0.25 and
            0.2 < mean_distance < 3.0 and
            abs(distance - mean_distance) < 0.3
        )
        
        return is_package_like
    
    def update_package_obstacles(self, package_obstacles):
        """更新包裹障碍物列表"""
        current_time = rospy.Time.now()
        
        # 清理旧的检测
        self.package_positions = [p for p in self.package_positions 
                                 if (current_time - p['timestamp']).to_sec() < 5.0]
        
        # 添加新检测的包裹障碍物
        for obstacle in package_obstacles:
            if self.current_position and len(self.current_position) >= 3:
                # 转换到世界坐标
                x = obstacle['distance'] * math.cos(obstacle['angle'])
                y = obstacle['distance'] * math.sin(obstacle['angle'])
                
                world_x = self.current_position[0] + x * math.cos(self.current_position[2]) - y * math.sin(self.current_position[2])
                world_y = self.current_position[1] + x * math.sin(self.current_position[2]) + y * math.cos(self.current_position[2])
                
                # 检查是否已存在
                is_new = True
                for existing in self.package_positions:
                    if self.calculate_distance([world_x, world_y], existing['position']) < 0.3:
                        is_new = False
                        existing['timestamp'] = current_time
                        break
                
                if is_new:
                    self.package_positions.append({
                        'position': [world_x, world_y],
                        'distance': obstacle['distance'],
                        'angle': obstacle['angle'],
                        'timestamp': current_time
                    })
    
    def evaluate_obstacle_threat(self, front_distance, package_obstacles):
        """评估障碍物威胁"""
        # 紧急停止条件
        if front_distance < self.emergency_stop_distance:
            if self.robot_state == "PICKING":
                return front_distance < 0.1
            return True
            
        # 在pickup区域内的特殊处理
        if self.in_pickup_area and self.pickup_mode_active:
            if self.robot_state == "SEARCHING":
                close_packages = [obs for obs in package_obstacles 
                                 if obs['distance'] < 0.3 and abs(obs['angle']) < 0.5]
                return len(close_packages) > 0 and min([pkg['distance'] for pkg in close_packages] + [float('inf')]) < 0.15
            elif self.robot_state == "APPROACHING":
                close_packages = [obs for obs in package_obstacles 
                                 if obs['distance'] < 0.2 and abs(obs['angle']) < 0.3]
                return len(close_packages) > 0 and min([pkg['distance'] for pkg in close_packages] + [float('inf')]) < 0.1
            elif self.robot_state == "PICKING":
                return False
        
        return front_distance < self.safe_distance_to_packages
    
    def check_packages_nearby(self):
        """检查是否有包裹在附近"""
        if not self.current_position:
            self.package_nearby = False
            return
            
        self.package_nearby = False
        
        for package in self.package_positions:
            distance = self.calculate_distance(self.current_position, package['position'])
            if distance < self.slow_down_distance:
                self.package_nearby = True
                break
    
    def package_detected_callback(self, msg):
        """包裹检测回调"""
        pass
    
    def package_position_callback(self, msg):
        """包裹位置回调"""
        pass
    
    def goal_callback(self, msg):
        """目标回调"""
        goal_node = msg.data
        
        rospy.loginfo("Navigation received goal: %s", goal_node)
        
        # 判断目标类型
        if "pickup" in goal_node.lower():
            self.goal_type = "PICKUP"
        elif any(color in goal_node.lower() for color in ["red", "blue", "green", "purple"]):
            self.goal_type = "DELIVERY"
        else:
            self.goal_type = "NONE"
        
        if self.current_position is None:
            rospy.logwarn("Current position unknown, cannot plan path")
            return
        
        # 尝试使用预定义路径，如果失败则使用A*
        success = self.try_predefined_path(goal_node)
        if not success:
            rospy.logwarn("Predefined path failed, using A* algorithm")
            success = self.plan_path_with_astar(goal_node)
            
        if not success:
            rospy.logerr("All path planning methods failed for goal: %s", goal_node)
            
    def try_predefined_path(self, goal_node):
        """尝试使用预定义车道线路径"""
        try:
            # 解析目标命令
            if goal_node == "goto_pickup_approach":
                path_key = None
                # 根据当前位置确定最佳pickup路径
                current_pos = self.current_position
                if current_pos[0] > 4:
                    path_key = 'purple_to_pickup'
                elif current_pos[0] < -4:
                    path_key = 'green_to_pickup'
                elif current_pos[1] > 4:
                    if current_pos[0] > 1:
                        path_key = 'blue_to_pickup'
                    else:
                        path_key = 'red_to_pickup'
                
                if path_key and path_key in self.lane_waypoints:
                    self.setup_waypoint_navigation(self.lane_waypoints[path_key], "pickup_approach")
                    return True
                    
            elif goal_node == "exit_pickup_to_safety":
                self.setup_waypoint_navigation(self.lane_waypoints['pickup_to_safety'], "safety_exit")
                return True
                
            elif goal_node.startswith("goto_") and goal_node.endswith("_drop"):
                color = goal_node.replace("goto_", "").replace("_drop", "")
                path_key = 'pickup_to_' + color
                if path_key in self.lane_waypoints:
                    self.setup_waypoint_navigation(self.lane_waypoints[path_key], color + "_drop")
                    return True
                    
            elif goal_node.startswith("return_from_") and goal_node.endswith("_to_pickup"):
                color = goal_node.replace("return_from_", "").replace("_to_pickup", "")
                path_key = color + '_to_pickup'
                if path_key in self.lane_waypoints:
                    self.setup_waypoint_navigation(self.lane_waypoints[path_key], "return_to_pickup")
                    return True
                elif path_key in self.backup_return_paths:
                    self.setup_waypoint_navigation(self.backup_return_paths[path_key], "return_to_pickup")
                    return True
                    
            return False
            
        except Exception as e:
            rospy.logerr("Predefined path planning error: %s", str(e))
            return False
    
    def setup_waypoint_navigation(self, waypoints, goal_name):
        """设置waypoint导航"""
        # 重置状态
        self.reset_recovery_state()
        self.navigation_start_time = rospy.Time.now()
        
        self.path = waypoints
        self.path_index = 0
        self.current_goal = goal_name
        self.navigation_active = True
        
        rospy.logwarn("Lane-aligned path to %s: %d waypoints", goal_name, len(waypoints))
        for i, wp in enumerate(waypoints):
            rospy.logwarn("  WP %d: [%.2f, %.2f, %.2f]", i+1, wp[0], wp[1], wp[2])
        
        return True
    
    # A*算法实现部分
    def plan_path_with_astar(self, goal_node):
        """使用A*算法进行路径规划"""
        try:
            rospy.logwarn("Starting A* path planning for goal: %s", goal_node)
            
            # 解析目标位置
            target_pos = self.parse_goal_to_position(goal_node)
            if target_pos is None:
                rospy.logerr("Cannot parse goal to position: %s", goal_node)
                return False
            
            start_pos = [self.current_position[0], self.current_position[1]]
            
            # 构建网格地图
            grid_map = self.build_grid_map()
            
            # 运行A*算法
            path = self.astar_search(start_pos, target_pos, grid_map)
            
            if path:
                # 转换路径为waypoints
                waypoints = self.convert_path_to_waypoints(path, target_pos)
                self.setup_waypoint_navigation(waypoints, goal_node)
                rospy.logwarn("A* planning successful: %d waypoints", len(waypoints))
                return True
            else:
                rospy.logerr("A* planning failed: No path found")
                return False
                
        except Exception as e:
            rospy.logerr("A* path planning error: %s", str(e))
            return False
    
    def parse_goal_to_position(self, goal_node):
        """解析目标命令到具体位置"""
        if goal_node == "goto_pickup_approach":
            return [0.0, -0.5]  # pickup接近点
        elif goal_node == "exit_pickup_to_safety":
            return [0.0, 1.0]   # 安全退出点
        elif goal_node.startswith("goto_") and goal_node.endswith("_drop"):
            color = goal_node.replace("goto_", "").replace("_drop", "")
            drop_positions = {
                'red': [1.0, 5.0],
                'blue': [3.0, 5.0], 
                'green': [-5.0, 1.0],
                'purple': [5.0, 1.0]
            }
            return drop_positions.get(color)
        elif goal_node.startswith("return_from_") and goal_node.endswith("_to_pickup"):
            return [0.0, -0.5]  # 返回pickup
        
        return None
    
    def build_grid_map(self):
        """构建用于A*算法的网格地图"""
        grid_width = int(self.map_width / self.grid_resolution)
        grid_height = int(self.map_height / self.grid_resolution)
        
        # 初始化网格（0=自由空间，1=障碍物）
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # 添加静态障碍物
        self.add_static_obstacles_to_grid(grid, self.map_origin_x, self.map_origin_y, self.grid_resolution)
        
        # 添加动态障碍物（包裹等）
        self.add_dynamic_obstacles_to_grid(grid, self.map_origin_x, self.map_origin_y, self.grid_resolution)
        
        return {
            'grid': grid,
            'width': grid_width,
            'height': grid_height,
            'resolution': self.grid_resolution,
            'origin_x': self.map_origin_x,
            'origin_y': self.map_origin_y
        }
    
    def add_static_obstacles_to_grid(self, grid, origin_x, origin_y, resolution):
        """添加静态障碍物到网格地图"""
        # 货架位置 - 从warehouse.world文件
        shelves = [
            [-3, -6], [-1, -6], [1, -6], [3, -6],  # 南侧货架
            [-3, 6], [-1, 6]                        # 北侧货架
        ]
        
        for shelf_x, shelf_y in shelves:
            # 货架尺寸: 1.5x0.5m
            self.add_rectangle_obstacle(grid, origin_x, origin_y, resolution,
                                      shelf_x, shelf_y, 1.5, 0.5)
        
        # 地图边界
        self.add_map_boundaries(grid)
    
    def add_rectangle_obstacle(self, grid, origin_x, origin_y, resolution, 
                              center_x, center_y, width, height):
        """在网格中添加矩形障碍物"""
        # 转换到网格坐标
        grid_x = int((center_x - origin_x) / resolution)
        grid_y = int((center_y - origin_y) / resolution)
        
        grid_width = int(width / resolution)
        grid_height = int(height / resolution)
        
        # 添加障碍物
        start_x = max(0, grid_x - grid_width//2)
        end_x = min(grid.shape[1], grid_x + grid_width//2 + 1)
        start_y = max(0, grid_y - grid_height//2)
        end_y = min(grid.shape[0], grid_y + grid_height//2 + 1)
        
        grid[start_y:end_y, start_x:end_x] = 1
    
    def add_map_boundaries(self, grid):
        """添加地图边界"""
        # 边界为障碍物
        grid[0, :] = 1      # 上边界
        grid[-1, :] = 1     # 下边界
        grid[:, 0] = 1      # 左边界
        grid[:, -1] = 1     # 右边界
    
    def add_dynamic_obstacles_to_grid(self, grid, origin_x, origin_y, resolution):
        """添加动态障碍物到网格地图"""
        # 添加检测到的包裹位置
        for package in self.package_positions:
            pos = package['position']
            grid_x = int((pos[0] - origin_x) / resolution)
            grid_y = int((pos[1] - origin_y) / resolution)
            
            # 包裹周围小区域标记为障碍
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    gx, gy = grid_x + dx, grid_y + dy
                    if 0 <= gx < grid.shape[1] and 0 <= gy < grid.shape[0]:
                        grid[gy, gx] = 1
    
    def astar_search(self, start_pos, goal_pos, grid_map):
        """A*算法搜索最优路径"""
        grid = grid_map['grid']
        resolution = grid_map['resolution']
        origin_x = grid_map['origin_x']
        origin_y = grid_map['origin_y']
        
        # 转换起点和终点到网格坐标
        start_grid = (
            int((start_pos[0] - origin_x) / resolution),
            int((start_pos[1] - origin_y) / resolution)
        )
        goal_grid = (
            int((goal_pos[0] - origin_x) / resolution),
            int((goal_pos[1] - origin_y) / resolution)
        )
        
        # 验证起点和终点
        if not self.is_valid_grid_point(start_grid, grid):
            rospy.logwarn("Invalid start point in grid")
            return None
        if not self.is_valid_grid_point(goal_grid, grid):
            rospy.logwarn("Invalid goal point in grid")
            return None
        
        # A*算法核心
        open_set = []
        closed_set = set()
        came_from = {}
        
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic_distance(start_grid, goal_grid)}
        
        heapq.heappush(open_set, (f_score[start_grid], start_grid))
        
        # 8方向移动
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal_grid:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                return list(reversed(path))
            
            closed_set.add(current)
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_grid_point(neighbor, grid):
                    continue
                if neighbor in closed_set:
                    continue
                
                # 计算移动代价
                if abs(dx) + abs(dy) == 2:  # 对角线移动
                    move_cost = 1.414
                else:  # 直线移动
                    move_cost = 1.0
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic_distance(neighbor, goal_grid)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 未找到路径
    
    def is_valid_grid_point(self, point, grid):
        """检查网格点是否有效"""
        x, y = point
        if x < 0 or x >= grid.shape[1] or y < 0 or y >= grid.shape[0]:
            return False
        return grid[y, x] == 0  # 0表示自由空间
    
    def heuristic_distance(self, point1, point2):
        """启发式距离函数（欧几里得距离）"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def convert_path_to_waypoints(self, grid_path, goal_pos):
        """将网格路径转换为世界坐标waypoints"""
        waypoints = []
        
        # 路径简化 - 移除不必要的中间点
        simplified_path = self.simplify_path(grid_path)
        
        for i, (grid_x, grid_y) in enumerate(simplified_path):
            # 转换回世界坐标
            world_x = grid_x * self.grid_resolution + self.map_origin_x
            world_y = grid_y * self.grid_resolution + self.map_origin_y
            
            # 计算朝向角度
            if i < len(simplified_path) - 1:
                next_point = simplified_path[i + 1]
                angle = math.atan2(next_point[1] - grid_y, next_point[0] - grid_x)
            else:
                # 最后一个点，面向目标
                angle = 0.0
            
            waypoints.append([world_x, world_y, angle])
        
        return waypoints
    
    def simplify_path(self, path):
        """简化路径，移除共线的中间点"""
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]
            
            # 检查是否共线
            if not self.points_collinear(prev_point, curr_point, next_point):
                simplified.append(curr_point)
        
        simplified.append(path[-1])
        return simplified
    
    def points_collinear(self, p1, p2, p3):
        """检查三点是否共线"""
        # 使用叉积判断
        return abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) < 0.1
    
    # 其他必要的方法
    def update_motion_detection(self):
        """更新运动检测和卡住检测"""
        if len(self.position_history) < 10:
            return
            
        # 检查最近的运动
        recent_positions = [p['position'] for p in list(self.position_history)[-10:]]
        max_distance = 0
        for i in range(len(recent_positions)-1):
            for j in range(i+1, len(recent_positions)):
                dist = self.calculate_distance(recent_positions[i], recent_positions[j])
                max_distance = max(max_distance, dist)
        
        # 卡住检测
        if (max_distance < self.stuck_threshold and 
            self.navigation_active and 
            self.robot_state not in ["PICKING", "DROPPING"]):
            
            self.stuck_timer += 0.1
            if self.stuck_timer > self.max_stuck_time and not self.recovery_active:
                rospy.logwarn("Robot stuck! Max movement: %.3fm", max_distance)
                self.initiate_recovery()
        else:
            self.stuck_timer = 0
    
    def initiate_recovery(self):
        """启动恢复程序"""
        self.recovery_active = True
        self.recovery_state = "LANE_RECOVERY"
        self.recovery_attempts += 1
        
        rospy.logwarn("Starting recovery attempt %d/%d", 
                     self.recovery_attempts, self.max_recovery_attempts)
    
    def reset_recovery_state(self):
        """重置恢复状态"""
        self.recovery_active = False
        self.recovery_state = "NONE"
        self.recovery_attempts = 0
        self.stuck_timer = 0
    
    def execute_recovery(self):
        """执行恢复操作"""
        if not self.recovery_active:
            return 0.0, 0.0
            
        if self.recovery_state == "LANE_RECOVERY":
            if self.lane_detection_active:
                rospy.loginfo("Recovery: Lane-based adjustment")
                linear_vel, angular_vel = self.calculate_lane_following_velocity()
                return linear_vel * 0.3, angular_vel * 2.0
            else:
                self.recovery_state = "GENTLE_BACKUP"
                return 0.0, 0.0
                
        elif self.recovery_state == "GENTLE_BACKUP":
            rospy.loginfo("Recovery: Gentle backup")
            return -0.1, 0.0
                
        elif self.recovery_state == "SMALL_ROTATE":
            rospy.loginfo("Recovery: Small rotation")
            return 0.0, 0.2 if self.recovery_attempts % 2 else -0.2
        
        # 恢复失败
        if self.recovery_attempts >= self.max_recovery_attempts:
            rospy.logerr("Recovery failed, stopping")
            self.reset_recovery_state()
            return 0.0, 0.0
            
        return 0.0, 0.0
    
    def calculate_safe_velocity(self):
        """计算安全速度"""
        base_linear = self.max_linear_vel
        base_angular = self.max_angular_vel
        
        # 根据机器人状态调整基础速度
        if self.robot_state == "DELIVERING":
            base_linear *= 1.2
            base_angular *= 1.1
        elif self.robot_state == "SEARCHING":
            base_linear *= 0.8
        elif self.robot_state == "APPROACHING":
            base_linear *= 0.6
        elif self.robot_state == "PICKING":
            base_linear *= 0.3
        
        # 车道偏离惩罚
        if self.lane_detection_active:
            lane_deviation_factor = max(0.4, 1.0 - abs(self.lane_error) * 3.0)
            base_linear *= lane_deviation_factor
            
            if abs(self.lane_error) > self.max_lane_deviation:
                base_linear *= 0.3
                base_angular *= 0.8
        
        # Pickup区域处理
        if self.in_pickup_area:
            base_linear *= 0.8
            base_angular *= 0.9
            
            if self.package_nearby and self.robot_state not in ["PICKING", "DELIVERING"]:
                base_linear *= 0.5
                base_angular *= 0.7
        elif self.package_nearby and self.robot_state != "DELIVERING":
            base_linear *= 0.8
            base_angular *= 0.9
            
        # 前方障碍物处理
        front_distance = self.obstacle_zones['front']['min_distance']
        if front_distance < self.slow_down_distance and self.robot_state != "DELIVERING":
            distance_factor = max(0.2, front_distance / self.slow_down_distance)
            base_linear *= distance_factor
            
        return base_linear, base_angular
    
    def calculate_distance(self, pos1, pos2):
        """计算距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_lane_following_velocity(self):
        """计算车道跟随速度"""
        if not self.lane_detection_active:
            return 0.0, 0.0
           
        # PD控制器
        error_derivative = 0.0
        if len(self.lane_error_history) > 1:
            error_derivative = self.lane_error - self.lane_error_history[-2]
       
        # 自适应增益
        if abs(self.lane_error) > self.max_lane_deviation:
            gain = self.emergency_lane_correction_gain
        else:
            gain = self.lane_correction_gain
       
        # 角速度控制
        angular_vel = -(gain * self.lane_error + 0.4 * error_derivative)
       
        # 基于误差的速度调整
        if self.in_pickup_area and self.pickup_mode_active:
            speed_factor = max(0.3, 1.0 - abs(self.lane_error) * 4.0)
            base_speed = self.pickup_creep_speed
        else:
            speed_factor = max(0.2, 1.0 - abs(self.lane_error) * 5.0)
            base_speed = self.max_linear_vel * 0.8
           
        linear_vel = base_speed * speed_factor
       
        return linear_vel, angular_vel
    
    def get_yaw_from_quaternion(self, quaternion):
        """从四元数获取yaw角度"""
        try:
            euler = tf.transformations.euler_from_quaternion([
                quaternion.x, quaternion.y, quaternion.z, quaternion.w
            ])
            return euler[2]
        except:
            return 0.0
    
    def normalize_angle(self, angle):
        """角度规范化"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def create_debug_image(self, original, mask, lane_center_x):
        """创建调试图像"""
        debug_image = original.copy()
        height, width = debug_image.shape[:2]
        
        # 绘制ROI区域
        cv2.rectangle(debug_image, (0, height//2), (width, height), (255, 255, 0), 2)
        
        # 绘制图像中心线
        cv2.line(debug_image, (self.image_center_x, 0), (self.image_center_x, height), (255, 255, 255), 2)
        
        # 绘制车道线中心
        if lane_center_x is not None:
            center_x = int(lane_center_x)
            cv2.line(debug_image, (center_x, height//2), (center_x, height), (0, 255, 0), 3)
            
            # 绘制偏移
            offset = center_x - self.image_center_x
            if abs(offset) > 20:
                arrow_start = (self.image_center_x, height//2 + 50)
                arrow_end = (center_x, height//2 + 50)
                cv2.arrowedLine(debug_image, arrow_start, arrow_end, (0, 255, 255), 3)
        
        # 显示状态信息
        status_text = "Lane Paths + A*: {}".format("ON" if self.lane_guidance_active else "OFF")
        cv2.putText(debug_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        nav_text = "Nav: {} | Goal: {}".format(
            "ACTIVE" if self.navigation_active else "IDLE", 
            self.current_goal or "None")
        cv2.putText(debug_image, nav_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if self.path:
            wp_text = "WP: {}/{} (Hybrid Planning)".format(
                self.path_index + 1, len(self.path))
            cv2.putText(debug_image, wp_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return debug_image
        
    def publish_debug_image(self, debug_image):
        """发布调试图像"""
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            rospy.logwarn("Debug image publish error: %s", str(e))
    
    def control_callback(self, event):
        """主控制回调"""
        if self.current_position is None:
            return
       
        # 发布导航状态
        status = "State:{} Lane:{} Error:{:.3f} Recovery:{}".format(
            self.robot_state, self.lane_detection_active, self.lane_error, self.recovery_active)
        self.navigation_status_pub.publish(String(status))
       
        # 检查导航超时
        if (self.navigation_active and self.navigation_start_time and
            (rospy.Time.now() - self.navigation_start_time).to_sec() > self.navigation_timeout):
            rospy.logwarn("Navigation timeout, resetting")
            self.navigation_active = False
       
        # 恢复操作优先
        if self.recovery_active:
            linear_vel, angular_vel = self.execute_recovery()
            self.publish_velocity(linear_vel, angular_vel)
            return
       
        # 紧急停止
        front_distance = self.obstacle_zones['front']['min_distance']
        
        if front_distance < self.emergency_stop_distance and self.robot_state not in ["PICKING", "DROPPING"]:
            rospy.logwarn_throttle(3.0, "OBSTACLE at %.2fm in state %s", 
                                  front_distance, self.robot_state)
            self.publish_velocity(0, 0)
            return
       
        # 检查导航路径
        if not self.navigation_active or not self.path or self.path_index >= len(self.path):
            # 无导航时的行为 - 使用车道跟随
            if (self.lane_detection_active and 
                self.robot_state not in ["PICKING", "DROPPING"]):
                
                linear_vel, angular_vel = self.calculate_lane_following_velocity()
                safe_linear, safe_angular = self.calculate_safe_velocity()
                
                linear_vel = min(linear_vel, safe_linear)
                angular_vel = np.clip(angular_vel, -safe_angular, safe_angular)
                
                self.publish_velocity(linear_vel, angular_vel)
            else:
                self.publish_velocity(0, 0)
            return
            
        # 路径跟随
        current_waypoint = self.path[self.path_index]
        waypoint_position = current_waypoint[:2]  # [x, y]
        
        self.current_waypoint_pub.publish(String("wp_{}".format(self.path_index)))
        
        # 计算到waypoint的距离
        distance = self.calculate_distance(self.current_position[:2], waypoint_position)
        
        # 检查waypoint是否到达
        tolerance = self.position_tolerance
        if self.in_pickup_area and self.pickup_mode_active:
            tolerance *= 1.2
            
        if distance < tolerance:
            self.path_index += 1
            rospy.loginfo("Reached waypoint: %d (distance: %.2fm)", self.path_index, distance)
            
            if self.path_index >= len(self.path):
                rospy.loginfo("GOAL REACHED: %s", self.current_goal)
                self.goal_reached_pub.publish(Bool(True))
                self.navigation_active = False
                self.publish_velocity(0, 0)
                return
            else:
                current_waypoint = self.path[self.path_index]
                waypoint_position = current_waypoint[:2]
                distance = self.calculate_distance(self.current_position[:2], waypoint_position)
        
        # 导航控制 - 优先车道跟随
        if self.lane_detection_active:
            # 主要使用车道跟随，轻微路点引导
            lane_linear, lane_angular = self.calculate_lane_following_velocity()
            
            # 计算到waypoint的角度引导
            angle_to_waypoint = math.atan2(
                waypoint_position[1] - self.current_position[1],
                waypoint_position[0] - self.current_position[0])
            angle_error = self.normalize_angle(angle_to_waypoint - self.current_position[2])
            
            # 车道跟随权重控制
            lane_weight = self.lane_following_weight
            waypoint_weight = self.waypoint_guidance_weight
            
            guidance_angular = angle_error * waypoint_weight * 0.3
            angular_vel = lane_angular * lane_weight + guidance_angular
            linear_vel = lane_linear
            
        else:
            # 车道丢失时的纯路点导航
            angle_to_waypoint = math.atan2(
                waypoint_position[1] - self.current_position[1],
                waypoint_position[0] - self.current_position[0])
            angle_error = self.normalize_angle(angle_to_waypoint - self.current_position[2])
            
            if abs(angle_error) > self.angle_tolerance:
                linear_vel = 0.1
                angular_vel = np.clip(angle_error * 0.8, -0.4, 0.4)
            else:
                linear_vel = min(0.3, distance * 0.8)
                angular_vel = angle_error * 0.4
        
        # 应用安全限制
        safe_linear, safe_angular = self.calculate_safe_velocity()
        linear_vel = min(linear_vel, safe_linear)
        angular_vel = np.clip(angular_vel, -safe_angular, safe_angular)
        
        # 最终速度限制
        linear_vel = np.clip(linear_vel, 0, self.max_linear_vel)
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        self.publish_velocity(linear_vel, angular_vel)
        
    def publish_velocity(self, linear, angular):
        """发布速度命令"""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)
        
    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = IntegratedNavigationController()
        controller.spin()
    except rospy.ROSInterruptException:
        pass