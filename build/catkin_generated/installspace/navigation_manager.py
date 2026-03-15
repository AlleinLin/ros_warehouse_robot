#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
import actionlib
import tf
import math
import cv2
import numpy as np
import time
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import GetModelState

class EnhancedNavigationManager(object):
    def __init__(self):
        rospy.init_node('enhanced_navigation_manager', anonymous=False)
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 等待Gazebo服务
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=10.0)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.gazebo_available = True
            rospy.loginfo("Gazebo service connected - using ground truth coordinates")
        except:
            rospy.logwarn("Gazebo service not available - using AMCL coordinates")
            self.gazebo_available = False
            self.get_model_state = None
        
        # 适应变粗车道线的检测参数 (0.6m宽度)
        self.yellow_lower = np.array([15, 80, 80])    # 放宽HSV范围
        self.yellow_upper = np.array([35, 255, 255])  
        
        # PID式车道线修正参数 - 温和但有效
        self.lane_center_tolerance = 0.45      # 增加容忍度适应粗车道线
        self.lane_guidance_active = True       # 车道线引导总是激活
        self.gentle_correction_enabled = True # 温和修正模式
        
        # PID参数 - 调整为更温和
        self.kp = 0.3  # 比例系数
        self.ki = 0.05 # 积分系数  
        self.kd = 0.1  # 微分系数
        
        # PID状态变量
        self.last_lane_error = 0.0
        self.integral_error = 0.0
        self.lane_history = []
        self.max_correction_angular = 0.4  # 最大角度修正
        
        # 车道线检测状态
        self.lane_center_x = None
        self.image_center_x = 320  # 640/2
        self.lane_detection_active = False
        self.last_lane_time = 0
        self.lane_timeout = 2.0
        
        # 车道线路径定义 - 包含快速EXIT路径
        self.lane_waypoints = {
            'pickup_to_red': [
                [0.0, 1.0, 0.0],     # 垂直车道线x=0到y=1
                [0.0, 3.0, 0.0],     # 继续沿垂直车道线到y=3
                [1.0, 3.0, 1.5708],  # 转向沿水平车道线y=3到x=1
                [1.0, 5.0, 0.0]      # 沿垂直线到red区域
            ],
            'pickup_to_blue': [
                [0.0, 1.0, 0.0],
                [0.0, 3.0, 0.0],
                [3.0, 3.0, 1.5708],
                [3.0, 5.0, 0.0]
            ],
            'pickup_to_green': [
                [0.0, 1.0, 0.0],
                [-5.0, 1.0, -1.5708]
            ],
            'pickup_to_purple': [
                [0.0, 1.0, 0.0],
                [5.0, 1.0, 1.5708]
            ],
            # 🔥 快速EXIT路径 - 从pickup沿车道线到safety
            'pickup_to_safety': [
                [0.0, -0.3, 0.0],   # pickup接近点（在车道线上）
                [0.0, 0.0, 0.0],    # pickup边缘（在车道线上）
                [0.0, 0.5, 0.0],    # 中间过渡点（在车道线上）
                [0.0, 1.0, 0.0]     # 安全检查点[0, 1.0]（在车道线上）
            ],
            # CRITICAL: 强制返回路径 - 必须先到[0, 1.0]安全点
            'red_to_pickup': [
                [1.0, 4.0, 3.14159],    # 从red区域先到更远的安全点
                [0.5, 3.5, 3.14159],    # 向中心靠近
                [0.0, 3.0, 3.14159],    # 到达主车道中心线
                [0.0, 1.5, 3.14159],    # 向南移动到中间点
                [0.0, 1.0, 3.14159],    # MANDATORY: 到达安全点[0, 1.0]
                [0.0, 0.5, 3.14159],    # 继续向南（在车道线上）
                [0.0, -0.5, 3.14159]    # 最终到达pickup接近点（在车道线上）
            ],
            'blue_to_pickup': [
                [3.0, 4.0, 3.14159],    # 从blue区域先到更远的安全点
                [2.0, 3.5, 3.14159],    # 向中心靠近
                [1.0, 3.0, 3.14159],    # 向左移动到主车道
                [0.0, 3.0, 3.14159],    # 到达主车道中心线
                [0.0, 1.5, 3.14159],    # 向南移动到中间点
                [0.0, 1.0, 3.14159],    # MANDATORY: 到达安全点[0, 1.0]
                [0.0, 0.5, 3.14159],    # 继续向南（在车道线上）
                [0.0, -0.5, 3.14159]    # 最终到达pickup接近点（在车道线上）
            ],
            'green_to_pickup': [
                [-3.5, 1.0, 1.5708],    # 从green先向右移到更安全位置
                [-2.0, 1.5, 1.5708],    # 向上移动到中间区域
                [-1.0, 2.0, 1.5708],    # 继续向上移动
                [0.0, 2.0, 3.14159],    # 转向到达中心线
                [0.0, 1.5, 3.14159],    # 向南移动
                [0.0, 1.0, 3.14159],    # MANDATORY: 到达安全点[0, 1.0]
                [0.0, 0.5, 3.14159],    # 继续向南（在车道线上）
                [0.0, -0.5, 3.14159]    # 最终到达pickup接近点（在车道线上）
            ],
            'purple_to_pickup': [
                [3.5, 1.0, -1.5708],    # 从purple先向左移到更安全位置
                [2.0, 1.5, -1.5708],    # 向上移动到中间区域
                [1.0, 2.0, -1.5708],    # 继续向上移动
                [0.0, 2.0, 3.14159],    # 转向到达中心线
                [0.0, 1.5, 3.14159],    # 向南移动
                [0.0, 1.0, 3.14159],    # MANDATORY: 到达安全点[0, 1.0]
                [0.0, 0.5, 3.14159],    # 继续向南（在车道线上）
                [0.0, -0.5, 3.14159]    # 最终到达pickup接近点（在车道线上）
            ]
        }
        
        # 🔥 安全的备用返回路径 - 确保在车道线上，避开包裹区域
        self.backup_return_paths = {
            'red_to_pickup': [
                [1.0, 4.5, 3.14159],    # 从red区域到安全的车道线位置
                [0.0, 4.0, 3.14159],    # 沿车道线向西到中心线
                [0.0, 3.0, 3.14159],    # 沿主车道线南下
                [0.0, 1.0, 3.14159],    # MANDATORY: 必须经过[0, 1.0]
                [0.0, -0.5, 3.14159]    # 进入pickup接近点（在车道线上）
            ],
            'blue_to_pickup': [
                [3.0, 4.5, 3.14159],    # 从blue区域到安全的车道线位置
                [1.0, 4.0, 3.14159],    # 沿车道线向西
                [0.0, 3.0, 3.14159],    # 到达主车道线
                [0.0, 1.0, 3.14159],    # MANDATORY: 必须经过[0, 1.0]
                [0.0, -0.5, 3.14159]    # 进入pickup接近点（在车道线上）
            ],
            'green_to_pickup': [
                [-4.0, 1.0, 1.5708],    # 从green沿车道线向东
                [-2.0, 1.0, 1.5708],    # 继续沿车道线
                [0.0, 1.0, 1.5708],     # MANDATORY: 到达安全点[0, 1.0]
                [0.0, -0.5, 3.14159]    # 沿主车道线进入pickup接近点
            ],
            'purple_to_pickup': [
                [4.0, 1.0, -1.5708],    # 从purple沿车道线向西
                [2.0, 1.0, -1.5708],    # 继续沿车道线
                [0.0, 1.0, -1.5708],    # MANDATORY: 到达安全点[0, 1.0]
                [0.0, -0.5, 3.14159]    # 沿主车道线进入pickup接近点
            ],
            # 🔥 快速EXIT备用路径 - 确保在车道线上
            'pickup_to_safety_backup': [
                [0.0, -0.2, 0.0],   # pickup边缘（车道线上）
                [0.0, 1.0, 0.0],    # 沿主车道线到安全检查点
                [0.0, 1.5, 0.0]     # 安全点（车道线上）
            ]
        }
        
        # MoveBase action client
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        rospy.loginfo("Waiting for move_base action server...")
        if not self.move_base_client.wait_for_server(timeout=rospy.Duration(60.0)):
            rospy.logerr("CRITICAL: Cannot connect to move_base server!")
            raise Exception("move_base server not available")
        rospy.loginfo("Connected to move_base server")
        
        # Publishers
        self.status_pub = rospy.Publisher('/navigation/arrived', String, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.lane_correction_pub = rospy.Publisher('/lane_correction_cmd', Twist, queue_size=10)
        self.debug_image_pub = rospy.Publisher('/navigation_debug_image', Image, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/navigation/command', String, self.command_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)  # 前置摄像头
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/robot_state', String, self.robot_state_callback)
        rospy.Subscriber('/current_zone', String, self.zone_callback)
        rospy.Subscriber('/fused_obstacle_info', String, self.obstacle_callback)
        
        # Navigation state
        self.current_goal = None
        self.navigation_active = False
        self.current_target = None
        self.current_waypoint_path = []
        self.current_waypoint_index = 0
        
        # 统一坐标系统
        self.robot_amcl_pose = None
        self.robot_odom_pose = None
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, yaw]
        
        self.robot_state = "INIT"
        self.current_zone = "unknown"
        self.obstacle_status = "clear"
        
        # 强制回归控制
        self.force_return_mode = False
        self.return_waypoint_index = 0
        self.max_return_time = 180.0  # 3分钟最大返回时间
        self.waypoint_reached_threshold = 0.4  # waypoint到达阈值
        
        # 返回导航控制
        self.return_attempt_count = 0
        self.max_return_attempts = 3
        self.using_backup_path = False
        self.return_start_time = None
        
        # 安全点[0, 1.0]检查
        self.safety_checkpoint_reached = False
        self.safety_checkpoint_position = [0.0, 1.0]
        self.safety_checkpoint_tolerance = 0.5
        
        # 导航监控和重启机制
        self.navigation_watchdog_enabled = True
        self.navigation_start_time = None
        self.max_navigation_time = 120.0  # 2分钟最大导航时间
        self.navigation_restart_count = 0
        self.max_restart_attempts = 3
        
        # 导航控制
        self.last_command = None
        self.last_command_time = 0
        self.command_attempts = 0
        self.max_command_attempts = 2
        
        # 车道线引导控制循环
        self.lane_guidance_timer = rospy.Timer(rospy.Duration(0.1), self.lane_guidance_loop)
        
        # 导航监控定时器
        self.watchdog_timer = rospy.Timer(rospy.Duration(5.0), self.navigation_watchdog)
        
        # waypoint进度监控
        self.waypoint_timer = rospy.Timer(rospy.Duration(2.0), self.check_waypoint_progress)
        
        # 状态一致性检查定时器
        self.consistency_timer = rospy.Timer(rospy.Duration(3.0), self.check_state_consistency)
        
        # 状态报告定时器
        self.status_report_timer = rospy.Timer(rospy.Duration(10.0), self.report_navigation_status)
        
        # 卡住检测
        self.last_position_check = [0.0, 0.0]
        self.last_position_time = rospy.Time.now()
        self.stuck_threshold = 0.1
        self.stuck_time_threshold = 15.0
        
        rospy.loginfo("EnhancedNavigationManager initialized with MANDATORY SAFETY CHECKPOINT [0, 1.0]")
        
    def camera_callback(self, msg):
        """前置摄像头车道线检测"""
        if not self.lane_guidance_active:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detect_lane_for_guidance(cv_image)
        except Exception as e:
            rospy.logwarn("Lane detection error: %s", str(e))
            
    def detect_lane_for_guidance(self, image):
        """温和的车道线检测用于导航引导"""
        height, width = image.shape[:2]
        
        # 检测图像下半部分
        roi_image = image[height//2:, :]
        
        # 转换到HSV
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # 适应变粗车道线的掩码
        mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # 增大核
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
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
        else:
            # 检查是否车道线检测超时
            if rospy.Time.now().to_sec() - self.last_lane_time > self.lane_timeout:
                self.lane_detection_active = False
                
        # 创建调试图像
        debug_image = self.create_debug_image(image, mask, lane_center_x)
        self.publish_debug_image(debug_image)
        
    def calculate_lane_center(self, contours, image_width):
        """计算车道线中心"""
        if not contours:
            return None
            
        total_area = 0
        weighted_x_sum = 0
        
        # 降低面积要求，适应各种车道线检测
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # 降低最小面积要求
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
        """车道线引导主循环 - PID式温和修正"""
        if not self.navigation_active or not self.gentle_correction_enabled:
            return
            
        if not self.lane_detection_active or self.lane_center_x is None:
            return
            
        # 计算车道线偏移
        lane_offset = self.lane_center_x - self.image_center_x
        offset_ratio = lane_offset / float(self.image_center_x)
        
        # 只在偏移较大时才进行修正
        if abs(offset_ratio) < self.lane_center_tolerance:
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
        
        # 温和的角度修正
        angular_correction = -pid_output  # 负号：向相反方向修正
        angular_correction = max(-self.max_correction_angular, 
                                min(self.max_correction_angular, angular_correction))
        
        # 发布温和的修正命令（不干扰move_base主导航）
        if abs(angular_correction) > 0.05:  # 只在需要时发布
            correction_cmd = Twist()
            correction_cmd.angular.z = angular_correction * 0.5  # 进一步减小修正强度
            self.lane_correction_pub.publish(correction_cmd)
            
            rospy.logdebug("Lane guidance: offset=%.3f, correction=%.3f", 
                         offset_ratio, angular_correction)
            
    def zone_callback(self, msg):
        old_zone = self.current_zone
        self.current_zone = msg.data
        if old_zone != self.current_zone:
            rospy.loginfo("Zone: %s -> %s", old_zone, self.current_zone)
        
    def robot_state_callback(self, msg):
        old_state = self.robot_state
        self.robot_state = msg.data
        
        # 根据状态启用/禁用车道线引导
        navigation_states = [
            "NAVIGATE_TO_PICKUP", "NAVIGATE_TO_DROP", 
            "RETURN_TO_PICKUP", "EXIT_PICKUP_ZONE"
        ]
        self.lane_guidance_active = (self.robot_state in navigation_states)
        
        if old_state != self.robot_state:
            rospy.loginfo("Robot state: %s -> %s, Lane guidance: %s", 
                         old_state, self.robot_state, self.lane_guidance_active)
        
    def obstacle_callback(self, msg):
        old_status = self.obstacle_status
        self.obstacle_status = msg.data
        if old_status != self.obstacle_status and self.obstacle_status != "clear":
            rospy.loginfo("Obstacle status: %s", self.obstacle_status)
        
    def pose_callback(self, msg):
        """统一的位姿回调"""
        # 优先使用Gazebo坐标
        if self.gazebo_available:
            try:
                response = self.get_model_state("warehouse_robot", "")
                if response.success:
                    pos = response.pose.position
                    orient = response.pose.orientation
                    yaw = self.get_yaw_from_quaternion(orient)
                    self.current_position = [pos.x, pos.y, yaw]
                    return
            except:
                pass
        
        # 备用：使用AMCL坐标
        self.robot_amcl_pose = msg.pose.pose
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        yaw = self.get_yaw_from_quaternion(orient)
        
        self.current_position = [pos.x, pos.y, yaw]
        
    def odom_callback(self, msg):
        """里程计回调"""
        self.robot_odom_pose = msg.pose.pose
        
        if self.robot_amcl_pose is None:
            pos = msg.pose.pose.position
            orient = msg.pose.pose.orientation
            yaw = self.get_yaw_from_quaternion(orient)
            self.current_position = [pos.x, pos.y, yaw]
    
    def command_callback(self, msg):
        """智能的导航命令处理 - 避免重复打断"""
        command = msg.data
        current_time = rospy.Time.now().to_sec()
        
        # 检查是否是重复命令且导航正在进行
        if (command == self.last_command and 
            self.navigation_active and 
            current_time - getattr(self, 'last_command_time', 0) < 5.0):
            rospy.logdebug("Ignoring duplicate command '%s' - navigation in progress", command)
            return
            
        rospy.logwarn("Navigation command: '%s' at position [%.2f, %.2f]", 
                     command, self.current_position[0], self.current_position[1])
        
        # 只有在必要时才取消导航
        should_cancel = False
        if self.navigation_active:
            # 检查是否是不同的命令类型
            if ((command == "exit_pickup_to_safety" and self.current_goal != "safety_exit") or
                (command.startswith("goto_") and not (self.current_goal and "drop" in self.current_goal)) or
                (command.startswith("return_") and self.current_goal != "return_to_pickup") or
                (command == "goto_pickup_approach" and self.current_goal != "pickup_approach")):
                should_cancel = True
        
        if should_cancel:
            rospy.logwarn("Canceling navigation for different command: %s -> %s", 
                         self.current_goal, command)
            self.move_base_client.cancel_all_goals()
            rospy.sleep(0.3)
            self.reset_navigation_state()
        
        # 记录命令时间
        self.last_command = command
        self.last_command_time = current_time
        self.command_attempts = 1
        
        if command == "stop":
            self.stop_navigation()
        elif command == "emergency_stop":
            self.emergency_stop()
        elif command == "goto_pickup_approach":
            if self.current_goal != "pickup_approach":
                self.navigate_to_pickup()
        elif command == "exit_pickup_to_safety":
            if self.current_goal != "safety_exit":
                rospy.logwarn("EXIT_PICKUP_TO_SAFETY command received!")
                self.fast_exit_pickup_zone()
        elif command.startswith("goto_") and command.endswith("_drop"):
            color = command.replace("goto_", "").replace("_drop", "")
            expected_goal = color + "_drop"
            if self.current_goal != expected_goal:
                self.navigate_to_drop_zone(color)
        elif command.startswith("return_from_") and command.endswith("_to_pickup"):
            color = command.replace("return_from_", "").replace("_to_pickup", "")
            if self.current_goal != "return_to_pickup":
                self.force_return_to_pickup(color)
        else:
            rospy.logwarn("Unknown navigation command: %s", command)

    def navigate_to_pickup(self):
        """导航到pickup区域 - 确保沿车道线安全接近"""
        rospy.loginfo("🚀 Navigating to pickup approach via lane lines")
        
        # 🔥 检查当前位置，确保使用安全的接近路径
        current_pos = self.current_position
        x, y = current_pos[0], current_pos[1]
        
        # 🔥 定义安全的pickup接近路径 - 沿车道线
        if abs(x) < 0.5 and y > 0.5:
            # 已经在主车道线上，直接南下
            pickup_approach_path = [
                [0.0, 1.0, 3.14159],    # 安全检查点
                [0.0, 0.5, 3.14159],    # 接近pickup边缘
                [0.0, -0.3, 3.14159]    # pickup接近点（在车道线上）
            ]
        else:
            # 从其他位置先到主车道线
            pickup_approach_path = [
                [0.0, 2.0, 3.14159],    # 先到主车道线
                [0.0, 1.0, 3.14159],    # 安全检查点
                [0.0, 0.5, 3.14159],    # 接近pickup边缘
                [0.0, -0.3, 3.14159]    # pickup接近点（在车道线上）
            ]
        
        self.current_goal = "pickup_approach"
        self.current_waypoint_path = pickup_approach_path
        self.current_waypoint_index = 0
        self.navigation_active = True
        self.navigation_start_time = rospy.Time.now()
        
        rospy.logwarn("🛣️ Safe pickup approach: %d waypoints along lane lines", len(pickup_approach_path))
        for i, wp in enumerate(pickup_approach_path):
            rospy.logwarn("  Approach WP %d: [%.2f, %.2f, %.2f] (on lane)", i+1, wp[0], wp[1], wp[2])
        
        # 开始waypoint导航
        self.navigate_to_next_waypoint()
        self.publish_status("navigating_to_pickup")

    
    def exit_pickup_zone(self):
        """快速退出pickup区域到安全点 - 使用预定义路径"""
        # 检查是否已经在执行退出导航
        if self.navigation_active and self.current_goal == "safety_exit":
            rospy.loginfo("Exit navigation already in progress, waiting...")
            return
            
        rospy.logwarn("FAST EXIT from pickup zone at [%.2f, %.2f] via waypoints", 
                     self.current_position[0], self.current_position[1])
        
        # 🔥 使用预定义的快速退出路径
        current_pos = self.current_position
        exit_waypoints = [
            [current_pos[0], current_pos[1], 0.0],  # 当前位置，调整朝向
            [0.0, -0.5, 0.0],   # pickup区域中心，朝北
            [0.0, 0.5, 0.0],    # 中间过渡点
            [0.0, 1.0, 0.0],    # 安全检查点[0, 1.0]
            [0.0, 1.5, 0.0]     # 最终安全点
        ]
        
        # 设置导航状态
        self.current_goal = "safety_exit"
        self.current_waypoint_path = exit_waypoints
        self.current_waypoint_index = 0
        self.navigation_active = True
        self.navigation_start_time = rospy.Time.now()
        
        # 立即发布正在退出的状态
        self.publish_status("exiting_pickup_zone")
        
        rospy.logwarn("Fast exit path: %d waypoints to safety [0, 1.5]", len(exit_waypoints))
        for i, wp in enumerate(exit_waypoints):
            is_safety = (abs(wp[0] - 0.0) < 0.1 and abs(wp[1] - 1.0) < 0.1)
            marker = " SAFETY CHECKPOINT" if is_safety else ""
            rospy.logwarn("  Exit WP %d: [%.2f, %.2f, %.2f]%s", i+1, wp[0], wp[1], wp[2], marker)
        
        # 开始waypoint导航
        self.navigate_to_next_waypoint()
        self.publish_status("navigating_to_safety")

    def emergency_exit_pickup(self):
        """紧急退出pickup区域 - 使用备用快速路径"""
        rospy.logwarn("EMERGENCY FAST EXIT from pickup zone!")
        
        # 🔥 使用备用快速EXIT路径
        if 'pickup_to_safety_backup' in self.backup_return_paths:
            current_pos = self.current_position
            emergency_exit_path = [
                [current_pos[0], current_pos[1], 0.0],  # 当前位置
                [0.0, 0.0, 0.0],    # pickup中心
                [0.0, 1.0, 0.0],    # 安全检查点
                [0.0, 1.5, 0.0]     # 安全点
            ]
            
            self.current_goal = "emergency_safety_exit"
            self.current_waypoint_path = emergency_exit_path
            self.current_waypoint_index = 0
            self.navigation_active = True
            self.navigation_start_time = rospy.Time.now()
            
            rospy.logwarn("Emergency exit via waypoints: %d points", len(emergency_exit_path))
            self.navigate_to_next_waypoint()
            self.publish_status("emergency_exiting_to_safety")
            return
        
        # 备用方案：直接cmd_vel控制
        rospy.logwarn("Using cmd_vel emergency exit")
        backup_cmd = Twist()
        backup_cmd.linear.x = 0.3  # 前进速度
        
        # 发布几次前进命令
        for i in range(20):  # 2秒前进
            self.cmd_vel_pub.publish(backup_cmd)
            rospy.sleep(0.1)
            
        # 停止
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        # 强制完成退出
        rospy.sleep(2.0)
        self.publish_status("arrived_safety_exit")

    def navigate_to_drop_zone(self, color):
        """使用waypoint导航到drop区域"""
        rospy.logwarn("Navigating to %s drop zone via waypoints", color)
        
        path_name = 'pickup_to_' + color
        if path_name not in self.lane_waypoints:
            rospy.logerr("Unknown path: %s", path_name)
            self.publish_status("failed_" + color + "_drop")
            return
        
        self.current_goal = color + "_drop"
        self.current_waypoint_path = self.lane_waypoints[path_name]
        self.current_waypoint_index = 0
        self.navigation_active = True
        self.navigation_start_time = rospy.Time.now()
        
        rospy.loginfo("Path to %s: %d waypoints", color, len(self.current_waypoint_path))
        
        # 开始导航到第一个waypoint
        self.navigate_to_next_waypoint()

    def force_return_to_pickup(self, color):
        """强制返回pickup区域 - 必须经过安全点[0, 1.0]"""
        rospy.logwarn("MANDATORY SAFE RETURN from %s to pickup via [0, 1.0] (attempt %d/%d)", 
                     color, self.return_attempt_count + 1, self.max_return_attempts)
        
        # 立即检查是否已经在pickup区域
        pickup_distance = self.distance_to_point([0.0, -1.5])
        if self.current_zone == "pickup_zone" or pickup_distance < 1.5:
            rospy.logwarn("Already in pickup area, completing return immediately")
            self.publish_status("arrived_return_to_pickup")
            self.reset_return_state()
            return
        
        # 检查是否已经在安全点[0, 1.0]附近
        safety_distance = self.distance_to_point(self.safety_checkpoint_position)
        if safety_distance < self.safety_checkpoint_tolerance:
            rospy.logwarn("Already at safety checkpoint [0, 1.0], proceeding to pickup")
            self.safety_checkpoint_reached = True
            # 直接导航到pickup
            self.navigate_to_pickup_from_safety_point()
            return
        
        # 选择返回路径 - 所有路径都必须经过[0, 1.0]
        path_name = color + '_to_pickup'
        if path_name not in self.lane_waypoints:
            rospy.logerr("Unknown return path: %s", path_name)
            self.emergency_direct_return()
            return
        
        # 根据尝试次数选择路径
        if self.return_attempt_count == 0:
            # 第一次尝试：使用完整路径（必须经过[0, 1.0]）
            waypoints = self.lane_waypoints[path_name]
            self.using_backup_path = False
            rospy.logwarn("Using FULL path with MANDATORY safety checkpoint [0, 1.0]")
        elif self.return_attempt_count == 1:
            # 第二次尝试：使用备用路径（也必须经过[0, 1.0]）
            if path_name in self.backup_return_paths:
                waypoints = self.backup_return_paths[path_name]
                self.using_backup_path = True
                rospy.logwarn("Using BACKUP path with MANDATORY safety checkpoint [0, 1.0]")
            else:
                waypoints = self.lane_waypoints[path_name]
                self.using_backup_path = False
        else:
            # 第三次尝试：沿车道线安全返回，避开包裹
            rospy.logwarn("Using SAFE LANE path via safety checkpoint [0, 1.0]")
            self.emergency_safe_return()
            return
        
        # 验证路径确实包含安全点[0, 1.0]
        if not self.verify_safety_checkpoint_in_path(waypoints):
            rospy.logerr("CRITICAL: Path does not contain mandatory safety checkpoint [0, 1.0]!")
            # 🔥 使用安全的车道线返回而不是应急路径
            rospy.logwarn("Using safe lane return instead of unsafe emergency path")
            self.emergency_safe_return()
            return
        
        # 设置返回导航状态
        self.force_return_mode = True
        self.current_goal = "return_to_pickup"
        self.current_waypoint_path = waypoints
        self.current_waypoint_index = 0
        self.return_waypoint_index = 0
        self.safety_checkpoint_reached = False  # 重置安全点标志
        if self.return_attempt_count == 0:
            self.return_start_time = rospy.Time.now()
        self.navigation_active = True
        self.navigation_start_time = rospy.Time.now()
        self.return_attempt_count += 1
        
        # 清除之前的导航状态
        self.move_base_client.cancel_all_goals()
        rospy.sleep(1.0)
        self.clear_costmaps()
        rospy.sleep(1.0)
        
        rospy.logwarn("SAFE RETURN path (%s): %d waypoints", 
                     "BACKUP" if self.using_backup_path else "FULL", len(waypoints))
        
        # 打印路径并标记安全点
        for i, wp in enumerate(waypoints):
            is_safety_point = (abs(wp[0] - 0.0) < 0.1 and abs(wp[1] - 1.0) < 0.1)
            marker = " SAFETY CHECKPOINT" if is_safety_point else ""
            rospy.logwarn("  Waypoint %d: [%.2f, %.2f, %.2f]%s", i+1, wp[0], wp[1], wp[2], marker)
        
        # 开始导航
        self.navigate_to_next_waypoint()
        self.publish_status("safe_returning_to_pickup")

    def verify_safety_checkpoint_in_path(self, waypoints):
        """验证路径是否包含安全点[0, 1.0]"""
        for wp in waypoints:
            if abs(wp[0] - 0.0) < 0.2 and abs(wp[1] - 1.0) < 0.2:
                rospy.loginfo("Safety checkpoint [0, 1.0] found in path")
                return True
        rospy.logerr("Safety checkpoint [0, 1.0] NOT found in path!")
        return False

    def navigate_to_pickup_from_safety_point(self):
        """从安全点[0, 1.0]导航到pickup区域"""
        rospy.logwarn("Navigating from safety checkpoint [0, 1.0] to pickup")
        
        # 设置简单的路径从[0, 1.0]到pickup
        safety_to_pickup_path = [
            [0.0, 0.5, 3.14159],    # 中间点
            [0.0, 0.0, 3.14159],    # 接近pickup
            [0.0, -0.3, 3.14159]    # pickup接近点（在车道线上）
        ]
        
        self.current_goal = "return_to_pickup"
        self.current_waypoint_path = safety_to_pickup_path
        self.current_waypoint_index = 0
        self.navigation_active = True
        self.navigation_start_time = rospy.Time.now()
        
        # 开始导航
        self.navigate_to_next_waypoint()

    def emergency_safe_return(self):
        """安全返回 - 始终沿车道线，避开包裹区域"""
        rospy.logwarn("SAFE RETURN via lane lines - avoiding package area!")
        
        # 🔥 根据当前位置创建安全的车道线路径
        current_pos = self.current_position
        x, y = current_pos[0], current_pos[1]
        
        # 根据当前大致位置创建安全路径
        if x > 2.5:  # purple区域附近
            safe_path = [
                [4.0, 1.0, -1.5708],   # 先到purple区域的车道线
                [2.0, 1.0, -1.5708],   # 沿车道线向西
                [0.0, 1.0, -1.5708],   # MANDATORY: 到达安全点[0, 1.0]
                [0.0, -0.3, 3.14159]   # 沿主车道线进入pickup接近点
            ]
        elif x < -2.5:  # green区域附近
            safe_path = [
                [-4.0, 1.0, 1.5708],   # 先到green区域的车道线
                [-2.0, 1.0, 1.5708],   # 沿车道线向东
                [0.0, 1.0, 1.5708],    # MANDATORY: 到达安全点[0, 1.0]
                [0.0, -0.3, 3.14159]   # 沿主车道线进入pickup接近点
            ]
        elif y > 4.0:  # red/blue区域附近
            if x > 1.5:  # blue区域
                safe_path = [
                    [3.0, 4.5, 3.14159],   # 先到blue区域车道线
                    [1.0, 4.0, 3.14159],   # 沿车道线向西
                    [0.0, 3.0, 3.14159],   # 到达主车道线
                    [0.0, 1.0, 3.14159],   # MANDATORY: 安全点[0, 1.0]
                    [0.0, -0.3, 3.14159]   # 进入pickup接近点
                ]
            else:  # red区域
                safe_path = [
                    [1.0, 4.5, 3.14159],   # 先到red区域车道线
                    [0.0, 4.0, 3.14159],   # 沿车道线到中心
                    [0.0, 3.0, 3.14159],   # 沿主车道线南下
                    [0.0, 1.0, 3.14159],   # MANDATORY: 安全点[0, 1.0]
                    [0.0, -0.3, 3.14159]   # 进入pickup接近点
                ]
        else:
            # 默认安全路径 - 直接沿主车道线
            safe_path = [
                [0.0, 2.0, 3.14159],    # 先到主车道线
                [0.0, 1.0, 3.14159],    # MANDATORY: 安全点[0, 1.0]
                [0.0, -0.3, 3.14159]    # 进入pickup接近点
            ]
        
        self.force_return_mode = True
        self.current_goal = "safe_lane_return"
        self.current_waypoint_path = safe_path
        self.current_waypoint_index = 0
        self.safety_checkpoint_reached = False
        self.navigation_active = True
        self.navigation_start_time = rospy.Time.now()
        
        rospy.logwarn("Safe lane return path: %d waypoints, avoiding packages", len(safe_path))
        for i, wp in enumerate(safe_path):
            is_safety_point = (abs(wp[0] - 0.0) < 0.1 and abs(wp[1] - 1.0) < 0.1)
            marker = " SAFETY CHECKPOINT" if is_safety_point else ""
            rospy.logwarn("  Safe WP %d: [%.2f, %.2f, %.2f] (on lane)%s", i+1, wp[0], wp[1], wp[2], marker)
        
        self.navigate_to_next_waypoint()

    def emergency_direct_return(self):
        """应急直接返回pickup - 沿车道线，避开包裹"""
        rospy.logwarn("EMERGENCY LANE RETURN to pickup avoiding packages!")
        
        self.force_return_mode = False
        self.move_base_client.cancel_all_goals()
        rospy.sleep(1.0)
        
        # 🔥 沿主车道线的安全直接路径
        direct_lane_path = [
            [0.0, 1.0, 3.14159],    # MANDATORY: 安全点[0, 1.0] (主车道线上)
            [0.0, -0.3, 3.14159]    # pickup接近点 (主车道线上)
        ]
        
        self.current_goal = "emergency_lane_pickup"
        self.current_waypoint_path = direct_lane_path
        self.current_waypoint_index = 0
        self.navigation_active = True
        self.navigation_start_time = rospy.Time.now()
        
        rospy.logwarn("Emergency lane path: staying on main lane line [x=0]")
        for i, wp in enumerate(direct_lane_path):
            rospy.logwarn("  Lane WP %d: [%.2f, %.2f, %.2f] (main lane)", i+1, wp[0], wp[1], wp[2])
        
        # 先导航到安全点
        self.navigate_to_next_waypoint()

    def reset_return_state(self):
        """重置返回状态"""
        self.return_attempt_count = 0
        self.using_backup_path = False
        self.force_return_mode = False
        self.return_start_time = None
        self.safety_checkpoint_reached = False

    def navigate_to_next_waypoint(self):
        """导航到下一个waypoint"""
        if self.current_waypoint_index >= len(self.current_waypoint_path):
            self.waypoint_navigation_completed()
            return
            
        waypoint = self.current_waypoint_path[self.current_waypoint_index]
        x, y, yaw = waypoint[0], waypoint[1], waypoint[2]
        
        # 检查是否到达安全点[0, 1.0]
        is_safety_checkpoint = (abs(x - 0.0) < 0.1 and abs(y - 1.0) < 0.1)
        if is_safety_checkpoint:
            rospy.logwarn("Navigating to SAFETY CHECKPOINT [0, 1.0] - waypoint %d/%d", 
                         self.current_waypoint_index + 1, len(self.current_waypoint_path))
        else:
            rospy.loginfo("Navigating to waypoint %d/%d: [%.2f, %.2f, %.2f]", 
                         self.current_waypoint_index + 1, len(self.current_waypoint_path),
                         x, y, yaw)
        
        success = self.send_movebase_goal(x, y, yaw)
        if not success:
            rospy.logwarn("Failed to send waypoint goal, retrying...")
            rospy.sleep(2.0)
            self.navigate_to_next_waypoint()

    def waypoint_navigation_completed(self):
        """Waypoint导航完成"""
        if self.force_return_mode:
            rospy.logwarn("SAFE RETURN completed! All waypoints reached via [0, 1.0].")
            self.publish_status("arrived_return_to_pickup")
            self.reset_return_state()
        elif self.current_goal and "drop" in self.current_goal:
            color = self.current_goal.replace("_drop", "")
            rospy.loginfo("Reached %s drop zone via waypoints", color)
            self.publish_status("arrived_" + color + "_drop")
        else:
            rospy.loginfo("Waypoint navigation completed")
            
        self.reset_navigation_state()

    def check_waypoint_progress(self, event):
        """检查waypoint进度 - 特别关注安全点"""
        if not self.navigation_active or not self.current_waypoint_path:
            return
            
        if self.current_waypoint_index >= len(self.current_waypoint_path):
            return
            
        current_waypoint = self.current_waypoint_path[self.current_waypoint_index]
        distance = self.distance_to_point(current_waypoint[:2])
        
        # 检查是否到达安全点[0, 1.0]
        is_safety_checkpoint = (abs(current_waypoint[0] - 0.0) < 0.1 and 
                               abs(current_waypoint[1] - 1.0) < 0.1)
        
        if distance < self.waypoint_reached_threshold:
            if is_safety_checkpoint:
                rospy.logwarn("SAFETY CHECKPOINT [0, 1.0] REACHED! Continuing to pickup...")
                self.safety_checkpoint_reached = True
            
            rospy.loginfo("Waypoint %d reached, advancing", self.current_waypoint_index + 1)
            self.current_waypoint_index += 1
            
            if self.current_waypoint_index < len(self.current_waypoint_path):
                # 继续到下一个waypoint
                rospy.sleep(1.0)  # 短暂停顿确保到达
                self.navigate_to_next_waypoint()
            else:
                # 所有waypoints完成
                self.waypoint_navigation_completed()

    def navigation_watchdog(self, event):
        """导航看门狗 - 监控和重启导航"""
        if not self.navigation_watchdog_enabled or not self.navigation_active:
            return
            
        if self.navigation_start_time is None:
            return
            
        elapsed_time = (rospy.Time.now() - self.navigation_start_time).to_sec()
        
        # 检查导航超时
        if elapsed_time > self.max_navigation_time:
            rospy.logwarn("Navigation timeout after %.1fs", elapsed_time)
            self.handle_navigation_timeout()
            return
            
        # 检查是否卡住
        if self.check_if_robot_stuck():
            rospy.logwarn("Robot stuck, restarting navigation")
            self.restart_navigation()
            return
            
        # 强制回归超时检查
        if self.force_return_mode and self.return_start_time:
            return_elapsed = (rospy.Time.now() - self.return_start_time).to_sec()
            if return_elapsed > self.max_return_time:
                rospy.logwarn("Return timeout after %.1fs, trying backup method", return_elapsed)
                if self.return_attempt_count < self.max_return_attempts:
                    # 尝试下一种方法
                    color = self.extract_color_from_current_state()
                    if color:
                        rospy.logwarn("Retrying return with different strategy")
                        self.force_return_to_pickup(color)
                else:
                    rospy.logwarn("All return attempts failed, using safe lane return")
                    self.emergency_safe_return()

    def extract_color_from_current_state(self):
        """从当前状态提取颜色信息"""
        # 从当前区域推断颜色
        if self.current_zone.endswith("_zone"):
            return self.current_zone.replace("_zone", "")
        
        # 从当前路径推断颜色
        if self.current_waypoint_path and len(self.current_waypoint_path) > 0:
            # 分析waypoint来推断来源
            first_wp = self.current_waypoint_path[0]
            x, y = first_wp[0], first_wp[1]
            
            if x > 2.5:  # purple区域
                return "purple"
            elif x < -2.5:  # green区域
                return "green"
            elif y > 4.0:  # red或blue区域
                if x > 1.5:
                    return "blue"
                else:
                    return "red"
        
        return None

    def handle_navigation_timeout(self):
        """处理导航超时"""
        rospy.logwarn("Handling navigation timeout")
        
        # 检查是否接近目标
        if self.check_if_close_to_target():
            rospy.logwarn("Close to target, forcing completion")
            self.force_navigation_completion()
            return
            
        # 尝试重启导航
        if self.navigation_restart_count < self.max_restart_attempts:
            self.restart_navigation()
        else:
            rospy.logwarn("Max restart attempts reached, forcing completion")
            self.force_navigation_completion()

    def restart_navigation(self):
        """重启导航"""
        self.navigation_restart_count += 1
        rospy.logwarn("Restarting navigation (attempt %d/%d)", 
                     self.navigation_restart_count, self.max_restart_attempts)
        
        # 取消当前goals
        self.move_base_client.cancel_all_goals()
        rospy.sleep(1.0)
        
        # 清除costmaps
        self.clear_costmaps()
        rospy.sleep(2.0)
        
        # 重新开始当前waypoint
        if self.current_waypoint_path and self.current_waypoint_index < len(self.current_waypoint_path):
            self.navigate_to_next_waypoint()
        else:
            rospy.logwarn("No valid waypoint to restart, forcing completion")
            self.force_navigation_completion()
            
        # 重置计时器
        self.navigation_start_time = rospy.Time.now()

    def check_if_close_to_target(self):
        """检查是否接近目标"""
        if not self.current_waypoint_path or self.current_waypoint_index >= len(self.current_waypoint_path):
            return False
            
        target_waypoint = self.current_waypoint_path[self.current_waypoint_index]
        distance = self.distance_to_point(target_waypoint[:2])
        
        return distance < 1.0  # 1米内认为接近

    def check_if_robot_stuck(self):
        """检查机器人是否卡住"""
        current_pos = [self.current_position[0], self.current_position[1]]
        current_time = rospy.Time.now()
        
        # 计算移动距离
        distance_moved = ((current_pos[0] - self.last_position_check[0])**2 + 
                         (current_pos[1] - self.last_position_check[1])**2)**0.5
        
        if distance_moved > self.stuck_threshold:
            # 机器人在移动
            self.last_position_check = current_pos[:]
            self.last_position_time = current_time
            return False
        else:
            # 检查卡住时间
            stuck_time = (current_time - self.last_position_time).to_sec()
            return stuck_time > self.stuck_time_threshold

    def force_navigation_completion(self):
        """强制完成导航"""
        rospy.logwarn("FORCING navigation completion")
        
        if self.force_return_mode:
            self.publish_status("arrived_return_to_pickup")
        elif self.current_goal == "pickup_approach":
            self.publish_status("arrived_pickup_approach")
        elif self.current_goal == "safety_exit":
            self.publish_status("arrived_safety_exit")
        elif self.current_goal and "drop" in self.current_goal:
            color = self.current_goal.replace("_drop", "")
            self.publish_status("arrived_" + color + "_drop")
        else:
            self.publish_status("arrived_" + (self.current_goal or "unknown"))
            
        self.reset_navigation_state()

    def clear_costmaps(self):
        """清除costmaps"""
        try:
            import rospy
            from std_srvs.srv import Empty
            
            rospy.wait_for_service('/move_base/clear_costmaps', timeout=2.0)
            clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
            clear_costmaps()
            rospy.loginfo("Costmaps cleared")
            rospy.sleep(1.0)
        except:
            rospy.logwarn("Could not clear costmaps")
        
    def send_movebase_goal(self, x, y, yaw):
        """发送MoveBase目标"""
        goal = self.create_movebase_goal(x, y, yaw)
        
        try:
            if not self.move_base_client.wait_for_server(timeout=rospy.Duration(2.0)):
                rospy.logerr("MoveBase server not available")
                return False
                
            self.move_base_client.send_goal(goal, done_cb=self.movebase_done_callback)
            rospy.loginfo("MoveBase goal sent: [%.2f, %.2f, %.2f]", x, y, yaw)
            return True
        except Exception as e:
            rospy.logerr("Failed to send goal: %s", str(e))
            return False

    def movebase_done_callback(self, status, result):
        """智能的MoveBase完成回调"""
        rospy.logwarn("MoveBase completed with status: %d (%s) for goal: %s", 
                     status, self.get_status_name(status), self.current_goal)
        
        if status == GoalStatus.SUCCEEDED:
            rospy.logwarn("MoveBase goal SUCCEEDED for: %s", self.current_goal)
            
            # 根据目标类型发布正确的完成状态
            if self.current_goal == "safety_exit":
                rospy.logwarn("Safety exit completed!")
                self.publish_status("arrived_safety_exit")
                self.reset_navigation_state()
                
            elif self.current_goal == "pickup_approach":
                rospy.logwarn("Pickup approach completed!")
                self.publish_status("arrived_pickup_approach")
                self.reset_navigation_state()
                
            elif self.current_goal and "drop" in self.current_goal:
                color = self.current_goal.replace("_drop", "")
                rospy.logwarn("Drop zone reached: %s", color)
                self.publish_status("arrived_" + color + "_drop")
                self.reset_navigation_state()
                
            # 对于waypoint导航，检查是否还有更多waypoints
            elif self.current_waypoint_path and self.current_waypoint_index < len(self.current_waypoint_path) - 1:
                # 还有更多waypoints，继续下一个
                self.current_waypoint_index += 1
                rospy.sleep(1.0)
                self.navigate_to_next_waypoint()
            else:
                # 完成所有waypoints
                self.waypoint_navigation_completed()
                
        elif status == GoalStatus.PREEMPTED:
            rospy.logwarn("MoveBase goal preempted for: %s", self.current_goal)
            # 对于preempted状态，检查是否是因为新命令取消的
            # 如果不是自主取消，可能是外部干扰，需要重新导航
            if hasattr(self, 'last_command_time'):
                time_since_command = rospy.Time.now().to_sec() - self.last_command_time
                if time_since_command > 2.0:  # 如果超过2秒没有新命令，可能是异常取消
                    rospy.logwarn("Unexpected preemption, will retry navigation")
                    rospy.sleep(1.0)
                    self.retry_current_navigation()
            
        elif status == GoalStatus.ABORTED:
            rospy.logwarn("MoveBase goal aborted for: %s", self.current_goal)
            
            # 对于安全退出，使用紧急退出
            if self.current_goal == "safety_exit":
                rospy.logwarn("Safety exit failed, using emergency exit")
                self.emergency_exit_pickup()
            else:
                # 尝试重启导航
                if self.navigation_restart_count < self.max_restart_attempts:
                    rospy.sleep(2.0)
                    self.restart_navigation()
                else:
                    self.force_navigation_completion()
        else:
            rospy.logwarn("MoveBase goal failed with status: %d for: %s", status, self.current_goal)
            
    def get_status_name(self, status):
        """获取状态名称"""
        status_names = {
            0: "PENDING",
            1: "ACTIVE", 
            2: "PREEMPTED",
            3: "SUCCEEDED",
            4: "ABORTED",
            5: "REJECTED",
            6: "PREEMPTING",
            7: "RECALLING",
            8: "RECALLED",
            9: "LOST"
        }
        return status_names.get(status, "UNKNOWN")
        
    def retry_current_navigation(self):
        """重试当前导航"""
        if not self.current_goal:
            return
            
        rospy.logwarn("Retrying navigation for goal: %s", self.current_goal)
        
        if self.current_goal == "safety_exit":
            safety_target = [0.0, 1.5, 0.0]
            self.send_movebase_goal(safety_target[0], safety_target[1], safety_target[2])
        elif self.current_goal == "pickup_approach":
            pickup_target = [0.0, -0.5, 3.14159]
            self.send_movebase_goal(pickup_target[0], pickup_target[1], pickup_target[2])
        elif self.current_waypoint_path and self.current_waypoint_index < len(self.current_waypoint_path):
            self.navigate_to_next_waypoint()

    def check_state_consistency(self, event):
        """检查状态一致性并修复不匹配"""
        # 检查机器人状态与导航状态是否匹配
        
        if self.robot_state == "EXIT_PICKUP_ZONE":
            # 机器人应该退出pickup区域
            if not self.navigation_active or self.current_goal != "safety_exit":
                rospy.logwarn("STATE MISMATCH: Robot in EXIT_PICKUP_ZONE but nav not exiting!")
                rospy.logwarn("Current nav goal: %s, active: %s", self.current_goal, self.navigation_active)
                rospy.logwarn("FORCING exit navigation...")
                self.exit_pickup_zone()
                
        elif self.robot_state == "NAVIGATE_TO_DROP":
            # 机器人应该导航到drop区域
            if not self.navigation_active or not (self.current_goal and "drop" in self.current_goal):
                rospy.logwarn("STATE MISMATCH: Robot should navigate to drop but nav inactive!")
                # 这种情况需要主控重新发送命令
                
        elif self.robot_state == "RETURN_TO_PICKUP":
            # 机器人应该返回pickup
            if not self.navigation_active or self.current_goal != "return_to_pickup":
                rospy.logwarn("STATE MISMATCH: Robot should return to pickup but nav inactive!")
                # 这种情况需要主控重新发送命令
                
        elif self.robot_state == "NAVIGATE_TO_PICKUP":
            # 机器人应该导航到pickup
            if not self.navigation_active or self.current_goal != "pickup_approach":
                rospy.logwarn("STATE MISMATCH: Robot should navigate to pickup but nav inactive!")
                rospy.logwarn("FORCING pickup navigation...")
                self.navigate_to_pickup()

    def report_navigation_status(self, event):
        """定期报告导航状态"""
        rospy.logwarn("=== NAVIGATION MANAGER STATUS ===")
        rospy.logwarn("Robot State: %s", self.robot_state)
        rospy.logwarn("Current Goal: %s", self.current_goal)
        rospy.logwarn("Navigation Active: %s", self.navigation_active)
        rospy.logwarn("Force Return Mode: %s", self.force_return_mode)
        rospy.logwarn("Safety Checkpoint Reached: %s", self.safety_checkpoint_reached)
        if self.force_return_mode:
            rospy.logwarn("Return Attempt: %d/%d | Using Backup: %s", 
                         self.return_attempt_count, self.max_return_attempts, self.using_backup_path)
        rospy.logwarn("Current Zone: %s", self.current_zone)
        rospy.logwarn("Position: [%.3f, %.3f, %.3f]", *self.current_position)
        rospy.logwarn("Lane Guidance Active: %s", self.lane_guidance_active)
        rospy.logwarn("Lane Detection Active: %s", self.lane_detection_active)
        if self.current_waypoint_path:
            rospy.logwarn("Waypoint Progress: %d/%d", 
                         self.current_waypoint_index + 1, len(self.current_waypoint_path))
            
            # 显示当前waypoint和安全点状态
            if self.current_waypoint_index < len(self.current_waypoint_path):
                current_wp = self.current_waypoint_path[self.current_waypoint_index]
                is_safety = (abs(current_wp[0] - 0.0) < 0.1 and abs(current_wp[1] - 1.0) < 0.1)
                rospy.logwarn("Current Waypoint: [%.2f, %.2f, %.2f] %s", 
                             current_wp[0], current_wp[1], current_wp[2],
                             "SAFETY CHECKPOINT" if is_safety else "")
        rospy.logwarn("===============================")

    def distance_to_point(self, point):
        """计算到指定点的距离"""
        robot_pos = [self.current_position[0], self.current_position[1]]
        return ((robot_pos[0] - point[0])**2 + (robot_pos[1] - point[1])**2)**0.5
        
    def get_yaw_from_quaternion(self, quaternion):
        """从四元数获取yaw角度"""
        try:
            euler = tf.transformations.euler_from_quaternion([
                quaternion.x, quaternion.y, quaternion.z, quaternion.w
            ])
            return euler[2]
        except:
            return 0.0
        
    def create_movebase_goal(self, x, y, yaw=0.0):
        """Create a MoveBase goal"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        
        return goal
        
    def normalize_angle(self, angle):
        """标准化角度到[-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
        
    def publish_status(self, status_text):
        """发布导航状态"""
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)
        rospy.logwarn("NAVIGATION STATUS: %s", status_text)
        
    def stop_navigation(self):
        """停止导航"""
        rospy.loginfo("Stopping navigation")
        
        self.move_base_client.cancel_all_goals()
        self.navigation_active = False
        self.force_return_mode = False
        
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
    def emergency_stop(self):
        """紧急停止"""
        rospy.logwarn("Emergency stop")
        
        self.move_base_client.cancel_all_goals()
        
        stop_cmd = Twist()
        for i in range(5):
            self.cmd_vel_pub.publish(stop_cmd)
            rospy.sleep(0.1)
            
        self.navigation_active = False
        self.force_return_mode = False
        self.lane_guidance_active = False
        
    def force_command_completion(self, command):
        """强制完成命令"""
        rospy.logwarn("FORCING completion of: %s", command)
        
        self.reset_navigation_state()
        rospy.sleep(0.5)
        
        if command.startswith("return_from_") and command.endswith("_to_pickup"):
            self.publish_status("arrived_return_to_pickup")
        elif command == "goto_pickup_approach":
            self.publish_status("arrived_pickup_approach")
        elif command == "exit_pickup_to_safety":
            self.publish_status("arrived_safety_exit")
        elif command.startswith("goto_") and command.endswith("_drop"):
            color = command.replace("goto_", "").replace("_drop", "")
            self.publish_status("arrived_" + color + "_drop")
            
        self.command_attempts = 0
        self.last_command = None

    def reset_navigation_state(self):
        """重置导航状态"""
        self.navigation_active = False
        self.force_return_mode = False
        self.current_goal = None
        self.navigation_start_time = None
        self.current_waypoint_path = []
        self.current_waypoint_index = 0
        self.return_waypoint_index = 0
        self.navigation_restart_count = 0
        self.command_attempts = 0
        
        # 重置PID状态
        self.last_lane_error = 0.0
        self.integral_error = 0.0
        self.lane_history = []
        
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
        status_text = "Lane Guidance: {}".format("ON" if self.lane_guidance_active else "OFF")
        cv2.putText(debug_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if lane_center_x is not None:
            offset_text = "Offset: {:.1f}px".format(lane_center_x - self.image_center_x)
            cv2.putText(debug_image, offset_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        nav_text = "Nav: {} | Goal: {}".format(
            "ACTIVE" if self.navigation_active else "IDLE", 
            self.current_goal or "None")
        cv2.putText(debug_image, nav_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if self.current_waypoint_path:
            wp_text = "Waypoint: {}/{}".format(
                self.current_waypoint_index + 1, len(self.current_waypoint_path))
            cv2.putText(debug_image, wp_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 显示安全点状态
        safety_text = "Safety [0,1.0]: {}".format("REACHED" if self.safety_checkpoint_reached else "PENDING")
        cv2.putText(debug_image, safety_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.safety_checkpoint_reached else (255, 255, 0), 2)
        
        if self.force_return_mode:
            cv2.rectangle(debug_image, (0, 0), (width, 40), (0, 0, 255), -1)
            cv2.putText(debug_image, "SAFE RETURN VIA [0,1.0]", (width//2 - 120, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_image
        
    def publish_debug_image(self, debug_image):
        """发布调试图像"""
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            rospy.logwarn("Debug image publish error: %s", str(e))

if __name__ == '__main__':
    try:
        manager = EnhancedNavigationManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("EnhancedNavigationManager terminated")
    except Exception as e:
        rospy.logerr("EnhancedNavigationManager fatal error: %s", str(e))