#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import time
import threading
import math
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

class ManipulationManager(object):
    def __init__(self):
        rospy.init_node('manipulation_manager', anonymous=False)
        
        # Status publishers
        self.status_pub = rospy.Publisher('/manipulation/status', String, queue_size=10)
        self.actual_package_color_pub = rospy.Publisher('/actual_package_color', String, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/manipulation/command', String, self.command_callback)
        rospy.Subscriber('/current_zone', String, self.zone_callback)
        rospy.Subscriber('/current_package_color', String, self.package_color_callback)
        rospy.Subscriber('/robot_state', String, self.robot_state_callback)
        # 统一使用amcl_pose作为主要坐标源
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)  # 仅作为备用
        
        # Wait for Gazebo services
        rospy.loginfo("Waiting for Gazebo services...")
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        rospy.loginfo("Gazebo services connected")
        
        # State tracking
        self.current_zone = "unknown"
        self.current_package_color = None
        self.robot_state = "INIT"
        self.attached_package = None
        self.attached_package_color = None
        self.manipulation_active = False
        
        # 状态重置和一致性管理
        self.last_robot_state = None
        self.state_change_time = 0
        self.manipulation_ready = True
        self.last_status_published = None
        
        # 统一坐标系统 - 主要使用amcl_pose
        self.robot_amcl_pose = None  # 主要坐标源
        self.robot_odom_pose = None  # 备用坐标源
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, z]
        self.current_orientation = None
        
        # 磁吸参数
        self.attach_offset_x = 0.0
        self.attach_offset_y = 0.0
        self.attach_offset_z = 1.2
        
        # DROP区域安全放置坐标 - 远离车道线和导航路径
        self.drop_zone_centers = {
            'red': {'x': 1.8, 'y': 6.2},      # 远离车道线，向外偏移
            'blue': {'x': 3.8, 'y': 6.2},    # 远离车道线，向外偏移  
            'green': {'x': -6.5, 'y': 1.8},  # 远离车道线，向外偏移
            'purple': {'x': 6.5, 'y': 1.8}   # 远离车道线，向外偏移
        }
        
        # 包裹放置随机偏移范围，避免堆叠和阻挡
        self.placement_offset_range = 0.8  # ±0.8米随机偏移
        self.safe_distance_from_lanes = 1.5  # 距离车道线最小安全距离
        
        # 超高频包裹跟随
        self.follow_thread = None
        self.follow_active = False
        self.follow_rate = 50
        
        # 包裹颜色映射
        self.package_color_map = {
            'package_1': 'red', 'package_2': 'blue', 'package_3': 'green', 'package_4': 'purple',
            'package_5': 'red', 'package_6': 'blue', 'package_7': 'green', 'package_8': 'purple',
            'package_9': 'red', 'package_10': 'blue', 'package_11': 'green', 'package_12': 'purple',
            'package_13': 'red', 'package_14': 'blue', 'package_15': 'green', 'package_16': 'purple',
            'package_17': 'red', 'package_18': 'blue', 'package_19': 'green', 'package_20': 'purple'
        }
        
        # 优雅的包裹检测和操作参数 - 平衡距离和成功率
        self.pickup_search_radius = 1.5      
        self.pickup_height_threshold = 0.8   # 排除高处的包裹
        self.attach_verification_attempts = 3
        
        # 包裹放置参数
        self.drop_height = 0.3
        self.drop_verification_attempts = 3
        self.drop_position_tolerance = 1.5
        
        # 智能命令处理
        self.last_command = None
        self.last_command_time = 0
        self.command_cooldown = 1.0  # 减少冷却时间，提高响应
        self.force_placement_mode = False
        
        # 任务统计
        self.total_packages_picked = 0
        self.total_packages_placed = 0
        self.failed_picks = 0
        self.failed_places = 0
        
        # 智能状态监控
        self.status_check_timer = rospy.Timer(rospy.Duration(3.0), self.smart_status_check)
        self.command_monitor_timer = rospy.Timer(rospy.Duration(1.0), self.monitor_command_execution)
        
        rospy.loginfo("ManipulationManager initialized with SMART STATE MANAGEMENT")
        
    def pose_callback(self, msg):
        """统一的位姿回调 - 使用amcl_pose作为主要坐标源"""
        # 为了保持坐标一致性，也从Gazebo获取当前位置
        try:
            response = self.get_model_state("warehouse_robot", "")
            if response.success:
                pos = response.pose.position
                self.current_position = [pos.x, pos.y, pos.z]
                self.current_orientation = response.pose.orientation
                return
        except:
            pass
            
        # 备用：从订阅的amcl_pose更新
        self.robot_amcl_pose = msg.pose.pose
        pos = msg.pose.pose.position
        self.current_position = [pos.x, pos.y, pos.z]
        self.current_orientation = msg.pose.pose.orientation
        
    def odom_callback(self, msg):
        """里程计回调 - 仅作为备用坐标源"""
        self.robot_odom_pose = msg.pose.pose
        
        # 如果amcl_pose不可用，使用odom作为备用
        if self.robot_amcl_pose is None:
            pos = msg.pose.pose.position
            self.current_position = [pos.x, pos.y, pos.z]
            self.current_orientation = msg.pose.pose.orientation
        
    def command_callback(self, msg):
        """确保命令被正确执行并解决状态冲突"""
        command = msg.data
        current_time = time.time()
        
        rospy.logwarn("MANIPULATION COMMAND: '%s' | Zone: %s | State: %s | Active: %s | Carrying: %s", 
                     command, self.current_zone, self.robot_state, self.manipulation_active, self.attached_package)
        
        # 智能重复命令处理
        if (command == self.last_command and 
            current_time - self.last_command_time < self.command_cooldown):
            # 检查是否需要强制执行（例如状态不匹配）
            if not self.should_force_command_execution(command):
                rospy.logdebug("Ignoring duplicate command: %s (cooldown active)", command)
                return
            else:
                rospy.logwarn("FORCING duplicate command execution due to state mismatch")
            
        self.last_command = command
        self.last_command_time = current_time
        
        # 处理各种命令
        if command == "pick_package":
            self.handle_pick_command_smart()
        elif command == "place_package":
            self.handle_place_command_smart()
        elif command == "emergency_stop":
            self.emergency_release()
        elif command == "release_package":
            self.force_release_package()
        else:
            rospy.logwarn("Unknown manipulation command: %s", command)

    def should_force_command_execution(self, command):
        """判断是否需要强制执行重复命令"""
        if command == "pick_package":
            # 如果robot_state是PICK_PACKAGE但manipulation不活跃，强制执行
            return (self.robot_state == "PICK_PACKAGE" and not self.manipulation_active)
        elif command == "place_package":
            # 如果robot_state是PLACE_PACKAGE但manipulation不活跃，强制执行
            return (self.robot_state == "PLACE_PACKAGE" and not self.manipulation_active)
        return False

    def handle_pick_command_smart(self):
        """智能抓取命令处理"""
        # 详细状态检查
        rospy.logwarn("PICK COMMAND ANALYSIS:")
        rospy.logwarn("  - Attached package: %s", self.attached_package)
        rospy.logwarn("  - Manipulation active: %s", self.manipulation_active)
        rospy.logwarn("  - Robot state: %s", self.robot_state)
        rospy.logwarn("  - Zone: %s", self.current_zone)
        rospy.logwarn("  - Manipulation ready: %s", self.manipulation_ready)
        
        # 检查是否已经携带包裹
        if self.attached_package is not None:
            rospy.logwarn("Already carrying package %s, cannot pick another", self.attached_package)
            self.publish_status("failed")
            return
            
        # 检查是否在正确的状态
        if self.robot_state != "PICK_PACKAGE":
            rospy.logwarn("Robot state is %s, not PICK_PACKAGE. Will proceed anyway.", self.robot_state)
            
        # 重置manipulation状态确保干净开始
        if self.manipulation_active and self.last_status_published in ["place_completed", "failed"]:
            rospy.logwarn("Resetting manipulation state from previous operation")
            self.manipulation_active = False
            self.manipulation_ready = True
            
        if not self.manipulation_ready:
            rospy.logwarn("Manipulation not ready, forcing reset")
            self.manipulation_ready = True
            
        rospy.logwarn("Starting intelligent pick sequence...")
        self.smart_pick_sequence()

    def handle_place_command_smart(self):
        """智能放置命令处理"""
        rospy.logwarn("🔧 PLACE COMMAND ANALYSIS:")
        rospy.logwarn("  - Attached package: %s (color: %s)", self.attached_package, self.attached_package_color)
        rospy.logwarn("  - Manipulation active: %s", self.manipulation_active)
        rospy.logwarn("  - Robot state: %s", self.robot_state)
        rospy.logwarn("  - Zone: %s", self.current_zone)
        
        if self.attached_package is None:
            rospy.logerr("No package attached for placement!")
            self.publish_status("failed")
            return
            
        # 检查zone匹配但允许强制放置
        expected_zone = self.attached_package_color + "_zone"
        if self.current_zone != expected_zone:
            rospy.logwarn("⚠️ Zone mismatch: In '%s' but expected '%s'. Proceeding anyway.", 
                         self.current_zone, expected_zone)
            self.force_placement_mode = True
            
        # 重置manipulation状态确保干净开始
        if self.manipulation_active and self.last_status_published in ["pick_completed", "failed"]:
            rospy.logwarn("🔧 Resetting manipulation state from previous operation")
            self.manipulation_active = False
            self.manipulation_ready = True
            
        rospy.logwarn("Starting intelligent place sequence...")
        self.smart_place_sequence()

    def smart_status_check(self, event):
        """智能状态检查和自动修复"""
        current_time = time.time()
        
        # 检查robot_state和manipulation状态的一致性
        if self.robot_state == "PICK_PACKAGE":
            if self.attached_package is None and not self.manipulation_active:
                rospy.logwarn("MISMATCH: Robot in PICK_PACKAGE but manipulation idle")
                # 自动触发pick序列
                if current_time - self.state_change_time > 3.0:  # 避免过于频繁
                    rospy.logwarn("AUTO-TRIGGERING pick sequence")
                    self.smart_pick_sequence()
                    
        elif self.robot_state == "PLACE_PACKAGE":
            if self.attached_package is not None and not self.manipulation_active:
                expected_zone = self.attached_package_color + "_zone"
                if self.current_zone == expected_zone or self.current_zone.endswith("_zone"):
                    rospy.logwarn("MISMATCH: Robot in PLACE_PACKAGE but manipulation idle")
                    # 自动触发place序列
                    if current_time - self.state_change_time > 3.0:
                        rospy.logwarn("AUTO-TRIGGERING place sequence")
                        self.force_placement_mode = True
                        self.smart_place_sequence()

    def monitor_command_execution(self, event):
        """监控命令执行状态"""
        if self.manipulation_active:
            # 检查manipulation是否卡住
            if (self.last_command_time > 0 and 
                time.time() - self.last_command_time > 30.0 and
                self.last_status_published not in ["pick_completed", "place_completed"]):
                rospy.logwarn("Manipulation seems stuck, resetting...")
                self.manipulation_active = False
                self.manipulation_ready = True
        
    def zone_callback(self, msg):
        """Track current robot zone"""
        old_zone = self.current_zone
        self.current_zone = msg.data
        if old_zone != self.current_zone:
            rospy.loginfo("Zone changed: %s -> %s", old_zone, self.current_zone)
        
    def package_color_callback(self, msg):
        """Track current package color"""
        self.current_package_color = msg.data
        
    def robot_state_callback(self, msg):
        """智能机器人状态跟踪"""
        old_state = self.robot_state
        self.robot_state = msg.data
        
        if old_state != self.robot_state:
            self.last_robot_state = old_state
            self.state_change_time = time.time()
            rospy.loginfo("🔄 Robot state: %s -> %s", old_state, self.robot_state)
            
            # 状态变化时的智能重置
            if self.robot_state == "DETECT_PACKAGE_COLOR":
                # 进入检测状态时，重置manipulation准备状态
                self.manipulation_ready = True
                if self.manipulation_active and self.last_status_published in ["failed", "pick_completed"]:
                    rospy.logwarn("🔧 Resetting manipulation for new detection cycle")
                    self.manipulation_active = False
        
    def publish_status(self, status):
        """智能状态发布"""
        if status != self.last_status_published:
            status_msg = String()
            status_msg.data = status
            self.status_pub.publish(status_msg)
            self.last_status_published = status
            rospy.logwarn("📡 Manipulation status: %s -> %s", self.last_status_published, status)
            
            # 状态发布后的智能处理
            if status in ["pick_completed", "place_completed"]:
                self.manipulation_ready = True
        
    def publish_actual_package_color(self, color):
        """发布实际抓取的包裹颜色"""
        if color:
            color_msg = String()
            color_msg.data = color
            self.actual_package_color_pub.publish(color_msg)
            rospy.loginfo("📦 Actual package color published: %s", color)
        
    def get_robot_position(self):
        """统一的机器人位置获取 - 优先使用Gazebo真实位置"""
        # 为了确保坐标一致性，优先使用Gazebo的真实位置
        try:
            response = self.get_model_state("warehouse_robot", "")
            if response.success:
                pos = response.pose.position
                orient = response.pose.orientation
                return pos, orient
        except Exception as e:
            rospy.logdebug("Gazebo position unavailable: %s", str(e))
            
        # 备用1：使用amcl_pose
        if self.robot_amcl_pose is not None:
            pos = self.robot_amcl_pose.position
            orient = self.robot_amcl_pose.orientation
            return pos, orient
            
        # 备用2：使用odom
        elif self.robot_odom_pose is not None:
            pos = self.robot_odom_pose.position  
            orient = self.robot_odom_pose.orientation
            return pos, orient
                
        rospy.logwarn("⚠️ No valid robot position available!")
        return None, None
        
    def get_package_color_from_name(self, package_name):
        """根据包裹名称获取颜色"""
        return self.package_color_map.get(package_name, 'unknown')
        
    def find_nearest_package(self):
        """智能包裹搜索"""
        if self.current_zone != "pickup_zone":
            rospy.logwarn("Not in pickup zone, cannot search for packages")
            return None, None
            
        robot_pos, _ = self.get_robot_position()
        if robot_pos is None:
            rospy.logerr("Cannot get robot position for package search")
            return None, None
            
        rospy.logwarn("SMART SEARCH for packages near [%.3f, %.3f] radius %.1fm", 
                     robot_pos.x, robot_pos.y, self.pickup_search_radius)
        
        nearest_package = None
        nearest_color = None
        min_distance = float('inf')
        target_color_package = None
        target_color_distance = float('inf')
        
        available_packages = []
        
        for i in range(1, 21):
            package_name = "package_{}".format(i)
            try:
                response = self.get_model_state(package_name, "")
                if response.success:
                    pkg_pos = response.pose.position
                    
                    in_pickup_area = (-2.0 <= pkg_pos.x <= 2.0 and -3.0 <= pkg_pos.y <= 0.0)
                    on_ground = pkg_pos.z < self.pickup_height_threshold
                    
                    if in_pickup_area and on_ground:
                        distance = ((pkg_pos.x - robot_pos.x)**2 + (pkg_pos.y - robot_pos.y)**2)**0.5
                        
                        if distance < self.pickup_search_radius:
                            package_color = self.get_package_color_from_name(package_name)
                            available_packages.append((package_name, package_color, distance, pkg_pos))
                            
                            # 优先选择期望颜色的包裹
                            if (self.current_package_color and 
                                package_color == self.current_package_color and
                                distance < target_color_distance):
                                target_color_package = package_name
                                target_color_distance = distance
                                
                            # 记录最近的包裹
                            if distance < min_distance:
                                min_distance = distance
                                nearest_package = package_name
                                nearest_color = package_color
                                
            except Exception as e:
                rospy.logdebug("Error checking package %s: %s", package_name, str(e))
                
        # 智能选择逻辑
        if len(available_packages) > 0:
            rospy.logwarn("Found %d packages in range:", len(available_packages))
            for pkg_name, pkg_color, pkg_dist, pkg_pos in available_packages:
                rospy.logwarn("  - %s (%s): %.3fm at [%.3f,%.3f,%.3f]", 
                             pkg_name, pkg_color, pkg_dist, pkg_pos.x, pkg_pos.y, pkg_pos.z)
        
        # 返回最佳选择
        if target_color_package:
            target_color = self.get_package_color_from_name(target_color_package)
            rospy.logwarn("SELECTED TARGET: %s (%s) at %.3fm", 
                         target_color_package, target_color, target_color_distance)
            return target_color_package, target_color
        elif nearest_package:
            rospy.logwarn("SELECTED NEAREST: %s (%s) at %.3fm", 
                         nearest_package, nearest_color, min_distance)
            return nearest_package, nearest_color
        else:
            rospy.logwarn("NO packages found in range %.1fm", self.pickup_search_radius)
            return None, None
        
    def verify_package_pickup(self, package_name):
        """智能包裹抓取验证"""
        try:
            response = self.get_model_state(package_name, "")
            if response.success:
                pkg_pos = response.pose.position
                robot_pos, _ = self.get_robot_position()
                
                if robot_pos is not None:
                    distance = ((pkg_pos.x - robot_pos.x)**2 + (pkg_pos.y - robot_pos.y)**2)**0.5
                    height_check = pkg_pos.z > 0.6  # 高度要求
                    proximity_check = distance < 0.8  # 距离要求
                    
                    rospy.logwarn("VERIFY %s: height=%.3f (>0.6?=%s), distance=%.3f (<0.8?=%s)", 
                                package_name, pkg_pos.z, height_check, distance, proximity_check)
                    
                    if height_check and proximity_check:
                        rospy.logwarn("Package %s pickup VERIFIED", package_name)
                        return True
                    else:
                        rospy.logwarn("Package %s pickup verification FAILED", package_name)
                        return False
                        
        except Exception as e:
            rospy.logerr("Pickup verification error: %s", str(e))
            
        return False

    def magnetic_attach_package(self, package_name):
        """智能磁性吸附包裹"""
        rospy.logwarn("SMART ATTACHMENT for package: %s", package_name)
        
        robot_pos, robot_orient = self.get_robot_position()
        if robot_pos is None:
            rospy.logerr("Cannot get robot position for attachment")
            return False
        
        for attempt in range(self.attach_verification_attempts):
            try:
                model_state = ModelState()
                model_state.model_name = package_name
                
                model_state.pose.position.x = robot_pos.x + self.attach_offset_x
                model_state.pose.position.y = robot_pos.y + self.attach_offset_y
                model_state.pose.position.z = robot_pos.z + self.attach_offset_z
                
                model_state.pose.orientation.x = 0.0
                model_state.pose.orientation.y = 0.0
                model_state.pose.orientation.z = 0.0
                model_state.pose.orientation.w = 1.0
                
                model_state.twist.linear.x = 0.0
                model_state.twist.linear.y = 0.0
                model_state.twist.linear.z = 0.0
                model_state.twist.angular.x = 0.0
                model_state.twist.angular.y = 0.0
                model_state.twist.angular.z = 0.0
                
                response = self.set_model_state(model_state)
                if response.success:
                    rospy.sleep(1.5)
                    
                    if self.verify_package_pickup(package_name):
                        self.attached_package = package_name
                        self.attached_package_color = self.get_package_color_from_name(package_name)
                        rospy.logwarn("Package %s (%s) ATTACHED successfully (attempt %d)", 
                                    package_name, self.attached_package_color, attempt + 1)
                        
                        self.start_ultra_stable_following()
                        self.total_packages_picked += 1
                        return True
                    else:
                        rospy.logwarn("Attachment verification failed (attempt %d)", attempt + 1)
                else:
                    rospy.logerr("Gazebo attachment failed (attempt %d)", attempt + 1)
                    
            except Exception as e:
                rospy.logerr("Attachment error (attempt %d): %s", attempt + 1, str(e))
                
            if attempt < self.attach_verification_attempts - 1:
                rospy.sleep(1.0)
                
        self.failed_picks += 1
        return False

    def smart_pick_sequence(self):
        """智能抓取序列"""
        if self.attached_package is not None:
            rospy.logwarn("Already carrying package %s", self.attached_package)
            self.publish_status("failed")
            return
            
        self.manipulation_active = True
        self.manipulation_ready = False
        self.publish_status("picking")
        
        try:
            rospy.logwarn("STARTING smart pick sequence")
            
            target_package, target_color = self.find_nearest_package()
            if target_package is None:
                rospy.logwarn("NO package found for picking")
                self.publish_status("failed")
                return
                
            rospy.logwarn("Target: %s (%s)", target_package, target_color)
            rospy.sleep(2.0)  # 稳定等待
            
            if self.magnetic_attach_package(target_package):
                rospy.logwarn("Pick sequence SUCCESS")
                self.publish_actual_package_color(self.attached_package_color)
                rospy.sleep(2.0)
                self.publish_status("pick_completed")
            else:
                rospy.logwarn("Pick sequence FAILED")
                self.publish_status("failed")
                
        except Exception as e:
            rospy.logerr("Pick sequence error: %s", str(e))
            self.publish_status("failed")
        finally:
            self.manipulation_active = False
            self.manipulation_ready = True

    def smart_place_sequence(self):
        """智能放置序列"""
        if self.attached_package is None:
            rospy.logwarn("No package to place")
            self.publish_status("failed")
            return
            
        self.manipulation_active = True
        self.manipulation_ready = False
        self.publish_status("placing")
        
        try:
            package_color = self.attached_package_color
            rospy.logwarn("STARTING smart place sequence for %s (%s)", 
                         self.attached_package, package_color)
            
            rospy.sleep(2.0)  # 稳定等待
            
            if self.magnetic_release_package():
                rospy.logwarn("Place sequence SUCCESS")
                rospy.sleep(2.0)
                self.publish_status("place_completed")
            else:
                rospy.logwarn("Place sequence FAILED")
                self.publish_status("failed")
                
        except Exception as e:
            rospy.logerr("Place sequence error: %s", str(e))
            self.publish_status("failed")
        finally:
            self.manipulation_active = False
            self.manipulation_ready = True
            self.force_placement_mode = False

    def get_drop_zone_center(self, package_color):
        """根据包裹颜色获取安全的drop区域坐标"""
        if package_color in self.drop_zone_centers:
            center = self.drop_zone_centers[package_color]
            rospy.loginfo("Safe drop zone for %s: [%.3f, %.3f]", package_color, center['x'], center['y'])
            return center['x'], center['y']
        else:
            rospy.logerr("Unknown package color for drop: %s", package_color)
            return None, None

    def verify_package_placement(self, package_name, target_x, target_y):
        """验证包裹是否成功放置在目标位置"""
        try:
            response = self.get_model_state(package_name, "")
            if response.success:
                pkg_pos = response.pose.position
                
                distance = ((pkg_pos.x - target_x)**2 + (pkg_pos.y - target_y)**2)**0.5
                height_check = pkg_pos.z < 0.5
                position_check = distance < self.drop_position_tolerance
                
                if height_check and position_check:
                    rospy.loginfo("Package %s placement verified: distance=%.3f, height=%.3f", 
                                package_name, distance, pkg_pos.z)
                    return True
                else:
                    rospy.logwarn("Package %s placement failed: distance=%.3f, height=%.3f", 
                                package_name, distance, pkg_pos.z)
                    return False
                    
        except Exception as e:
            rospy.logerr("Placement verification error: %s", str(e))
            
        return False

    def magnetic_release_package(self):
        """磁性释放包裹 - 安全距离放置"""
        if self.attached_package is None:
            rospy.logwarn("No package attached for release")
            return True
            
        package_color = self.attached_package_color
        if not package_color:
            rospy.logerr("Cannot determine package color for drop placement")
            return False
            
        rospy.logwarn("🔧 RELEASING %s (%s) from [%.3f, %.3f]", 
                     self.attached_package, package_color,
                     self.current_position[0], self.current_position[1])
        
        # 获取安全的放置位置
        safe_drop_pos = self.get_safe_drop_position(package_color)
        if not safe_drop_pos:
            rospy.logerr("Cannot get safe drop position for color: %s", package_color)
            return False
            
        target_x, target_y = safe_drop_pos
        
        for attempt in range(self.drop_verification_attempts):
            try:
                self.stop_package_following()
                
                model_state = ModelState()
                model_state.model_name = self.attached_package
                
                model_state.pose.position.x = target_x
                model_state.pose.position.y = target_y
                model_state.pose.position.z = self.drop_height
                
                model_state.pose.orientation.x = 0.0
                model_state.pose.orientation.y = 0.0
                model_state.pose.orientation.z = 0.0
                model_state.pose.orientation.w = 1.0
                
                model_state.twist.linear.x = 0.0
                model_state.twist.linear.y = 0.0
                model_state.twist.linear.z = 0.0
                model_state.twist.angular.x = 0.0
                model_state.twist.angular.y = 0.0
                model_state.twist.angular.z = 0.0
                
                response = self.set_model_state(model_state)
                if response.success:
                    rospy.logwarn("Package placed at [%.3f, %.3f] - attempt %d", 
                                target_x, target_y, attempt + 1)
                    
                    rospy.sleep(3.0)
                    
                    if self.verify_package_placement(self.attached_package, target_x, target_y):
                        rospy.logwarn("Package %s (%s) placed successfully!", 
                                    self.attached_package, self.attached_package_color)
                        
                        self.attached_package = None
                        self.attached_package_color = None
                        self.total_packages_placed += 1
                        self.force_placement_mode = False
                        
                        rospy.loginfo("Total: picked=%d, placed=%d", 
                                    self.total_packages_picked, self.total_packages_placed)
                        return True
                    else:
                        rospy.logwarn("Placement verification failed (attempt %d)", attempt + 1)
                        
                        if attempt < self.drop_verification_attempts - 1:
                            safe_drop_pos = self.get_safe_drop_position(package_color)
                            if safe_drop_pos:
                                target_x, target_y = safe_drop_pos
                else:
                    rospy.logerr("Gazebo placement failed (attempt %d)", attempt + 1)
                    
            except Exception as e:
                rospy.logerr("Package release error (attempt %d): %s", attempt + 1, str(e))
                
            if attempt < self.drop_verification_attempts - 1:
                rospy.sleep(2.0)
                
        rospy.logerr("Failed to place package after %d attempts", self.drop_verification_attempts)
        self.failed_places += 1
        self.force_placement_mode = False
        return False
        
    def get_safe_drop_position(self, package_color):
        """获取安全的包裹放置位置 - 远离车道线和导航路径"""
        if package_color not in self.drop_zone_centers:
            rospy.logerr("Unknown package color: %s", package_color)
            return None
            
        base_center = self.drop_zone_centers[package_color]
        base_x, base_y = base_center['x'], base_center['y']
        
        # 添加随机偏移避免包裹堆叠
        import random
        offset_x = random.uniform(-self.placement_offset_range, self.placement_offset_range)
        offset_y = random.uniform(-self.placement_offset_range, self.placement_offset_range)
        
        candidate_x = base_x + offset_x
        candidate_y = base_y + offset_y
        
        # 确保不会太接近车道线
        safe_x, safe_y = self.ensure_safe_distance_from_lanes(candidate_x, candidate_y)
        
        rospy.loginfo("Safe drop position for %s: [%.3f, %.3f]", package_color, safe_x, safe_y)
        
        return [safe_x, safe_y]
        
    def ensure_safe_distance_from_lanes(self, x, y):
        """确保位置远离车道线"""
        # 主要车道线位置
        vertical_lanes = [0.0, 1.0, 3.0, -5.0, 5.0]  # x坐标
        horizontal_lanes = [-1.0, 1.0, 3.0, 5.0]     # y坐标
        
        safe_x = x
        safe_y = y
        
        # 检查并调整x坐标，确保远离垂直车道线
        for lane_x in vertical_lanes:
            if abs(safe_x - lane_x) < self.safe_distance_from_lanes:
                # 向远离车道线方向移动
                if safe_x > lane_x:
                    safe_x = lane_x + self.safe_distance_from_lanes
                else:
                    safe_x = lane_x - self.safe_distance_from_lanes
                rospy.logdebug("Adjusted x from %.3f to %.3f (avoiding lane at %.3f)", x, safe_x, lane_x)
                
        # 检查并调整y坐标，确保远离水平车道线
        for lane_y in horizontal_lanes:
            if abs(safe_y - lane_y) < self.safe_distance_from_lanes:
                # 向远离车道线方向移动
                if safe_y > lane_y:
                    safe_y = lane_y + self.safe_distance_from_lanes
                else:
                    safe_y = lane_y - self.safe_distance_from_lanes
                rospy.logdebug("Adjusted y from %.3f to %.3f (avoiding lane at %.3f)", y, safe_y, lane_y)
                
        return safe_x, safe_y
        
    def calculate_distance_to_nearest_lane(self, x, y):
        """计算到最近车道线的距离"""
        vertical_lanes = [0.0, 1.0, 3.0, -5.0, 5.0]
        horizontal_lanes = [-1.0, 1.0, 3.0, 5.0]
        
        min_distance = float('inf')
        
        # 检查到垂直车道线的距离
        for lane_x in vertical_lanes:
            distance = abs(x - lane_x)
            min_distance = min(min_distance, distance)
            
        # 检查到水平车道线的距离
        for lane_y in horizontal_lanes:
            distance = abs(y - lane_y)
            min_distance = min(min_distance, distance)
            
        return min_distance
            
    def start_ultra_stable_following(self):
        """启动超稳定包裹跟随"""
        if self.attached_package is None:
            return
            
        self.stop_package_following()
        rospy.sleep(0.2)
        
        self.follow_active = True
        self.follow_thread = threading.Thread(target=self.ultra_stable_follow_loop)
        self.follow_thread.daemon = True
        self.follow_thread.start()
        rospy.loginfo("Package following started for %s", self.attached_package)
        
    def stop_package_following(self):
        """停止包裹跟随"""
        self.follow_active = False
        if self.follow_thread is not None:
            self.follow_thread.join(timeout=3.0)
            self.follow_thread = None
        rospy.loginfo("Package following stopped")
        
    def ultra_stable_follow_loop(self):
        """包裹跟随循环 - 使用统一坐标系"""
        rate = rospy.Rate(self.follow_rate)
        consecutive_errors = 0
        max_errors = 15
        
        rospy.loginfo("Starting stable follow loop for package: %s", self.attached_package)
        
        while (self.follow_active and self.attached_package is not None and 
               not rospy.is_shutdown() and consecutive_errors < max_errors):
            try:
                robot_pos, robot_orient = self.get_robot_position()
                if robot_pos is not None:
                    model_state = ModelState()
                    model_state.model_name = self.attached_package
                    
                    model_state.pose.position.x = robot_pos.x + self.attach_offset_x
                    model_state.pose.position.y = robot_pos.y + self.attach_offset_y
                    model_state.pose.position.z = robot_pos.z + self.attach_offset_z
                    
                    model_state.pose.orientation.x = 0.0
                    model_state.pose.orientation.y = 0.0
                    model_state.pose.orientation.z = 0.0
                    model_state.pose.orientation.w = 1.0
                    
                    model_state.twist.linear.x = 0.0
                    model_state.twist.linear.y = 0.0
                    model_state.twist.linear.z = 0.0
                    model_state.twist.angular.x = 0.0
                    model_state.twist.angular.y = 0.0
                    model_state.twist.angular.z = 0.0
                    
                    self.set_model_state(model_state)
                    consecutive_errors = 0
                    
                else:
                    consecutive_errors += 1
                    if consecutive_errors % 5 == 0:
                        rospy.logwarn("Cannot get robot position for following, error count: %d", 
                                    consecutive_errors)
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors % 5 == 0:
                    rospy.logwarn("Package follow error: %s, error count: %d", str(e), consecutive_errors)
                
            rate.sleep()
            
        if consecutive_errors >= max_errors:
            rospy.logerr("Too many consecutive errors in package following, stopping")
            self.follow_active = False
            
        rospy.loginfo("Stable follow loop ended for package: %s", self.attached_package)
            
    def force_release_package(self):
        """强制释放包裹"""
        if self.attached_package is not None:
            rospy.logwarn("Force releasing package: %s", self.attached_package)
            self.stop_package_following()
            self.attached_package = None
            self.attached_package_color = None
            self.publish_status("force_released")
        
    def emergency_release(self):
        """紧急释放"""
        rospy.logwarn("Emergency release activated")
        
        if self.attached_package is not None:
            rospy.loginfo("Emergency: maintaining package %s connection for safety", self.attached_package)
            if not self.follow_active:
                self.start_ultra_stable_following()
        
        self.manipulation_active = False
        self.publish_status("emergency_stopped")
        
    def get_manipulation_stats(self):
        """获取操作统计信息"""
        # 确定坐标来源
        coord_source = "unknown"
        try:
            response = self.get_model_state("warehouse_robot", "")
            if response.success:
                coord_source = "gazebo"
        except:
            if self.robot_amcl_pose:
                coord_source = "amcl"
            elif self.robot_odom_pose:
                coord_source = "odom"
        
        return {
            'total_picked': self.total_packages_picked,
            'total_placed': self.total_packages_placed,
            'failed_picks': self.failed_picks,
            'failed_places': self.failed_places,
            'currently_carrying': self.attached_package is not None,
            'current_package': self.attached_package,
            'current_color': self.attached_package_color,
            'robot_position': self.current_position,
            'coordinate_source': coord_source,
            'manipulation_active': self.manipulation_active,
            'manipulation_ready': self.manipulation_ready,
            'last_status': self.last_status_published,
            'pickup_search_radius': self.pickup_search_radius
        }

if __name__ == '__main__':
    try:
        manager = ManipulationManager()
        
        def print_stats(event):
            stats = manager.get_manipulation_stats()
            rospy.loginfo("SMART MANIPULATION Stats: Picked=%d, Placed=%d, Failed_picks=%d, Failed_places=%d, Carrying=%s, Active=%s, Ready=%s, Status=%s", 
                         stats['total_picked'], stats['total_placed'], 
                         stats['failed_picks'], stats['failed_places'], stats['current_package'],
                         stats['manipulation_active'], stats['manipulation_ready'], stats['last_status'])
                         
        stats_timer = rospy.Timer(rospy.Duration(30.0), print_stats)
        
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ManipulationManager node terminated")
    except Exception as e:
        rospy.logerr("ManipulationManager fatal error: %s", str(e))