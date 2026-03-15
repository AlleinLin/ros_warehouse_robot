#!/usr/bin/env python2
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
        # 🔥 统一使用amcl_pose作为主要坐标源
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
        
        # 🔥 统一坐标系统 - 主要使用amcl_pose
        self.robot_amcl_pose = None  # 主要坐标源
        self.robot_odom_pose = None  # 备用坐标源
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, z]
        self.current_orientation = None
        
        # 超稳定磁吸参数
        self.attach_offset_x = 0.0
        self.attach_offset_y = 0.0
        self.attach_offset_z = 1.2
        
        # 🔥 DROP区域安全放置坐标 - 远离车道线和导航路径
        self.drop_zone_centers = {
            'red': {'x': 1.8, 'y': 6.2},      # 远离车道线，向外偏移
            'blue': {'x': 3.8, 'y': 6.2},    # 远离车道线，向外偏移  
            'green': {'x': -6.5, 'y': 1.8},  # 远离车道线，向外偏移
            'purple': {'x': 6.5, 'y': 1.8}   # 远离车道线，向外偏移
        }
        
        # 🔥 包裹放置随机偏移范围，避免堆叠和阻挡
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
        
        # 包裹检测和操作参数
        self.pickup_search_radius = 2.0
        self.pickup_height_threshold = 0.5
        self.attach_verification_attempts = 3
        
        # 包裹放置参数
        self.drop_height = 0.3
        self.drop_verification_attempts = 3
        self.drop_position_tolerance = 1.5  # 🔥 增加容忍度，因为包裹现在放置范围更大
        
        # 🔥 增强的命令处理
        self.last_command = None
        self.last_command_time = 0
        self.command_cooldown = 2.0  # 命令冷却时间
        self.force_placement_mode = False  # 强制放置模式
        
        # 任务统计
        self.total_packages_picked = 0
        self.total_packages_placed = 0
        self.failed_picks = 0
        self.failed_places = 0
        
        # 🔥 添加定期状态检查
        self.status_check_timer = rospy.Timer(rospy.Duration(5.0), self.periodic_status_check)
        
        rospy.loginfo("ManipulationManager initialized with SAFE DISTANCE PACKAGE PLACEMENT (Ground Truth Coordinates)")
        
    def pose_callback(self, msg):
        """🔥 统一的位姿回调 - 使用amcl_pose作为主要坐标源"""
        # 🔥 为了保持坐标一致性，也从Gazebo获取当前位置
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
        """🔥 增强的命令处理 - 确保命令被正确执行"""
        command = msg.data
        current_time = time.time()
        
        rospy.logwarn("🔧 MANIPULATION COMMAND RECEIVED: '%s'", command)
        rospy.logwarn("🔧 Current State - Zone: %s, Robot State: %s, Position: [%.2f, %.2f], Attached: %s, Active: %s", 
                     self.current_zone, self.robot_state, self.current_position[0], self.current_position[1],
                     self.attached_package, self.manipulation_active)
        
        # 🔥 避免重复命令
        if (command == self.last_command and 
            current_time - self.last_command_time < self.command_cooldown):
            rospy.logwarn("⚠️ Ignoring duplicate command: %s (cooldown active)", command)
            return
            
        self.last_command = command
        self.last_command_time = current_time
        
        # 🔥 处理各种命令
        if command == "pick_package":
            self.handle_pick_command()
        elif command == "place_package":
            self.handle_place_command()
        elif command == "emergency_stop":
            self.emergency_release()
        elif command == "release_package":
            self.force_release_package()
        else:
            rospy.logwarn("⚠️ Unknown manipulation command: %s", command)

    def handle_pick_command(self):
        """🔥 处理抓取命令"""
        if self.attached_package is not None:
            rospy.logwarn("⚠️ Already carrying package %s, cannot pick another", self.attached_package)
            self.publish_status("failed")
            return
            
        if self.manipulation_active:
            rospy.logwarn("⚠️ Manipulation already active, ignoring pick command")
            return
            
        rospy.loginfo("✅ Starting pick sequence...")
        self.magnetic_pick_sequence()

    def handle_place_command(self):
        """🔥 增强的放置命令处理 - 强制执行"""
        rospy.logwarn("🔧 PLACE COMMAND HANDLER ACTIVATED")
        
        if self.attached_package is None:
            rospy.logerr("❌ No package attached for placement!")
            self.publish_status("failed")
            return
            
        # 🔥 强制检查当前状态
        rospy.logwarn("🔧 Current attached package: %s (color: %s)", 
                     self.attached_package, self.attached_package_color)
        rospy.logwarn("🔧 Current zone: %s, Position: [%.2f, %.2f]", 
                     self.current_zone, self.current_position[0], self.current_position[1])
        rospy.logwarn("🔧 Manipulation active: %s", self.manipulation_active)
        
        # 🔥 即使manipulation_active也要执行（可能是状态残留）
        if self.manipulation_active:
            rospy.logwarn("⚠️ Manipulation active but forcing place command execution")
            
        rospy.loginfo("✅ Starting FORCED place sequence...")
        self.force_placement_mode = True
        self.magnetic_place_sequence()

    def periodic_status_check(self, event):
        """🔥 定期检查状态并处理异常"""
        if self.attached_package and self.robot_state == "PLACE_PACKAGE":
            if not self.manipulation_active:
                rospy.logwarn("🔧 DETECTED: Robot should be placing but manipulation not active!")
                rospy.logwarn("🔧 Attached package: %s, Zone: %s, Position: [%.2f, %.2f]", 
                             self.attached_package, self.current_zone, 
                             self.current_position[0], self.current_position[1])
                
                # 自动触发放置序列
                expected_zone = self.attached_package_color + "_zone"
                if self.current_zone == expected_zone or self.current_zone.endswith("_zone"):
                    rospy.logwarn("🔧 AUTO-TRIGGERING place sequence due to status mismatch")
                    self.force_placement_mode = True
                    self.magnetic_place_sequence()
        
    def zone_callback(self, msg):
        """Track current robot zone"""
        old_zone = self.current_zone
        self.current_zone = msg.data
        if old_zone != self.current_zone:
            rospy.loginfo("🔍 Zone changed: %s -> %s", old_zone, self.current_zone)
        
    def package_color_callback(self, msg):
        """Track current package color"""
        self.current_package_color = msg.data
        
    def robot_state_callback(self, msg):
        """Track robot state"""
        old_state = self.robot_state
        self.robot_state = msg.data
        
        # 🔥 当进入PLACE_PACKAGE状态时，自动检查是否需要触发放置
        if (old_state != self.robot_state and 
            self.robot_state == "PLACE_PACKAGE" and 
            self.attached_package is not None and
            not self.manipulation_active):
            rospy.logwarn("🔧 AUTO-DETECT: Entered PLACE_PACKAGE state with package attached")
            rospy.sleep(2.0)  # 等待状态稳定
            self.handle_place_command()
        
    def publish_status(self, status):
        """Publish manipulation status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)
        rospy.loginfo("📡 Manipulation status: %s", status)
        
    def publish_actual_package_color(self, color):
        """发布实际抓取的包裹颜色"""
        if color:
            color_msg = String()
            color_msg.data = color
            self.actual_package_color_pub.publish(color_msg)
            rospy.loginfo("📦 Actual package color published: %s", color)
        
    def get_robot_position(self):
        """🔥 统一的机器人位置获取 - 优先使用Gazebo真实位置"""
        # 🔥 为了确保坐标一致性，优先使用Gazebo的真实位置
        try:
            response = self.get_model_state("warehouse_robot", "")
            if response.success:
                pos = response.pose.position
                orient = response.pose.orientation
                rospy.logdebug("Using Gazebo pose (ground truth): [%.3f, %.3f, %.3f]", pos.x, pos.y, pos.z)
                return pos, orient
        except Exception as e:
            rospy.logdebug("Gazebo position unavailable: %s", str(e))
            
        # 备用1：使用amcl_pose
        if self.robot_amcl_pose is not None:
            pos = self.robot_amcl_pose.position
            orient = self.robot_amcl_pose.orientation
            rospy.logdebug("Using AMCL pose: [%.3f, %.3f, %.3f]", pos.x, pos.y, pos.z)
            return pos, orient
            
        # 备用2：使用odom
        elif self.robot_odom_pose is not None:
            pos = self.robot_odom_pose.position  
            orient = self.robot_odom_pose.orientation
            rospy.logdebug("Using ODOM pose: [%.3f, %.3f, %.3f]", pos.x, pos.y, pos.z)
            return pos, orient
                
        rospy.logwarn("⚠️ No valid robot position available!")
        return None, None
        
    def get_package_color_from_name(self, package_name):
        """根据包裹名称获取颜色"""
        return self.package_color_map.get(package_name, 'unknown')
        
    def find_nearest_package(self):
        """包裹搜索"""
        if self.current_zone != "pickup_zone":
            rospy.logwarn("Not in pickup zone, cannot search for packages")
            return None, None
            
        robot_pos, _ = self.get_robot_position()
        if robot_pos is None:
            rospy.logerr("Cannot get robot position for package search")
            return None, None
            
        rospy.loginfo("Searching for packages near robot position: (%.3f, %.3f)", 
                     robot_pos.x, robot_pos.y)
        
        nearest_package = None
        nearest_color = None
        min_distance = float('inf')
        target_color_package = None
        target_color_distance = float('inf')
        
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
                            
                            if (self.current_package_color and 
                                package_color == self.current_package_color and
                                distance < target_color_distance):
                                target_color_package = package_name
                                target_color_distance = distance
                                
                            if distance < min_distance:
                                min_distance = distance
                                nearest_package = package_name
                                nearest_color = package_color
                                
            except Exception as e:
                rospy.logdebug("Error checking package %s: %s", package_name, str(e))
                continue
                
        if target_color_package:
            target_color = self.get_package_color_from_name(target_color_package)
            rospy.loginfo("Found TARGET COLOR package: %s (color: %s) at distance %.3f", 
                         target_color_package, target_color, target_color_distance)
            return target_color_package, target_color
        elif nearest_package:
            rospy.loginfo("Found nearest package: %s (color: %s) at distance %.3f", 
                         nearest_package, nearest_color, min_distance)
            return nearest_package, nearest_color
        else:
            rospy.logwarn("No packages found in pickup area")
            return None, None
        
    def verify_package_pickup(self, package_name):
        """验证包裹是否成功被抓取"""
        try:
            response = self.get_model_state(package_name, "")
            if response.success:
                pkg_pos = response.pose.position
                robot_pos, _ = self.get_robot_position()
                
                if robot_pos is not None:
                    distance = ((pkg_pos.x - robot_pos.x)**2 + (pkg_pos.y - robot_pos.y)**2)**0.5
                    height_check = pkg_pos.z > 1.0
                    proximity_check = distance < 0.3
                    
                    if height_check and proximity_check:
                        rospy.loginfo("Package %s pickup verified: height=%.3f, distance=%.3f", 
                                    package_name, pkg_pos.z, distance)
                        return True
                    else:
                        rospy.logwarn("Package %s pickup failed: height=%.3f, distance=%.3f", 
                                    package_name, pkg_pos.z, distance)
                        return False
                        
        except Exception as e:
            rospy.logerr("Pickup verification error: %s", str(e))
            
        return False

    def get_drop_zone_center(self, package_color):
        """🔥 根据包裹颜色获取安全的drop区域坐标"""
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
        
    def magnetic_attach_package(self, package_name):
        """超稳定磁性吸附包裹"""
        rospy.loginfo("Attempting magnetic attachment of package: %s", package_name)
        
        for attempt in range(self.attach_verification_attempts):
            try:
                robot_pos, robot_orient = self.get_robot_position()
                if robot_pos is None:
                    rospy.logerr("Cannot get robot position for attachment")
                    return False
                    
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
                        rospy.loginfo("Package %s (color: %s) UNIFIED attached at robot pos [%.3f, %.3f] (attempt %d)", 
                                    package_name, self.attached_package_color, 
                                    robot_pos.x, robot_pos.y, attempt + 1)
                        
                        self.start_ultra_stable_following()
                        self.total_packages_picked += 1
                        return True
                    else:
                        rospy.logwarn("Package attachment verification failed (attempt %d)", attempt + 1)
                else:
                    rospy.logerr("Failed to attach package %s - Gazebo error (attempt %d)", 
                               package_name, attempt + 1)
                    
            except Exception as e:
                rospy.logerr("Magnetic attachment error (attempt %d): %s", attempt + 1, str(e))
                
            if attempt < self.attach_verification_attempts - 1:
                rospy.sleep(1.0)
                
        self.failed_picks += 1
        return False

    def magnetic_release_package(self):
        """🔥 增强的磁性释放包裹 - 安全距离放置"""
        if self.attached_package is None:
            rospy.logwarn("No package attached for release")
            return True
            
        package_color = self.attached_package_color
        if not package_color:
            rospy.logerr("Cannot determine package color for drop placement")
            return False
            
        rospy.logwarn("🔧 RELEASING PACKAGE: %s (color: %s) to %s drop zone from robot pos [%.3f, %.3f]", 
                     self.attached_package, package_color, package_color,
                     self.current_position[0], self.current_position[1])
        
        # 🔥 强制检查当前zone是否合适
        expected_zone = package_color + "_zone"
        if self.current_zone != expected_zone and not self.force_placement_mode:
            rospy.logwarn("⚠️ Warning: In zone '%s' but expected '%s'. Forcing placement anyway.", 
                         self.current_zone, expected_zone)
        
        # 🔥 获取安全的放置位置
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
                
                # 🔥 放置在安全的远离位置
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
                    rospy.logwarn("🔧 Package %s placed SAFELY at [%.3f, %.3f] (%.1fm from lanes) - attempt %d", 
                                self.attached_package, target_x, target_y, 
                                self.calculate_distance_to_nearest_lane(target_x, target_y), attempt + 1)
                    
                    rospy.sleep(3.0)
                    
                    if self.verify_package_placement(self.attached_package, target_x, target_y):
                        rospy.logwarn("✅ Package %s (color: %s) SAFELY placed in %s drop zone!", 
                                    self.attached_package, self.attached_package_color, package_color)
                        
                        self.attached_package = None
                        self.attached_package_color = None
                        self.total_packages_placed += 1
                        self.force_placement_mode = False
                        
                        rospy.loginfo("Safe package drop completed. Total packages: picked=%d, placed=%d", 
                                    self.total_packages_picked, self.total_packages_placed)
                        return True
                    else:
                        rospy.logwarn("Package placement verification failed (attempt %d)", attempt + 1)
                        
                        if attempt < self.drop_verification_attempts - 1:
                            rospy.loginfo("Retrying package placement...")
                            # 🔥 尝试稍微不同的位置
                            safe_drop_pos = self.get_safe_drop_position(package_color)
                            if safe_drop_pos:
                                target_x, target_y = safe_drop_pos
                            continue
                else:
                    rospy.logerr("Failed to place package %s - Gazebo error (attempt %d)", 
                               self.attached_package, attempt + 1)
                    
            except Exception as e:
                rospy.logerr("Package release error (attempt %d): %s", attempt + 1, str(e))
                
            if attempt < self.drop_verification_attempts - 1:
                rospy.sleep(2.0)
                
        rospy.logerr("Failed to place package safely after %d attempts", self.drop_verification_attempts)
        self.failed_places += 1
        self.force_placement_mode = False
        return False
        
    def get_safe_drop_position(self, package_color):
        """🔥 获取安全的包裹放置位置 - 远离车道线和导航路径"""
        if package_color not in self.drop_zone_centers:
            rospy.logerr("Unknown package color: %s", package_color)
            return None
            
        base_center = self.drop_zone_centers[package_color]
        base_x, base_y = base_center['x'], base_center['y']
        
        # 🔥 添加随机偏移避免包裹堆叠
        import random
        offset_x = random.uniform(-self.placement_offset_range, self.placement_offset_range)
        offset_y = random.uniform(-self.placement_offset_range, self.placement_offset_range)
        
        candidate_x = base_x + offset_x
        candidate_y = base_y + offset_y
        
        # 🔥 确保不会太接近车道线
        safe_x, safe_y = self.ensure_safe_distance_from_lanes(candidate_x, candidate_y)
        
        rospy.loginfo("Safe drop position for %s: [%.3f, %.3f] (base: [%.3f, %.3f], offset: [%.3f, %.3f])",
                     package_color, safe_x, safe_y, base_x, base_y, offset_x, offset_y)
        
        return [safe_x, safe_y]
        
    def ensure_safe_distance_from_lanes(self, x, y):
        """🔥 确保位置远离车道线"""
        # 主要车道线位置
        vertical_lanes = [0.0, 1.0, 3.0, -5.0, 5.0]  # x坐标
        horizontal_lanes = [-1.0, 1.0, 3.0, 5.0]     # y坐标
        
        safe_x = x
        safe_y = y
        
        # 🔥 检查并调整x坐标，确保远离垂直车道线
        for lane_x in vertical_lanes:
            if abs(safe_x - lane_x) < self.safe_distance_from_lanes:
                # 向远离车道线方向移动
                if safe_x > lane_x:
                    safe_x = lane_x + self.safe_distance_from_lanes
                else:
                    safe_x = lane_x - self.safe_distance_from_lanes
                rospy.logdebug("Adjusted x from %.3f to %.3f (avoiding lane at %.3f)", x, safe_x, lane_x)
                
        # 🔥 检查并调整y坐标，确保远离水平车道线
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
        """🔥 计算到最近车道线的距离"""
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
        rospy.loginfo("UNIFIED package following started at %dHz for package %s", 
                     self.follow_rate, self.attached_package)
        
    def stop_package_following(self):
        """停止包裹跟随"""
        self.follow_active = False
        if self.follow_thread is not None:
            self.follow_thread.join(timeout=3.0)
            self.follow_thread = None
        rospy.loginfo("Package following stopped")
        
    def ultra_stable_follow_loop(self):
        """🔥 超稳定包裹跟随循环 - 使用统一坐标系"""
        rate = rospy.Rate(self.follow_rate)
        consecutive_errors = 0
        max_errors = 15
        
        rospy.loginfo("Starting UNIFIED stable follow loop for package: %s", self.attached_package)
        
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
            
        rospy.loginfo("UNIFIED stable follow loop ended for package: %s", self.attached_package)
            
    def magnetic_pick_sequence(self):
        """磁性抓取序列"""
        if self.attached_package is not None:
            rospy.logwarn("Already carrying package %s, cannot pick another", self.attached_package)
            self.publish_status("failed")
            return
            
        self.manipulation_active = True
        self.publish_status("picking")
        
        try:
            rospy.loginfo("Starting UNIFIED magnetic pickup sequence")
            
            target_package, target_color = self.find_nearest_package()
            if target_package is None:
                rospy.logwarn("No package found for picking in current area")
                self.publish_status("failed")
                return
                
            rospy.loginfo("Target package selected: %s (color: %s)", target_package, target_color)
            
            rospy.loginfo("Waiting for robot stabilization...")
            rospy.sleep(3.0)
            
            if self.magnetic_attach_package(target_package):
                rospy.loginfo("Package %s UNIFIED attached (actual color: %s)", 
                            target_package, self.attached_package_color)
                
                self.publish_actual_package_color(self.attached_package_color)
                
                rospy.sleep(4.0)
                self.publish_status("pick_completed")
                
                rospy.loginfo("Pickup sequence completed successfully")
            else:
                rospy.logwarn("Failed to attach package after multiple attempts")
                self.publish_status("failed")
                
        except Exception as e:
            rospy.logerr("Magnetic pickup sequence error: %s", str(e))
            self.publish_status("failed")
        finally:
            self.manipulation_active = False

    def magnetic_place_sequence(self):
        """🔥 增强的磁性放置序列"""
        if self.attached_package is None:
            rospy.logwarn("No package attached for placement")
            self.publish_status("failed")
            return
            
        # 🔥 强制设置active状态
        self.manipulation_active = True
        self.publish_status("placing")
        
        try:
            package_color = self.attached_package_color
            rospy.logwarn("🔧 STARTING SAFE PLACEMENT for package %s (color: %s) from pos [%.3f, %.3f]", 
                         self.attached_package, package_color,
                         self.current_position[0], self.current_position[1])
            
            expected_zone = package_color + "_zone"
            if self.current_zone != expected_zone:
                rospy.logwarn("⚠️ Warning: Robot in zone '%s' but expected '%s' for package color '%s'", 
                             self.current_zone, expected_zone, package_color)
                rospy.logwarn("⚠️ Proceeding with safe placement anyway...")
            
            rospy.loginfo("Waiting for robot stabilization before placement...")
            rospy.sleep(3.0)
            
            if self.magnetic_release_package():
                rospy.logwarn("✅ Package safely placed in %s drop zone!", package_color)
                rospy.sleep(2.0)
                self.publish_status("place_completed")
            else:
                rospy.logwarn("❌ Failed to release package to safe drop zone")
                self.publish_status("failed")
                
        except Exception as e:
            rospy.logerr("Magnetic placement sequence error: %s", str(e))
            self.publish_status("failed")
        finally:
            self.manipulation_active = False
            self.force_placement_mode = False
            
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
            'safe_placement_enabled': True,
            'min_lane_distance': self.safe_distance_from_lanes
        }

if __name__ == '__main__':
    try:
        manager = ManipulationManager()
        
        def print_stats(event):
            stats = manager.get_manipulation_stats()
            rospy.loginfo("SAFE PLACEMENT Stats: Picked=%d, Placed=%d, Failed_picks=%d, Failed_places=%d, Carrying=%s, Pos=[%.3f,%.3f], Source=%s, SafeDist=%.1fm", 
                         stats['total_picked'], stats['total_placed'], 
                         stats['failed_picks'], stats['failed_places'], stats['current_package'],
                         stats['robot_position'][0], stats['robot_position'][1], stats['coordinate_source'],
                         stats['min_lane_distance'])
                         
        stats_timer = rospy.Timer(rospy.Duration(30.0), print_stats)
        
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ManipulationManager node terminated")
    except Exception as e:
        rospy.logerr("ManipulationManager fatal error: %s", str(e))