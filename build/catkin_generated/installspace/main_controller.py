#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
import time
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from gazebo_msgs.srv import GetModelState

# Python 2.7 compatible enum replacement
class RobotState(object):
    INIT = "INIT"
    NAVIGATE_TO_PICKUP = "NAVIGATE_TO_PICKUP"
    DETECT_PACKAGE_COLOR = "DETECT_PACKAGE_COLOR"
    PICK_PACKAGE = "PICK_PACKAGE"
    EXIT_PICKUP_ZONE = "EXIT_PICKUP_ZONE"
    NAVIGATE_TO_DROP = "NAVIGATE_TO_DROP"
    PLACE_PACKAGE = "PLACE_PACKAGE"
    RETURN_TO_PICKUP = "RETURN_TO_PICKUP"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

class OptimizedMainController(object):
    def __init__(self):
        rospy.init_node('optimized_main_controller', anonymous=False)
        
        # 等待Gazebo服务
        rospy.loginfo("Waiting for Gazebo get_model_state service...")
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=10.0)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.gazebo_available = True
            rospy.loginfo("Gazebo service connected - using ground truth coordinates")
        except:
            rospy.logwarn("Gazebo service not available - using AMCL coordinates")
            self.gazebo_available = False
            self.get_model_state = None
        
        # State management
        self.current_state = RobotState.INIT
        self.previous_state = None
        self.current_package_color = None
        self.actual_package_color = None
        self.task_start_time = None
        self.state_start_time = None
        
        # 🔥 快速状态转换控制
        self.fast_transition_mode = True
        self.last_placed_package_color = None
        self.placement_completed_time = None
        self.quick_return_delay = 1.0  # 减少到1秒
        
        # Publishers
        self.state_pub = rospy.Publisher('/robot_state', String, queue_size=10)
        self.package_color_pub = rospy.Publisher('/current_package_color', String, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/current_zone', String, self.zone_callback)
        rospy.Subscriber('/package_color', String, self.package_color_callback)
        rospy.Subscriber('/actual_package_color', String, self.actual_package_color_callback)
        rospy.Subscriber('/navigation/arrived', String, self.navigation_callback)
        rospy.Subscriber('/manipulation/status', String, self.manipulation_callback)
        rospy.Subscriber('/fused_obstacle_info', String, self.obstacle_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/front_lane_guidance', String, self.lane_guidance_callback)
        
        # Status tracking
        self.current_zone = "unknown"
        self.navigation_status = "idle"
        self.manipulation_status = "idle"
        self.obstacle_detected = False
        self.lane_guidance = "no_lane"
        
        # 统一坐标系统
        self.robot_amcl_pose = None
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, yaw]
        
        # 🔥 快速超时时间
        self.pickup_timeout = 45.0          # 减少pickup超时
        self.navigation_timeout = 120.0     # 减少导航超时
        self.drop_navigation_timeout = 150.0 # 减少drop导航超时
        self.return_navigation_timeout = 180.0 # 减少返回导航超时
        self.exit_timeout = 60.0            # 减少退出超时
        self.place_timeout = 30.0           # 减少放置超时
        
        # Commands publishers
        self.navigation_cmd_pub = rospy.Publisher('/navigation/command', String, queue_size=10)
        self.manipulation_cmd_pub = rospy.Publisher('/manipulation/command', String, queue_size=10)
        
        # 🔥 快速返回控制
        self.return_attempts = 0
        self.max_return_attempts = 2  # 减少重试次数
        self.return_command_sent = False
        self.return_command_time = None
        self.return_command_timeout = 15.0  # 减少返回命令超时
        
        # Exit control
        self.exit_attempts = 0
        self.max_exit_attempts = 2
        
        # Task completion tracking
        self.completed_packages = 0
        self.target_packages = rospy.get_param('~target_packages', 20)
        
        # 🔥 智能状态检查
        self.last_position = [0.0, 0.0, 0.0]
        self.position_change_threshold = 0.05
        self.stuck_check_time = 20.0  # 减少卡住检查时间
        self.last_movement_time = time.time()
        
        # 🔥 快速错误恢复
        self.error_recovery_enabled = True
        self.consecutive_errors = 0
        self.max_consecutive_errors = 2
        
        # Debug timer
        self.debug_timer = rospy.Timer(rospy.Duration(10.0), self.debug_status)
        
        rospy.loginfo("OptimizedMainController initialized with FAST STATE TRANSITIONS")
        
    def lane_guidance_callback(self, msg):
        """处理车道线引导信息"""
        self.lane_guidance = msg.data
        
    def debug_status(self, event):
        """定期打印调试信息"""
        rospy.loginfo("=== OPTIMIZED CONTROLLER STATUS ===")
        rospy.loginfo("State: %s | Zone: %s | Position: [%.2f, %.2f, %.2f]", 
                     self.current_state, self.current_zone,
                     self.current_position[0], self.current_position[1], self.current_position[2])
        rospy.loginfo("Navigation: %s | Manipulation: %s | Lane: %s", 
                     self.navigation_status, self.manipulation_status, self.lane_guidance)
        rospy.loginfo("Packages: Expected=%s | Actual=%s | Last=%s | Done=%d/%d", 
                     self.current_package_color, self.actual_package_color,
                     self.last_placed_package_color, self.completed_packages, self.target_packages)
        
        # 🔥 快速位置和状态检查
        self.check_position_and_state_consistency()
        
        if self.state_start_time:
            elapsed = time.time() - self.state_start_time
            timeout = self.get_current_timeout()
            remaining = timeout - elapsed
            rospy.loginfo("State elapsed: %.1fs | Remaining: %.1fs", elapsed, remaining)
            
        rospy.loginfo("========================================")

    def check_position_and_state_consistency(self):
        """🔥 快速检查位置和状态一致性"""
        x, y = self.current_position[0], self.current_position[1]
        
        # 检查是否卡在不合适的位置
        if self.current_state == RobotState.RETURN_TO_PICKUP:
            if self.is_in_pickup_edge_area():
                rospy.logwarn("🔧 FAST FIX: Already in pickup area, switching to DETECT")
                self.change_state(RobotState.DETECT_PACKAGE_COLOR)
                return
                
        elif self.current_state == RobotState.PLACE_PACKAGE:
            expected_color = self.actual_package_color if self.actual_package_color else self.current_package_color
            if expected_color:
                expected_zone = expected_color + "_zone"
                if self.current_zone != expected_zone:
                    rospy.logwarn("🔧 ZONE MISMATCH: In %s but should be in %s", 
                                 self.current_zone, expected_zone)
                    # 重新导航到正确区域
                    self.change_state(RobotState.NAVIGATE_TO_DROP)

    def get_current_timeout(self):
        """根据当前状态获取合适的超时时间"""
        timeout_map = {
            RobotState.NAVIGATE_TO_DROP: self.drop_navigation_timeout,
            RobotState.RETURN_TO_PICKUP: self.return_navigation_timeout,
            RobotState.NAVIGATE_TO_PICKUP: self.navigation_timeout,
            RobotState.EXIT_PICKUP_ZONE: self.exit_timeout,
            RobotState.PICK_PACKAGE: self.pickup_timeout,
            RobotState.PLACE_PACKAGE: self.place_timeout,
            RobotState.DETECT_PACKAGE_COLOR: self.pickup_timeout
        }
        return timeout_map.get(self.current_state, self.navigation_timeout)

    def check_if_robot_stuck(self):
        """🔥 快速检查机器人是否卡住"""
        current_pos = self.current_position
        
        position_change = ((current_pos[0] - self.last_position[0])**2 + 
                          (current_pos[1] - self.last_position[1])**2)**0.5
        
        if position_change > self.position_change_threshold:
            self.last_movement_time = time.time()
            self.last_position = current_pos[:]
            return False
        else:
            stuck_time = time.time() - self.last_movement_time
            return stuck_time > self.stuck_check_time

    def smart_stop_if_needed(self, reason=""):
        """智能停止"""
        if self.is_navigation_active():
            rospy.logwarn("Smart stop: %s", reason)
            self.send_navigation_command("stop")
            rospy.sleep(0.3)  # 减少等待时间
        
    def zone_callback(self, msg):
        old_zone = self.current_zone
        self.current_zone = msg.data
        if old_zone != self.current_zone:
            rospy.loginfo("Zone: %s -> %s at [%.2f, %.2f]", 
                         old_zone, self.current_zone, self.current_position[0], self.current_position[1])
        
    def package_color_callback(self, msg):
        if msg.data in ['red', 'blue', 'green', 'purple'] and self.current_state == RobotState.DETECT_PACKAGE_COLOR:
            self.current_package_color = msg.data
            rospy.loginfo("Package detected: %s at [%.2f, %.2f]", 
                         self.current_package_color, self.current_position[0], self.current_position[1])
            
    def actual_package_color_callback(self, msg):
        """处理实际抓取的包裹颜色"""
        self.actual_package_color = msg.data
        rospy.loginfo("Actual package picked: %s at [%.2f, %.2f]", 
                     self.actual_package_color, self.current_position[0], self.current_position[1])
        
    def navigation_callback(self, msg):
        """导航状态处理"""
        old_status = self.navigation_status
        self.navigation_status = msg.data
        
        if old_status != self.navigation_status:
            rospy.loginfo("Navigation: %s -> %s at [%.2f, %.2f]", 
                         old_status, self.navigation_status, self.current_position[0], self.current_position[1])
        
    def manipulation_callback(self, msg):
        old_status = self.manipulation_status
        self.manipulation_status = msg.data
        if old_status != self.manipulation_status:
            rospy.loginfo("Manipulation: %s -> %s at [%.2f, %.2f]", 
                         old_status, self.manipulation_status, self.current_position[0], self.current_position[1])
        
    def obstacle_callback(self, msg):
        self.obstacle_detected = (msg.data == "obstacle_detected")
        
    def pose_callback(self, msg):
        """统一的位姿回调"""
        # 优先使用Gazebo坐标
        if self.gazebo_available:
            try:
                response = self.get_model_state("warehouse_robot", "")
                if response.success:
                    pos = response.pose.position
                    orient = response.pose.orientation
                    
                    import tf
                    try:
                        euler = tf.transformations.euler_from_quaternion([
                            orient.x, orient.y, orient.z, orient.w
                        ])
                        yaw = euler[2]
                    except:
                        yaw = 0.0
                        
                    self.current_position = [pos.x, pos.y, yaw]
                    return
            except:
                pass
        
        # 备用：使用AMCL坐标
        self.robot_amcl_pose = msg.pose.pose
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        
        import tf
        try:
            euler = tf.transformations.euler_from_quaternion([
                orient.x, orient.y, orient.z, orient.w
            ])
            yaw = euler[2]
        except:
            yaw = 0.0
            
        self.current_position = [pos.x, pos.y, yaw]
        
    def publish_state(self):
        """Publish current robot state"""
        state_msg = String()
        state_msg.data = self.current_state
        self.state_pub.publish(state_msg)
        
        # 发布当前包裹颜色
        color_to_publish = self.actual_package_color if self.actual_package_color else self.current_package_color
        if color_to_publish:
            color_msg = String()
            color_msg.data = color_to_publish
            self.package_color_pub.publish(color_msg)
            
    def change_state(self, new_state):
        """🔥 优化的状态转换"""
        if self.current_state != new_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_start_time = time.time()
            
            # 🔥 关键修复：彻底清理包裹信息
            if new_state == RobotState.DETECT_PACKAGE_COLOR:
                rospy.logwarn("🔧 CLEARING all package info for fresh detection")
                self.current_package_color = None
                self.actual_package_color = None
                # 发布清空信号
                empty_msg = String()
                empty_msg.data = ""
                self.package_color_pub.publish(empty_msg)
                
            # 🔥 快速重置状态特定变量
            elif new_state == RobotState.RETURN_TO_PICKUP:
                self.return_attempts = 0
                self.return_command_sent = False
                self.return_command_time = None
                rospy.logwarn("🔥 FAST RETURN STATE at [%.2f, %.2f]", 
                             self.current_position[0], self.current_position[1])
            
            # 重置移动检测
            self.last_movement_time = time.time()
            self.last_position = self.current_position[:]
            
            # 🔥 特定状态的快速准备
            if new_state in [RobotState.PICK_PACKAGE, RobotState.PLACE_PACKAGE]:
                self.ensure_robot_stopped_for_manipulation()
                
            elif new_state == RobotState.EXIT_PICKUP_ZONE:
                self.exit_attempts += 1
                
            rospy.loginfo("🔥 STATE: %s -> %s at [%.2f, %.2f] (timeout: %.1fs)", 
                         self.previous_state if self.previous_state else "None", 
                         self.current_state, self.current_position[0], self.current_position[1],
                         self.get_current_timeout())
            self.publish_state()
    
    def ensure_robot_stopped_for_manipulation(self):
        """确保机器人为操作做好准备"""
        if self.is_navigation_active():
            rospy.loginfo("Preparing for manipulation...")
            self.send_navigation_command("stop")
            rospy.sleep(0.5)  # 减少等待时间
            
    def check_timeout(self, timeout_duration=None):
        """🔥 优化的超时检查"""
        if self.state_start_time is None:
            return False
            
        if timeout_duration is None:
            timeout_duration = self.get_current_timeout()
            
        elapsed = time.time() - self.state_start_time
        
        # 🔥 特殊处理返回导航超时
        if self.current_state == RobotState.RETURN_TO_PICKUP:
            # 如果已经在pickup区域，直接完成
            if self.is_in_pickup_edge_area():
                rospy.logwarn("🔥 FAST RETURN: Already in pickup area")
                return False  # 不超时，让正常流程处理
                
            # 如果命令超时未响应
            if (self.return_command_sent and self.return_command_time and 
                time.time() - self.return_command_time > self.return_command_timeout):
                rospy.logwarn("🔥 Return command timeout")
                return True
            
            # 检查是否卡住
            if self.check_if_robot_stuck():
                rospy.logwarn("🔥 Robot stuck during return")
                return True
                
            return elapsed > timeout_duration
        
        # 其他状态的超时检查
        navigation_states = [RobotState.NAVIGATE_TO_DROP, RobotState.NAVIGATE_TO_PICKUP, 
                           RobotState.EXIT_PICKUP_ZONE]
        
        if self.current_state in navigation_states:
            if self.is_navigation_active():
                if self.check_if_robot_stuck():
                    rospy.logwarn("🔥 Robot stuck, timeout triggered")
                    return True
                return False
            else:
                return elapsed > timeout_duration
        else:
            return elapsed > timeout_duration
        
    def send_navigation_command(self, command):
        """发送导航命令"""
        cmd_msg = String()
        cmd_msg.data = command
        self.navigation_cmd_pub.publish(cmd_msg)
        rospy.logwarn("🚀 Navigation command: %s from [%.2f, %.2f]", 
                     command, self.current_position[0], self.current_position[1])
        
        # 🔥 特殊跟踪返回命令
        if command.startswith("return_from_") and command.endswith("_to_pickup"):
            self.return_command_sent = True
            self.return_command_time = time.time()
            rospy.logwarn("🔥 FAST RETURN command sent: %s", command)
        
    def send_manipulation_command(self, command):
        """发送操作命令"""
        cmd_msg = String()
        cmd_msg.data = command
        self.manipulation_cmd_pub.publish(cmd_msg)
        rospy.loginfo("🔧 Manipulation: %s at [%.2f, %.2f]", 
                     command, self.current_position[0], self.current_position[1])
        
    def is_navigation_active(self):
        """检查导航是否活跃"""
        active_statuses = [
            "navigating_to_pickup", "navigating_to_safety", "navigating_to_drop"
        ]
        return any(status in self.navigation_status for status in active_statuses)
        
    def is_pickup_approach_completed(self):
        """检查是否到达pickup接近位置"""
        completed_statuses = [
            "arrived_pickup_approach", "timeout_pickup_approach"
        ]
        position_check = (
            -0.8 <= self.current_position[0] <= 0.8 and
            -1.5 <= self.current_position[1] <= 0.5
        )
        
        return (any(status in self.navigation_status for status in completed_statuses) or
                position_check)
    
    def is_safety_exit_completed(self):
        """检查安全退出是否完成"""
        completed_statuses = [
            "arrived_safety_exit", "timeout_safety_exit", "exit_completed"
        ]
        position_check = (
            -1.0 <= self.current_position[0] <= 1.0 and
            self.current_position[1] > 0.5
        )
        
        return (any(status in self.navigation_status for status in completed_statuses) or
                position_check)
    
    def is_drop_navigation_completed(self):
        """检查放置导航是否完成"""
        target_color = self.actual_package_color if self.actual_package_color else self.current_package_color
        if not target_color:
            return False
        completed_statuses = [
            "arrived_" + target_color + "_drop",
            "timeout_" + target_color + "_drop"
        ]
        return any(status in self.navigation_status for status in completed_statuses)

    def is_return_navigation_completed(self):
        """🔥 快速返回导航完成检查"""
        completed_statuses = [
            "arrived_return_to_pickup", "timeout_return_to_pickup",
            "arrived_pickup_approach", "timeout_pickup_approach"
        ]
        
        # 检查导航状态
        status_completed = any(status in self.navigation_status for status in completed_statuses)
        
        if status_completed:
            rospy.logwarn("🔥 Return completed by status: %s", self.navigation_status)
            return True
            
        # 检查位置
        position_completed = self.is_in_pickup_edge_area()
        
        if position_completed:
            rospy.logwarn("🔥 Return completed by position at [%.2f, %.2f]", 
                         self.current_position[0], self.current_position[1])
            return True
            
        return False
        
    def is_in_pickup_edge_area(self):
        """检查是否在pickup边缘区域"""
        x, y = self.current_position[0], self.current_position[1]
        
        # pickup边缘区域
        edge_area = (
            -1.5 <= x <= 1.5 and
            -2.0 <= y <= 0.8
        )
        
        return edge_area

    def is_in_correct_drop_zone(self):
        """检查是否在正确的drop区域"""
        target_color = self.actual_package_color if self.actual_package_color else self.current_package_color
        if not target_color:
            return False
            
        expected_zone = target_color + "_zone"
        return self.current_zone == expected_zone
        
    def handle_init_state(self):
        """Handle initialization state"""
        rospy.loginfo("🔥 Robot initializing at [%.2f, %.2f]", 
                     self.current_position[0], self.current_position[1])
        self.task_start_time = time.time()
        
        rospy.sleep(2.0)  # 减少初始化时间
        self.change_state(RobotState.NAVIGATE_TO_PICKUP)
        
    def handle_navigate_to_pickup_state(self):
        """Handle navigation to pickup"""
        if self.is_pickup_approach_completed():
            rospy.loginfo("Pickup approach completed at [%.2f, %.2f]", 
                         self.current_position[0], self.current_position[1])
            self.change_state(RobotState.DETECT_PACKAGE_COLOR)
            return
            
        if not self.is_navigation_active():
            rospy.loginfo("Starting navigation to pickup from [%.2f, %.2f]", 
                         self.current_position[0], self.current_position[1])
            self.send_navigation_command("goto_pickup_approach")
            
        if self.check_timeout():
            rospy.logwarn("Navigation to pickup timed out")
            self.change_state(RobotState.ERROR)
            
    def handle_detect_package_color_state(self):
        """Handle package detection"""
        if not self.is_in_pickup_edge_area():
            rospy.logwarn("🔥 Not in pickup area at [%.2f, %.2f], returning", 
                         self.current_position[0], self.current_position[1])
            self.change_state(RobotState.NAVIGATE_TO_PICKUP)
            return
            
        if self.current_package_color:
            rospy.loginfo("🔥 Package detected at [%.2f, %.2f]: %s", 
                         self.current_position[0], self.current_position[1], self.current_package_color)
            self.change_state(RobotState.PICK_PACKAGE)
        elif self.check_timeout():
            rospy.logwarn("🔥 Package detection timed out, clearing and retrying")
            # 🔥 修复：超时时清理包裹信息重新开始
            self.current_package_color = None
            self.actual_package_color = None
            self.change_state(RobotState.NAVIGATE_TO_PICKUP)

    def handle_pick_package_state(self):
        """包裹抓取处理"""
        if self.manipulation_status == "pick_completed":
            rospy.loginfo("🔥 Package picked at [%.2f, %.2f] - quick exit", 
                         self.current_position[0], self.current_position[1])
            self.change_state(RobotState.EXIT_PICKUP_ZONE)
        elif self.manipulation_status == "idle":
            rospy.loginfo("🔥 Starting pickup at [%.2f, %.2f]", 
                         self.current_position[0], self.current_position[1])
            self.send_manipulation_command("pick_package")
        elif self.manipulation_status == "failed":
            rospy.logwarn("🔥 Pickup failed, clearing info and retrying detection")
            # 🔥 修复：抓取失败时清理包裹信息
            self.current_package_color = None
            self.actual_package_color = None
            self.change_state(RobotState.DETECT_PACKAGE_COLOR)
        elif self.check_timeout():
            rospy.logwarn("🔥 Pickup timed out")
            self.change_state(RobotState.ERROR)
            
    def handle_exit_pickup_zone_state(self):
        """处理退出pickup区域"""
        if self.is_safety_exit_completed():
            rospy.loginfo("🔥 Safe exit completed at [%.2f, %.2f]", 
                         self.current_position[0], self.current_position[1])
            self.exit_attempts = 0
            self.change_state(RobotState.NAVIGATE_TO_DROP)
            return
            
        if self.exit_attempts > self.max_exit_attempts:
            rospy.logwarn("🔥 Max exit attempts, forcing completion")
            self.exit_attempts = 0
            self.change_state(RobotState.NAVIGATE_TO_DROP)
            return
            
        if not self.is_navigation_active():
            rospy.logwarn("🔥 Sending exit command from [%.2f, %.2f]", 
                         self.current_position[0], self.current_position[1])
            self.send_navigation_command("exit_pickup_to_safety")
            return
            
        if self.check_timeout():
            rospy.logwarn("🔥 Exit timed out, forcing completion")
            self.exit_attempts = 0
            self.change_state(RobotState.NAVIGATE_TO_DROP)
            
    def handle_navigate_to_drop_state(self):
        """处理到drop区域的导航"""
        target_color = self.actual_package_color if self.actual_package_color else self.current_package_color
        if not target_color:
            rospy.logerr("No package color for drop navigation")
            self.change_state(RobotState.ERROR)
            return
            
        target_zone = target_color + "_zone"
        
        if self.current_zone == target_zone:
            rospy.loginfo("Arrived at %s zone at [%.2f, %.2f]", 
                         target_color, self.current_position[0], self.current_position[1])
            self.change_state(RobotState.PLACE_PACKAGE)
            return
            
        if self.is_drop_navigation_completed():
            rospy.loginfo("Drop navigation completed")
            rospy.sleep(1.0)
            if self.current_zone == target_zone:
                self.change_state(RobotState.PLACE_PACKAGE)
            else:
                self.send_navigation_command("goto_" + target_color + "_drop")
            return
            
        if not self.is_navigation_active():
            rospy.loginfo("Starting navigation to %s drop from [%.2f, %.2f]", 
                         target_color, self.current_position[0], self.current_position[1])
            self.send_navigation_command("goto_" + target_color + "_drop")
            
        if self.check_timeout():
            rospy.logwarn("Drop navigation timed out")
            self.change_state(RobotState.ERROR)

    def handle_place_package_state(self):
        """🔥 优化的包裹放置处理"""
        if not self.is_in_correct_drop_zone():
            target_color = self.actual_package_color if self.actual_package_color else self.current_package_color
            rospy.logwarn("🔥 Wrong zone at [%.2f, %.2f]! Current: %s, Expected: %s_zone", 
                         self.current_position[0], self.current_position[1], 
                         self.current_zone, target_color)
            self.change_state(RobotState.NAVIGATE_TO_DROP)
            return
        
        if self.manipulation_status == "place_completed":
            rospy.logwarn("🔥 PACKAGE PLACED at [%.2f, %.2f] - FAST RETURN STARTING", 
                         self.current_position[0], self.current_position[1])
            
            self.completed_packages += 1
            
            # 🔥 记录放置信息
            self.last_placed_package_color = self.actual_package_color if self.actual_package_color else self.current_package_color
            self.placement_completed_time = time.time()
            
            # 🔥 彻底清除包裹信息
            rospy.logwarn("🔧 CLEARING package info after placement")
            self.current_package_color = None
            self.actual_package_color = None
            
            if self.completed_packages >= self.target_packages:
                rospy.loginfo("🔥 All packages completed!")
                self.change_state(RobotState.COMPLETED)
            else:
                rospy.logwarn("🔥 FAST TRANSITION: Package %d/%d done, quick return", 
                             self.completed_packages, self.target_packages)
                # 🔥 立即切换到返回状态
                rospy.sleep(self.quick_return_delay)
                self.change_state(RobotState.RETURN_TO_PICKUP)
                
        elif self.manipulation_status == "idle":
            rospy.loginfo("🔥 Starting placement in %s zone at [%.2f, %.2f]", 
                         self.current_zone, self.current_position[0], self.current_position[1])
            self.send_manipulation_command("place_package")
        elif self.manipulation_status == "failed":
            rospy.logwarn("🔥 Placement failed")
            self.change_state(RobotState.ERROR)
        elif self.check_timeout():
            rospy.logwarn("🔥 Placement timed out")
            self.change_state(RobotState.ERROR)
            
    def handle_return_to_pickup_state(self):
        """🔥 快速返回pickup区域处理"""
        rospy.logwarn("FAST RETURN handler (attempt %d/%d) from [%.2f, %.2f]", 
                     self.return_attempts + 1, self.max_return_attempts,
                     self.current_position[0], self.current_position[1])
        
        # 检查返回导航是否完成
        if self.is_return_navigation_completed():
            rospy.logwarn("Return navigation completed at [%.2f, %.2f]", 
                         self.current_position[0], self.current_position[1])
            self.return_attempts = 0
            self.return_command_sent = False
            self.change_state(RobotState.DETECT_PACKAGE_COLOR)
            return
            
        # 发送返回命令
        if not self.is_navigation_active() and not self.return_command_sent:
            if self.last_placed_package_color:
                return_command = "return_from_" + self.last_placed_package_color + "_to_pickup"
                rospy.logwarn("SENDING RETURN: %s from [%.2f, %.2f]", 
                             return_command, self.current_position[0], self.current_position[1])
                self.send_navigation_command(return_command)
                self.return_attempts += 1
            else:
                rospy.logwarn("Using fallback approach from [%.2f, %.2f]", 
                             self.current_position[0], self.current_position[1])
                self.send_navigation_command("goto_pickup_approach")
                self.return_attempts += 1
            return
            
        # 检查超时或最大重试
        if self.check_timeout() or self.return_attempts >= self.max_return_attempts:
            if self.return_attempts >= self.max_return_attempts:
                rospy.logwarn("Max return attempts, emergency fallback")
                self.emergency_return_fallback()
            else:
                rospy.logwarn("Return timeout, retrying")
                self.retry_return_navigation()

    def retry_return_navigation(self):
        """🔥 快速重试返回导航"""
        rospy.logwarn("🔥 FAST RETRY return from [%.2f, %.2f]", 
                     self.current_position[0], self.current_position[1])
        
        # 重置状态
        self.return_command_sent = False
        self.return_command_time = None
        
        # 快速停止并重新开始
        self.send_navigation_command("stop")
        rospy.sleep(1.0)  # 减少等待时间
        
        if self.last_placed_package_color and self.return_attempts < self.max_return_attempts:
            return_command = "return_from_" + self.last_placed_package_color + "_to_pickup"
            self.send_navigation_command(return_command)
            self.return_attempts += 1
        else:
            self.emergency_return_fallback()

    def emergency_return_fallback(self):
        """🔥 紧急返回备用方案"""
        rospy.logwarn("🔥 EMERGENCY FALLBACK from [%.2f, %.2f]", 
                     self.current_position[0], self.current_position[1])
        
        # 使用简单approach
        self.send_navigation_command("goto_pickup_approach")
        
        # 短暂等待后强制完成
        rospy.sleep(3.0)
        
        if self.is_in_pickup_edge_area():
            rospy.logwarn("🔥 Emergency success at [%.2f, %.2f]", 
                         self.current_position[0], self.current_position[1])
            self.change_state(RobotState.DETECT_PACKAGE_COLOR)
        else:
            rospy.logwarn("🔥 Emergency failed, switching to ERROR")
            self.change_state(RobotState.ERROR)
            
    def handle_completed_state(self):
        """Handle mission completion"""
        rospy.loginfo("🔥 Mission completed at [%.2f, %.2f]! Processed %d packages.", 
                     self.current_position[0], self.current_position[1], self.completed_packages)
        total_time = time.time() - self.task_start_time if self.task_start_time else 0
        rospy.loginfo("🔥 Total mission time: %.1f seconds", total_time)
        
        self.send_navigation_command("stop")
        rospy.sleep(3.0)

    def handle_error_state(self):
        """🔥 优化的错误状态处理"""
        self.consecutive_errors += 1
        rospy.logwarn("🔥 ERROR STATE at [%.2f, %.2f] (error #%d)", 
                     self.current_position[0], self.current_position[1], self.consecutive_errors)
        
        if self.consecutive_errors > self.max_consecutive_errors:
            rospy.logerr("🔥 Too many consecutive errors, stopping")
            self.send_navigation_command("stop")
            return
        
        # 快速停止并等待
        self.send_navigation_command("stop")
        rospy.sleep(2.0)  # 减少等待时间
        
        # 🔥 清理包裹信息防止残留
        rospy.logwarn("🔧 CLEARING package info in error recovery")
        self.current_package_color = None
        self.actual_package_color = None
        
        # 🔥 智能恢复策略
        if self.is_in_pickup_edge_area():
            rospy.loginfo("🔥 In pickup area - continuing with detection")
            self.change_state(RobotState.DETECT_PACKAGE_COLOR)
        elif self.actual_package_color and self.is_in_correct_drop_zone():
            rospy.loginfo("🔥 Carrying package in correct zone - placing")
            self.change_state(RobotState.PLACE_PACKAGE)
        elif self.actual_package_color or self.current_package_color:
            rospy.loginfo("🔥 Carrying package - navigating to drop")
            self.change_state(RobotState.NAVIGATE_TO_DROP)
        else:
            rospy.loginfo("🔥 No package - returning to pickup")
            self.change_state(RobotState.NAVIGATE_TO_PICKUP)
        
        # 重置错误计数如果成功恢复
        if self.current_state != RobotState.ERROR:
            self.consecutive_errors = 0
        
    def run(self):
        """Main state machine execution loop"""
        rate = rospy.Rate(20)  # 提高到20 Hz for faster response
        
        rospy.loginfo("🔥 Starting OPTIMIZED warehouse robot with FAST STATE TRANSITIONS")
        
        while not rospy.is_shutdown():
            try:
                if self.current_state == RobotState.INIT:
                    self.handle_init_state()
                elif self.current_state == RobotState.NAVIGATE_TO_PICKUP:
                    self.handle_navigate_to_pickup_state()
                elif self.current_state == RobotState.DETECT_PACKAGE_COLOR:
                    self.handle_detect_package_color_state()
                elif self.current_state == RobotState.PICK_PACKAGE:
                    self.handle_pick_package_state()
                elif self.current_state == RobotState.EXIT_PICKUP_ZONE:
                    self.handle_exit_pickup_zone_state()
                elif self.current_state == RobotState.NAVIGATE_TO_DROP:
                    self.handle_navigate_to_drop_state()
                elif self.current_state == RobotState.PLACE_PACKAGE:
                    self.handle_place_package_state()
                elif self.current_state == RobotState.RETURN_TO_PICKUP:
                    self.handle_return_to_pickup_state()
                elif self.current_state == RobotState.COMPLETED:
                    self.handle_completed_state()
                elif self.current_state == RobotState.ERROR:
                    self.handle_error_state()
                    
            except Exception as e:
                rospy.logerr("🔥 State machine error: %s", str(e))
                self.change_state(RobotState.ERROR)
                
            # Publish current state
            self.publish_state()
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = OptimizedMainController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("OptimizedMainController terminated")
    except Exception as e:
        rospy.logerr("OptimizedMainController fatal error: %s", str(e))