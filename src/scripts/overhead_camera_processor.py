#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetModelState

class OverheadCameraProcessor(object):
    def __init__(self):
        rospy.init_node('overhead_camera_processor', anonymous=False)
        
        self.bridge = CvBridge()
        
        # 等待Gazebo服务以获取准确坐标
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.gazebo_available = True
        except:
            self.gazebo_available = False
            self.get_model_state = None
        
        # 车道线检测参数
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])
        
        # 导航控制参数
        self.lane_width = 0.15  # 车道线宽度
        self.safe_distance_from_lane = 0.05  # 距离车道线的安全距离
        self.position_tolerance = 0.1  # 位置容忍度
        self.angle_tolerance = 0.087  # 角度容忍度（5度）
        
        # 像素到世界坐标转换参数
        self.pixels_per_meter = 100.0  # 每米100像素
        self.image_center_x = 960  # 图像中心X
        self.image_center_y = 540  # 图像中心Y
        
        # 控制状态 - 恢复车道线跟随控制
        self.lane_following_active = False
        self.target_angle = None
        self.current_phase = "idle"  # idle, rotating, moving, completed
        self.phase_start_time = None
        
        # Publishers - 恢复车道线跟随控制
        self.lane_guidance_pub = rospy.Publisher('/lane_guidance', String, queue_size=10)
        self.lane_correction_pub = rospy.Publisher('/lane_correction', Twist, queue_size=10)
        self.lane_status_pub = rospy.Publisher('/lane_following_status', String, queue_size=10)
        self.lane_debug_pub = rospy.Publisher('/overhead_lane_debug', Image, queue_size=1)
        
        # Subscribers - 恢复车道线跟随控制
        rospy.Subscriber('/overhead_camera/image_raw', Image, self.image_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/robot_state', String, self.robot_state_callback)
        rospy.Subscriber('/navigation/command', String, self.navigation_command_callback)
        
        # 机器人状态
        self.robot_amcl_pose = None
        self.robot_odom_pose = None
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, yaw]
        self.robot_state = "INIT"
        
        # AMCL误差校正
        self.amcl_correction_x = 0.0
        self.amcl_correction_y = -0.162  # 校正偏差 1.5 - 1.338 = 0.162
        
        # 车道线路径定义
        self.lane_paths = {
            'pickup_to_red': [
                {'target': [0.0, 1.0], 'angle': 0.0},
                {'target': [0.0, 3.0], 'angle': 0.0},
                {'target': [0.0, 3.0], 'angle': 1.5708},
                {'target': [1.0, 3.0], 'angle': 1.5708},
                {'target': [1.0, 3.0], 'angle': 0.0},
                {'target': [1.0, 5.0], 'angle': 0.0}
            ],
            'pickup_to_blue': [
                {'target': [0.0, 1.0], 'angle': 0.0},
                {'target': [0.0, 3.0], 'angle': 0.0},
                {'target': [0.0, 3.0], 'angle': 1.5708},
                {'target': [3.0, 3.0], 'angle': 1.5708},
                {'target': [3.0, 3.0], 'angle': 0.0},
                {'target': [3.0, 5.0], 'angle': 0.0}
            ],
            'pickup_to_green': [
                {'target': [0.0, 1.0], 'angle': 0.0},
                {'target': [0.0, 1.0], 'angle': -1.5708},
                {'target': [-5.0, 1.0], 'angle': -1.5708}
            ],
            'pickup_to_purple': [
                {'target': [0.0, 1.0], 'angle': 0.0},
                {'target': [0.0, 1.0], 'angle': 1.5708},
                {'target': [5.0, 1.0], 'angle': 1.5708}
            ],
            'red_to_pickup': [
                {'target': [1.0, 3.0], 'angle': 3.14159},
                {'target': [1.0, 3.0], 'angle': -1.5708},
                {'target': [0.0, 3.0], 'angle': -1.5708},
                {'target': [0.0, 3.0], 'angle': 3.14159},
                {'target': [0.0, 1.0], 'angle': 3.14159},
                {'target': [0.0, -0.5], 'angle': 3.14159}
            ],
            'blue_to_pickup': [
                {'target': [3.0, 3.0], 'angle': 3.14159},
                {'target': [3.0, 3.0], 'angle': -1.5708},
                {'target': [0.0, 3.0], 'angle': -1.5708},
                {'target': [0.0, 3.0], 'angle': 3.14159},
                {'target': [0.0, 1.0], 'angle': 3.14159},
                {'target': [0.0, -0.5], 'angle': 3.14159}
            ],
            'green_to_pickup': [
                {'target': [0.0, 1.0], 'angle': 1.5708},
                {'target': [0.0, 1.0], 'angle': 3.14159},
                {'target': [0.0, -0.5], 'angle': 3.14159}
            ],
            'purple_to_pickup': [
                {'target': [0.0, 1.0], 'angle': -1.5708},
                {'target': [0.0, 1.0], 'angle': 3.14159},
                {'target': [0.0, -0.5], 'angle': 3.14159}
            ]
        }
        
        self.current_path = None
        self.current_waypoint_index = 0
        
        # 控制定时器 - 恢复车道线跟随控制
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("OverheadCameraProcessor initialized for LANE DETECTION ASSISTANCE (MoveBase handles navigation)")
        
    def pose_callback(self, msg):
        """处理AMCL位姿回调"""
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
        
        # 使用AMCL坐标并应用误差校正
        self.robot_amcl_pose = msg.pose.pose
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        yaw = self.get_yaw_from_quaternion(orient)
        
        # 应用AMCL误差校正
        corrected_x = pos.x + self.amcl_correction_x
        corrected_y = pos.y + self.amcl_correction_y
        
        self.current_position = [corrected_x, corrected_y, yaw]
        
    def odom_callback(self, msg):
        """里程计回调"""
        self.robot_odom_pose = msg.pose.pose
        if self.robot_amcl_pose is None:
            pos = msg.pose.pose.position
            orient = msg.pose.pose.orientation
            yaw = self.get_yaw_from_quaternion(orient)
            self.current_position = [pos.x, pos.y, yaw]
        
    def robot_state_callback(self, msg):
        """机器人状态回调"""
        self.robot_state = msg.data
        
    def navigation_command_callback(self, msg):
        """处理导航命令,车道线检测辅助"""
        command = msg.data
        rospy.logdebug("OVERHEAD CAMERA received command: %s (assist mode)", command)
        
        if command.startswith("assist_goto_") or command.startswith("assist_return_"):
            # 启动车道线检测辅助模式
            self.lane_following_active = True
            self.publish_lane_status("lane_assistance_started")
            rospy.logwarn("Overhead camera: Lane detection assistance started")
        elif command.startswith("goto_") and command.endswith("_drop"):
            # 兼容模式：如果需要可以启动完整车道线跟随
            color = command.replace("goto_", "").replace("_drop", "")
            rospy.logdebug("Overhead camera: Drop command received but MoveBase will handle navigation")
        elif command.startswith("return_from_") and command.endswith("_to_pickup"):
            # 兼容模式：如果需要可以启动完整车道线跟随
            color = command.replace("return_from_", "").replace("_to_pickup", "")
            rospy.logdebug("Overhead camera: Return command received but MoveBase will handle navigation")
        elif command == "stop":
            self.stop_lane_following()
        else:
            rospy.logwarn("OVERHEAD CAMERA ignoring command: %s", command)
            
    def start_lane_following(self, path_name):
        """开始车道线跟随"""
        if path_name in self.lane_paths:
            self.current_path = self.lane_paths[path_name]
            self.current_waypoint_index = 0
            self.lane_following_active = True
            self.current_phase = "rotating"
            self.phase_start_time = rospy.Time.now()
            
            rospy.logwarn("OVERHEAD CAMERA: Started lane following for path: %s with %d waypoints", 
                         path_name, len(self.current_path))
            
            # 打印完整路径
            for i, wp in enumerate(self.current_path):
                rospy.logwarn("Waypoint %d: pos=[%.3f, %.3f], angle=%.3f", 
                             i+1, wp['target'][0], wp['target'][1], wp['angle'])
            
            self.publish_lane_status("lane_following_started")
        else:
            rospy.logerr("OVERHEAD CAMERA: Unknown path: %s", path_name)
            
    def stop_lane_following(self):
        """停止车道线跟随"""
        self.lane_following_active = False
        self.current_path = None
        self.current_phase = "idle"
        
        # 发送停止命令
        stop_cmd = Twist()
        self.lane_correction_pub.publish(stop_cmd)
        self.publish_lane_status("lane_following_stopped")
        
    def image_callback(self, msg):
        """处理俯视摄像头图像"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detect_lane_lines(cv_image)
        except Exception as e:
            rospy.logerr("Overhead camera processing error: %s", str(e))
            
    def detect_lane_lines(self, image):
        """检测车道线"""
        height, width = image.shape[:2]
        
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建黄色车道线掩码
        mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
        try:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建调试图像
        debug_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 绘制机器人位置
        robot_pixel_x, robot_pixel_y = self.world_to_pixel(self.current_position[0], self.current_position[1])
        if 0 <= robot_pixel_x < width and 0 <= robot_pixel_y < height:
            cv2.circle(debug_image, (int(robot_pixel_x), int(robot_pixel_y)), 10, (0, 0, 255), -1)
            
            # 绘制机器人方向
            yaw = self.current_position[2]
            end_x = int(robot_pixel_x + 30 * math.cos(yaw))
            end_y = int(robot_pixel_y - 30 * math.sin(yaw))
            cv2.arrowedLine(debug_image, (int(robot_pixel_x), int(robot_pixel_y)), 
                           (end_x, end_y), (0, 0, 255), 3)
        
        # 绘制当前路径
        if self.current_path and self.lane_following_active:
            for i, waypoint in enumerate(self.current_path):
                wp_x, wp_y = self.world_to_pixel(waypoint['target'][0], waypoint['target'][1])
                if 0 <= wp_x < width and 0 <= wp_y < height:
                    color = (0, 255, 0) if i == self.current_waypoint_index else (255, 0, 0)
                    cv2.circle(debug_image, (int(wp_x), int(wp_y)), 8, color, -1)
                    cv2.putText(debug_image, str(i), (int(wp_x), int(wp_y)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制车道线
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                cv2.drawContours(debug_image, [contour], -1, (0, 255, 255), 2)
        
        # 发布调试图像
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.lane_debug_pub.publish(debug_msg)
        except Exception as e:
            rospy.logerr("Debug image publish error: %s", str(e))
            
    def control_loop(self, event):
        """主控制循环 - 现在只做可视化和车道线检测辅助"""
        if not self.lane_following_active:
            return
            
        # 只做车道线检测和可视化，不控制机器人移动
        # MoveBase负责实际的导航控制
        rospy.logdebug("Overhead camera: Lane detection assistance active")
            
    def handle_rotation_phase(self, angle_diff, target_yaw):
        """处理旋转阶段"""
        if abs(angle_diff) < self.angle_tolerance:
            # 旋转完成，开始移动
            self.current_phase = "moving"
            self.phase_start_time = rospy.Time.now()
            rospy.logwarn("Rotation completed for WP %d, starting movement", self.current_waypoint_index + 1)
        else:
            # 继续旋转
            angular_velocity = self.calculate_rotation_velocity(angle_diff)
            cmd = Twist()
            cmd.angular.z = angular_velocity
            self.lane_correction_pub.publish(cmd)
            
            if abs(angle_diff) > 0.2:  # 只在大角度时打印
                rospy.logdebug("Rotating WP %d: angle_diff=%.3f, vel=%.3f", 
                              self.current_waypoint_index + 1, angle_diff, angular_velocity)
            
    def handle_movement_phase(self, distance, dx, dy):
        """处理移动阶段"""
        if distance < self.position_tolerance:
            # 移动完成
            self.current_phase = "completed"
            self.phase_start_time = rospy.Time.now()
            
            # 停止机器人
            stop_cmd = Twist()
            self.lane_correction_pub.publish(stop_cmd)
            rospy.logwarn("Movement completed for WP %d", self.current_waypoint_index + 1)
        else:
            # 继续直线移动
            linear_velocity = self.calculate_linear_velocity(distance)
            cmd = Twist()
            cmd.linear.x = linear_velocity
            
            # 小幅角度校正保持直线
            target_heading = math.atan2(dy, dx)
            current_heading = self.current_position[2]
            heading_error = self.normalize_angle(target_heading - current_heading)
            cmd.angular.z = 0.5 * heading_error  # 小幅校正
            
            self.lane_correction_pub.publish(cmd)
            
            if distance > 0.5:  # 只在远距离时打印
                rospy.logdebug("Moving WP %d: dist=%.3f, vel=%.3f", 
                              self.current_waypoint_index + 1, distance, linear_velocity)
            
    def advance_to_next_waypoint(self):
        """前进到下一个路径点"""
        self.current_waypoint_index += 1
        self.current_phase = "rotating"
        self.phase_start_time = rospy.Time.now()
        
        rospy.loginfo("Advancing to waypoint %d/%d", 
                     self.current_waypoint_index + 1, len(self.current_path))
        
    def complete_lane_following(self):
        """完成车道线跟随"""
        self.lane_following_active = False
        self.current_phase = "idle"
        
        # 停止机器人
        stop_cmd = Twist()
        self.lane_correction_pub.publish(stop_cmd)
        
        self.publish_lane_status("lane_following_completed")
        rospy.loginfo("Lane following completed")
        
    def calculate_rotation_velocity(self, angle_diff):
        """计算旋转速度"""
        max_angular_vel = 0.5
        min_angular_vel = 0.1
        
        abs_angle = abs(angle_diff)
        if abs_angle > 0.5:
            angular_vel = max_angular_vel
        else:
            angular_vel = max(min_angular_vel, abs_angle * 0.8)
            
        return angular_vel if angle_diff > 0 else -angular_vel
        
    def calculate_linear_velocity(self, distance):
        """计算线性速度"""
        max_linear_vel = 0.3
        min_linear_vel = 0.05
        
        if distance > 1.0:
            return max_linear_vel
        else:
            return max(min_linear_vel, distance * 0.3)
            
    def world_to_pixel(self, world_x, world_y):
        """世界坐标转像素坐标"""
        pixel_x = self.image_center_x + world_x * self.pixels_per_meter
        pixel_y = self.image_center_y - world_y * self.pixels_per_meter
        return pixel_x, pixel_y
        
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
        """标准化角度到[-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
        
    def publish_lane_status(self, status):
        """发布车道线跟随状态"""
        status_msg = String()
        status_msg.data = status
        self.lane_status_pub.publish(status_msg)

if __name__ == '__main__':
    try:
        processor = OverheadCameraProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("OverheadCameraProcessor node terminated")