#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

class PackageDetector(object):
    def __init__(self):
        rospy.init_node('package_detector', anonymous=False)
        
        self.bridge = CvBridge()
        
        # 颜色检测参数
        self.color_ranges = {
            'red': {
                'lower': np.array([0, 40, 40]),      
                'upper': np.array([15, 255, 255]),   
                'lower2': np.array([165, 40, 40]),   
                'upper2': np.array([180, 255, 255])
            },
            'blue': {
                'lower': np.array([90, 40, 40]),     
                'upper': np.array([140, 255, 255])
            },
            'green': {
                'lower': np.array([35, 40, 40]),     
                'upper': np.array([85, 255, 255])
            },
            'purple': {
                'lower': np.array([125, 40, 40]),    
                'upper': np.array([165, 255, 255])
            }
        }
        
        # 检测参数
        self.min_package_area = rospy.get_param('~min_package_area', 500)   
        self.max_package_area = rospy.get_param('~max_package_area', 100000) 
        
        # Publishers
        self.package_color_pub = rospy.Publisher('/package_color', String, queue_size=10)
        self.package_debug_pub = rospy.Publisher('/package_debug_image', Image, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/overhead_camera/image_raw', Image, self.overhead_image_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.robot_camera_callback)
        rospy.Subscriber('/current_zone', String, self.zone_callback)
        rospy.Subscriber('/robot_state', String, self.robot_state_callback)
        rospy.Subscriber('/current_package_color', String, self.current_package_color_callback)
        
        # 检测启用逻辑
        self.current_zone = "unknown"
        self.robot_state = "INIT"
        self.detection_enabled = False
        self.last_detected_color = None
        
        # 只在pickup_zone和DETECT_PACKAGE_COLOR状态启用检测
        self.valid_detection_combinations = [
            ("DETECT_PACKAGE_COLOR", "pickup_zone"),     # 只在pickup_zone检测
        ]
        
        # 监听主控发来的包裹颜色清理信号
        self.current_package_color_from_main = None
        
        # 检测结果稳定化
        self.detection_history = []
        self.history_size = 3  # 减少历史大小，提高响应速度
        self.confidence_threshold = 2  # 降低阈值，但保持稳定性
        
        # 调试信息
        self.detection_attempts = 0
        self.successful_detections = 0
        
        # 添加检测状态重置机制
        self.last_state_change_time = rospy.Time.now()
        self.detection_reset_delay = 1.0  # 状态变化后1秒重置检测
        
        rospy.loginfo("PackageDetector initialized with ENHANCED detection control and reset mechanism")
        
    def current_package_color_callback(self, msg):
        """监听主控发来的包裹颜色信息"""
        if msg.data == "":
            # 空消息表示主控清理了包裹信息
            rospy.logwarn("Main controller cleared package info - resetting detector")
            self.reset_detection_state()
            self.current_package_color_from_main = None
        elif msg.data in ['red', 'blue', 'green', 'purple']:
            self.current_package_color_from_main = msg.data
            #rospy.loginfo("Received package color from main: %s", msg.data)
        
    def robot_state_callback(self, msg):
        """机器人状态回调 - 严格控制检测启用"""
        old_state = self.robot_state
        self.robot_state = msg.data
        
        # 只在状态变化时更新检测状态
        if old_state != self.robot_state:
            self.last_state_change_time = rospy.Time.now()
            old_detection_enabled = self.detection_enabled
            self.update_detection_status()
            
            rospy.loginfo("State change: %s -> %s, Detection: %s -> %s", 
                         old_state, self.robot_state, old_detection_enabled, self.detection_enabled)
            
            # 修复：进入DETECT_PACKAGE_COLOR时强制重置检测状态
            if (self.robot_state == "DETECT_PACKAGE_COLOR" and 
                old_state != "DETECT_PACKAGE_COLOR"):
                rospy.logwarn("🔧 ENTERING DETECT_PACKAGE_COLOR - forcing detection reset")
                rospy.sleep(self.detection_reset_delay)  # 等待状态稳定
                self.reset_detection_state()
        
    def zone_callback(self, msg):
        """区域变化回调"""
        old_zone = self.current_zone
        self.current_zone = msg.data
        
        # 只在区域变化时更新检测状态
        if old_zone != self.current_zone:
            old_detection_enabled = self.detection_enabled
            self.update_detection_status()
            
            rospy.loginfo("Zone change: %s -> %s, Detection: %s -> %s", 
                         old_zone, self.current_zone, old_detection_enabled, self.detection_enabled)
    
    def update_detection_status(self):
        """严格的检测状态更新逻辑"""
        # 检查当前(状态, 区域)组合是否在允许列表中
        current_combination = (self.robot_state, self.current_zone)
        combination_valid = current_combination in self.valid_detection_combinations
        
        # 额外检查：确保不在navigation_area
        if self.current_zone == "navigation_area":
            combination_valid = False
            rospy.logwarn("Blocking detection in navigation_area")
        
        # 最终决定
        should_enable = combination_valid
        
        if should_enable != self.detection_enabled:
            self.detection_enabled = should_enable
            rospy.logwarn("Detection status changed: %s for combination (%s, %s)", 
                         self.detection_enabled, self.robot_state, self.current_zone)
            
            if not self.detection_enabled:
                self.reset_detection_state()
    
    def reset_detection_state(self):
        """重置检测状态"""
        rospy.logwarn("RESETTING detection state")
        self.detection_history = []
        self.last_detected_color = None
        
    def overhead_image_callback(self, msg):
        """俯视摄像头图像处理 - 增强安全检查"""
        # 检查状态变化后的延迟
        if (rospy.Time.now() - self.last_state_change_time).to_sec() < self.detection_reset_delay:
            return
            
        # 三重安全检查
        if not self.detection_enabled:
            return
        
        # 额外检查：确保不在navigation_area
        if self.current_zone == "navigation_area":
            rospy.logwarn_throttle(5, "Blocking detection in navigation_area (state: %s)", self.robot_state)
            return
            
        # 额外检查：确保状态正确
        if self.robot_state != "DETECT_PACKAGE_COLOR":
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            detected_color = self.detect_package_color(cv_image, camera_type="overhead")
            
            if detected_color:
                self.process_detection_result(detected_color, "overhead")
                
        except Exception as e:
            rospy.logerr("Overhead package detection error: %s", str(e))
            
    def robot_camera_callback(self, msg):
        """机器人前置摄像头图像处理 - 增强安全检查"""
        # 检查状态变化后的延迟
        if (rospy.Time.now() - self.last_state_change_time).to_sec() < self.detection_reset_delay:
            return
            
        # 三重安全检查
        if not self.detection_enabled:
            return
            
        # 额外检查：确保不在navigation_area
        if self.current_zone == "navigation_area":
            rospy.logwarn_throttle(5, "Blocking detection in navigation_area (state: %s)", self.robot_state)
            return
            
        # 额外检查：确保状态正确
        if self.robot_state != "DETECT_PACKAGE_COLOR":
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            detected_color = self.detect_package_color(cv_image, camera_type="robot", close_range=True)
            
            if detected_color:
                self.process_detection_result(detected_color, "robot")
                
        except Exception as e:
            rospy.logerr("Robot camera package detection error: %s", str(e))
    
    def process_detection_result(self, detected_color, camera_source):
        """修复：处理检测结果 - 更严格的确认逻辑"""
        self.detection_attempts += 1
        
        # 添加到历史记录
        self.detection_history.append(detected_color)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # 修复：只有在有足够历史记录时才分析
        if len(self.detection_history) >= self.confidence_threshold:
            color_counts = {}
            for color in self.detection_history:
                color_counts[color] = color_counts.get(color, 0) + 1
            
            # 找到最频繁的颜色
            most_frequent_color = max(color_counts, key=color_counts.get)
            most_frequent_count = color_counts[most_frequent_color]
            
            # 修复：更严格的确认条件
            confidence_ratio = float(most_frequent_count) / len(self.detection_history)
            if (confidence_ratio >= 0.6 and  # 至少60%的一致性
                most_frequent_color != self.last_detected_color):
                
                self.last_detected_color = most_frequent_color
                self.successful_detections += 1
                
                # 修复：检查是否与主控期望的颜色一致
                if (self.current_package_color_from_main and 
                    most_frequent_color != self.current_package_color_from_main):
                    rospy.logwarn("⚠️ Detected color %s doesn't match main controller expectation %s", 
                                 most_frequent_color, self.current_package_color_from_main)
                
                # 发布检测结果
                color_msg = String()
                color_msg.data = most_frequent_color
                self.package_color_pub.publish(color_msg)
                
                rospy.logwarn("PACKAGE DETECTED: %s (from %s, confidence: %.1f%%, state: %s, zone: %s)", 
                             most_frequent_color, camera_source, confidence_ratio*100, 
                             self.robot_state, self.current_zone)
                
                # 修复：检测成功后清除历史，避免重复检测
                self.detection_history = []
                
    def detect_package_color(self, image, camera_type="overhead", close_range=False):
        """改进的包裹颜色检测"""
        if image is None:
            return None
            
        # 根据摄像头类型调整ROI
        height, width = image.shape[:2]
        
        if camera_type == "robot" and close_range:
            # 机器人摄像头：关注中心区域
            roi_image = image[height//4:3*height//4, width//4:3*width//4]
        elif camera_type == "overhead":
            # 修复：俯视摄像头ROI更保守，只关注真正的pickup区域
            roi_image = image[2*height//3:, width//3:2*width//3]
        else:
            roi_image = image
            
        # 转换到HSV
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        best_color = None
        max_area = 0
        max_confidence = 0
        
        # 创建调试图像
        debug_image = roi_image.copy()
        
        # 检查每个颜色
        for color_name, color_range in self.color_ranges.items():
            # 创建颜色掩码
            if 'lower2' in color_range:  # 红色有两个范围
                mask1 = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 寻找轮廓 - OpenCV 3.2兼容
            try:
                _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 分析轮廓
            total_area = 0
            valid_contours = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_package_area < area < self.max_package_area:
                    total_area += area
                    valid_contours += 1
                    
                    # 在调试图像上绘制轮廓
                    cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 3)
                    
                    # 获取轮廓中心
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(debug_image, (cx, cy), 10, (0, 0, 255), -1)
                        cv2.putText(debug_image, color_name, (cx-30, cy-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 计算置信度
            if total_area > 0:
                confidence = total_area * valid_contours
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    max_area = total_area
                    best_color = color_name
                    
        # 发布调试图像
        self.publish_debug_image(debug_image, camera_type)
        
        # 返回最佳检测结果
        if max_area > self.min_package_area:
            return best_color
        
        return None
        
    def publish_debug_image(self, debug_image, camera_type):
        """发布调试图像"""
        try:
            height, width = debug_image.shape[:2]
            
            # 显示检测状态
            status_text = "Detection: {} | State: {} | Zone: {}".format(
                "ACTIVE" if self.detection_enabled else "INACTIVE", 
                self.robot_state, self.current_zone)
            cv2.putText(debug_image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 显示统计信息
            stats_text = "Attempts: {} | Success: {} | Camera: {}".format(
                self.detection_attempts, self.successful_detections, camera_type)
            cv2.putText(debug_image, stats_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 显示最近检测结果
            if self.last_detected_color:
                result_text = "Last detected: {}".format(self.last_detected_color)
                cv2.putText(debug_image, result_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示主控期望的颜色
            if self.current_package_color_from_main:
                expect_text = "Expected: {}".format(self.current_package_color_from_main)
                cv2.putText(debug_image, expect_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # 显示时间信息
            time_since_state_change = (rospy.Time.now() - self.last_state_change_time).to_sec()
            time_text = "Time since state change: {:.1f}s".format(time_since_state_change)
            cv2.putText(debug_image, time_text, (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
            
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.package_debug_pub.publish(debug_msg)
        except Exception as e:
            rospy.logerr("Debug image publish error: %s", str(e))

if __name__ == '__main__':
    try:
        detector = PackageDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("PackageDetector node terminated")