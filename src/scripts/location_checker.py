#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import tf
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetModelState

class LocationChecker(object):
    def __init__(self):
        rospy.init_node('location_checker', anonymous=False)
        
        # 优先等待Gazebo服务以获取准确坐标
        rospy.loginfo("Waiting for Gazebo get_model_state service...")
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=10.0)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.gazebo_available = True
            rospy.loginfo("Gazebo service connected - will use ground truth coordinates")
        except:
            rospy.logwarn("Gazebo service not available - will use AMCL coordinates")
            self.gazebo_available = False
            self.get_model_state = None
        
        # 区域判断
        self.pickup_zone = rospy.get_param('/pickup_zone', {
            'center': [0.0, -1.5],
            'radius': 1.5,  # 增加半径
            'safe_exit_point': [0.0, 1.0]
        })
        
        # drop区域配置
        self.drop_zones = rospy.get_param('/drop_zones', {
            'red': {'center': [1.0, 5.0], 'radius': 1.8},   
            'blue': {'center': [3.0, 5.0], 'radius': 1.8},  
            'green': {'center': [-5.0, 1.0], 'radius': 1.8},
            'purple': {'center': [5.0, 1.0], 'radius': 1.8} 
        })
        
        self.safe_points = rospy.get_param('/safe_navigation_points', {
            'home': [0.0, 1.0],
            'checkpoint_1': [0.0, 3.0],
            'checkpoint_2': [-5.0, 3.0],
            'checkpoint_3': [5.0, 3.0]
        })
        
        # 确保安全退出点存在
        if 'safe_exit_point' not in self.pickup_zone:
            self.pickup_zone['safe_exit_point'] = [0.0, 1.0]
            rospy.logwarn("Adding default safe_exit_point: [0.0, 1.0]")
        
        # Publishers
        self.zone_pub = rospy.Publisher('/current_zone', String, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # 统一坐标系统 - 优先使用Gazebo
        self.robot_amcl_pose = None
        self.robot_odom_pose = None
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, yaw]
        self.coordinate_source = "none"
        self.current_zone = "unknown"
        
        # AMCL误差校正 - 使用Gazebo地面真实坐标
        self.amcl_correction_x = 0.0
        self.amcl_correction_y = -0.183  # 手动校正
        
        # 区域判断稳定性控制
        self.zone_history = ["unknown"] * 5
        self.zone_history_index = 0
        self.last_stable_zone = "unknown"
        
        # 智能区域切换
        self.zone_change_threshold = 3  # 需要3次确认才切换区域
        self.position_stability_check = True
        
        rospy.loginfo("LocationChecker initialized with UNIFIED COORDINATE SYSTEM (Gazebo priority)")
        
    def amcl_callback(self, msg):
        """统一的AMCL位姿处理 - 优先使用Gazebo地面真实坐标"""
        # 优先使用Gazebo的准确坐标
        if self.gazebo_available:
            try:
                response = self.get_model_state("warehouse_robot", "")
                if response.success:
                    pos = response.pose.position
                    orient = response.pose.orientation
                    
                    # 计算yaw角度
                    yaw = self.get_yaw_from_quaternion(orient)
                    self.current_position = [pos.x, pos.y, yaw]
                    self.coordinate_source = "gazebo"
                    self.check_zone_stable()
                    return
            except Exception as e:
                rospy.logdebug("Gazebo coordinate access failed: %s", str(e))
        
        # 备用：使用AMCL坐标
        self.robot_amcl_pose = msg.pose.pose
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        
        # 计算yaw角度
        yaw = self.get_yaw_from_quaternion(orient)
        
        # 信任AMCL定位
        self.current_position = [pos.x, pos.y, yaw]
        self.coordinate_source = "amcl"
        self.check_zone_stable()
        
    def odom_callback(self, msg):
        """里程计回调 - 仅作为最后备用坐标源"""
        self.robot_odom_pose = msg.pose.pose
        
        # 如果amcl_pose和gazebo都不可用，使用odom作为备用
        if self.robot_amcl_pose is None and not self.gazebo_available:
            pos = msg.pose.pose.position
            orient = msg.pose.pose.orientation
            yaw = self.get_yaw_from_quaternion(orient)
            self.current_position = [pos.x, pos.y, yaw]
            self.coordinate_source = "odom"
            self.check_zone_stable()
            
    def get_yaw_from_quaternion(self, quaternion):
        """从四元数获取yaw角度"""
        try:
            euler = tf.transformations.euler_from_quaternion([
                quaternion.x, quaternion.y, quaternion.z, quaternion.w
            ])
            return euler[2]  # yaw
        except:
            return 0.0
            
    def calculate_distance(self, point1, point2):
        """计算两点间欧几里得距离"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
    def check_zone_stable(self):
        """稳定的区域检测 - 基于world文件的实际尺寸"""
        if not self.current_position:
            return
            
        # 检测当前区域
        current_zone = self.detect_current_zone()
        
        # 使用历史数据稳定区域检测
        stable_zone = self.stabilize_zone_detection(current_zone)
        
        # 发布区域变化
        if stable_zone != self.current_zone:
            old_zone = self.current_zone
            self.current_zone = stable_zone
            zone_msg = String()
            zone_msg.data = self.current_zone
            self.zone_pub.publish(zone_msg)
            
            rospy.loginfo("Robot zone: %s -> %s at position [%.3f, %.3f, %.3f] (%s)", 
                         old_zone, self.current_zone, 
                         self.current_position[0], self.current_position[1], self.current_position[2],
                         self.coordinate_source)
    
    def detect_current_zone(self):
        """检测当前区域"""
        if not self.current_position:
            return "unknown"
            
        x, y = self.current_position[0], self.current_position[1]
        
        # 首先检查drop区域 - 使用圆形检测，增加半径
        for color, zone_info in self.drop_zones.items():
            drop_distance = self.calculate_distance([x, y], zone_info['center'])
            
            if drop_distance <= zone_info['radius']:
                rospy.logdebug("Detected in %s zone at [%.3f, %.3f]! Distance: %.3f, Radius: %.3f", 
                             color, x, y, drop_distance, zone_info['radius'])
                return color + "_zone"
        
        # 检查pickup区域
        # pickup_area在world文件中是3x2米，中心在[0, -1.5]
        pickup_bounds = {
            'x_min': -2.0, 
            'x_max': 2.0,  
            'y_min': -3.0, 
            'y_max': 0.0   
        }
        
        in_pickup_rect = (pickup_bounds['x_min'] <= x <= pickup_bounds['x_max'] and
                         pickup_bounds['y_min'] <= y <= pickup_bounds['y_max'])
        
        if in_pickup_rect:
            rospy.logdebug("Robot in pickup zone (rectangular check) at [%.3f, %.3f]", x, y)
            return "pickup_zone"
        
        # 检查安全退出点区域
        safe_exit_point = self.pickup_zone['safe_exit_point']
        exit_distance = self.calculate_distance([x, y], safe_exit_point)
        if exit_distance <= 0.8:  # 增加安全退出点半径
            rospy.logdebug("Robot at safety exit point [%.3f, %.3f]", x, y)
            return "safe_area"
        
        # 检查其他安全导航点
        for point_name, point_coords in self.safe_points.items():
            point_distance = self.calculate_distance([x, y], point_coords)
            if point_distance <= 0.8:  # 增加安全点半径
                return "safe_area"
        
        # 检查是否在主要导航区域内
        navigation_bounds = {
            'x_min': -7.0,
            'x_max': 7.0,
            'y_min': -4.0,
            'y_max': 7.0
        }
        
        if (navigation_bounds['x_min'] <= x <= navigation_bounds['x_max'] and
            navigation_bounds['y_min'] <= y <= navigation_bounds['y_max']):
            return "navigation_area"
        else:
            return "safe_area"
    
    def stabilize_zone_detection(self, current_zone):
        """ 稳定化区域检测 """
        # 更新历史记录
        self.zone_history[self.zone_history_index] = current_zone
        self.zone_history_index = (self.zone_history_index + 1) % len(self.zone_history)
        
        # 分析历史数据
        zone_counts = {}
        for zone in self.zone_history:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        # 找到最频繁的区域
        if zone_counts:
            most_frequent_zone = max(zone_counts, key=zone_counts.get)
            most_frequent_count = zone_counts[most_frequent_zone]
            
            # 需要足够的确认才切换区域
            if most_frequent_count >= self.zone_change_threshold:
                return most_frequent_zone
            else:
                # 不够确认时保持当前稳定区域
                return self.current_zone if self.current_zone != "unknown" else most_frequent_zone
        
        return current_zone
    
    def get_position_info(self):
        """获取位置信息用于调试"""
        return {
            'position': self.current_position,
            'coordinate_source': self.coordinate_source,
            'current_zone': self.current_zone,
            'zone_history': self.zone_history,
            'gazebo_available': self.gazebo_available
        }
    
    def is_in_pickup_area(self):
        """检查是否在pickup区域 - 供其他节点调用"""
        return self.current_zone == "pickup_zone"
    
    def is_in_drop_area(self, color=None):
        """检查是否在drop区域 - 供其他节点调用"""
        if color:
            return self.current_zone == (color + "_zone")
        else:
            return self.current_zone.endswith("_zone") and self.current_zone != "pickup_zone"
    
    def is_in_navigation_area(self):
        """检查是否在导航区域 - 供其他节点调用"""
        return self.current_zone == "navigation_area"
    
    def get_distance_to_zone(self, zone_name):
        """计算到指定区域的距离"""
        if not self.current_position:
            return float('inf')
            
        robot_pos = [self.current_position[0], self.current_position[1]]
        
        if zone_name == "pickup_zone":
            pickup_center = self.pickup_zone['center']
            return self.calculate_distance(robot_pos, pickup_center)
        
        elif zone_name.endswith("_zone") and zone_name != "pickup_zone":
            color = zone_name.replace("_zone", "")
            if color in self.drop_zones:
                drop_center = self.drop_zones[color]['center']
                return self.calculate_distance(robot_pos, drop_center)
        
        return float('inf')

if __name__ == '__main__':
    try:
        checker = LocationChecker()
        
        def print_position_info(event):
            info = checker.get_position_info()
            rospy.loginfo("Position Info: Pos=[%.3f,%.3f,%.3f], Source=%s, Zone=%s", 
                         info['position'][0], info['position'][1], info['position'][2],
                         info['coordinate_source'], info['current_zone'])
        
        debug_timer = rospy.Timer(rospy.Duration(30.0), print_position_info)
        
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("LocationChecker node terminated")