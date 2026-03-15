#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
导航系统诊断脚本
用于检查move_base导航系统的各种问题
"""

import rospy
import tf
import math
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from actionlib_msgs.msg import GoalStatusArray
import actionlib

class NavigationDiagnostics(object):
    def __init__(self):
        rospy.init_node('navigation_diagnostics', anonymous=False)
        
        print("=== 导航系统诊断工具 ===")
        print("检查move_base导航系统的问题")
        
        # Subscribers for diagnostics
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)
        rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, self.global_costmap_callback)
        rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.local_costmap_callback)
        rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.global_plan_callback)
        rospy.Subscriber('/move_base/TrajectoryPlannerROS/local_plan', Path, self.local_plan_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        # MoveBase client for testing
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        # Data storage
        self.robot_pose = None
        self.global_costmap = None
        self.local_costmap = None
        self.global_plan = None
        self.local_plan = None
        self.laser_data = None
        self.last_cmd_vel = None
        self.move_base_status = None
        
        # Diagnostics flags
        self.server_connected = False
        self.pose_received = False
        self.laser_received = False
        self.costmap_received = False
        
        self.run_diagnostics()
        
    def status_callback(self, msg):
        self.move_base_status = msg
        
    def global_costmap_callback(self, msg):
        self.global_costmap = msg
        self.costmap_received = True
        
    def local_costmap_callback(self, msg):
        self.local_costmap = msg
        
    def global_plan_callback(self, msg):
        self.global_plan = msg
        
    def local_plan_callback(self, msg):
        self.local_plan = msg
        
    def pose_callback(self, msg):
        self.robot_pose = msg.pose.pose
        self.pose_received = True
        
    def laser_callback(self, msg):
        self.laser_data = msg
        self.laser_received = True
        
    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = msg
        
    def run_diagnostics(self):
        print("\n1. 检查move_base服务器连接...")
        if self.move_base_client.wait_for_server(timeout=rospy.Duration(5.0)):
            print("✓ move_base服务器连接正常")
            self.server_connected = True
        else:
            print("✗ move_base服务器连接失败")
            
        # Wait for data
        print("\n2. 等待传感器数据...")
        rospy.sleep(2.0)
        
        self.check_localization()
        self.check_sensors()
        self.check_costmaps()
        self.check_planning()
        self.test_simple_navigation()
        
        print("\n=== 诊断完成 ===")
        
    def check_localization(self):
        print("\n3. 检查定位系统...")
        if self.pose_received and self.robot_pose:
            x = self.robot_pose.position.x
            y = self.robot_pose.position.y
            print("✓ 机器人位置: ({:.2f}, {:.2f})".format(x, y))
            
            # Check if position is reasonable
            if abs(x) > 100 or abs(y) > 100:
                print("⚠ 警告: 机器人位置可能异常")
        else:
            print("✗ 未收到定位数据")
            
    def check_sensors(self):
        print("\n4. 检查传感器...")
        if self.laser_received and self.laser_data:
            valid_readings = [r for r in self.laser_data.ranges 
                            if self.laser_data.range_min < r < self.laser_data.range_max]
            print("✓ 激光雷达: {}/{} 有效读数".format(len(valid_readings), len(self.laser_data.ranges)))
            
            if len(valid_readings) < len(self.laser_data.ranges) * 0.5:
                print("⚠ 警告: 激光雷达有效读数过少")
        else:
            print("✗ 未收到激光雷达数据")
            
    def check_costmaps(self):
        print("\n5. 检查代价地图...")
        if self.costmap_received:
            if self.global_costmap:
                print("✓ 全局代价地图: {}x{} 分辨率:{:.3f}".format(
                    self.global_costmap.info.width,
                    self.global_costmap.info.height,
                    self.global_costmap.info.resolution))
                    
            if self.local_costmap:
                print("✓ 局部代价地图: {}x{} 分辨率:{:.3f}".format(
                    self.local_costmap.info.width,
                    self.local_costmap.info.height,
                    self.local_costmap.info.resolution))
        else:
            print("✗ 未收到代价地图数据")
            
    def check_planning(self):
        print("\n6. 检查路径规划...")
        if self.global_plan and len(self.global_plan.poses) > 0:
            print("✓ 全局路径: {} 个路径点".format(len(self.global_plan.poses)))
        else:
            print("- 当前无全局路径")
            
        if self.local_plan and len(self.local_plan.poses) > 0:
            print("✓ 局部路径: {} 个路径点".format(len(self.local_plan.poses)))
        else:
            print("- 当前无局部路径")
            
        if self.last_cmd_vel:
            print("✓ 速度命令: 线速度={:.2f}, 角速度={:.2f}".format(
                self.last_cmd_vel.linear.x, self.last_cmd_vel.angular.z))
        else:
            print("- 当前无速度命令")
            
    def test_simple_navigation(self):
        print("\n7. 测试简单导航...")
        if not self.server_connected or not self.robot_pose:
            print("跳过导航测试 - 前置条件不满足")
            return
            
        # Test navigation to a nearby point
        current_x = self.robot_pose.position.x
        current_y = self.robot_pose.position.y
        
        # Target 1 meter ahead
        target_x = current_x + 1.0
        target_y = current_y
        
        print("测试导航到: ({:.2f}, {:.2f})".format(target_x, target_y))
        
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = target_x
        goal.target_pose.pose.position.y = target_y
        goal.target_pose.pose.orientation.w = 1.0
        
        self.move_base_client.send_goal(goal)
        
        # Wait for result for 10 seconds
        result = self.move_base_client.wait_for_result(timeout=rospy.Duration(10.0))
        
        if result:
            state = self.move_base_client.get_state()
            if state == 3:  # SUCCEEDED
                print("✓ 导航测试成功")
            else:
                print("✗ 导航测试失败, 状态码: {}".format(state))
        else:
            print("⚠ 导航测试超时")
            self.move_base_client.cancel_goal()

if __name__ == '__main__':
    try:
        diagnostics = NavigationDiagnostics()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("诊断中断")
