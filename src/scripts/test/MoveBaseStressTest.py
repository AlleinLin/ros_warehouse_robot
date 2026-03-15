#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
move_base压力测试脚本
测试导航系统在各种条件下的表现
"""

import rospy
import tf
import math
import time
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib

class MoveBaseStressTest(object):
    def __init__(self):
        rospy.init_node('movebase_stress_test', anonymous=False)
        
        print("=== move_base 压力测试 ===")
        
        # MoveBase client
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        
        # Test parameters
        self.robot_pose = None
        self.test_results = []
        
        # Test locations (仓库环境中的关键点)
        self.test_locations = [
            [0.0, 1.5, "home"],
            [0.0, -0.5, "pickup_approach"], 
            [1.0, 4.0, "red_drop_approach"],
            [3.0, 4.0, "blue_drop_approach"],
            [-4.0, 1.0, "green_drop_approach"],
            [4.0, 1.0, "purple_drop_approach"],
            [0.0, 3.0, "checkpoint_1"],
        ]
        
        if self.move_base_client.wait_for_server(timeout=rospy.Duration(5.0)):
            print("✓ move_base服务器连接成功")
            self.run_tests()
        else:
            print("✗ move_base服务器连接失败")
            
    def pose_callback(self, msg):
        self.robot_pose = msg.pose.pose
        
    def create_goal(self, x, y, yaw=0.0):
        """创建导航目标"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        
        return goal
        
    def test_navigation_to_point(self, x, y, name, timeout=30.0):
        """测试导航到指定点"""
        print("\n--- 测试导航到 {} [{:.1f}, {:.1f}] ---".format(name, x, y))
        
        start_time = time.time()
        goal = self.create_goal(x, y)
        
        # 记录起始位置
        start_pose = None
        if self.robot_pose:
            start_pose = [self.robot_pose.position.x, self.robot_pose.position.y]
            print("起始位置: [{:.2f}, {:.2f}]".format(start_pose[0], start_pose[1]))
        
        # 发送目标
        self.move_base_client.send_goal(goal)
        print("目标已发送，等待结果...")
        
        # 等待结果
        result = self.move_base_client.wait_for_result(timeout=rospy.Duration(timeout))
        end_time = time.time()
        
        # 分析结果
        test_result = {
            'target': name,
            'target_pos': [x, y],
            'start_pos': start_pose,
            'duration': end_time - start_time,
            'success': False,
            'status': 'TIMEOUT',
            'final_pos': None
        }
        
        if result:
            state = self.move_base_client.get_state()
            test_result['status'] = self.get_status_name(state)
            test_result['success'] = (state == GoalStatus.SUCCEEDED)
            
            if self.robot_pose:
                test_result['final_pos'] = [self.robot_pose.position.x, self.robot_pose.position.y]
                
                # 计算到达精度
                dx = self.robot_pose.position.x - x
                dy = self.robot_pose.position.y - y
                error = math.sqrt(dx*dx + dy*dy)
                test_result['position_error'] = error
                
                print("最终位置: [{:.2f}, {:.2f}]".format(
                    self.robot_pose.position.x, self.robot_pose.position.y))
                print("位置误差: {:.2f}m".format(error))
        else:
            print("导航超时")
            self.move_base_client.cancel_goal()
            
        print("结果: {} ({:.1f}秒)".format(test_result['status'], test_result['duration']))
        self.test_results.append(test_result)
        
        return test_result['success']
        
    def get_status_name(self, status):
        """获取状态名称"""
        status_names = {
            0: 'PENDING',
            1: 'ACTIVE', 
            2: 'PREEMPTED',
            3: 'SUCCEEDED',
            4: 'ABORTED',
            5: 'REJECTED',
            6: 'PREEMPTING',
            7: 'RECALLING',
            8: 'RECALLED',
            9: 'LOST'
        }
        return status_names.get(status, 'UNKNOWN')
        
    def test_rapid_goals(self):
        """测试快速发送多个目标"""
        print("\n--- 快速目标测试 ---")
        
        goals = [
            [0.5, 1.0],
            [1.0, 1.0], 
            [1.0, 1.5],
            [0.5, 1.5]
        ]
        
        for i, (x, y) in enumerate(goals):
            print("发送快速目标 {}: [{}, {}]".format(i+1, x, y))
            goal = self.create_goal(x, y)
            self.move_base_client.send_goal(goal)
            rospy.sleep(0.5)  # 快速发送
            
        # 等待最后一个目标完成
        result = self.move_base_client.wait_for_result(timeout=rospy.Duration(20.0))
        print("快速目标测试完成")
        
    def test_impossible_goal(self):
        """测试不可达目标"""
        print("\n--- 不可达目标测试 ---")
        
        # 发送到墙内的目标
        impossible_goals = [
            [-10.0, -10.0, "墙外远点"],
            [0.0, -4.0, "pickup区域内部"],
        ]
        
        for x, y, desc in impossible_goals:
            print("测试不可达目标: {} [{}, {}]".format(desc, x, y))
            goal = self.create_goal(x, y)
            self.move_base_client.send_goal(goal)
            
            result = self.move_base_client.wait_for_result(timeout=rospy.Duration(15.0))
            if result:
                state = self.move_base_client.get_state()
                print("结果: {}".format(self.get_status_name(state)))
            else:
                print("超时，取消目标")
                self.move_base_client.cancel_goal()
                
    def test_with_interference(self):
        """测试有干扰的导航"""
        print("\n--- 干扰测试 ---")
        
        # 发送一个目标
        goal = self.create_goal(2.0, 2.0)
        self.move_base_client.send_goal(goal)
        print("发送目标到 [2.0, 2.0]")
        
        # 等待2秒后发送干扰速度命令
        rospy.sleep(2.0)
        print("发送干扰速度命令...")
        
        interference_cmd = Twist()
        interference_cmd.angular.z = 1.0
        
        for _ in range(20):  # 2秒的干扰
            self.cmd_vel_pub.publish(interference_cmd)
            rospy.sleep(0.1)
            
        print("干扰结束，等待导航完成...")
        result = self.move_base_client.wait_for_result(timeout=rospy.Duration(20.0))
        
        if result:
            state = self.move_base_client.get_state()
            print("干扰测试结果: {}".format(self.get_status_name(state)))
        else:
            print("干扰测试超时")
            self.move_base_client.cancel_goal()
            
    def run_tests(self):
        """运行所有测试"""
        print("开始压力测试...\n")
        
        # 等待初始定位
        print("等待机器人定位...")
        while not self.robot_pose and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print("机器人当前位置: [{:.2f}, {:.2f}]".format(
            self.robot_pose.position.x, self.robot_pose.position.y))
        
        # 1. 基础导航测试
        print("\n=== 基础导航测试 ===")
        success_count = 0
        for x, y, name in self.test_locations:
            if self.test_navigation_to_point(x, y, name):
                success_count += 1
                
        print("\n基础导航成功率: {}/{} ({:.1f}%)".format(
            success_count, len(self.test_locations), 
            100.0 * success_count / len(self.test_locations)))
        
        # 2. 快速目标测试
        self.test_rapid_goals()
        
        # 3. 不可达目标测试  
        self.test_impossible_goal()
        
        # 4. 干扰测试
        self.test_with_interference()
        
        # 5. 生成报告
        self.generate_report()
        
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*50)
        print("压力测试报告")
        print("="*50)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r['success']])
        
        print("总测试次数: {}".format(total_tests))
        print("成功次数: {}".format(successful_tests))
        print("成功率: {:.1f}%".format(100.0 * successful_tests / total_tests if total_tests > 0 else 0))
        
        # 分析失败原因
        failures = [r for r in self.test_results if not r['success']]
        if failures:
            print("\n失败分析:")
            failure_reasons = {}
            for failure in failures:
                reason = failure['status']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                
            for reason, count in failure_reasons.items():
                print("  {}: {} 次".format(reason, count))
                
        # 性能分析
        if self.test_results:
            durations = [r['duration'] for r in self.test_results if r['success']]
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                min_duration = min(durations)
                
                print("\n性能分析 (成功导航):")
                print("  平均耗时: {:.1f}秒".format(avg_duration))
                print("  最长耗时: {:.1f}秒".format(max_duration))
                print("  最短耗时: {:.1f}秒".format(min_duration))
                
        print("\n详细结果:")
        for result in self.test_results:
            print("  {} -> {} ({:.1f}s)".format(
                result['target'], result['status'], result['duration']))

if __name__ == '__main__':
    try:
        test = MoveBaseStressTest()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("测试中断")
