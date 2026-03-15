#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
代价地图分析脚本
分析move_base的代价地图是否有问题
"""

import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

class CostmapAnalyzer(object):
    def __init__(self):
        rospy.init_node('costmap_analyzer', anonymous=False)
        
        print("=== 代价地图分析工具 ===")
        
        # Subscribers
        rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, self.global_costmap_callback)
        rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.local_costmap_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        
        # Data storage
        self.global_costmap = None
        self.local_costmap = None
        self.robot_pose = None
        self.laser_data = None
        
        # Analysis results
        self.analysis_results = {}
        
        self.run_analysis()
        
    def global_costmap_callback(self, msg):
        self.global_costmap = msg
        
    def local_costmap_callback(self, msg):
        self.local_costmap = msg
        
    def pose_callback(self, msg):
        self.robot_pose = msg.pose.pose
        
    def laser_callback(self, msg):
        self.laser_data = msg
        
    def costmap_to_image(self, costmap):
        """将代价地图转换为OpenCV图像"""
        if costmap is None:
            return None
            
        # 获取地图数据
        width = costmap.info.width
        height = costmap.info.height
        data = costmap.data
        
        # 转换为numpy数组
        grid = np.array(data, dtype=np.int8).reshape((height, width))
        
        # 转换为0-255的图像 (-1表示未知区域，0表示自由，100表示障碍)
        image = np.zeros((height, width), dtype=np.uint8)
        
        # 未知区域 (-1) -> 128 (灰色)
        image[grid == -1] = 128
        
        # 自由区域 (0) -> 255 (白色)  
        image[grid == 0] = 255
        
        # 障碍区域 (>0) -> 按比例设置 (黑色到深灰)
        obstacle_mask = grid > 0
        image[obstacle_mask] = 255 - (grid[obstacle_mask] * 255 / 100)
        
        return image
        
    def analyze_costmap(self, costmap, name):
        """分析代价地图"""
        if costmap is None:
            print("✗ {}: 未收到数据".format(name))
            return None
            
        print("\n--- {} 分析 ---".format(name))
        
        width = costmap.info.width
        height = costmap.info.height
        resolution = costmap.info.resolution
        origin_x = costmap.info.origin.position.x
        origin_y = costmap.info.origin.position.y
        
        print("尺寸: {}x{} (分辨率: {:.3f}m/pixel)".format(width, height, resolution))
        print("原点: ({:.2f}, {:.2f})".format(origin_x, origin_y))
        print("覆盖范围: {:.1f}m x {:.1f}m".format(width * resolution, height * resolution))
        
        # 分析数据统计
        data = np.array(costmap.data)
        
        unknown_cells = np.sum(data == -1)
        free_cells = np.sum(data == 0)
        obstacle_cells = np.sum(data > 0)
        total_cells = len(data)
        
        print("单元格统计:")
        print("  未知: {} ({:.1f}%)".format(unknown_cells, 100.0 * unknown_cells / total_cells))
        print("  自由: {} ({:.1f}%)".format(free_cells, 100.0 * free_cells / total_cells))
        print("  障碍: {} ({:.1f}%)".format(obstacle_cells, 100.0 * obstacle_cells / total_cells))
        
        # 检查机器人周围区域
        if self.robot_pose:
            robot_result = self.check_robot_area(costmap, self.robot_pose)
            print("机器人周围区域: {}".format(robot_result))
            
        # 生成可视化图像
        image = self.costmap_to_image(costmap)
        if image is not None:
            # 保存图像用于调试
            filename = "/tmp/{}_costmap.png".format(name.lower().replace(' ', '_'))
            cv2.imwrite(filename, image)
            print("代价地图图像已保存: {}".format(filename))
            
        analysis = {
            'width': width,
            'height': height,
            'resolution': resolution,
            'unknown_ratio': unknown_cells / float(total_cells),
            'obstacle_ratio': obstacle_cells / float(total_cells),
            'free_ratio': free_cells / float(total_cells)
        }
        
        return analysis
        
    def check_robot_area(self, costmap, pose):
        """检查机器人周围区域的代价"""
        # 将机器人世界坐标转换为地图坐标
        map_x = int((pose.position.x - costmap.info.origin.position.x) / costmap.info.resolution)
        map_y = int((pose.position.y - costmap.info.origin.position.y) / costmap.info.resolution)
        
        width = costmap.info.width
        height = costmap.info.height
        
        # 检查边界
        if map_x < 0 or map_x >= width or map_y < 0 or map_y >= height:
            return "机器人位置超出地图范围"
            
        # 检查机器人位置的代价值
        robot_cell_index = map_y * width + map_x
        robot_cost = costmap.data[robot_cell_index]
        
        # 检查机器人周围3x3区域
        area_costs = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                check_x = map_x + dx
                check_y = map_y + dy
                
                if 0 <= check_x < width and 0 <= check_y < height:
                    cell_index = check_y * width + check_x
                    area_costs.append(costmap.data[cell_index])
                    
        if robot_cost == -1:
            return "未知区域 (代价: {})".format(robot_cost)
        elif robot_cost > 90:
            return "高代价区域 (代价: {}) - 可能被视为障碍".format(robot_cost)
        elif robot_cost > 50:
            return "中等代价区域 (代价: {})".format(robot_cost)
        else:
            return "自由区域 (代价: {})".format(robot_cost)
            
    def analyze_path_blockage(self):
        """分析路径是否被阻塞"""
        print("\n--- 路径阻塞分析 ---")
        
        if not self.global_costmap or not self.robot_pose:
            print("缺少数据，无法分析路径")
            return
            
        # 目标点列表 (仓库中的关键位置)
        targets = [
            [1.0, 5.0, "红色区域"],
            [3.0, 5.0, "蓝色区域"], 
            [-5.0, 1.0, "绿色区域"],
            [5.0, 1.0, "紫色区域"],
            [0.0, 1.5, "家位置"]
        ]
        
        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y
        
        for target_x, target_y, name in targets:
            blocked = self.check_line_of_sight(robot_x, robot_y, target_x, target_y)
            if blocked:
                print("到 {} 的直线路径被阻塞".format(name))
            else:
                print("到 {} 的直线路径畅通".format(name))
                
    def check_line_of_sight(self, x1, y1, x2, y2):
        """检查两点间是否有直线视线"""
        if not self.global_costmap:
            return True
            
        # 使用Bresenham算法检查直线路径
        costmap = self.global_costmap
        resolution = costmap.info.resolution
        origin_x = costmap.info.origin.position.x
        origin_y = costmap.info.origin.position.y
        width = costmap.info.width
        height = costmap.info.height
        
        # 转换为地图坐标
        map_x1 = int((x1 - origin_x) / resolution)
        map_y1 = int((y1 - origin_y) / resolution)
        map_x2 = int((x2 - origin_x) / resolution)
        map_y2 = int((y2 - origin_y) / resolution)
        
        # Bresenham直线算法
        dx = abs(map_x2 - map_x1)
        dy = abs(map_y2 - map_y1)
        
        x_step = 1 if map_x1 < map_x2 else -1
        y_step = 1 if map_y1 < map_y2 else -1
        
        error = dx - dy
        x, y = map_x1, map_y1
        
        while True:
            # 检查边界
            if x < 0 or x >= width or y < 0 or y >= height:
                return True  # 超出边界视为阻塞
                
            # 检查代价值
            cell_index = y * width + x
            cost = costmap.data[cell_index]
            
            if cost > 90:  # 高代价视为阻塞
                return True
                
            if x == map_x2 and y == map_y2:
                break
                
            error2 = 2 * error
            
            if error2 > -dy:
                error -= dy
                x += x_step
                
            if error2 < dx:
                error += dx
                y += y_step
                
        return False
        
    def compare_laser_and_costmap(self):
        """比较激光雷达数据与代价地图"""
        print("\n--- 激光雷达与代价地图对比 ---")
        
        if not self.laser_data or not self.local_costmap or not self.robot_pose:
            print("缺少数据，无法对比")
            return
            
        # 统计激光雷达检测到的障碍物
        ranges = self.laser_data.ranges
        valid_ranges = [r for r in ranges if self.laser_data.range_min < r < self.laser_data.range_max]
        close_obstacles = [r for r in valid_ranges if r < 1.0]  # 1米内的障碍物
        
        print("激光雷达统计:")
        print("  总读数: {}".format(len(ranges)))
        print("  有效读数: {}".format(len(valid_ranges)))
        print("  1米内障碍: {}".format(len(close_obstacles)))
        
        # 检查局部代价地图中机器人周围的障碍
        costmap = self.local_costmap
        width = costmap.info.width  
        height = costmap.info.height
        
        # 机器人在局部地图中心
        robot_map_x = width // 2
        robot_map_y = height // 2
        
        # 统计机器人周围的高代价单元格
        radius_cells = int(1.0 / costmap.info.resolution)  # 1米对应的单元格数
        high_cost_cells = 0
        
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                check_x = robot_map_x + dx
                check_y = robot_map_y + dy
                
                if 0 <= check_x < width and 0 <= check_y < height:
                    cell_index = check_y * width + check_x
                    if costmap.data[cell_index] > 90:
                        high_cost_cells += 1
                        
        print("局部代价地图统计:")
        print("  1米内高代价单元格: {}".format(high_cost_cells))
        
        # 判断一致性
        laser_detects_obstacles = len(close_obstacles) > 10
        costmap_shows_obstacles = high_cost_cells > 5
        
        if laser_detects_obstacles and costmap_shows_obstacles:
            print("✓ 激光雷达与代价地图一致：都检测到障碍物")
        elif not laser_detects_obstacles and not costmap_shows_obstacles:
            print("✓ 激光雷达与代价地图一致：都未检测到障碍物")
        elif laser_detects_obstacles and not costmap_shows_obstacles:
            print("⚠ 不一致：激光雷达检测到障碍物，但代价地图显示清空")
        else:
            print("⚠ 不一致：代价地图显示障碍物，但激光雷达未检测到")
            
    def run_analysis(self):
        """运行完整分析"""
        print("等待数据...")
        
        # 等待数据
        start_time = rospy.Time.now()
        while (not self.global_costmap or not self.local_costmap or 
               not self.robot_pose or not self.laser_data) and not rospy.is_shutdown():
            rospy.sleep(0.1)
            if (rospy.Time.now() - start_time).to_sec() > 10.0:
                break
                
        print("开始分析...\n")
        
        # 分析全局代价地图
        global_analysis = self.analyze_costmap(self.global_costmap, "全局代价地图")
        
        # 分析局部代价地图
        local_analysis = self.analyze_costmap(self.local_costmap, "局部代价地图")
        
        # 路径阻塞分析
        self.analyze_path_blockage()
        
        # 激光雷达对比
        self.compare_laser_and_costmap()
        
        # 生成诊断建议
        self.generate_recommendations(global_analysis, local_analysis)
        
    def generate_recommendations(self, global_analysis, local_analysis):
        """生成诊断建议"""
        print("\n" + "="*50)
        print("诊断建议")
        print("="*50)
        
        recommendations = []
        
        if global_analysis:
            if global_analysis['obstacle_ratio'] > 0.3:
                recommendations.append("全局代价地图障碍物比例过高 ({:.1f}%)，检查传感器数据".format(
                    global_analysis['obstacle_ratio'] * 100))
                    
            if global_analysis['unknown_ratio'] > 0.5:
                recommendations.append("全局代价地图未知区域过多 ({:.1f}%)，可能需要重新建图".format(
                    global_analysis['unknown_ratio'] * 100))
                    
        if local_analysis:
            if local_analysis['obstacle_ratio'] > 0.4:
                recommendations.append("局部代价地图障碍物密度过高，可能导航困难")
                
        if not recommendations:
            recommendations.append("代价地图看起来正常")
            
        for i, rec in enumerate(recommendations, 1):
            print("{}. {}".format(i, rec))
            
        print("\n建议的解决方案:")
        print("1. 检查激光雷达是否正常工作")
        print("2. 清除代价地图: rosservice call /move_base/clear_costmaps")
        print("3. 重启move_base节点")
        print("4. 检查机器人是否卡在障碍物中")
        print("5. 调整代价地图参数 (inflation_radius, cost_scaling_factor)")

if __name__ == '__main__':
    try:
        analyzer = CostmapAnalyzer()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("分析中断")
