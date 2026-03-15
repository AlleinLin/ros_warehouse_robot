#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import Tkinter as tk
import ttk
import threading
import time
import math
import tf
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetModelState
import cv2
from cv_bridge import CvBridge
import numpy as np

class MonitorGUI(object):
    def __init__(self):
        rospy.init_node('monitor_gui', anonymous=False)
        
        self.bridge = CvBridge()
        
        # 等待Gazebo服务以获取准确坐标
        rospy.loginfo("Waiting for Gazebo get_model_state service...")
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.gazebo_available = True
            rospy.loginfo("Gazebo service connected - will use ground truth coordinates")
        except:
            rospy.logwarn("Gazebo service not available - will use AMCL coordinates")
            self.gazebo_available = False
            self.get_model_state = None
        
        # Data storage
        self.robot_state = "UNKNOWN"
        self.current_zone = "unknown"
        self.package_color = "none"
        self.actual_package_color = "none"
        self.navigation_status = "idle"
        self.manipulation_status = "idle"
        self.obstacle_status = "clear"
        self.lane_center = "no_lane"
        
        # 统一坐标系统
        self.robot_amcl_pose = None   # 主要坐标源
        self.robot_odom_pose = None   # 备用坐标源
        self.robot_position = [0.0, 0.0, 0.0]  # [x, y, yaw]
        self.coordinate_source = "none"  # 跟踪坐标来源
        
        # Images
        self.lane_debug_image = None
        self.package_debug_image = None
        self.front_camera_image = None
        
        # Laser data
        self.laser_ranges = []
        self.laser_min_distance = float('inf')
        
        # Create GUI
        self.setup_gui()
        
        # ROS subscribers
        self.setup_subscribers()
        
        # Start GUI update thread
        self.gui_update_thread = threading.Thread(target=self.update_gui_loop)
        self.gui_update_thread.daemon = True
        self.gui_update_thread.start()
        
        rospy.loginfo("MonitorGUI initialized with GROUND TRUTH COORDINATE SYSTEM")
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Warehouse Robot Monitor - UNIFIED COORDINATES")
        self.root.geometry("1200x800")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status tab
        self.status_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.status_frame, text="Robot Status")
        self.setup_status_tab()
        
        # Vision tab
        self.vision_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.vision_frame, text="Vision Debug")
        self.setup_vision_tab()
        
        # Sensors tab
        self.sensors_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sensors_frame, text="Sensors")
        self.setup_sensors_tab()
        
    def setup_status_tab(self):
        """Setup robot status display tab"""
        # Main status information
        status_group = ttk.LabelFrame(self.status_frame, text="Robot Status", padding=10)
        status_group.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.status_labels = {}
        
        # Robot state
        tk.Label(status_group, text="Robot State:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.status_labels['state'] = tk.Label(status_group, text="UNKNOWN", font=("Arial", 12), fg="blue")
        self.status_labels['state'].grid(row=0, column=1, sticky="w", padx=10)
        
        # Current zone
        tk.Label(status_group, text="Current Zone:", font=("Arial", 10)).grid(row=1, column=0, sticky="w")
        self.status_labels['zone'] = tk.Label(status_group, text="unknown", font=("Arial", 10))
        self.status_labels['zone'].grid(row=1, column=1, sticky="w", padx=10)
        
        # Package color (expected)
        tk.Label(status_group, text="Expected Package:", font=("Arial", 10)).grid(row=2, column=0, sticky="w")
        self.status_labels['package'] = tk.Label(status_group, text="none", font=("Arial", 10))
        self.status_labels['package'].grid(row=2, column=1, sticky="w", padx=10)
        
        # 实际包裹颜色
        tk.Label(status_group, text="Actual Package:", font=("Arial", 10)).grid(row=3, column=0, sticky="w")
        self.status_labels['actual_package'] = tk.Label(status_group, text="none", font=("Arial", 10))
        self.status_labels['actual_package'].grid(row=3, column=1, sticky="w", padx=10)
        
        # 统一坐标显示
        tk.Label(status_group, text="Position (X, Y, θ):", font=("Arial", 10)).grid(row=4, column=0, sticky="w")
        self.status_labels['position'] = tk.Label(status_group, text="0.000, 0.000, 0.000", font=("Arial", 10))
        self.status_labels['position'].grid(row=4, column=1, sticky="w", padx=10)
        
        # 坐标来源显示
        tk.Label(status_group, text="Coordinate Source:", font=("Arial", 10)).grid(row=5, column=0, sticky="w")
        self.status_labels['coord_source'] = tk.Label(status_group, text="none", font=("Arial", 10))
        self.status_labels['coord_source'].grid(row=5, column=1, sticky="w", padx=10)
        
        # Navigation and manipulation status
        nav_group = ttk.LabelFrame(self.status_frame, text="Navigation & Manipulation", padding=10)
        nav_group.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        tk.Label(nav_group, text="Navigation:", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        self.status_labels['navigation'] = tk.Label(nav_group, text="idle", font=("Arial", 10))
        self.status_labels['navigation'].grid(row=0, column=1, sticky="w", padx=10)
        
        tk.Label(nav_group, text="Manipulation:", font=("Arial", 10)).grid(row=1, column=0, sticky="w")
        self.status_labels['manipulation'] = tk.Label(nav_group, text="idle", font=("Arial", 10))
        self.status_labels['manipulation'].grid(row=1, column=1, sticky="w", padx=10)
        
        tk.Label(nav_group, text="Obstacles:", font=("Arial", 10)).grid(row=2, column=0, sticky="w")
        self.status_labels['obstacles'] = tk.Label(nav_group, text="clear", font=("Arial", 10))
        self.status_labels['obstacles'].grid(row=2, column=1, sticky="w", padx=10)
        
        tk.Label(nav_group, text="Lane Center:", font=("Arial", 10)).grid(row=3, column=0, sticky="w")
        self.status_labels['lane'] = tk.Label(nav_group, text="no_lane", font=("Arial", 10))
        self.status_labels['lane'].grid(row=3, column=1, sticky="w", padx=10)
        
        # Task progress
        task_group = ttk.LabelFrame(self.status_frame, text="Task Progress", padding=10)
        task_group.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        self.progress_text = tk.Text(task_group, height=10, width=60)
        self.progress_text.grid(row=0, column=0, sticky="nsew")
        
        progress_scroll = ttk.Scrollbar(task_group, orient="vertical", command=self.progress_text.yview)
        progress_scroll.grid(row=0, column=1, sticky="ns")
        self.progress_text.configure(yscrollcommand=progress_scroll.set)
        
    def setup_vision_tab(self):
        """Setup vision debug display tab"""
        # Lane detection
        lane_group = ttk.LabelFrame(self.vision_frame, text="Lane Detection", padding=10)
        lane_group.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.lane_canvas = tk.Canvas(lane_group, width=400, height=300, bg="black")
        self.lane_canvas.pack()
        
        # Package detection
        package_group = ttk.LabelFrame(self.vision_frame, text="Package Detection", padding=10)
        package_group.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.package_canvas = tk.Canvas(package_group, width=400, height=300, bg="black")
        self.package_canvas.pack()
        
        # Front camera
        front_group = ttk.LabelFrame(self.vision_frame, text="Front Camera", padding=10)
        front_group.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        self.front_canvas = tk.Canvas(front_group, width=640, height=240, bg="black")
        self.front_canvas.pack()
        
    def setup_sensors_tab(self):
        """Setup sensors display tab"""
        # Laser scan visualization
        laser_group = ttk.LabelFrame(self.sensors_frame, text="Laser Scanner", padding=10)
        laser_group.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.laser_canvas = tk.Canvas(laser_group, width=500, height=500, bg="black")
        self.laser_canvas.pack()
        
        # Sensor readings
        readings_group = ttk.LabelFrame(self.sensors_frame, text="Sensor Readings", padding=10)
        readings_group.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.sensor_labels = {}
        
        tk.Label(readings_group, text="Min Laser Distance:", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        self.sensor_labels['laser_min'] = tk.Label(readings_group, text="∞", font=("Arial", 10))
        self.sensor_labels['laser_min'].grid(row=0, column=1, sticky="w", padx=10)
        
    def setup_subscribers(self):
        """Setup ROS subscribers with unified coordinate system"""
        rospy.Subscriber('/robot_state', String, self.robot_state_callback)
        rospy.Subscriber('/current_zone', String, self.zone_callback)
        rospy.Subscriber('/current_package_color', String, self.package_color_callback)
        rospy.Subscriber('/actual_package_color', String, self.actual_package_color_callback)  # 🔥 新增
        rospy.Subscriber('/navigation/arrived', String, self.navigation_callback)
        rospy.Subscriber('/manipulation/status', String, self.manipulation_callback)
        rospy.Subscriber('/fused_obstacle_info', String, self.obstacle_callback)
        rospy.Subscriber('/lane_center', String, self.lane_callback)
        
        # 统一使用amcl_pose作为主要坐标源
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)  # 仅作为备用
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        
        # Vision debug topics
        rospy.Subscriber('/lane_debug_image', Image, self.lane_debug_callback)
        rospy.Subscriber('/package_debug_image', Image, self.package_debug_callback)
        rospy.Subscriber('/front_camera_debug', Image, self.front_debug_callback)
        
    def robot_state_callback(self, msg):
        old_state = self.robot_state
        self.robot_state = msg.data
        if old_state != self.robot_state:
            self.log_progress("State: {} -> {} at [{}]".format(
                old_state, self.robot_state, self.format_position()))
        
    def zone_callback(self, msg):
        old_zone = self.current_zone
        self.current_zone = msg.data
        if old_zone != self.current_zone:
            self.log_progress("Zone: {} -> {} at [{}]".format(
                old_zone, self.current_zone, self.format_position()))
        
    def package_color_callback(self, msg):
        old_color = self.package_color
        self.package_color = msg.data
        if old_color != self.package_color:
            self.log_progress("Expected package: {} at [{}]".format(
                self.package_color, self.format_position()))
        
    def actual_package_color_callback(self, msg):
        """处理实际包裹颜色"""
        old_color = self.actual_package_color
        self.actual_package_color = msg.data
        if old_color != self.actual_package_color:
            self.log_progress("Actual package picked: {} at [{}]".format(
                self.actual_package_color, self.format_position()))
        
    def navigation_callback(self, msg):
        old_status = self.navigation_status
        self.navigation_status = msg.data
        if old_status != self.navigation_status:
            self.log_progress("Navigation: {} at [{}]".format(
                self.navigation_status, self.format_position()))
        
    def manipulation_callback(self, msg):
        old_status = self.manipulation_status
        self.manipulation_status = msg.data
        if old_status != self.manipulation_status:
            self.log_progress("Manipulation: {} at [{}]".format(
                self.manipulation_status, self.format_position()))
        
    def obstacle_callback(self, msg):
        old_status = self.obstacle_status
        self.obstacle_status = msg.data
        if old_status != self.obstacle_status and self.obstacle_status != "clear":
            self.log_progress("Obstacles: {} at [{}]".format(
                self.obstacle_status, self.format_position()))
        
    def lane_callback(self, msg):
        self.lane_center = msg.data
        
    def pose_callback(self, msg):
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
                    self.robot_position = [pos.x, pos.y, yaw]
                    self.coordinate_source = "gazebo"
                    return
            except:
                pass
        
        # 备用：使用AMCL坐标
        self.robot_amcl_pose = msg.pose.pose
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        
        # 计算yaw角度
        yaw = self.get_yaw_from_quaternion(orient)
        self.robot_position = [pos.x, pos.y, yaw]
        self.coordinate_source = "amcl"
        
    def odom_callback(self, msg):
        """里程计回调 - 仅作为备用坐标源"""
        self.robot_odom_pose = msg.pose.pose
        
        # 如果amcl_pose不可用，使用odom作为备用
        if self.robot_amcl_pose is None:
            pos = msg.pose.pose.position
            orient = msg.pose.pose.orientation
            yaw = self.get_yaw_from_quaternion(orient)
            self.robot_position = [pos.x, pos.y, yaw]
            self.coordinate_source = "odom"
            
    def get_yaw_from_quaternion(self, quaternion):
        """从四元数获取yaw角度"""
        try:
            euler = tf.transformations.euler_from_quaternion([
                quaternion.x, quaternion.y, quaternion.z, quaternion.w
            ])
            return euler[2]  # yaw
        except:
            return 0.0
            
    def format_position(self):
        """格式化位置信息用于显示"""
        return "{:.3f},{:.3f},{:.3f}".format(
            self.robot_position[0], self.robot_position[1], self.robot_position[2])
            
    def laser_callback(self, msg):
        self.laser_ranges = list(msg.ranges)
        valid_ranges = [r for r in self.laser_ranges if msg.range_min < r < msg.range_max]
        self.laser_min_distance = min(valid_ranges) if valid_ranges else float('inf')
        
    def lane_debug_callback(self, msg):
        try:
            self.lane_debug_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass
            
    def package_debug_callback(self, msg):
        try:
            self.package_debug_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass
            
    def front_debug_callback(self, msg):
        try:
            self.front_camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass
            
    def log_progress(self, message):
        """Add message to progress log"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = "[{}] {}\n".format(timestamp, message)
        
        def update_log():
            self.progress_text.insert(tk.END, log_message)
            self.progress_text.see(tk.END)
            # 限制日志长度，避免内存溢出
            if self.progress_text.index('end-1c').split('.')[0] > "100":
                self.progress_text.delete('1.0', '50.0')
            
        self.root.after(0, update_log)
        
    def update_gui_loop(self):
        """GUI update loop running in separate thread"""
        while not rospy.is_shutdown():
            try:
                self.root.after(0, self.update_display)
                time.sleep(0.1)
            except:
                break
                
    def update_display(self):
        """Update GUI display with unified coordinate system"""
        # Update status labels
        self.status_labels['state'].config(text=self.robot_state)
        
        # 状态颜色编码
        if self.robot_state == "EXIT_PICKUP_ZONE":
            self.status_labels['state'].config(fg="red")
        elif self.robot_state in ["PICK_PACKAGE", "PLACE_PACKAGE"]:
            self.status_labels['state'].config(fg="orange")
        elif self.robot_state in ["NAVIGATE_TO_PICKUP", "NAVIGATE_TO_DROP", "RETURN_TO_PICKUP"]:
            self.status_labels['state'].config(fg="green")
        else:
            self.status_labels['state'].config(fg="blue")
            
        self.status_labels['zone'].config(text=self.current_zone)
        self.status_labels['package'].config(text=self.package_color)
        self.status_labels['actual_package'].config(text=self.actual_package_color)  # 🔥 显示实际包裹颜色
        
        # 显示统一坐标系统的位置信息
        position_text = "{:.3f}, {:.3f}, {:.3f}".format(*self.robot_position)
        self.status_labels['position'].config(text=position_text)
        
        # 显示坐标来源
        coord_source_text = self.coordinate_source
        if self.coordinate_source == "gazebo":
            self.status_labels['coord_source'].config(text=coord_source_text + " (ground truth)", fg="blue")
        elif self.coordinate_source == "amcl":
            self.status_labels['coord_source'].config(text=coord_source_text, fg="green")
        elif self.coordinate_source == "odom":
            self.status_labels['coord_source'].config(text=coord_source_text + " (backup)", fg="orange")
        else:
            self.status_labels['coord_source'].config(text="no data", fg="red")
            
        self.status_labels['navigation'].config(text=self.navigation_status)
        self.status_labels['manipulation'].config(text=self.manipulation_status)
        self.status_labels['obstacles'].config(text=self.obstacle_status)
        self.status_labels['lane'].config(text=self.lane_center)
        
        # Update sensor readings
        if hasattr(self, 'sensor_labels'):
            laser_text = "{:.3f}m".format(self.laser_min_distance) if self.laser_min_distance != float('inf') else "∞"
            self.sensor_labels['laser_min'].config(text=laser_text)
        
        # Update color coding based on status
        if self.obstacle_status == "clear":
            self.status_labels['obstacles'].config(fg="green")
        elif self.obstacle_status == "obstacle_detected":
            self.status_labels['obstacles'].config(fg="orange")
        else:
            self.status_labels['obstacles'].config(fg="red")
            
        # 实际包裹颜色编码
        if self.actual_package_color != "none":
            color_map = {"red": "red", "blue": "blue", "green": "green", "purple": "purple"}
            self.status_labels['actual_package'].config(fg=color_map.get(self.actual_package_color, "black"))
        else:
            self.status_labels['actual_package'].config(fg="gray")
            
        # Update images
        self.update_vision_displays()
        self.update_laser_display()
        
    def update_vision_displays(self):
        """Update vision debug displays"""
        # Update lane debug image
        if self.lane_debug_image is not None:
            self.display_image_on_canvas(self.lane_debug_image, self.lane_canvas, (400, 300))
            
        # Update package debug image
        if self.package_debug_image is not None:
            self.display_image_on_canvas(self.package_debug_image, self.package_canvas, (400, 300))
            
        # Update front camera image
        if self.front_camera_image is not None:
            self.display_image_on_canvas(self.front_camera_image, self.front_canvas, (640, 240))
            
    def display_image_on_canvas(self, cv_image, canvas, size):
        """Display OpenCV image on Tkinter canvas"""
        try:
            # Resize image
            resized = cv2.resize(cv_image, size)
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL format
            try:
                from PIL import Image, ImageTk
                pil_image = Image.fromarray(rgb_image)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update canvas
                canvas.delete("all")
                canvas.create_image(size[0]//2, size[1]//2, anchor=tk.CENTER, image=photo)
                canvas.image = photo  # Keep a reference
            except ImportError:
                # PIL not available, skip image display
                pass
            
        except Exception as e:
            rospy.logwarn("Image display error: %s", str(e))
            
    def update_laser_display(self):
        """Update laser scan visualization"""
        if not self.laser_ranges:
            return
            
        try:
            self.laser_canvas.delete("all")
            
            # Canvas properties
            canvas_size = 500
            center = canvas_size // 2
            scale = 50  # pixels per meter
            
            # Draw grid
            for i in range(1, 6):
                radius = i * scale
                self.laser_canvas.create_oval(center - radius, center - radius,
                                            center + radius, center + radius,
                                            outline="gray", width=1)
                                            
            # Draw laser points
            num_points = len(self.laser_ranges)
            for i, range_val in enumerate(self.laser_ranges):
                if 0.1 < range_val < 10:  # Valid range
                    angle = -3.14159 + (i * 2 * 3.14159 / num_points)
                    x = center + range_val * scale * np.cos(angle)
                    y = center - range_val * scale * np.sin(angle)
                    
                    # Color based on distance
                    if range_val < 0.5:
                        color = "red"
                    elif range_val < 1.0:
                        color = "orange"
                    else:
                        color = "green"
                        
                    self.laser_canvas.create_oval(x-2, y-2, x+2, y+2, fill=color, outline=color)
                    
            # Draw robot with position info
            self.laser_canvas.create_oval(center-10, center-10, center+10, center+10,
                                        fill="blue", outline="blue")
                                        
            # 显示机器人位置信息
            pos_text = "Pos: {:.2f}, {:.2f}, {:.2f}°".format(
                self.robot_position[0], self.robot_position[1], 
                math.degrees(self.robot_position[2]))
            self.laser_canvas.create_text(10, 10, anchor="nw", text=pos_text, 
                                        fill="white", font=("Arial", 10))
            
            # 显示坐标来源
            if self.coordinate_source == "gazebo":
                source_text = "Source: {} (ground truth)".format(self.coordinate_source)
                source_color = "blue"
            elif self.coordinate_source == "amcl":
                source_text = "Source: {}".format(self.coordinate_source)
                source_color = "green"
            else:
                source_text = "Source: {} (backup)".format(self.coordinate_source)
                source_color = "orange"
            self.laser_canvas.create_text(10, 25, anchor="nw", text=source_text, 
                                        fill=source_color, font=("Arial", 10))
                                        
        except Exception as e:
            rospy.logwarn("Laser display error: %s", str(e))
            
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

if __name__ == '__main__':
    try:
        gui = MonitorGUI()
        gui.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("MonitorGUI node terminated")