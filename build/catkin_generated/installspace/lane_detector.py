#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point

class LaneDetector(object):
    def __init__(self):
        rospy.init_node('lane_detector', anonymous=False)
        
        self.bridge = CvBridge()
        
        # Load parameters
        self.lower_hsv = np.array(rospy.get_param('~lane_detection/color/lower_hsv', [20, 100, 100]))
        self.upper_hsv = np.array(rospy.get_param('~lane_detection/color/upper_hsv', [30, 255, 255]))
        self.blur_kernel = rospy.get_param('~lane_detection/processing/blur_kernel_size', 5)
        self.min_area = rospy.get_param('~lane_detection/processing/min_contour_area', 500)
        self.max_area = rospy.get_param('~lane_detection/processing/max_contour_area', 50000)
        
        # Camera parameters
        self.img_width = rospy.get_param('~lane_detection/camera_params/image_width', 1920)
        self.img_height = rospy.get_param('~lane_detection/camera_params/image_height', 1080)
        self.roi_y_start = rospy.get_param('~lane_detection/camera_params/roi_y_start', 400)
        self.roi_y_end = rospy.get_param('~lane_detection/camera_params/roi_y_end', 800)
        
        # Publishers
        self.lane_center_pub = rospy.Publisher('/lane_center', String, queue_size=10)
        self.lane_debug_pub = rospy.Publisher('/lane_debug_image', Image, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/overhead_camera/image_raw', Image, self.image_callback)
        
        # Lane tracking
        self.lane_centers = []
        self.lane_detected = False
        
        rospy.loginfo("LaneDetector initialized")
        
    def image_callback(self, msg):
        """Process overhead camera image for lane detection"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Process image for lane detection
            lane_center = self.detect_lane_center(cv_image)
            
            # Publish lane center information
            if lane_center is not None:
                center_msg = String()
                center_msg.data = "{:.2f},{:.2f}".format(lane_center[0], lane_center[1])
                self.lane_center_pub.publish(center_msg)
                self.lane_detected = True
            else:
                # Publish no lane detected
                center_msg = String()
                center_msg.data = "no_lane"
                self.lane_center_pub.publish(center_msg)
                self.lane_detected = False
                
        except Exception as e:
            rospy.logerr("Lane detection error: %s", str(e))
            
    def detect_lane_center(self, image):
        """Detect lane center from overhead camera image"""
        # Apply ROI
        height, width = image.shape[:2]
        roi_image = image[self.roi_y_start:self.roi_y_end, :]
        
        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for yellow lane lines
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(mask, (self.blur_kernel, self.blur_kernel), 0)
        
        # Find contours - OpenCV 3.2 compatible
        try:
            # OpenCV 3.x returns image, contours, hierarchy
            _, contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            # OpenCV 4.x returns contours, hierarchy
            contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                valid_contours.append(contour)
                
        if not valid_contours:
            return None
            
        # Find lane center from valid contours
        lane_center = self.calculate_lane_center(valid_contours, roi_image)
        
        # Create debug image
        debug_image = self.create_debug_image(roi_image, mask, valid_contours, lane_center)
        self.publish_debug_image(debug_image)
        
        return lane_center
        
    def calculate_lane_center(self, contours, image):
        """Calculate the center of detected lane lines"""
        if not contours:
            return None
            
        # Get image center
        height, width = image.shape[:2]
        image_center_x = width / 2.0
        
        # Find the main lane contours
        lane_points = []
        
        for contour in contours:
            # Get contour moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lane_points.append((cx, cy))
                
        if not lane_points:
            return None
            
        # Calculate average center point
        avg_x = sum([p[0] for p in lane_points]) / float(len(lane_points))
        avg_y = sum([p[1] for p in lane_points]) / float(len(lane_points))
        
        # Convert to world coordinates (simplified)
        # This assumes the overhead camera provides a top-down view
        # Scale factors would need calibration in real implementation
        world_x = (avg_x - image_center_x) * 0.01  # Convert pixels to meters
        world_y = (height - avg_y) * 0.01  # Convert pixels to meters
        
        return (world_x, world_y)
        
    def create_debug_image(self, original, mask, contours, center):
        """Create debug visualization image"""
        debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Draw contours
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
        
        # Draw center point
        if center is not None:
            height, width = original.shape[:2]
            image_center_x = width / 2.0
            center_x = int(center[0] / 0.01 + image_center_x)
            center_y = int(height - center[1] / 0.01)
            cv2.circle(debug, (center_x, center_y), 10, (0, 0, 255), -1)
            
        return debug
        
    def publish_debug_image(self, debug_image):
        """Publish debug image for visualization"""
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.lane_debug_pub.publish(debug_msg)
        except Exception as e:
            rospy.logerr("Debug image publish error: %s", str(e))

if __name__ == '__main__':
    try:
        detector = LaneDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("LaneDetector node terminated")
