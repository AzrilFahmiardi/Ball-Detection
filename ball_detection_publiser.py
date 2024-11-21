import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import cv2
import numpy as np
import pyrealsense2 as rs

class BallTracker(Node):
    def __init__(self):
        super().__init__('ball_tracker')
        self.publisher = self.create_publisher(Point, 'ball_position', 10)

        # Konfigurasi kamera RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # Parameter deteksi bola
        self.lower_orange = np.array([0, 120, 100])
        self.upper_orange = np.array([20, 255, 255])
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.camera_matrix = np.array([[607.1673, 0, 319.5],
                                       [0, 607.1673, 239.5],
                                       [0, 0, 1]])

        self.timer = self.create_timer(0.1, self.track_ball)

    def calculate_3d_position(self, center, depth):
        x_pixel, y_pixel = center
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        z = depth / 1000.0  # Convert to meters
        x_camera = (cx - x_pixel) * z / fx
        y_camera = (cy - y_pixel) * z / fy
        return np.array([z, x_camera, y_camera])

    def track_ball(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = cv2.GaussianBlur(color_image, (5, 5), 0)

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        mask = cv2.erode(mask, self.kernel)
        mask = cv2.dilate(mask, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                distance = depth_image[int(y), int(x)]

                if distance > 0:
                    ball_3d = self.calculate_3d_position(center, distance)
                    point_msg = Point()
                    point_msg.x = ball_3d[0]  # z (depth)
                    point_msg.y = ball_3d[1]  # x (horizontal)
                    point_msg.z = ball_3d[2]  # y (vertical)
                    self.publisher.publish(point_msg)
                    self.get_logger().info(f'Published: {point_msg}')