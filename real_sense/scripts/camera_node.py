#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import pyrealsense2 as rs

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import time


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # QoS (match your C++ best effort)
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT
        )

        # Publishers
        self.compressed_pub = self.create_publisher(
            CompressedImage,
            '/camera/rgb/compressed',
            qos
        )

        self.raw_pub = self.create_publisher(
            Image,
            '/camera/rgb/raw',
            10
        )

        self.bridge = CvBridge()

        # Reset any stale device locks from previous runs
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            self.get_logger().error("No RealSense device found!")
            raise RuntimeError("No RealSense device")

        self.get_logger().info("Resetting camera hardware...")
        devices[0].hardware_reset()
        time.sleep(3)  # wait for device to reinitialize after reset

        # Now safe to start
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # --- RealSense setup ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            self.pipeline.start(config)
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense pipeline: {e}")
            raise

        # Single loop at ~30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.loop)

    def loop(self):
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                return

            frame = np.asanyarray(color_frame.get_data())
            stamp = self.get_clock().now().to_msg()

            # --- Compressed ---
            success, buffer = cv2.imencode(
                '.jpg',
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            )

            if success:
                comp_msg = CompressedImage()
                comp_msg.header.stamp = stamp
                comp_msg.format = "jpeg"
                comp_msg.data = buffer.tobytes()
                self.compressed_pub.publish(comp_msg)

            # --- Raw ---
            raw_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            raw_msg.header.stamp = stamp
            self.raw_pub.publish(raw_msg)

        except Exception as e:
            self.get_logger().error(f"Frame processing error: {e}")

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = CameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()