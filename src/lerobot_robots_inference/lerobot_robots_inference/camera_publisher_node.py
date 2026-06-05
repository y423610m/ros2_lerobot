#!/usr/bin/env python3
"""Grabs frames from a USB webcam via cv2.VideoCapture and publishes them as
sensor_msgs/Image on a configurable topic. One instance per camera.

Frames are published at their native resolution — downstream consumers (e.g.
policy_node) resize as needed.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__('camera_publisher')

        self.declare_parameter('device_index', 0)
        self.declare_parameter('topic_name', 'image_raw')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'camera')

        device_index = int(self.get_parameter('device_index').value)
        topic_name = str(self.get_parameter('topic_name').value)
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        fps = float(self.get_parameter('fps').value)
        self._frame_id = str(self.get_parameter('frame_id').value)

        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            raise RuntimeError(f'Failed to open camera device {device_index}')
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)

        self._bridge = CvBridge()
        self._publisher = self.create_publisher(Image, topic_name, 10)
        self._timer = self.create_timer(1.0 / fps, self._on_timer)

        self.get_logger().info(
            f'camera_publisher_node device={device_index} topic={topic_name} '
            f'{width}x{height}@{fps}Hz'
        )

    def _on_timer(self) -> None:
        ok, frame_bgr = self._capture.read()
        if not ok:
            self.get_logger().warn('frame grab failed')
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        msg = self._bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        self._publisher.publish(msg)

    def destroy_node(self) -> bool:
        self._capture.release()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
