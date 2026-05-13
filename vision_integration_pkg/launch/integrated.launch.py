from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision_integration_pkg',
            executable='integrated.py',
            name='vision_integration_runner',
            output='screen'
        )
    ])
