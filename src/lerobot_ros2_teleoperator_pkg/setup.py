from setuptools import find_packages, setup

package_name = 'lerobot_ros2_teleoperator_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mujin',
    maintainer_email='yoshito.mori@mujin.co.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lerobot_ros2_teleoperator_node = lerobot_ros2_teleoperator_pkg.lerobot_ros2_teleoperator_node:main'
        ],
    },
)
