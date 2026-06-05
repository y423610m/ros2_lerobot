from setuptools import find_packages, setup

package_name = 'lerobot_robots_inference'

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
    maintainer='yoshito.mori',
    maintainer_email='y423610m@icloud.com',
    description='Runs a trained mjlab vision policy on the real SO-101 arm.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_publisher_node = lerobot_robots_inference.camera_publisher_node:main',
            'policy_node = lerobot_robots_inference.policy_node:main',
        ],
    },
)
