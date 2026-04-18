from setuptools import find_packages, setup

package_name = 'lerobot_robots_bringup'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'so101_joint_state_to_trajectory = lerobot_robots_bringup.so101_joint_state_to_trajectory:main'
        ],
    },
)