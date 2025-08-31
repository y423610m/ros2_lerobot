FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python3-pip

# lerobot specific
RUN python3 -m pip install lerobot && \
    python3 -m pip install 'lerobot[feetech]'

# non headless opencv 
RUN python3 -m pip uninstall -y opencv-python && \
    python3 -m pip uninstall -y opencv-python-headless && \
    python3 -m pip install opencv-python

RUN apt update && apt install -y libxcb-xinerama0 libxcb-render0 libxcb-shape0 libxcb-randr0 libxkbcommon-x11-0

# ROS
RUN apt update && apt install -y \
    ros-humble-rviz2 \
    ros-humble-rqt-graph \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-rqt-tf-tree \
     && \
    rm -rf /var/lib/apt/lists/*

# python common
RUN python3 -m pip install urdf-parser-py ipython matplotlib

# 
RUN apt update && apt install -y \
    vim \
    less \
    usbutils \
    libgtk2.0-dev \
    pkg-config

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

RUN mkdir -p /ros2_lerobot_ws/src

WORKDIR /ros2_lerobot_ws

CMD ["bash"]