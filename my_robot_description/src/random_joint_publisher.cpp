#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <urdf/model.h>
#include <random>

class RandomJointPublisher : public rclcpp::Node {
public:
  RandomJointPublisher() : Node("random_joint_publisher") {
    this->declare_parameter<std::string>("robot_description", "");
    std::string urdf_xml;
    this->get_parameter("robot_description", urdf_xml);

    if (!model_.initString(urdf_xml)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF");
      rclcpp::shutdown();
      return;
    }

    for (const auto &joint_pair : model_.joints_) {
      const auto &joint = joint_pair.second;
      if (joint->type != urdf::Joint::REVOLUTE && joint->type != urdf::Joint::PRISMATIC)
        continue;

      joint_names_.push_back(joint->name);
      double lower = joint->limits ? joint->limits->lower : -1.0;
      double upper = joint->limits ? joint->limits->upper : 1.0;
      joint_limits_.push_back(std::make_pair(lower, upper));
    }

    pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&RandomJointPublisher::publish_joint_states, this));
  }

private:
  void publish_joint_states() {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = this->now();
    msg.name = joint_names_;
    msg.position.resize(joint_names_.size());

    for (size_t i = 0; i < joint_names_.size(); ++i) {
      std::uniform_real_distribution<double> dist(joint_limits_[i].first, joint_limits_[i].second);
      msg.position[i] = dist(rng_);
    }

    pub_->publish(msg);
  }

  urdf::Model model_;
  std::vector<std::string> joint_names_;
  std::vector<std::pair<double, double>> joint_limits_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::default_random_engine rng_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RandomJointPublisher>());
  rclcpp::shutdown();
  return 0;
}
