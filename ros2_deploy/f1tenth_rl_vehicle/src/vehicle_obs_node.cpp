// vehicle_obs_node: build the 380-dim policy observation on the real car.
//
// Merges the sim stack's track_server + observation_builder (+ odom adapter) into a
// single lean C++ node. It loads the training centerline CSV directly, samples the
// latest map-frame pose (particle filter) and body-frame twist (VESC odom) on a
// fixed 10 Hz timer, reconstructs the exact training observation via rl_obs_core,
// and publishes /rl/observation.
//
// Tyre-slip observation dims [372:380] are published as zeros (no per-wheel sensing
// on the car), matching the gym deploy. Acceleration is a fixed-step finite
// difference of body velocity at the control rate (not a per-callback differencing).
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include "f1tenth_rl_vehicle/rl_obs_core.hpp"

namespace f1tenth_rl_vehicle
{

namespace
{
double quatToYaw(double x, double y, double z, double w)
{
  double siny_cosp = 2.0 * (w * z + x * y);
  double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}
}  // namespace

class VehicleObsNode : public rclcpp::Node
{
public:
  VehicleObsNode()
  : rclcpp::Node("vehicle_obs")
  {
    const std::string track_csv = declare_parameter<std::string>("track_csv", "");
    control_hz_ = declare_parameter<double>("control_hz", 10.0);
    const std::string pose_topic =
      declare_parameter<std::string>("pose_topic", "/pf/pose/odom");
    const std::string twist_topic =
      declare_parameter<std::string>("twist_topic", "/odom");
    const std::string action_topic =
      declare_parameter<std::string>("action_topic", "/rl/action");
    const std::string obs_topic =
      declare_parameter<std::string>("obs_topic", "/rl/observation");
    twist_in_world_frame_ = declare_parameter<bool>("twist_in_world_frame", false);

    ObsConfig cfg;
    cfg.num_obs = static_cast<int>(declare_parameter<int>("num_obs", 380));
    cfg.future_track_num_points =
      static_cast<int>(declare_parameter<int>("future_track_num_points", 60));
    cfg.future_track_horizon_s = declare_parameter<double>("future_track_horizon_s", 6.0);
    cfg.future_track_width = declare_parameter<double>("future_track_width", 2.2);
    cfg.contact_margin_m = declare_parameter<double>("contact_margin_m", 0.08);
    cfg.clip_obs = declare_parameter<double>("clip_obs", 50.0);
    cfg.lin_vel_scale = declare_parameter<double>("lin_vel_scale", 1.0);
    cfg.ang_vel_scale = declare_parameter<double>("ang_vel_scale", 1.0);
    cfg.lin_acc_scale = declare_parameter<double>("lin_acc_scale", 1.0);
    const int coarse_stride = static_cast<int>(declare_parameter<int>("coarse_stride", 10));

    if (track_csv.empty()) {
      throw std::runtime_error("vehicle_obs: 'track_csv' parameter is required");
    }
    TrackData track = loadTrackCsv(track_csv);
    builder_ = std::make_unique<TrackObservationBuilder>(
      track.x, track.y, track.w_left, track.w_right, cfg, coarse_stride);
    RCLCPP_INFO(
      get_logger(), "Loaded track '%s' (%d points); obs dim %d at %.1f Hz",
      track_csv.c_str(), builder_->numCenterlinePoints(), cfg.num_obs, control_hz_);

    obs_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(obs_topic, 10);

    pose_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      pose_topic, 10,
      [this](nav_msgs::msg::Odometry::SharedPtr msg) {this->onPose(*msg);});
    twist_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      twist_topic, 10,
      [this](nav_msgs::msg::Odometry::SharedPtr msg) {this->onTwist(*msg);});
    action_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      action_topic, 10,
      [this](std_msgs::msg::Float32MultiArray::SharedPtr msg) {this->onAction(*msg);});

    const double period = (control_hz_ > 0.0) ? 1.0 / control_hz_ : 0.1;
    timer_ = create_wall_timer(
      std::chrono::duration<double>(period),
      [this]() {this->onTimer();});
  }

private:
  void onPose(const nav_msgs::msg::Odometry & msg)
  {
    pos_x_ = msg.pose.pose.position.x;
    pos_y_ = msg.pose.pose.position.y;
    const auto & q = msg.pose.pose.orientation;
    yaw_ = quatToYaw(q.x, q.y, q.z, q.w);
    have_pose_ = true;
  }

  void onTwist(const nav_msgs::msg::Odometry & msg)
  {
    twist_vx_ = msg.twist.twist.linear.x;
    twist_vy_ = msg.twist.twist.linear.y;
    wz_ = msg.twist.twist.angular.z;
    have_twist_ = true;
  }

  void onAction(const std_msgs::msg::Float32MultiArray & msg)
  {
    if (msg.data.size() >= 2) {
      last_throttle_ = msg.data[0];
      last_steer_ = msg.data[1];
    }
  }

  void onTimer()
  {
    if (!have_pose_ || !have_twist_) {
      return;
    }

    double vx = twist_vx_;
    double vy = twist_vy_;
    if (twist_in_world_frame_) {
      double c = std::cos(yaw_);
      double s = std::sin(yaw_);
      double bx = c * vx + s * vy;
      double by = -s * vx + c * vy;
      vx = bx;
      vy = by;
    }

    // Fixed-step body acceleration at the control rate (not per odom callback).
    const double dt = (control_hz_ > 0.0) ? 1.0 / control_hz_ : 0.1;
    double ax = 0.0;
    double ay = 0.0;
    if (have_prev_vel_) {
      ax = (vx - prev_vx_) / dt;
      ay = (vy - prev_vy_) / dt;
    }
    prev_vx_ = vx;
    prev_vy_ = vy;
    have_prev_vel_ = true;

    VehicleState st;
    st.pos_x = pos_x_;
    st.pos_y = pos_y_;
    st.yaw = yaw_;
    st.vx = vx;
    st.vy = vy;
    st.wz = wz_;
    st.ax = ax;
    st.ay = ay;
    st.last_throttle = last_throttle_;
    st.last_steer = last_steer_;
    // tyre_slip defaults to zeros (no per-wheel sensing on the car).

    std::vector<float> obs = builder_->build(st);
    std_msgs::msg::Float32MultiArray out;
    out.data = std::move(obs);
    obs_pub_->publish(out);
  }

  std::unique_ptr<TrackObservationBuilder> builder_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr obs_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr twist_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr action_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  double control_hz_ = 10.0;
  bool twist_in_world_frame_ = false;

  bool have_pose_ = false;
  bool have_twist_ = false;
  double pos_x_ = 0.0, pos_y_ = 0.0, yaw_ = 0.0;
  double twist_vx_ = 0.0, twist_vy_ = 0.0, wz_ = 0.0;
  double last_throttle_ = 0.0, last_steer_ = 0.0;

  bool have_prev_vel_ = false;
  double prev_vx_ = 0.0, prev_vy_ = 0.0;
};

}  // namespace f1tenth_rl_vehicle

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<f1tenth_rl_vehicle::VehicleObsNode>());
  rclcpp::shutdown();
  return 0;
}
