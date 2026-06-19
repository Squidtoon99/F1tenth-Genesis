#include "f1tenth_rl_vehicle/rl_obs_core.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace f1tenth_rl_vehicle
{

namespace
{
constexpr double kTwoPi = 2.0 * M_PI;

double clampd(double v, double lo, double hi)
{
  return std::max(lo, std::min(hi, v));
}

// Python floor-modulo (result has the sign of the divisor), for index wraparound.
int floorMod(int a, int m)
{
  int r = a % m;
  return (r < 0) ? r + m : r;
}
}  // namespace

TrackData loadTrackCsv(const std::string & path)
{
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("loadTrackCsv: cannot open " + path);
  }

  std::string line;
  std::vector<std::string> names;
  TrackData track;

  auto split = [](const std::string & s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      size_t a = tok.find_first_not_of(" \t\r\n");
      size_t b = tok.find_last_not_of(" \t\r\n");
      out.push_back(a == std::string::npos ? "" : tok.substr(a, b - a + 1));
    }
    return out;
  };

  bool have_header = false;
  int ix = -1, iy = -1, il = -1, ir = -1;
  while (std::getline(f, line)) {
    if (line.empty()) {
      continue;
    }
    size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      continue;
    }
    if (line[first] == '#' && !have_header) {
      names = split(line.substr(first + 1));
      for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == "x_m") ix = static_cast<int>(i);
        else if (names[i] == "y_m") iy = static_cast<int>(i);
        else if (names[i] == "w_tr_left_m") il = static_cast<int>(i);
        else if (names[i] == "w_tr_right_m") ir = static_cast<int>(i);
      }
      have_header = true;
      continue;
    }
    if (line[first] == '#') {
      continue;
    }
    std::vector<std::string> cols = split(line);
    if (ix < 0 || iy < 0 || il < 0 || ir < 0) {
      throw std::runtime_error(
              "loadTrackCsv: missing required columns x_m,y_m,w_tr_left_m,w_tr_right_m in " + path);
    }
    track.x.push_back(std::stod(cols[ix]));
    track.y.push_back(std::stod(cols[iy]));
    track.w_left.push_back(std::stod(cols[il]));
    track.w_right.push_back(std::stod(cols[ir]));
  }

  if (track.x.size() < 2) {
    throw std::runtime_error("loadTrackCsv: fewer than 2 points in " + path);
  }
  return track;
}

std::array<double, 8> computeTyreSlip(
  const std::array<double, 4> & v_fwd,
  const std::array<double, 4> & v_lat,
  const std::array<double, 4> & spin_rate,
  double wheel_radius,
  double slip_eps)
{
  std::array<double, 8> out{};
  for (int i = 0; i < 4; ++i) {
    double slip_angle = std::atan2(v_lat[i], std::max(std::abs(v_fwd[i]), slip_eps));
    double wheel_speed = wheel_radius * spin_rate[i];
    double denom = std::max(std::max(std::abs(wheel_speed), std::abs(v_fwd[i])), slip_eps);
    double slip_ratio = (wheel_speed - v_fwd[i]) / denom;
    out[i] = slip_ratio;
    out[i + 4] = slip_angle;
  }
  return out;
}

std::pair<double, double> mapActionToDrive(
  double throttle,
  double steering,
  double max_speed,
  double max_steer,
  double clip_actions,
  const std::string & brake_behavior)
{
  throttle = clampd(throttle, -clip_actions, clip_actions);
  steering = clampd(steering, -clip_actions, clip_actions);

  double speed;
  if (brake_behavior == "reverse") {
    speed = throttle * max_speed;
  } else {  // "stop": negative throttle commands a stop, not reverse
    speed = std::max(throttle, 0.0) * max_speed;
  }
  double steering_angle = steering * max_steer;
  return {speed, steering_angle};
}

TrackObservationBuilder::TrackObservationBuilder(
  const std::vector<double> & xs,
  const std::vector<double> & ys,
  const std::vector<double> & w_left,
  const std::vector<double> & w_right,
  const ObsConfig & cfg,
  int coarse_stride,
  int window)
: cfg_(cfg), coarse_stride_(coarse_stride), window_(window)
{
  n_ = static_cast<int>(xs.size());
  if (n_ < 2 || static_cast<int>(ys.size()) != n_) {
    throw std::runtime_error("TrackObservationBuilder: invalid centerline size");
  }
  wl_ = w_left;
  wr_ = w_right;
  nw_ = static_cast<int>(w_left.size());

  // Open-polyline cache (original centerline) used by the future-track points.
  ox_ = xs;
  oy_ = ys;
  oseg_len_.resize(n_ - 1);
  ocumlen_.assign(n_, 0.0);
  for (int i = 0; i < n_ - 1; ++i) {
    double dx = ox_[i + 1] - ox_[i];
    double dy = oy_[i + 1] - oy_[i];
    oseg_len_[i] = std::sqrt(dx * dx + dy * dy);
    ocumlen_[i + 1] = ocumlen_[i] + oseg_len_[i];
  }

  // Closed-loop cache used by Frenet projection: append the first point if the
  // centerline is not already closed (mirrors build_track_cache).
  std::vector<double> clx = xs;
  std::vector<double> cly = ys;
  double dx0 = clx.front() - clx.back();
  double dy0 = cly.front() - cly.back();
  if (std::sqrt(dx0 * dx0 + dy0 * dy0) > 1e-6) {
    clx.push_back(clx.front());
    cly.push_back(cly.front());
  }
  m_ = static_cast<int>(clx.size()) - 1;
  cx_.resize(m_);
  cy_.resize(m_);
  seg_x_.resize(m_);
  seg_y_.resize(m_);
  seg_len_.resize(m_);
  cumlen_.assign(m_, 0.0);
  for (int i = 0; i < m_; ++i) {
    cx_[i] = clx[i];
    cy_[i] = cly[i];
    seg_x_[i] = clx[i + 1] - clx[i];
    seg_y_[i] = cly[i + 1] - cly[i];
    seg_len_[i] = std::max(std::sqrt(seg_x_[i] * seg_x_[i] + seg_y_[i] * seg_y_[i]), 1e-8);
  }
  // cumlen[1:] = cumsum(seg_len[:-1]); cumlen[0] = 0.
  for (int i = 1; i < m_; ++i) {
    cumlen_[i] = cumlen_[i - 1] + seg_len_[i - 1];
  }
  total_len_ = 0.0;
  for (int i = 0; i < m_; ++i) {
    total_len_ += seg_len_[i];
  }
  for (int i = 0; i < m_; i += coarse_stride_) {
    coarse_idx_.push_back(i);
  }
}

FrenetState TrackObservationBuilder::project(double px, double py) const
{
  // Coarse nearest centerline vertex.
  int i0 = 0;
  double best_c = std::numeric_limits<double>::infinity();
  for (int idx : coarse_idx_) {
    double dx = cx_[idx] - px;
    double dy = cy_[idx] - py;
    double d2 = dx * dx + dy * dy;
    if (d2 < best_c) {
      best_c = d2;
      i0 = idx;
    }
  }

  // Fine search within +/- window segments of the coarse pick.
  FrenetState fr;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (int off = -window_; off <= window_; ++off) {
    int idx = floorMod(i0 + off, m_);
    double cxi = cx_[idx], cyi = cy_[idx];
    double sx = seg_x_[idx], sy = seg_y_[idx];
    double seg_len2 = std::max(sx * sx + sy * sy, 1e-10);
    double t = ((px - cxi) * sx + (py - cyi) * sy) / seg_len2;
    t = clampd(t, 0.0, 1.0);
    double projx = cxi + t * sx;
    double projy = cyi + t * sy;
    double dx = projx - px;
    double dy = projy - py;
    double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      fr.best_idx = idx;
      fr.best_t = t;
      fr.proj_x = projx;
      fr.proj_y = projy;
    }
  }

  double bsx = seg_x_[fr.best_idx];
  double bsy = seg_y_[fr.best_idx];
  double norm = std::max(std::sqrt(bsx * bsx + bsy * bsy), 1e-8);
  fr.dir_x = bsx / norm;
  fr.dir_y = bsy / norm;
  fr.s = cumlen_[fr.best_idx] + fr.best_t * seg_len_[fr.best_idx];
  fr.L = total_len_;
  return fr;
}

std::vector<float> TrackObservationBuilder::build(const VehicleState & st) const
{
  std::vector<float> obs(static_cast<size_t>(cfg_.num_obs), 0.0f);
  const FrenetState fr = project(st.pos_x, st.pos_y);

  // [0:5] body-frame kinematics; [5:7] last action.
  obs[0] = static_cast<float>(st.vx * cfg_.lin_vel_scale);
  obs[1] = static_cast<float>(st.vy * cfg_.lin_vel_scale);
  obs[2] = static_cast<float>(st.wz * cfg_.ang_vel_scale);
  obs[3] = static_cast<float>(st.ax * cfg_.lin_acc_scale);
  obs[4] = static_cast<float>(st.ay * cfg_.lin_acc_scale);
  obs[5] = static_cast<float>(st.last_throttle);
  obs[6] = static_cast<float>(st.last_steer);

  // [7:9] track progress (cos, sin) of Frenet s/L.
  double track_len = std::max(fr.L, 1e-6);
  double angle = kTwoPi * (fr.s / track_len);
  obs[7] = static_cast<float>(std::cos(angle));
  obs[8] = static_cast<float>(std::sin(angle));

  // [9] centerline heading error (wrapped).
  double track_angle = std::atan2(fr.dir_y, fr.dir_x);
  double theta_err = st.yaw - track_angle;
  theta_err = std::atan2(std::sin(theta_err), std::cos(theta_err));
  obs[9] = static_cast<float>(theta_err);

  // [10] signed lateral error ey; [11] wall-contact flag.
  double nx = -fr.dir_y;
  double ny = fr.dir_x;
  double ey = (st.pos_x - fr.proj_x) * nx + (st.pos_y - fr.proj_y) * ny;
  obs[10] = static_cast<float>(ey);

  int i0 = fr.best_idx;
  int i1 = (nw_ > 0) ? (fr.best_idx + 1) % nw_ : fr.best_idx;
  double w_l = wl_[i0] + fr.best_t * (wl_[i1] - wl_[i0]);
  double w_r = wr_[i0] + fr.best_t * (wr_[i1] - wr_[i0]);
  double d_left = w_l - ey;
  double d_right = w_r + ey;
  double boundary_dist = std::min(d_left, d_right);
  obs[11] = (boundary_dist < cfg_.contact_margin_m) ? 1.0f : 0.0f;

  // [12:372] future track points: center/left/right x samples x 2D in ego frame.
  const int samples = cfg_.future_track_num_points;
  const double total = std::max(fr.L, 1e-6);
  const double speed = std::sqrt(st.vx * st.vx + st.vy * st.vy);
  const double lookahead = speed * cfg_.future_track_horizon_s;
  const double half_w = 0.5 * cfg_.future_track_width;
  const double cos_y = std::cos(st.yaw);
  const double sin_y = std::sin(st.yaw);

  const int base_center = 12;
  const int base_left = base_center + samples * 2;
  const int base_right = base_left + samples * 2;

  for (int k = 1; k <= samples; ++k) {
    double step = static_cast<double>(k) / samples;
    double s_target = std::fmod(fr.s + lookahead * step, total);
    if (s_target < 0.0) {
      s_target += total;
    }
    // searchsorted(ocumlen, s_target, right=True) - 1, clamped to [0, n-2].
    int seg_idx =
      static_cast<int>(std::upper_bound(ocumlen_.begin(), ocumlen_.end(), s_target) -
      ocumlen_.begin()) - 1;
    seg_idx = std::max(0, std::min(seg_idx, n_ - 2));

    double p0x = ox_[seg_idx], p0y = oy_[seg_idx];
    double p1x = ox_[seg_idx + 1], p1y = oy_[seg_idx + 1];
    double seg_len_sel = std::max(oseg_len_[seg_idx], 1e-8);
    double alpha = (s_target - ocumlen_[seg_idx]) / seg_len_sel;

    double cx = p0x + alpha * (p1x - p0x);
    double cy = p0y + alpha * (p1y - p0y);
    double tx = (p1x - p0x) / seg_len_sel;
    double ty = (p1y - p0y) / seg_len_sel;
    double n2x = -ty;
    double n2y = tx;

    double lx = cx + half_w * n2x;
    double ly = cy + half_w * n2y;
    double rx = cx - half_w * n2x;
    double ry = cy - half_w * n2y;

    auto to_ego = [&](double wx, double wy, double & ox, double & oy) {
      double dx = wx - st.pos_x;
      double dy = wy - st.pos_y;
      ox = cos_y * dx + sin_y * dy;
      oy = -sin_y * dx + cos_y * dy;
    };

    double ex, ey2;
    int slot = (k - 1) * 2;
    to_ego(cx, cy, ex, ey2);
    obs[base_center + slot] = static_cast<float>(ex);
    obs[base_center + slot + 1] = static_cast<float>(ey2);
    to_ego(lx, ly, ex, ey2);
    obs[base_left + slot] = static_cast<float>(ex);
    obs[base_left + slot + 1] = static_cast<float>(ey2);
    to_ego(rx, ry, ex, ey2);
    obs[base_right + slot] = static_cast<float>(ex);
    obs[base_right + slot + 1] = static_cast<float>(ey2);
  }

  // [372:380] tyre slip.
  const int slip_base = 372;
  for (int i = 0; i < 8 && slip_base + i < cfg_.num_obs; ++i) {
    obs[slip_base + i] = static_cast<float>(st.tyre_slip[i]);
  }

  if (cfg_.clip_obs > 0.0) {
    float c = static_cast<float>(cfg_.clip_obs);
    for (auto & v : obs) {
      v = std::max(-c, std::min(c, v));
    }
  }
  return obs;
}

}  // namespace f1tenth_rl_vehicle
