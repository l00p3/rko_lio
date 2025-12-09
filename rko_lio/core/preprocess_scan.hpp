#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <optional>
#include <random>
#include <sophus/se3.hpp>
#include <vector>

#include "lio.hpp"

// For voxel downsampling
template <>
struct std::hash<Eigen::Vector3i> {
  std::size_t operator()(const Eigen::Vector3i& voxel) const {
    const uint32_t* vec = reinterpret_cast<const uint32_t*>(voxel.data());
    return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
  }
};

namespace rko_lio::core {

struct PreprocessingResult {
  Vector3dVector filtered_frame;
  std::optional<Vector3dVector> map_frame;
  Vector3dVector keypoints;

  const Vector3dVector& map_update_frame() const { return map_frame ? *map_frame : keypoints; }
};

PreprocessingResult preprocess_scan(const Vector3dVector& frame,
                                    const LIO::Config& config,
                                    const std::optional<TimestampVector>& timestamps = std::nullopt,
                                    const std::optional<Secondsd>& end_time = std::nullopt,
                                    const std::optional<Eigen::Vector3d>& avg_body_accel = std::nullopt,
                                    const std::optional<Eigen::Vector3d>& avg_ang_vel = std::nullopt,
                                    const std::optional<State>& lidar_state = std::nullopt);
} // namespace rko_lio::core
