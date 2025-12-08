#include "preprocess_scan.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <optional>

namespace {

using Voxel = Eigen::Vector3i;

inline Voxel PointToVoxel(const Eigen::Vector3d& point, const double voxel_size) {
  return {static_cast<int>(std::floor(point.x() / voxel_size)), static_cast<int>(std::floor(point.y() / voxel_size)),
          static_cast<int>(std::floor(point.z() / voxel_size))};
}

// if you need even better runtime-performance, consider using Luca Lobefaro's version of one cycle downsampling here:
// https://github.com/PRBonn/kiss-icp/pull/347
// although it does lead to worse odometry performance in certain situations
std::vector<Eigen::Vector3d> voxel_down_sample(const std::vector<Eigen::Vector3d>& frame, const double voxel_size) {
  std::unordered_map<Voxel, Eigen::Vector3d> grid;
  grid.reserve(frame.size());
  std::for_each(frame.cbegin(), frame.cend(), [&](const auto& point) {
    const auto voxel = PointToVoxel(point, voxel_size);
    if (!grid.contains(voxel)) {
      grid.insert({voxel, point});
    }
  });
  std::vector<Eigen::Vector3d> frame_dowsampled;
  frame_dowsampled.reserve(grid.size());
  std::for_each(grid.cbegin(), grid.cend(),
                [&](const auto& voxel_and_point) { frame_dowsampled.emplace_back(voxel_and_point.second); });
  return frame_dowsampled;
}

inline std::vector<Eigen::Vector3d>
clip_frame(const std::vector<Eigen::Vector3d>& frame, const double min_range, const double max_range) {
  std::vector<Eigen::Vector3d> clipped_frame;
  clipped_frame.reserve(frame.size());
  std::for_each(frame.cbegin(), frame.cend(), [&](const auto& point) {
    const double point_range = point.norm();
    if (point_range > min_range && point_range < max_range) {
      clipped_frame.emplace_back(point);
    }
  });
  clipped_frame.shrink_to_fit();
  return clipped_frame;
}

} // namespace

namespace rko_lio::core {

PreprocessingResult preprocess_scan(const Vector3dVector& frame, const LIO::Config& config) {
  Vector3dVector downsampled_frame = voxel_down_sample(frame, config.voxel_size * 0.5);
  downsampled_frame = clip_frame(downsampled_frame, config.min_range, config.max_range);
  if (config.double_downsample) {
    const Vector3dVector keypoints = voxel_down_sample(downsampled_frame, config.voxel_size * 1.5);
    return {.filtered_frame = frame, .map_frame = downsampled_frame, .keypoints = keypoints};
  }
  return {.filtered_frame = frame, .map_frame = std::nullopt, .keypoints = downsampled_frame};
}

} // namespace rko_lio::core
