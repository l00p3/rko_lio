#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "preprocess_scan.hpp"

namespace {
using Voxel = Eigen::Vector3i;

inline Voxel PointToVoxel(const Eigen::Vector3d& point, const double voxel_size) {
  return {static_cast<int>(std::floor(point.x() / voxel_size)), static_cast<int>(std::floor(point.y() / voxel_size)),
          static_cast<int>(std::floor(point.z() / voxel_size))};
}
} // namespace

namespace rko_lio::core {

PreprocessingResult preprocess_scan(const Vector3dVector& frame,
                                    const LIO::Config& config,
                                    const std::optional<TimestampVector>& timestamps,
                                    const std::optional<Secondsd>& end_time,
                                    const std::optional<Eigen::Vector3d>& avg_body_accel,
                                    const std::optional<Eigen::Vector3d>& avg_ang_vel,
                                    const std::optional<State>& lidar_state) {
  // TODO: this is repeated here, we should fix it (for now just for testing)
  auto relative_pose_at_time = [&](const Secondsd time) -> Sophus::SE3d {
    const double dt = (time - lidar_state->time).count();
    Eigen::Matrix<double, 6, 1> tau;
    tau.head<3>() = lidar_state->velocity * dt + (*avg_body_accel * square(dt) / 2);
    tau.tail<3>() = *avg_ang_vel * dt;
    return Sophus::SE3d::exp(tau);
  };

  // Initialization
  const Sophus::SE3d scan_to_scan_motion_inverse =
      end_time ? relative_pose_at_time(*end_time).inverse() : Sophus::SE3d{};
  Vector3dVector clipped_frame;
  std::unordered_map<Voxel, Eigen::Vector3d> grid;

  clipped_frame.reserve(frame.size());
  for (size_t i = 0; i < frame.size(); ++i) {
    // Initialization
    Eigen::Vector3d point = frame[i];

    // DESKEW
    if (config.deskew and timestamps) {
      const auto pose = scan_to_scan_motion_inverse * relative_pose_at_time((*timestamps)[i]);
      point = pose * point;
    }

    // CLIPPING
    const double point_range = point.norm();
    if (point_range <= config.min_range or point_range >= config.max_range) {
      continue;
    }
    clipped_frame.emplace_back(point);

    // DOWNSAMPLING
    const auto voxel = PointToVoxel(point, config.voxel_size);
    if (!grid.contains(voxel)) {
      grid.insert({voxel, point});
    }
  }
  /*clipped_frame.shrink_to_fit();*/

  // Extract points from grid (to preserve implicit random shuffling and avod fake regularities)
  std::vector<Eigen::Vector3d> downsampled_frame;
  downsampled_frame.reserve(grid.size());
  std::for_each(grid.cbegin(), grid.cend(),
                [&](const auto& voxel_and_point) { downsampled_frame.emplace_back(voxel_and_point.second); });

  // DOUBLE DOWNSAMPLING
  if (config.double_downsample) {
    // Downsample again
    Vector3dVector keypoints;
    keypoints.reserve(downsampled_frame.size());
    std::unordered_set<Voxel> grid_double_downsampling;
    std::for_each(downsampled_frame.cbegin(), downsampled_frame.cend(), [&](const auto& point) {
      const auto voxel = PointToVoxel(point, config.voxel_size);
      if (!grid_double_downsampling.contains(voxel)) {
        grid_double_downsampling.insert(voxel);
        keypoints.emplace_back(point);
      }
    });
    keypoints.shrink_to_fit();

    return {.filtered_frame = clipped_frame, .map_frame = downsampled_frame, .keypoints = keypoints};
  }

  // TODO: we can consider to drop the return of clipped frame
  return {.filtered_frame = clipped_frame, .map_frame = std::nullopt, .keypoints = downsampled_frame};
}

} // namespace rko_lio::core
