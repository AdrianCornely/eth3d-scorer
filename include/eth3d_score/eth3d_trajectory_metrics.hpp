#pragma once

#include "eth3d_score/trajectory.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

namespace eth3d_score {

struct RelativeTrajectoryError {
    std::size_t pair_count = 0;
    std::optional<double> translation_percent;
    std::optional<double> rotation_degrees_per_meter;
};

struct Eth3dTrajectoryScore {
    std::size_t evaluated_frames = 0;
    std::size_t dataset_frames = 0;
    double dataset_coverage = 0.0;
    double first_timestamp = 0.0;
    double last_timestamp = 0.0;
    Eigen::Matrix4d sim3 = Eigen::Matrix4d::Identity();
    double scale = 1.0;
    double ate_rmse_centimeters = 0.0;
    std::array<RelativeTrajectoryError, 4> relative_errors;
};

/// the trajectory numbers ETH3D itself reports
/// this lines up camera positions only and does not touch the cloud alignment
[[nodiscard]] Eth3dTrajectoryScore score_eth3d_trajectory(
    const std::vector<double>& timestamps,
    const std::vector<Pose>& estimated,
    const std::vector<Pose>& ground_truth,
    std::size_t dataset_frame_count);
[[nodiscard]] nlohmann::json to_json(const Eth3dTrajectoryScore& score);

}  // namespace eth3d_score
