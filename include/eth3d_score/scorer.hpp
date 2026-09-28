#pragma once

#include "eth3d_score/cloud.hpp"
#include "eth3d_score/config.hpp"
#include "eth3d_score/eth3d_trajectory_metrics.hpp"

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace eth3d_score {

struct PoseAlignmentScore {
    std::vector<int> frame_indices;
    Eigen::Matrix4d sim3 = Eigen::Matrix4d::Identity();
    double ate_mean_meters = 0.0;
    double ate_median_meters = 0.0;
    double ate_rmse_meters = 0.0;
    double rotation_mean_degrees = 0.0;
    double rotation_median_degrees = 0.0;
};

struct ScoreSummary {
    std::string manifest;
    std::string pose_file_layout;
    Eth3dTrajectoryScore eth3d_trajectory;
    std::optional<std::string> reconstruction = std::nullopt;
    std::optional<std::string> ground_truth_cloud = std::nullopt;
    std::optional<PoseAlignmentScore> pose_alignment = std::nullopt;
    std::optional<CloudScore> geometry = std::nullopt;
};

[[nodiscard]] nlohmann::json to_json(const ScoreSummary& summary);

/// run the actual scoring job
/// main handles the report file and this handles the optional aligned cloud
[[nodiscard]] ScoreSummary score(const Config& config);

}  // namespace eth3d_score

