#pragma once

#include "eth3d_score/math.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>

namespace eth3d_score {

struct DistanceStats {
    double mean = 0.0;
    double median = 0.0;
    double rmse = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
};

struct ThresholdScore {
    double threshold = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

struct CloudScore {
    std::size_t reconstruction_points_scored = 0;
    std::size_t ground_truth_points_scored = 0;
    std::size_t worker_threads = 0;
    DistanceStats recon_to_ground_truth;
    DistanceStats ground_truth_to_recon;
    double chamfer_mean_unsquared = 0.0;
    double chamfer_mean_squared = 0.0;
    std::vector<ThresholdScore> threshold_metrics;
};

/// read XYZ from a PLY and ignore stuff this scorer does not use like color or normals
[[nodiscard]] Points load_ply_xyz(const std::filesystem::path& path);
/// save XYZ as a smaller binary PLY
void save_ply_xyz(const std::filesystem::path& path, const Points& points);
/// compare the clouds both ways since close recon points do not prove the whole scene got covered
[[nodiscard]] CloudScore score_clouds(
    const Points& reconstruction, const Points& ground_truth,
    const std::vector<double>& thresholds, std::size_t metric_points, int threads = 0);
[[nodiscard]] nlohmann::json to_json(const CloudScore& score);

}  // namespace eth3d_score
