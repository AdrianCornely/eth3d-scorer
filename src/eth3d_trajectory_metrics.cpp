#include "eth3d_score/eth3d_trajectory_metrics.hpp"

#include "eth3d_score/math.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace eth3d_score {
namespace {

struct RelativeTotals {
    double translation_percent = 0.0;
    double rotation_degrees_per_meter = 0.0;
    std::size_t count = 0;
};

nlohmann::json sim3_json(const Eigen::Matrix4d& sim3) {
    nlohmann::json result = nlohmann::json::array();
    for (Eigen::Index row = 0; row < 4; ++row) {
        nlohmann::json values = nlohmann::json::array();
        for (Eigen::Index column = 0; column < 4; ++column) {
            values.push_back(sim3(row, column));
        }
        result.push_back(values);
    }
    return result;
}

}  // namespace

Eth3dTrajectoryScore score_eth3d_trajectory(
    const std::vector<double>& timestamps,
    const std::vector<Pose>& estimated,
    const std::vector<Pose>& ground_truth,
    const std::size_t dataset_frame_count) {
    if (estimated.size() < 3 || estimated.size() != ground_truth.size() ||
        estimated.size() != timestamps.size()) {
        throw std::invalid_argument(
            "ETH3D trajectory scoring needs at least three matching poses and timestamps");
    }

    const Eigen::Index count = static_cast<Eigen::Index>(estimated.size());
    Points estimated_centers(count, 3);
    Points ground_truth_centers(count, 3);
    for (Eigen::Index index = 0; index < count; ++index) {
        estimated_centers.row(index) = estimated[static_cast<std::size_t>(index)].center;
        ground_truth_centers.row(index) = ground_truth[static_cast<std::size_t>(index)].center;
    }

    // line up the camera paths with one rotation one scale and one offset
    // ETH3D only cares where the cameras are for this part not where they point
    const Eigen::Matrix4d sim3 = estimate_similarity(estimated_centers, ground_truth_centers);
    const Points aligned_centers = transform_points(estimated_centers, sim3);
    // yes this 3x3 block still has scale baked into it
    // ETH3D does the same weird thing and cleaning it up changes the published numbers
    const Eigen::Quaterniond alignment_rotation(sim3.topLeftCorner<3, 3>());
    std::vector<Eigen::Quaterniond> aligned_rotations;
    aligned_rotations.reserve(estimated.size());
    double squared_error_sum = 0.0;
    for (Eigen::Index index = 0; index < count; ++index) {
        const std::size_t pose_index = static_cast<std::size_t>(index);
        aligned_rotations.push_back(
            alignment_rotation * Eigen::Quaterniond(estimated[pose_index].rotation));
        squared_error_sum +=
            (ground_truth_centers.row(index) - aligned_centers.row(index)).squaredNorm();
    }

    // measure along the real path so 0.5 m still means 0.5 m when the camera changes speed
    std::vector<double> distance_from_start(ground_truth.size(), 0.0);
    for (std::size_t index = 1; index < ground_truth.size(); ++index) {
        distance_from_start[index] = distance_from_start[index - 1] +
            (ground_truth[index].center - ground_truth[index - 1].center).norm();
    }

    constexpr std::array<double, 4> evaluation_distances = {0.5, 1.0, 1.5, 2.0};
    constexpr double allowed_distance_miss = 0.025;
    std::array<RelativeTrajectoryError, 4> relative_errors;
    for (std::size_t distance_index = 0; distance_index < evaluation_distances.size(); ++distance_index) {
        const double evaluation_distance = evaluation_distances[distance_index];
        RelativeTotals totals;
        std::size_t second_pose = 0;
        for (std::size_t first_pose = 0; first_pose < ground_truth.size(); ++first_pose) {
            const double target_distance = distance_from_start[first_pose] + evaluation_distance;
            double best_difference = std::abs(target_distance - distance_from_start[second_pose]);
            bool finished = false;
            while (second_pose + 1 < ground_truth.size()) {
                const double next_difference =
                    std::abs(target_distance - distance_from_start[second_pose + 1]);
                if (next_difference >= best_difference) {
                    break;
                }
                // do not update best_difference here even though it looks wrong
                // this is what ETH3D does and changing it changes their score
                ++second_pose;
                if (second_pose >= ground_truth.size() - 1) {
                    finished = true;
                    break;
                }
            }
            if (finished) {
                break;
            }
            if (first_pose == second_pose) {
                continue;
            }

            const double pose_distance =
                distance_from_start[second_pose] - distance_from_start[first_pose];
            if (std::abs(pose_distance - evaluation_distance) > allowed_distance_miss) {
                continue;
            }

            // describe camera one from camera two's point of view in both paths
            // this removes where the whole path happened to sit in the room
            const Eigen::Quaterniond ground_truth_first_rotation(ground_truth[first_pose].rotation);
            const Eigen::Quaterniond ground_truth_second_rotation(ground_truth[second_pose].rotation);
            const Eigen::Quaterniond ground_truth_relative_rotation =
                ground_truth_second_rotation.inverse() * ground_truth_first_rotation;
            const Eigen::Vector3d ground_truth_relative_translation =
                ground_truth_second_rotation.inverse() *
                (ground_truth[first_pose].center - ground_truth[second_pose].center);
            const Eigen::Quaterniond estimated_relative_rotation =
                aligned_rotations[second_pose].inverse() * aligned_rotations[first_pose];
            const Eigen::Vector3d estimated_relative_translation =
                aligned_rotations[second_pose].inverse() *
                (aligned_centers.row(static_cast<Eigen::Index>(first_pose)).transpose() -
                 aligned_centers.row(static_cast<Eigen::Index>(second_pose)).transpose());
            totals.translation_percent += 100.0 *
                (ground_truth_relative_translation - estimated_relative_translation).norm() / pose_distance;
            const Eigen::Quaterniond rotation_error =
                ground_truth_relative_rotation.inverse() * estimated_relative_rotation;
            double angle_error = std::abs(Eigen::AngleAxisd(rotation_error).angle());
            if (angle_error > std::numbers::pi_v<double>) {
                angle_error = 2.0 * std::numbers::pi_v<double> - angle_error;
            }
            totals.rotation_degrees_per_meter +=
                180.0 * angle_error / (std::numbers::pi_v<double> * pose_distance);
            ++totals.count;
        }

        RelativeTrajectoryError& result = relative_errors[distance_index];
        result.pair_count = totals.count;
        if (totals.count > 0) {
            result.translation_percent =
                totals.translation_percent / static_cast<double>(totals.count);
            result.rotation_degrees_per_meter =
                totals.rotation_degrees_per_meter / static_cast<double>(totals.count);
        }
    }

    double coverage = 0.0;
    if (dataset_frame_count != 0) {
        coverage = static_cast<double>(estimated.size()) /
                   static_cast<double>(dataset_frame_count);
    }
    return {
        .evaluated_frames = estimated.size(),
        .dataset_frames = dataset_frame_count,
        .dataset_coverage = coverage,
        .first_timestamp = timestamps.front(),
        .last_timestamp = timestamps.back(),
        .sim3 = sim3,
        .scale = similarity_scale(sim3),
        .ate_rmse_centimeters =
            100.0 * std::sqrt(squared_error_sum / static_cast<double>(estimated.size())),
        .relative_errors = relative_errors,
    };
}

nlohmann::json to_json(const Eth3dTrajectoryScore& score) {
    constexpr std::array<const char*, 4> distance_names = {"0.5m", "1.0m", "1.5m", "2.0m"};
    nlohmann::json relative = nlohmann::json::object();
    for (std::size_t index = 0; index < distance_names.size(); ++index) {
        const RelativeTrajectoryError& error = score.relative_errors[index];
        nlohmann::json translation = nullptr;
        if (error.translation_percent) {
            translation = *error.translation_percent;
        }
        nlohmann::json rotation = nullptr;
        if (error.rotation_degrees_per_meter) {
            rotation = *error.rotation_degrees_per_meter;
        }
        relative[distance_names[index]] = {
            {"pair_count", error.pair_count},
            {"translation_percent", translation},
            {"rotation_deg_per_m", rotation},
        };
    }
    return {
        {"alignment", "sim3_camera_centers_only"},
        {"evaluated_frames", score.evaluated_frames},
        {"dataset_frames", score.dataset_frames},
        {"dataset_coverage", score.dataset_coverage},
        {"first_timestamp", score.first_timestamp},
        {"last_timestamp", score.last_timestamp},
        {"sim3", sim3_json(score.sim3)},
        {"scale", score.scale},
        {"ate_rmse_cm", score.ate_rmse_centimeters},
        {"relative_errors", relative},
    };
}

}  // namespace eth3d_score
