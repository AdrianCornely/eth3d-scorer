#include "eth3d_score/scorer.hpp"

#include "eth3d_score/cloud.hpp"
#include "eth3d_score/dataset_input.hpp"
#include "eth3d_score/eth3d_trajectory_metrics.hpp"
#include "eth3d_score/math.hpp"
#include "eth3d_score/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace eth3d_score {
namespace {

struct MatchingPoses {
    std::vector<int> frame_indices;
    std::vector<double> timestamps;
    std::vector<Pose> estimated_poses;
    std::vector<Pose> ground_truth_poses;
    Points estimated_centers;
    Points ground_truth_centers;
    std::vector<Eigen::Matrix3d> estimated_rotations;
    std::vector<Eigen::Matrix3d> ground_truth_rotations;
};

nlohmann::json matrix_json(const Eigen::Matrix4d& matrix) {
    nlohmann::json result = nlohmann::json::array();
    for (Eigen::Index row = 0; row < 4; ++row) {
        nlohmann::json values = nlohmann::json::array();
        for (Eigen::Index column = 0; column < 4; ++column) {
            values.push_back(matrix(row, column));
        }
        result.push_back(values);
    }
    return result;
}

double sorted_median(const std::vector<double>& sorted) {
    const std::size_t middle = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        return 0.5 * (sorted[middle - 1] + sorted[middle]);
    }
    return sorted[middle];
}

MatchingPoses match_trajectory(
    const std::vector<RgbEntry>& rgb_entries,
    const GroundTruth& ground_truth,
    const std::map<int, Pose>& estimated) {
    MatchingPoses result;
    for (const std::pair<const int, Pose>& entry : estimated) {
        if (entry.first < 0 ||
            static_cast<std::size_t>(entry.first) >= rgb_entries.size()) {
            continue;
        }
        const double timestamp = rgb_entries[static_cast<std::size_t>(entry.first)].timestamp;
        if (has_eth3d_ground_truth(timestamp, ground_truth)) {
            result.frame_indices.push_back(entry.first);
        }
    }
    if (result.frame_indices.size() < 3) {
        throw std::runtime_error("Need at least three corresponding trajectory frames");
    }

    const Eigen::Index count = static_cast<Eigen::Index>(result.frame_indices.size());
    result.estimated_centers.resize(count, 3);
    result.ground_truth_centers.resize(count, 3);
    result.timestamps.reserve(result.frame_indices.size());
    result.estimated_poses.reserve(result.frame_indices.size());
    result.ground_truth_poses.reserve(result.frame_indices.size());
    result.estimated_rotations.reserve(result.frame_indices.size());
    result.ground_truth_rotations.reserve(result.frame_indices.size());
    for (Eigen::Index index = 0; index < count; ++index) {
        const int frame = result.frame_indices[static_cast<std::size_t>(index)];
        const double timestamp = rgb_entries[static_cast<std::size_t>(frame)].timestamp;
        const Pose& estimated_pose = estimated.at(frame);
        const Pose ground_truth_pose = interpolate_ground_truth(timestamp, ground_truth);
        result.timestamps.push_back(timestamp);
        result.estimated_poses.push_back(estimated_pose);
        result.ground_truth_poses.push_back(ground_truth_pose);
        result.estimated_centers.row(index) = estimated_pose.center;
        result.ground_truth_centers.row(index) = ground_truth_pose.center;
        result.estimated_rotations.push_back(estimated_pose.rotation);
        result.ground_truth_rotations.push_back(ground_truth_pose.rotation);
    }
    return result;
}

PoseAlignmentScore compute_pose_alignment(const MatchingPoses& trajectory) {
    const Eigen::Index count = trajectory.estimated_centers.rows();
    const double estimated_epsilon = pose_epsilon(trajectory.estimated_centers);
    const double ground_truth_epsilon = pose_epsilon(trajectory.ground_truth_centers);
    Points estimated_pose_points(count * 2, 3);
    Points ground_truth_pose_points(count * 2, 3);
    for (Eigen::Index index = 0; index < count; ++index) {
        estimated_pose_points.row(index * 2) = trajectory.estimated_centers.row(index);
        estimated_pose_points.row(index * 2 + 1) = trajectory.estimated_centers.row(index) +
            estimated_epsilon *
                trajectory.estimated_rotations[static_cast<std::size_t>(index)].col(2).transpose();
        ground_truth_pose_points.row(index * 2) = trajectory.ground_truth_centers.row(index);
        ground_truth_pose_points.row(index * 2 + 1) =
            trajectory.ground_truth_centers.row(index) +
            ground_truth_epsilon *
                trajectory.ground_truth_rotations[static_cast<std::size_t>(index)].col(2).transpose();
    }

    PoseAlignmentScore result;
    result.frame_indices = trajectory.frame_indices;
    result.sim3 = estimate_similarity(estimated_pose_points, ground_truth_pose_points);
    const Points aligned_centers = transform_points(trajectory.estimated_centers, result.sim3);
    std::vector<double> center_errors;
    std::vector<double> rotation_errors;
    center_errors.reserve(trajectory.frame_indices.size());
    rotation_errors.reserve(trajectory.frame_indices.size());
    double center_sum = 0.0;
    double center_squared_sum = 0.0;
    double rotation_sum = 0.0;
    for (Eigen::Index index = 0; index < count; ++index) {
        const double center_error =
            (aligned_centers.row(index) - trajectory.ground_truth_centers.row(index)).norm();
        const double rotation_error = rotation_error_degrees(
            transform_rotation(
                trajectory.estimated_rotations[static_cast<std::size_t>(index)], result.sim3),
            trajectory.ground_truth_rotations[static_cast<std::size_t>(index)]);
        center_errors.push_back(center_error);
        rotation_errors.push_back(rotation_error);
        center_sum += center_error;
        center_squared_sum += center_error * center_error;
        rotation_sum += rotation_error;
    }
    std::sort(center_errors.begin(), center_errors.end());
    std::sort(rotation_errors.begin(), rotation_errors.end());
    const double sample_count = static_cast<double>(center_errors.size());
    result.ate_mean_meters = center_sum / sample_count;
    result.ate_median_meters = sorted_median(center_errors);
    result.ate_rmse_meters = std::sqrt(center_squared_sum / sample_count);
    result.rotation_mean_degrees = rotation_sum / sample_count;
    result.rotation_median_degrees = sorted_median(rotation_errors);
    return result;
}

nlohmann::json pose_alignment_json(const PoseAlignmentScore& alignment) {
    return {
        {"frame_indices", alignment.frame_indices},
        {"sim3", matrix_json(alignment.sim3)},
        {"scale", similarity_scale(alignment.sim3)},
        {"ate_mean_m", alignment.ate_mean_meters},
        {"ate_median_m", alignment.ate_median_meters},
        {"ate_rmse_m", alignment.ate_rmse_meters},
        {"rotation_error_mean_deg", alignment.rotation_mean_degrees},
        {"rotation_error_median_deg", alignment.rotation_median_degrees},
    };
}

}  // namespace

nlohmann::json to_json(const ScoreSummary& summary) {
    nlohmann::json result = {
        {"manifest", summary.manifest},
        {"manifest_type", summary.pose_file_layout},
        {"eth3d_trajectory_metrics", to_json(summary.eth3d_trajectory)},
    };
    if (summary.reconstruction) {
        result["reconstruction"] = *summary.reconstruction;
    }
    if (summary.ground_truth_cloud) {
        result["ground_truth_cloud"] = *summary.ground_truth_cloud;
    }
    if (summary.pose_alignment) {
        result["pose_alignment"] = pose_alignment_json(*summary.pose_alignment);
    }
    if (summary.geometry) {
        result["geometry_metrics"] = to_json(*summary.geometry);
    }
    return result;
}

ScoreSummary score(const Config& config) {
    const Eth3dInput eth3d(config.mono_root, config.gt);
    const DatasetInput& dataset = eth3d;
    // TODO: choose the dataset implementation here once ETH3D is not the only option
    // everything below this should stay completely unaware of which dataset was picked

    // frame 12 in the JSON means row 12 in rgb.txt
    // that row gives us the timestamp needed to find the matching real camera pose
    std::vector<RgbEntry> rgb_entries = dataset.frames();
    const std::size_t dataset_frame_count = rgb_entries.size();
    if (config.max_images && rgb_entries.size() > *config.max_images) {
        rgb_entries.resize(*config.max_images);
    }
    const GroundTruth ground_truth = dataset.camera_ground_truth();
    const std::map<int, Pose> estimated =
        load_estimated_trajectory(config.manifest, config.pose_file_layout);
    const MatchingPoses trajectory = match_trajectory(rgb_entries, ground_truth, estimated);
    const Eth3dTrajectoryScore eth3d_trajectory_score = score_eth3d_trajectory(
        trajectory.timestamps, trajectory.estimated_poses,
        trajectory.ground_truth_poses, dataset_frame_count);
    if (config.save_trajectory) {
        save_eth3d_trajectory(
            *config.save_trajectory, trajectory.timestamps, trajectory.estimated_poses);
    }
    const char* layout_name = "pairwise";
    if (config.pose_file_layout == PoseFileLayout::baseline) {
        layout_name = "baseline";
    }
    if (config.trajectory_only) {
        return {
            .manifest = config.manifest.string(),
            .pose_file_layout = layout_name,
            .eth3d_trajectory = eth3d_trajectory_score,
        };
    }
    const PoseAlignmentScore pose_alignment = compute_pose_alignment(trajectory);

    // line up the full cloud first then thin it out since the transform does not change point order
    const Points reconstruction = load_ply_xyz(config.recon);
    const Points gt_cloud = dataset.reference_cloud();
    const Points aligned_reconstruction = transform_points(reconstruction, pose_alignment.sim3);
    if (config.save_aligned) {
        save_ply_xyz(*config.save_aligned, aligned_reconstruction);
    }
    if (config.metric_points <= 0) {
        throw std::runtime_error("metric-points must be positive");
    }
    const CloudScore geometry = score_clouds(
        aligned_reconstruction, gt_cloud, config.thresholds,
        static_cast<std::size_t>(config.metric_points), config.threads);

    return {
        .manifest = config.manifest.string(),
        .pose_file_layout = layout_name,
        .eth3d_trajectory = eth3d_trajectory_score,
        .reconstruction = config.recon.string(),
        .ground_truth_cloud = config.gt.string(),
        .pose_alignment = pose_alignment,
        .geometry = geometry,
    };
}

}  // namespace eth3d_score

