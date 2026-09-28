#pragma once

#include "eth3d_score/config.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace eth3d_score {

/// one line from rgb.txt
struct RgbEntry { double timestamp; std::string path; };
/// the file and Eigen order quaternions differently so these cannot just be copied straight over
struct GroundTruth {
    std::vector<double> timestamps;
    std::vector<Eigen::Vector3d> translations;
    std::vector<Eigen::Quaterniond> rotations;
};
/// where the camera is and which way it points
struct Pose { Eigen::Vector3d center; Eigen::Matrix3d rotation; };

[[nodiscard]] std::vector<RgbEntry> load_rgb_entries(const std::filesystem::path& path);
[[nodiscard]] GroundTruth load_ground_truth(const std::filesystem::path& path);
/// fill in a camera pose between two known timestamps
[[nodiscard]] Pose interpolate_ground_truth(double timestamp, const GroundTruth& ground_truth);
/// ETH3D skips a frame if the ground truth gap around it is over 1/75 second
[[nodiscard]] bool has_eth3d_ground_truth(
    double timestamp, const GroundTruth& ground_truth, double maximum_timespan = 1.0 / 75.0);
/// pull the estimated camera path out of either JSON layout
[[nodiscard]] std::map<int, Pose> load_estimated_trajectory(
    const std::filesystem::path& path, PoseFileLayout pose_file_layout);
/// write the camera path in the exact text layout the ETH3D evaluator accepts
void save_eth3d_trajectory(
    const std::filesystem::path& path,
    const std::vector<double>& timestamps,
    const std::vector<Pose>& poses);

}  // namespace eth3d_score
