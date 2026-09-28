#include "eth3d_score/trajectory.hpp"

#include "eth3d_score/math.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace eth3d_score {
namespace {

Eigen::Matrix4d json_transform(const nlohmann::json& value) {
    if (!value.is_array() || value.size() != 4) {
        throw std::invalid_argument("Expected 4x4 transform");
    }
    Eigen::Matrix4d result;
    for (std::size_t row = 0; row < 4; ++row) {
        if (!value[row].is_array() || value[row].size() != 4) {
            throw std::invalid_argument("Expected 4x4 transform");
        }
        for (std::size_t column = 0; column < 4; ++column) {
            result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
                value[row][column].get<double>();
        }
    }
    if (!result.allFinite()) {
        throw std::invalid_argument("Transform contains a nonfinite value");
    }
    return result;
}

}  // namespace

std::vector<RgbEntry> load_rgb_entries(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    std::vector<RgbEntry> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream stream(line);
        RgbEntry entry;
        std::string extra;
        if (!(stream >> entry.timestamp >> entry.path) || stream >> extra) {
            throw std::runtime_error(
                "Malformed RGB entry in " + path.string() + " at line " +
                std::to_string(line_number));
        }
        if (!rows.empty() && entry.timestamp <= rows.back().timestamp) {
            throw std::runtime_error(
                "RGB timestamps must increase in " + path.string() + " at line " +
                std::to_string(line_number));
        }
        rows.push_back(entry);
    }
    if (rows.empty()) {
        throw std::runtime_error("No RGB entries found in " + path.string());
    }
    return rows;
}

GroundTruth load_ground_truth(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    GroundTruth result;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream stream(line);
        double timestamp = 0.0;
        double tx = 0.0; double ty = 0.0; double tz = 0.0;
        double qx = 0.0; double qy = 0.0; double qz = 0.0; double qw = 0.0;
        std::string extra;
        if (!(stream >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) ||
            stream >> extra) {
            throw std::runtime_error(
                "Malformed ground-truth pose in " + path.string() + " at line " +
                std::to_string(line_number));
        }
        if (!result.timestamps.empty() && timestamp <= result.timestamps.back()) {
            throw std::runtime_error(
                "Ground-truth timestamps must increase in " + path.string() + " at line " +
                std::to_string(line_number));
        }
        const Eigen::Quaterniond rotation(qw, qx, qy, qz);
        if (!Eigen::Vector3d(tx, ty, tz).allFinite() || !rotation.coeffs().allFinite() ||
            rotation.norm() <= 1e-15) {
            throw std::runtime_error(
                "Nonfinite or invalid ground-truth pose in " + path.string() + " at line " +
                std::to_string(line_number));
        }
        result.timestamps.push_back(timestamp);
        result.translations.emplace_back(tx, ty, tz);
        result.rotations.push_back(rotation.normalized());
    }
    if (result.timestamps.size() < 2) {
        throw std::runtime_error("Insufficient ground truth in " + path.string());
    }
    return result;
}

Pose interpolate_ground_truth(const double timestamp, const GroundTruth& ground_truth) {
    if (timestamp < ground_truth.timestamps.front() || timestamp > ground_truth.timestamps.back()) {
        throw std::out_of_range("Timestamp outside GT range");
    }
    const std::vector<double>::const_iterator position = std::lower_bound(
        ground_truth.timestamps.begin(), ground_truth.timestamps.end(), timestamp);
    const std::size_t right = static_cast<std::size_t>(position - ground_truth.timestamps.begin());
    if (right == 0 || right == ground_truth.timestamps.size() ||
        std::abs(ground_truth.timestamps[right] - timestamp) < 1e-12) {
        std::size_t index = right;
        if (right == ground_truth.timestamps.size()) {
            index = right - 1;
        }
        return {ground_truth.translations[index], ground_truth.rotations[index].toRotationMatrix()};
    }
    const std::size_t left = right - 1;
    const double alpha = (timestamp - ground_truth.timestamps[left]) /
                         (ground_truth.timestamps[right] - ground_truth.timestamps[left]);
    const Eigen::Vector3d translation =
        (1.0 - alpha) * ground_truth.translations[left] + alpha * ground_truth.translations[right];
    // rotations do not average like normal numbers so take the shortest smooth turn between them
    const Eigen::Quaterniond rotation =
        ground_truth.rotations[left].slerp(alpha, ground_truth.rotations[right]);
    return {translation, rotation.toRotationMatrix()};
}

bool has_eth3d_ground_truth(
    const double timestamp,
    const GroundTruth& ground_truth,
    const double maximum_timespan) {
    const std::vector<double>::const_iterator next = std::upper_bound(
        ground_truth.timestamps.begin(), ground_truth.timestamps.end(), timestamp);
    if (next == ground_truth.timestamps.begin() || next == ground_truth.timestamps.end()) {
        return false;
    }
    const std::vector<double>::const_iterator previous = next - 1;
    return *next - *previous <= maximum_timespan;
}

std::map<int, Pose> load_estimated_trajectory(
    const std::filesystem::path& path, const PoseFileLayout pose_file_layout) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    nlohmann::json manifest;
    input >> manifest;
    std::map<int, Pose> trajectory;
    if (pose_file_layout == PoseFileLayout::baseline) {
        if (!manifest.contains("camera_poses")) {
            throw std::runtime_error("Baseline manifest has no camera_poses.");
        }
        const nlohmann::json& poses = manifest.at("camera_poses");
        for (std::size_t frame = 0; frame < poses.size(); ++frame) {
            const Eigen::Matrix4d transform = json_transform(poses[frame]);
            trajectory.emplace(static_cast<int>(frame), Pose{
                transform.topRightCorner<3, 1>(), similarity_rotation(transform)});
        }
        return trajectory;
    }
    if (!manifest.contains("accepted_pairs") || manifest.at("accepted_pairs").empty()) {
        throw std::runtime_error("Pairwise manifest has no accepted_pairs.");
    }
    for (const nlohmann::json& record : manifest.at("accepted_pairs")) {
        // TODO: if both camera poses get saved later decide whether both should count
        // keeping only the left one would preserve the scores from the current files
        const int pair = record.at("pair").get<int>();
        int left = pair;
        if (record.contains("frames")) {
            left = record.at("frames")[0].get<int>();
        }
        const nlohmann::json* value = nullptr;
        if (record.contains("transform") && !record.at("transform").is_null()) {
            value = &record.at("transform");
        } else if (record.contains("provisional_transform") &&
                   !record.at("provisional_transform").is_null()) {
            value = &record.at("provisional_transform");
        }
        if (value == nullptr) {
            continue;
        }
        const Eigen::Matrix4d transform = json_transform(*value);
        // this is the left camera transform so its translation is that camera's center
        // remove scale before treating the 3x3 part like an actual rotation
        trajectory[left] = {transform.topRightCorner<3, 1>(), similarity_rotation(transform)};
    }
    if (trajectory.size() < 3) {
        throw std::runtime_error("Fewer than three pairwise camera centers recovered.");
    }
    return trajectory;
}

void save_eth3d_trajectory(
    const std::filesystem::path& path,
    const std::vector<double>& timestamps,
    const std::vector<Pose>& poses) {
    if (timestamps.size() != poses.size()) {
        throw std::invalid_argument("Trajectory timestamps and poses must have the same size");
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    output << std::setprecision(17);
    for (std::size_t index = 0; index < poses.size(); ++index) {
        const Pose& pose = poses[index];
        const Eigen::Quaterniond rotation(pose.rotation);
        output << timestamps[index] << ' '
               << pose.center.x() << ' ' << pose.center.y() << ' ' << pose.center.z() << ' '
               << rotation.x() << ' ' << rotation.y() << ' ' << rotation.z() << ' '
               << rotation.w() << '\n';
    }
    if (!output) {
        throw std::runtime_error("Failed while writing trajectory: " + path.string());
    }
}

}  // namespace eth3d_score
