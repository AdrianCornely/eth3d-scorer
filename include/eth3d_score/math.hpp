#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace eth3d_score {

/// one XYZ point per row
using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

/// how much this transform grows or shrinks everything
[[nodiscard]] double similarity_scale(const Eigen::Matrix4d& transform);
/// pull out the rotation without letting scale or a mirror sneak through
[[nodiscard]] Eigen::Matrix3d similarity_rotation(const Eigen::Matrix4d& transform);
/// move every point with the same transform
[[nodiscard]] Points transform_points(const Points& xyz, const Eigen::Matrix4d& transform);
/// rotate a camera by the rotation part of the same transform
[[nodiscard]] Eigen::Matrix3d transform_rotation(
    const Eigen::Matrix3d& rotation, const Eigen::Matrix4d& sim3);
/// find the one rotation scale and offset that best line up matching points
[[nodiscard]] Eigen::Matrix4d estimate_similarity(const Points& source, const Points& target);
/// pick a short step in front of each camera
/// using 1/100 of normal camera spacing keeps it sensible for any scene size
[[nodiscard]] double pose_epsilon(const Points& centers);
/// how far apart two camera directions are in degrees
[[nodiscard]] double rotation_error_degrees(
    const Eigen::Matrix3d& estimated, const Eigen::Matrix3d& ground_truth);
/// thin a cloud out evenly so the same input always gets the same score
[[nodiscard]] Points deterministic_sample(const Points& xyz, std::optional<std::size_t> maximum);

}  // namespace eth3d_score
