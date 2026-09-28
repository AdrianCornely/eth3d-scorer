#include "eth3d_score/math.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <Eigen/SVD>

namespace eth3d_score {

double similarity_scale(const Eigen::Matrix4d& transform) {
    return std::cbrt(std::abs(transform.topLeftCorner<3, 3>().determinant()));
}

Eigen::Matrix3d similarity_rotation(const Eigen::Matrix4d& transform) {
    const double scale = similarity_scale(transform);
    if (scale <= 1e-15) {
        throw std::invalid_argument("Degenerate Sim(3) scale.");
    }
    const Eigen::Matrix3d scaled_rotation = transform.topLeftCorner<3, 3>() / scale;
    // removing scale should leave a rotation but floating point math leaves it a little crooked
    // SVD straightens it back out and flips one axis if the result accidentally becomes a mirror
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        scaled_rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d u = svd.matrixU();
    Eigen::Matrix3d rotation = u * svd.matrixV().transpose();
    if (rotation.determinant() < 0.0) {
        u.col(2) *= -1.0;
        rotation = u * svd.matrixV().transpose();
    }
    return rotation;
}

Points transform_points(const Points& xyz, const Eigen::Matrix4d& transform) {
    Points transformed = xyz * transform.topLeftCorner<3, 3>().transpose();
    transformed.rowwise() += transform.topRightCorner<3, 1>().transpose();
    return transformed;
}

Eigen::Matrix3d transform_rotation(
    const Eigen::Matrix3d& rotation, const Eigen::Matrix4d& sim3) {
    return similarity_rotation(sim3) * rotation;
}

Eigen::Matrix4d estimate_similarity(const Points& source, const Points& target) {
    if (source.rows() < 3 || source.rows() != target.rows()) {
        throw std::invalid_argument("Need >=3 paired camera centers.");
    }
    const Eigen::RowVector3d source_mean = source.colwise().mean();
    const Eigen::RowVector3d target_mean = target.colwise().mean();
    const Points source_centered = source.rowwise() - source_mean;
    const Points target_centered = target.rowwise() - target_mean;
    const Eigen::Matrix3d covariance =
        target_centered.transpose() * source_centered / static_cast<double>(source.rows());
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // move both sets around zero first so location cannot affect the rotation
    // then find the rotation that makes the shapes agree without allowing a mirror image
    Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
    if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0) {
        correction(2, 2) = -1.0;
    }
    const Eigen::Matrix3d rotation = svd.matrixU() * correction * svd.matrixV().transpose();
    const double variance = source_centered.rowwise().squaredNorm().mean();
    if (variance <= 1e-15) {
        throw std::invalid_argument("Degenerate estimated trajectory.");
    }
    const double scale = svd.singularValues().dot(correction.diagonal()) / variance;
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.topLeftCorner<3, 3>() = scale * rotation;
    transform.topRightCorner<3, 1>() =
        target_mean.transpose() - scale * rotation * source_mean.transpose();
    return transform;
}

double pose_epsilon(const Points& centers) {
    if (centers.rows() <= 1) {
        return 1e-3;
    }
    std::vector<double> positive;
    positive.reserve(static_cast<std::size_t>(centers.rows() - 1));
    for (Eigen::Index i = 1; i < centers.rows(); ++i) {
        const double distance = (centers.row(i) - centers.row(i - 1)).norm();
        if (distance > 1e-12) {
            positive.push_back(distance);
        }
    }
    if (positive.empty()) {
        return 1e-3;
    }
    const std::vector<double>::iterator middle =
        positive.begin() + static_cast<std::ptrdiff_t>(positive.size() / 2);
    std::nth_element(positive.begin(), middle, positive.end());
    double median = *middle;
    if (positive.size() % 2 == 0) {
        median = 0.5 * (median + *std::max_element(positive.begin(), middle));
    }
    return std::max(median / 100.0, 1e-6);
}

double rotation_error_degrees(
    const Eigen::Matrix3d& estimated, const Eigen::Matrix3d& ground_truth) {
    const Eigen::Matrix3d relative = ground_truth.transpose() * estimated;
    const double cosine = std::clamp((relative.trace() - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(cosine) * 180.0 / std::numbers::pi_v<double>;
}

Points deterministic_sample(const Points& xyz, const std::optional<std::size_t> maximum) {
    if (!maximum || xyz.rows() <= static_cast<Eigen::Index>(*maximum)) {
        return xyz;
    }
    Points sampled(static_cast<Eigen::Index>(*maximum), 3);
    if (*maximum == 1) {
        sampled.row(0) = xyz.row(0);
        return sampled;
    }
    const std::size_t last = static_cast<std::size_t>(xyz.rows() - 1);
    for (std::size_t i = 0; i < *maximum; ++i) {
        // walk evenly from first to last and round down so this picks the same points every time
        const std::size_t index = static_cast<std::size_t>(
            static_cast<double>(i) * static_cast<double>(last) /
            static_cast<double>(*maximum - 1));
        sampled.row(static_cast<Eigen::Index>(i)) = xyz.row(static_cast<Eigen::Index>(index));
    }
    return sampled;
}

}  // namespace eth3d_score
