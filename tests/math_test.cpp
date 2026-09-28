#include "eth3d_score/math.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <numbers>

TEST(MathTest, UmeyamaRecoversSourceToTargetSim3) {
    eth3d_score::Points source(4, 3);
    source << 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3;
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Matrix4d expected = Eigen::Matrix4d::Identity();
    expected.topLeftCorner<3, 3>() = 2.5 * rotation;
    expected.topRightCorner<3, 1>() << 4.0, -2.0, 1.5;
    const eth3d_score::Points target = eth3d_score::transform_points(source, expected);
    const Eigen::Matrix4d actual = eth3d_score::estimate_similarity(source, target);
    EXPECT_NEAR((actual - expected).norm(), 0.0, 1e-12);
}

TEST(MathTest, PoseEpsilonMatchesMedianConsecutiveDistanceOver100) {
    eth3d_score::Points centers(4, 3);
    centers << 0, 0, 0, 2, 0, 0, 6, 0, 0, 12, 0, 0;
    EXPECT_DOUBLE_EQ(eth3d_score::pose_epsilon(centers), 0.04);
}

TEST(MathTest, RotationErrorIsReportedInDegrees) {
    const Eigen::Matrix3d quarter_turn =
        Eigen::AngleAxisd(std::numbers::pi_v<double> / 2.0,
                          Eigen::Vector3d::UnitX()).toRotationMatrix();
    EXPECT_NEAR(eth3d_score::rotation_error_degrees(quarter_turn, Eigen::Matrix3d::Identity()), 90.0, 1e-12);
}

TEST(MathTest, DeterministicSamplingUsesEvenlySpacedIntegerIndices) {
    eth3d_score::Points points(10, 3);
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        points.row(i).setConstant(i);
    }
    const eth3d_score::Points sampled = eth3d_score::deterministic_sample(points, 4);
    ASSERT_EQ(sampled.rows(), 4);
    EXPECT_EQ(sampled(0, 0), 0);
    EXPECT_EQ(sampled(1, 0), 3);
    EXPECT_EQ(sampled(2, 0), 6);
    EXPECT_EQ(sampled(3, 0), 9);
}
