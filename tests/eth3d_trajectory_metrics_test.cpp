#include "eth3d_score/eth3d_trajectory_metrics.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

TEST(Eth3dTrajectoryMetricsTest, RecoversAPerfectTrajectoryAfterSim3Alignment) {
    std::vector<double> timestamps;
    std::vector<eth3d_score::Pose> estimated;
    std::vector<eth3d_score::Pose> ground_truth;
    const Eigen::Matrix3d offset_rotation =
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    for (int index = 0; index < 8; ++index) {
        const double distance = 0.5 * static_cast<double>(index);
        timestamps.push_back(static_cast<double>(index));
        estimated.push_back({Eigen::Vector3d(distance, 0.0, 0.0), Eigen::Matrix3d::Identity()});
        ground_truth.push_back({
            offset_rotation * Eigen::Vector3d(distance, 0.0, 0.0) + Eigen::Vector3d(1.0, 2.0, 3.0),
            offset_rotation,
        });
    }

    const eth3d_score::Eth3dTrajectoryScore result =
        eth3d_score::score_eth3d_trajectory(timestamps, estimated, ground_truth, 10);
    EXPECT_NEAR(result.ate_rmse_centimeters, 0.0, 1e-10);
    EXPECT_DOUBLE_EQ(result.dataset_coverage, 0.8);
    const eth3d_score::RelativeTrajectoryError& half_meter = result.relative_errors[0];
    EXPECT_GT(half_meter.pair_count, 0);
    ASSERT_TRUE(half_meter.translation_percent);
    ASSERT_TRUE(half_meter.rotation_degrees_per_meter);
    EXPECT_NEAR(*half_meter.translation_percent, 0.0, 1e-10);
    EXPECT_NEAR(*half_meter.rotation_degrees_per_meter, 0.0, 1e-5);

    const nlohmann::json json = eth3d_score::to_json(result);
    EXPECT_EQ(json.at("alignment"), "sim3_camera_centers_only");
    EXPECT_EQ(json.at("relative_errors").at("0.5m").at("pair_count"),
              half_meter.pair_count);
}
