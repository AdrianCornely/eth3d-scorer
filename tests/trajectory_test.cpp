#include "eth3d_score/trajectory.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

TEST(TrajectoryTest, Eth3dGroundTruthCutoffIncludesTheBoundary) {
    eth3d_score::GroundTruth ground_truth;
    ground_truth.timestamps = {0.0, 1.0 / 75.0, 2.0 / 75.0 + 1e-9};
    ground_truth.translations = {
        Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(), 2.0 * Eigen::Vector3d::UnitX()};
    ground_truth.rotations = {
        Eigen::Quaterniond::Identity(), Eigen::Quaterniond::Identity(),
        Eigen::Quaterniond::Identity()};

    EXPECT_TRUE(eth3d_score::has_eth3d_ground_truth(0.5 / 75.0, ground_truth));
    EXPECT_FALSE(eth3d_score::has_eth3d_ground_truth(1.5 / 75.0, ground_truth));
    EXPECT_FALSE(eth3d_score::has_eth3d_ground_truth(-0.1, ground_truth));
    EXPECT_FALSE(eth3d_score::has_eth3d_ground_truth(1.0, ground_truth));
}

TEST(TrajectoryTest, BaselineLayoutRemovesUniformScaleFromRotation) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "eth3d_scaled_baseline_manifest.json";
    {
        std::ofstream output(path);
        output << R"({"camera_poses":[
            [[2,0,0,1],[0,2,0,2],[0,0,2,3],[0,0,0,1]],
            [[2,0,0,2],[0,2,0,2],[0,0,2,3],[0,0,0,1]],
            [[2,0,0,3],[0,2,0,2],[0,0,2,3],[0,0,0,1]]
        ]})";
    }
    const std::map<int, eth3d_score::Pose> poses =
        eth3d_score::load_estimated_trajectory(path, eth3d_score::PoseFileLayout::baseline);
    std::filesystem::remove(path);
    ASSERT_EQ(poses.size(), 3);
    EXPECT_NEAR((poses.at(0).rotation - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12);
    EXPECT_DOUBLE_EQ(poses.at(0).center.x(), 1.0);
}

TEST(TrajectoryTest, MalformedDataRowsFailInsteadOfDisappearing) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "eth3d_bad_rgb.txt";
    {
        std::ofstream output(path);
        output << "1.0 rgb/1.png\nthis is not an rgb row\n";
    }
    EXPECT_THROW(
        {
            const std::vector<eth3d_score::RgbEntry> entries =
                eth3d_score::load_rgb_entries(path);
            (void)entries;
        },
        std::runtime_error);
    std::filesystem::remove(path);
}
