#include "eth3d_score/config.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(ConfigTest, ReadsEverySupportedOption) {
    const std::vector<std::string> args = {
        "--recon", "recon.ply", "--manifest", "manifest.json",
        "--manifest-type", "pairwise", "--mono-root", "mono",
        "--gt", "reference.ply", "--out", "scores.json",
        "--save-aligned", "aligned.ply", "--max-images", "25",
        "--save-trajectory", "trajectory.txt", "--metric-points", "100000",
        "--threads", "4", "--thresholds", "0.01", "0.03"
    };
    const eth3d_score::ParseResult result = eth3d_score::parse_command_line(args);
    ASSERT_TRUE(result.config.has_value());
    EXPECT_EQ(result.config->recon, "recon.ply");
    EXPECT_EQ(result.config->manifest, "manifest.json");
    EXPECT_EQ(result.config->pose_file_layout, eth3d_score::PoseFileLayout::pairwise);
    EXPECT_EQ(result.config->mono_root, "mono");
    EXPECT_EQ(result.config->gt, "reference.ply");
    EXPECT_EQ(result.config->out, "scores.json");
    EXPECT_EQ(result.config->save_aligned, "aligned.ply");
    EXPECT_EQ(result.config->save_trajectory, "trajectory.txt");
    EXPECT_EQ(result.config->max_images, std::optional<std::size_t>{25});
    EXPECT_EQ(result.config->metric_points, 100000);
    EXPECT_EQ(result.config->threads, 4);
    EXPECT_EQ(result.config->thresholds, (std::vector<double>{0.01, 0.03}));
}

TEST(ConfigTest, RejectsBadLayoutsAndUnknownOptions) {
    EXPECT_FALSE(eth3d_score::parse_command_line({"--manifest-type", "other"}).config);
    EXPECT_FALSE(eth3d_score::parse_command_line({"--wat"}).config);
    EXPECT_FALSE(eth3d_score::parse_command_line({"--threads", "-1"}).config);
    EXPECT_FALSE(eth3d_score::parse_command_line({"--max-images", "0"}).config);
    EXPECT_FALSE(eth3d_score::parse_command_line({"--max-images", "-1"}).config);
}

TEST(ConfigTest, UsesTheDocumentedDefaults) {
    const eth3d_score::ParseResult result = eth3d_score::parse_command_line({
        "--recon", "r.ply", "--manifest", "m.json", "--manifest-type", "baseline",
        "--mono-root", "mono", "--gt", "gt.ply", "--out", "out.json"
    });
    ASSERT_TRUE(result.config);
    EXPECT_EQ(result.config->max_images, std::optional<std::size_t>{50});
    EXPECT_EQ(result.config->metric_points, 200000);
    EXPECT_EQ(result.config->threads, 0);
    EXPECT_EQ(result.config->thresholds, (std::vector<double>{0.01, 0.02, 0.05, 0.10}));
}

TEST(ConfigTest, AllImagesRemovesTheDefaultLimit) {
    const eth3d_score::ParseResult result = eth3d_score::parse_command_line({
        "--recon", "r.ply", "--manifest", "m.json", "--manifest-type", "baseline",
        "--mono-root", "mono", "--gt", "gt.ply", "--out", "out.json", "--all-images"
    });
    ASSERT_TRUE(result.config);
    EXPECT_FALSE(result.config->max_images.has_value());
}

TEST(ConfigTest, HelpExitsBeforePositionalValidation) {
    const eth3d_score::ParseResult result = eth3d_score::parse_command_line({"--help"});
    EXPECT_TRUE(result.help_requested);
    EXPECT_FALSE(result.config);
}

TEST(ConfigTest, TrajectoryOnlyDoesNotRequirePointClouds) {
    const eth3d_score::ParseResult result = eth3d_score::parse_command_line({
        "--trajectory-only", "--manifest", "m.json", "--manifest-type", "pairwise",
        "--mono-root", "mono", "--out", "out.json"
    });
    ASSERT_TRUE(result.config);
    EXPECT_TRUE(result.config->trajectory_only);
    EXPECT_TRUE(result.config->recon.empty());
    EXPECT_TRUE(result.config->gt.empty());
}

