#include "eth3d_score/scorer.hpp"

#include <gtest/gtest.h>

TEST(ScorerTest, ScoreSummaryHasStableJsonFieldNames) {
    eth3d_score::ScoreSummary summary{
        .manifest = "manifest.json",
        .pose_file_layout = "pairwise",
    };
    const nlohmann::json json = eth3d_score::to_json(summary);
    EXPECT_EQ(json.at("manifest"), "manifest.json");
    EXPECT_EQ(json.at("manifest_type"), "pairwise");
    EXPECT_TRUE(json.contains("eth3d_trajectory_metrics"));
    EXPECT_FALSE(json.contains("geometry_metrics"));
}

