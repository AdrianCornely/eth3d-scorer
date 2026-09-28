#include "eth3d_score/cloud.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

std::vector<double> brute_force_distances(
    const eth3d_score::Points& queries, const eth3d_score::Points& references) {
    std::vector<double> distances(static_cast<std::size_t>(queries.rows()));
    for (Eigen::Index query = 0; query < queries.rows(); ++query) {
        double best_squared = std::numeric_limits<double>::infinity();
        for (Eigen::Index reference = 0; reference < references.rows(); ++reference) {
            best_squared = std::min(
                best_squared, (queries.row(query) - references.row(reference)).squaredNorm());
        }
        distances[static_cast<std::size_t>(query)] = std::sqrt(best_squared);
    }
    return distances;
}

double brute_force_mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

}  // namespace

TEST(CloudTest, BinaryLittleEndianPlyLoadsReorderedScalarProperties) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "eth3d_scorer_binary_test.ply";
    {
        std::ofstream output(path, std::ios::binary);
        output << "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
               << "property uchar red\nproperty float z\nproperty float x\nproperty float y\nend_header\n";
        const std::uint8_t red = 7;
        const float values[6] = {3.0F, 1.0F, 2.0F, 6.0F, 4.0F, 5.0F};
        for (std::size_t row = 0; row < 2; ++row) {
            output.write(reinterpret_cast<const char*>(&red), sizeof(red));
            output.write(reinterpret_cast<const char*>(&values[row * 3]), sizeof(float) * 3);
        }
    }
    const eth3d_score::Points points = eth3d_score::load_ply_xyz(path);
    std::filesystem::remove(path);
    ASSERT_EQ(points.rows(), 2);
    EXPECT_DOUBLE_EQ(points(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(points(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(points(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(points(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(points(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(points(1, 2), 6.0);
}

TEST(CloudTest, MalformedAsciiVertexFailsInsteadOfDisappearing) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "eth3d_scorer_bad_ascii_test.ply";
    {
        std::ofstream output(path);
        output << "ply\nformat ascii 1.0\nelement vertex 2\n"
               << "property float x\nproperty float y\nproperty float z\nend_header\n"
               << "1 2 3\n4 broken 6\n";
    }
    EXPECT_THROW(
        {
            const eth3d_score::Points points = eth3d_score::load_ply_xyz(path);
            (void)points;
        },
        std::runtime_error);
    std::filesystem::remove(path);
}

TEST(CloudTest, KdTreeMetricsMatchExactNearestDistances) {
    eth3d_score::Points reconstruction(2, 3);
    reconstruction << 0, 0, 0, 2, 0, 0;
    eth3d_score::Points ground_truth(2, 3);
    ground_truth << 0, 0, 0, 3, 0, 0;
    const nlohmann::json metrics =
        eth3d_score::to_json(eth3d_score::score_clouds(
            reconstruction, ground_truth, {0.5, 1.0}, 10));
    EXPECT_DOUBLE_EQ(metrics.at("accuracy_recon_to_gt").at("mean").get<double>(), 0.5);
    EXPECT_DOUBLE_EQ(metrics.at("completeness_gt_to_recon").at("mean").get<double>(), 0.5);
    EXPECT_DOUBLE_EQ(metrics.at("threshold_metrics").at("0.5").at("f1").get<double>(), 0.5);
    EXPECT_DOUBLE_EQ(metrics.at("threshold_metrics").at("1").at("f1").get<double>(), 1.0);
}

TEST(CloudTest, ParallelQueriesMatchSerialQueriesExactly) {
    eth3d_score::Points reconstruction(257, 3);
    eth3d_score::Points ground_truth(193, 3);
    for (Eigen::Index index = 0; index < reconstruction.rows(); ++index) {
        reconstruction.row(index) << 0.01 * static_cast<double>(index),
            std::sin(static_cast<double>(index)), std::cos(0.5 * static_cast<double>(index));
    }
    for (Eigen::Index index = 0; index < ground_truth.rows(); ++index) {
        ground_truth.row(index) << 0.012 * static_cast<double>(index),
            std::sin(0.7 * static_cast<double>(index)), std::cos(static_cast<double>(index));
    }
    const nlohmann::json serial = eth3d_score::to_json(
        eth3d_score::score_clouds(reconstruction, ground_truth, {0.01, 0.05}, 300, 1));
    const nlohmann::json parallel = eth3d_score::to_json(
        eth3d_score::score_clouds(reconstruction, ground_truth, {0.01, 0.05}, 300, 4));
    nlohmann::json parallel_without_thread_count = parallel;
    parallel_without_thread_count["worker_threads"] = 1;
    EXPECT_EQ(serial, parallel_without_thread_count);
}

TEST(CloudTest, KdTreeMatchesBruteForceAcrossVariedClouds) {
    for (int case_index = 0; case_index < 12; ++case_index) {
        const Eigen::Index reference_count = 1 + case_index * 7;
        const Eigen::Index query_count = 3 + case_index * 5;
        eth3d_score::Points references(reference_count, 3);
        eth3d_score::Points queries(query_count, 3);
        for (Eigen::Index index = 0; index < reference_count; ++index) {
            const double value = static_cast<double>(index + case_index);
            double z = std::sin(0.11 * value);
            if (case_index % 3 == 0) {
                z = 0.0;
            }
            references.row(index) << std::sin(value), std::cos(0.37 * value),
                z;
        }
        if (reference_count > 2) {
            references.row(1) = references.row(0);
        }
        for (Eigen::Index index = 0; index < query_count; ++index) {
            const double value = static_cast<double>(index - case_index);
            queries.row(index) << 1.7 * std::sin(0.23 * value),
                -1.4 * std::cos(0.41 * value), 0.05 * value;
        }

        const std::vector<double> expected = brute_force_distances(queries, references);
        const nlohmann::json metrics = eth3d_score::to_json(
            eth3d_score::score_clouds(queries, references, {0.1}, 1000, 3));
        EXPECT_NEAR(metrics.at("accuracy_recon_to_gt").at("mean").get<double>(),
                    brute_force_mean(expected), 1e-15);
        double squared_sum = 0.0;
        for (const double value : expected) {
            squared_sum += value * value;
        }
        EXPECT_NEAR(metrics.at("accuracy_recon_to_gt").at("rmse").get<double>(),
                    std::sqrt(squared_sum / static_cast<double>(expected.size())), 1e-15);
    }
}
