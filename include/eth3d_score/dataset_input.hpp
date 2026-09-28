#pragma once

#include "eth3d_score/math.hpp"
#include "eth3d_score/trajectory.hpp"

#include <filesystem>
#include <string_view>
#include <vector>

namespace eth3d_score {

/// the small set of things the scorer needs from any dataset
/// another dataset should plug in here instead of leaking into the scoring math
class DatasetInput {
public:
    virtual ~DatasetInput() = default;

    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual std::vector<RgbEntry> frames() const = 0;
    [[nodiscard]] virtual GroundTruth camera_ground_truth() const = 0;
    [[nodiscard]] virtual Points reference_cloud() const = 0;
};

/// reads the particular files ETH3D gives us
class Eth3dInput final : public DatasetInput {
public:
    Eth3dInput(std::filesystem::path mono_root, std::filesystem::path ground_truth_cloud);

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<RgbEntry> frames() const override;
    [[nodiscard]] GroundTruth camera_ground_truth() const override;
    [[nodiscard]] Points reference_cloud() const override;

private:
    std::filesystem::path mono_root_;
    std::filesystem::path ground_truth_cloud_;
};

}  // namespace eth3d_score
