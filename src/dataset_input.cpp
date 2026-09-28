#include "eth3d_score/dataset_input.hpp"

#include "eth3d_score/cloud.hpp"

#include <utility>

namespace eth3d_score {

Eth3dInput::Eth3dInput(
    std::filesystem::path mono_root, std::filesystem::path ground_truth_cloud)
    : mono_root_(std::move(mono_root)),
      ground_truth_cloud_(std::move(ground_truth_cloud)) {}

std::string_view Eth3dInput::name() const {
    return "eth3d";
}

std::vector<RgbEntry> Eth3dInput::frames() const {
    return load_rgb_entries(mono_root_ / "rgb.txt");
}

GroundTruth Eth3dInput::camera_ground_truth() const {
    return load_ground_truth(mono_root_ / "groundtruth.txt");
}

Points Eth3dInput::reference_cloud() const {
    return load_ply_xyz(ground_truth_cloud_);
}

}  // namespace eth3d_score
