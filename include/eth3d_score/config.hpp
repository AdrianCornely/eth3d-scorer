#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace eth3d_score {

/// which JSON layout the camera poses are using
/// baseline means one camera_poses entry per frame
/// pairwise means the left camera pose from each accepted_pairs entry
/// TODO: figure out whether buffering needs another layout once it actually exists
enum class PoseFileLayout { baseline, pairwise };

/// everything needed for one scoring run
struct Config {
    std::filesystem::path recon;
    std::filesystem::path manifest;
    PoseFileLayout pose_file_layout = PoseFileLayout::baseline;
    std::filesystem::path mono_root;
    std::filesystem::path gt;
    std::filesystem::path out;
    std::optional<std::filesystem::path> save_aligned;
    std::optional<std::filesystem::path> save_trajectory;
    bool trajectory_only = false;
    std::optional<std::size_t> max_images = 50;
    int metric_points = 200000;
    int threads = 0;
    std::vector<double> thresholds = {0.01, 0.02, 0.05, 0.10};
};

struct ParseResult {
    std::optional<Config> config;
    bool help_requested = false;
    std::string error;
};

/// turn the command line into settings and explain anything that does not make sense
[[nodiscard]] ParseResult parse_command_line(const std::vector<std::string>& args);
[[nodiscard]] std::string usage(std::string_view executable = "eth3d-scorer");
/// catch missing inputs before doing any of the expensive stuff
[[nodiscard]] std::vector<std::string> validate(const Config& config);

}  // namespace eth3d_score

