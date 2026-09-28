#include "eth3d_score/config.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace eth3d_score {
namespace {

ParseResult parse_error(std::string message) {
    return {.config = std::nullopt, .help_requested = false, .error = std::move(message)};
}

const std::string* value_after(
    const std::vector<std::string>& args, std::size_t& index) {
    if (index + 1 >= args.size()) {
        return nullptr;
    }
    ++index;
    return &args[index];
}

template <typename T>
bool parse_number(std::string_view text, T& output) {
    const char* first = text.data();
    const char* last = first + text.size();
    const std::from_chars_result result = std::from_chars(first, last, output);
    return result.ec == std::errc{} && result.ptr == last;
}

void require_file(
    const std::filesystem::path& path,
    const std::string_view name,
    std::vector<std::string>& errors) {
    if (!std::filesystem::is_regular_file(path)) {
        errors.emplace_back(std::string{name} + " is not a readable file: " + path.string());
    }
}

}  // namespace

std::string usage(const std::string_view executable) {
    std::ostringstream out;
    out << "Usage: " << executable << " --recon PATH --manifest PATH --manifest-type "
        << "{baseline,pairwise} --mono-root PATH --gt PATH --out PATH [options]\n"
        << "       " << executable << " --trajectory-only --manifest PATH --manifest-type "
        << "{baseline,pairwise} --mono-root PATH --out PATH [options]\n\n"
        << "Required arguments:\n"
        << "  --recon PATH                  reconstructed point cloud unless trajectory-only\n"
        << "  --manifest PATH               evaluation manifest\n"
        << "  --manifest-type TYPE          baseline or pairwise\n"
        << "  --mono-root PATH              directory containing rgb.txt and groundtruth.txt\n"
        << "  --gt PATH                     reference cloud unless trajectory-only\n"
        << "  --out PATH                    destination for score results\n\n"
        << "Optional arguments:\n"
        << "  --save-aligned PATH           write aligned reconstruction\n"
        << "  --save-trajectory PATH        write estimated poses in ETH3D text format\n"
        << "  --trajectory-only             only write the official ETH3D trajectory report\n"
        << "  --max-images N                first N RGB frames (default 50)\n"
        << "  --all-images                  use every RGB frame in the dataset\n"
        << "  --metric-points N             points sampled per cloud (default 200000)\n"
        << "  --threads N                   nearest-neighbor threads, 0 uses available CPUs\n"
        << "  --thresholds VALUE [...]      distance thresholds (defaults 0.01 0.02 0.05 0.10)\n"
        << "  -h, --help                    show this help\n";
    return out.str();
}

ParseResult parse_command_line(const std::vector<std::string>& args) {
    Config config;
    bool have_recon = false;
    bool have_manifest = false;
    bool have_pose_file_layout = false;
    bool have_mono_root = false;
    bool have_gt = false;
    bool have_out = false;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "-h" || arg == "--help") {
            return {.config = std::nullopt, .help_requested = true, .error = {}};
        }
        if (arg == "--trajectory-only") {
            config.trajectory_only = true;
            continue;
        }
        if (arg == "--all-images") {
            config.max_images = std::nullopt;
            continue;
        }
        if (arg == "--recon" || arg == "--manifest" || arg == "--mono-root" ||
            arg == "--gt" || arg == "--out" || arg == "--save-aligned" ||
            arg == "--save-trajectory") {
            const std::string* value = value_after(args, i);
            if (value == nullptr) {
                return parse_error("missing value for " + arg);
            }
            if (arg == "--recon") {
                config.recon = *value;
                have_recon = true;
            } else if (arg == "--manifest") {
                config.manifest = *value;
                have_manifest = true;
            } else if (arg == "--mono-root") {
                config.mono_root = *value;
                have_mono_root = true;
            } else if (arg == "--gt") {
                config.gt = *value;
                have_gt = true;
            } else if (arg == "--out") {
                config.out = *value;
                have_out = true;
            } else if (arg == "--save-aligned") {
                config.save_aligned = *value;
            } else {
                config.save_trajectory = *value;
            }
        } else if (arg == "--manifest-type") {
            const std::string* value = value_after(args, i);
            if (value == nullptr) {
                return parse_error("missing value for " + arg);
            }
            if (*value == "baseline") {
                config.pose_file_layout = PoseFileLayout::baseline;
            } else if (*value == "pairwise") {
                config.pose_file_layout = PoseFileLayout::pairwise;
            } else {
                return parse_error("--manifest-type must be baseline or pairwise");
            }
            have_pose_file_layout = true;
        } else if (arg == "--max-images" || arg == "--metric-points" || arg == "--threads") {
            const std::string* value = value_after(args, i);
            if (value == nullptr) {
                return parse_error("missing value for " + arg);
            }
            int parsed = 0;
            if (!parse_number(*value, parsed)) {
                return parse_error(arg + " must be an integer");
            }
            if (arg == "--max-images") {
                if (parsed <= 0) {
                    return parse_error("--max-images must be positive");
                }
                config.max_images = static_cast<std::size_t>(parsed);
            } else if (arg == "--metric-points") {
                config.metric_points = parsed;
            } else {
                if (parsed < 0) {
                    return parse_error("--threads must be zero or greater");
                }
                config.threads = parsed;
            }
        } else if (arg == "--thresholds") {
            std::vector<double> parsed_thresholds;
            while (i + 1 < args.size() && !args[i + 1].starts_with('-')) {
                double parsed = 0.0;
                ++i;
                if (!parse_number(args[i], parsed)) {
                    return parse_error("--thresholds values must be numbers");
                }
                parsed_thresholds.push_back(parsed);
            }
            if (parsed_thresholds.empty()) {
                return parse_error("missing value for --thresholds");
            }
            config.thresholds = std::move(parsed_thresholds);
        } else if (arg.starts_with('-')) {
            return parse_error("unknown option: " + arg);
        } else {
            return parse_error("unexpected positional argument: " + arg);
        }
    }

    std::vector<std::string> missing;
    if (!have_recon && !config.trajectory_only) {
        missing.emplace_back("--recon");
    }
    if (!have_manifest) {
        missing.emplace_back("--manifest");
    }
    if (!have_pose_file_layout) {
        missing.emplace_back("--manifest-type");
    }
    if (!have_mono_root) {
        missing.emplace_back("--mono-root");
    }
    if (!have_gt && !config.trajectory_only) {
        missing.emplace_back("--gt");
    }
    if (!have_out) {
        missing.emplace_back("--out");
    }
    if (!missing.empty()) {
        std::ostringstream message;
        message << "missing required arguments:";
        for (const std::string& option : missing) {
            message << ' ' << option;
        }
        return parse_error(message.str());
    }
    return {.config = std::move(config), .help_requested = false, .error = {}};
}

std::vector<std::string> validate(const Config& config) {
    std::vector<std::string> errors;
    if (!config.trajectory_only) {
        require_file(config.recon, "reconstruction PLY", errors);
    }
    require_file(config.manifest, "manifest JSON", errors);
    if (!config.trajectory_only) {
        require_file(config.gt, "reference cloud", errors);
    }
    if (!std::filesystem::is_directory(config.mono_root)) {
        errors.emplace_back("ETH3D mono root is not a directory: " + config.mono_root.string());
    } else {
        require_file(config.mono_root / "rgb.txt", "rgb.txt", errors);
        require_file(config.mono_root / "groundtruth.txt", "groundtruth.txt", errors);
    }
    return errors;
}

}  // namespace eth3d_score

