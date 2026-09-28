#include "eth3d_score/cloud.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace eth3d_score {
namespace {

// split by X then Y then Z and keep repeating that pattern
// each split lets the search ignore a whole chunk when it cannot contain anything closer
class KdTree {
public:
    explicit KdTree(const Points& points) : points_(points) {
        indices_.resize(static_cast<std::size_t>(points.rows()));
        std::iota(indices_.begin(), indices_.end(), Eigen::Index{0});
        nodes_.reserve(indices_.size());
        root_ = build(0, indices_.size(), 0);
    }

    [[nodiscard]] double nearest_distance(const Eigen::RowVector3d& query) const {
        double best_squared = std::numeric_limits<double>::infinity();
        search(root_, query, best_squared);
        return std::sqrt(best_squared);
    }

private:
    struct AxisLess {
        const Points& points;
        int axis;

        bool operator()(const Eigen::Index left, const Eigen::Index right) const {
            return points(left, axis) < points(right, axis);
        }
    };

    struct Node {
        Eigen::Index point;
        int left;
        int right;
        int axis;
    };

    int build(const std::size_t begin, const std::size_t end, const int depth) {
        if (begin >= end) {
            return -1;
        }
        const int axis = depth % 3;
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(indices_.begin() + static_cast<std::ptrdiff_t>(begin),
                         indices_.begin() + static_cast<std::ptrdiff_t>(middle),
                         indices_.begin() + static_cast<std::ptrdiff_t>(end),
                         AxisLess{points_, axis});
        const int node_index = static_cast<int>(nodes_.size());
        nodes_.push_back({indices_[middle], -1, -1, axis});
        const int left = build(begin, middle, depth + 1);
        const int right = build(middle + 1, end, depth + 1);
        nodes_[static_cast<std::size_t>(node_index)].left = left;
        nodes_[static_cast<std::size_t>(node_index)].right = right;
        return node_index;
    }

    void search(const int node_index, const Eigen::RowVector3d& query, double& best_squared) const {
        if (node_index < 0) {
            return;
        }
        const Node& node = nodes_[static_cast<std::size_t>(node_index)];
        const Eigen::RowVector3d difference = query - points_.row(node.point);
        best_squared = std::min(best_squared, difference.squaredNorm());
        const double split_difference = query(node.axis) - points_(node.point, node.axis);
        int near_child = node.right;
        int far_child = node.left;
        if (split_difference < 0.0) {
            near_child = node.left;
            far_child = node.right;
        }
        search(near_child, query, best_squared);
        // the other side only matters when it could still beat the closest point found so far
        if (split_difference * split_difference <= best_squared) {
            search(far_child, query, best_squared);
        }
    }

    const Points& points_;
    std::vector<Eigen::Index> indices_;
    std::vector<Node> nodes_;
    int root_ = -1;
};

void fill_nearest_distances(
    const KdTree& tree,
    const Points& queries,
    std::vector<double>& distances,
    const std::size_t begin,
    const std::size_t end) {
    for (std::size_t query = begin; query < end; ++query) {
        distances[query] = tree.nearest_distance(
            queries.row(static_cast<Eigen::Index>(query)));
    }
}

std::size_t resolve_thread_count(const int requested, const std::size_t query_count) {
    std::size_t count = static_cast<std::size_t>(requested);
    if (requested == 0) {
        count = static_cast<std::size_t>(std::thread::hardware_concurrency());
    }
    if (count == 0) {
        count = 1;
    }
    return std::min(count, std::max<std::size_t>(query_count, 1));
}

std::vector<double> nearest_distances(
    const Points& queries, const Points& references, const std::size_t requested_workers) {
    if (references.rows() == 0) {
        throw std::runtime_error("Cannot query an empty point cloud");
    }
    const KdTree tree(references);
    const std::size_t query_count = static_cast<std::size_t>(queries.rows());
    std::vector<double> distances(query_count);
    const std::size_t worker_count = std::min(requested_workers, std::max<std::size_t>(query_count, 1));
    const std::size_t chunk_size = (query_count + worker_count - 1) / worker_count;
    {
        // leaving this scope joins every worker so none of them can outlive distances
        std::vector<std::jthread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            const std::size_t begin = worker * chunk_size;
            const std::size_t end = std::min(begin + chunk_size, query_count);
            if (begin >= end) {
                break;
            }
            workers.emplace_back(
                fill_nearest_distances,
                std::cref(tree),
                std::cref(queries),
                std::ref(distances),
                begin,
                end);
        }
    }
    return distances;
}

double quantile(const std::vector<double>& sorted_values, const double probability) {
    // blend the two neighbors when the percentile lands between them
    const double position = probability * static_cast<double>(sorted_values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    return sorted_values[lower] +
           (position - static_cast<double>(lower)) *
               (sorted_values[upper] - sorted_values[lower]);
}

DistanceStats describe(const std::vector<double>& values) {
    double sum = 0.0;
    double squared_sum = 0.0;
    for (const double value : values) {
        sum += value;
        squared_sum += value * value;
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t size = sorted.size();
    double median = sorted[size / 2];
    if (size % 2 == 0) {
        median = 0.5 * (sorted[size / 2 - 1] + sorted[size / 2]);
    }
    return {
        .mean = sum / static_cast<double>(size),
        .median = median,
        .rmse = std::sqrt(squared_sum / static_cast<double>(size)),
        .p90 = quantile(sorted, 0.90),
        .p95 = quantile(sorted, 0.95),
    };
}

std::string threshold_label(const double threshold) {
    // keep JSON keys readable like 0.01 instead of 0.010000
    std::ostringstream stream;
    stream << std::setprecision(6) << std::defaultfloat << threshold;
    return stream.str();
}

std::size_t count_at_or_below(
    const std::vector<double>& values, const double threshold) {
    std::size_t count = 0;
    for (const double value : values) {
        if (value <= threshold) {
            ++count;
        }
    }
    return count;
}

double read_binary_scalar(std::istream& input, const std::string& type) {
    if (type == "char" || type == "int8") {
        std::int8_t value = 0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    if (type == "uchar" || type == "uint8") {
        std::uint8_t value = 0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    if (type == "short" || type == "int16") {
        std::int16_t value = 0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    if (type == "ushort" || type == "uint16") {
        std::uint16_t value = 0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    if (type == "int" || type == "int32") {
        std::int32_t value = 0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    if (type == "uint" || type == "uint32") {
        std::uint32_t value = 0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    if (type == "float" || type == "float32") {
        float value = 0.0F;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    if (type == "double" || type == "float64") {
        double value = 0.0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    throw std::runtime_error("Unsupported PLY scalar type: " + type);
}

}  // namespace

Points load_ply_xyz(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open point cloud: " + path.string());
    }
    std::string line;
    std::size_t vertex_count = 0;
    bool ascii = false;
    bool binary_little_endian = false;
    std::vector<std::string> properties;
    std::vector<std::string> property_types;
    bool vertices = false;
    // binary rows have no labels so the header is the only map back to X Y and Z
    while (std::getline(input, line)) {
        if (line == "format ascii 1.0") {
            ascii = true;
        }
        if (line == "format binary_little_endian 1.0") {
            binary_little_endian = true;
        }
        if (line.rfind("element ", 0) == 0) {
            vertices = line.rfind("element vertex ", 0) == 0;
            if (vertices) {
                vertex_count = static_cast<std::size_t>(std::stoull(line.substr(15)));
            }
        } else if (vertices && line.rfind("property ", 0) == 0) {
            std::istringstream property_stream(line);
            std::string keyword;
            std::string type;
            std::string name;
            property_stream >> keyword >> type >> name;
            if (type == "list") {
                throw std::runtime_error(
                    "List property in PLY vertex element is unsupported");
            }
            property_types.push_back(type);
            properties.push_back(name);
        } else if (line == "end_header") {
            break;
        }
    }
    if (!ascii && !binary_little_endian) {
        throw std::runtime_error("Unsupported PLY format: " + path.string());
    }
    const std::vector<std::string>::const_iterator x_position =
        std::find(properties.begin(), properties.end(), "x");
    const std::vector<std::string>::const_iterator y_position =
        std::find(properties.begin(), properties.end(), "y");
    const std::vector<std::string>::const_iterator z_position =
        std::find(properties.begin(), properties.end(), "z");
    if (x_position == properties.end() || y_position == properties.end() || z_position == properties.end()) {
        throw std::runtime_error("PLY vertex element has no xyz properties");
    }
    const std::size_t xi = static_cast<std::size_t>(x_position - properties.begin());
    const std::size_t yi = static_cast<std::size_t>(y_position - properties.begin());
    const std::size_t zi = static_cast<std::size_t>(z_position - properties.begin());
    std::vector<Eigen::Vector3d> valid;
    valid.reserve(vertex_count);
    for (std::size_t row = 0; row < vertex_count; ++row) {
        std::vector<double> values(properties.size());
        bool parsed = true;
        if (ascii) {
            if (!std::getline(input, line)) {
                break;
            }
            std::istringstream stream(line);
            for (double& value : values) {
                if (!(stream >> value)) {
                    parsed = false;
                }
            }
            std::string extra;
            if (stream >> extra) {
                parsed = false;
            }
        } else {
            for (std::size_t property = 0; property < properties.size(); ++property) {
                values[property] = read_binary_scalar(input, property_types[property]);
            }
            parsed = static_cast<bool>(input);
        }
        if (!parsed) {
            throw std::runtime_error(
                "Malformed PLY vertex in " + path.string() + " at row " +
                std::to_string(row + 1));
        }
        const Eigen::Vector3d point(values[xi], values[yi], values[zi]);
        if (!point.allFinite()) {
            throw std::runtime_error(
                "Nonfinite PLY vertex in " + path.string() + " at row " +
                std::to_string(row + 1));
        }
        valid.push_back(point);
    }
    Points points(static_cast<Eigen::Index>(valid.size()), 3);
    for (std::size_t index = 0; index < valid.size(); ++index) {
        points.row(static_cast<Eigen::Index>(index)) = valid[index];
    }
    return points;
}

void save_ply_xyz(const std::filesystem::path& path, const Points& points) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot write point cloud: " + path.string());
    }
    output << "ply\nformat binary_little_endian 1.0\nelement vertex " << points.rows()
           << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (Eigen::Index row = 0; row < points.rows(); ++row) {
        const float xyz[3] = {
            static_cast<float>(points(row, 0)),
            static_cast<float>(points(row, 1)),
            static_cast<float>(points(row, 2)),
        };
        output.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
    }
    if (!output) {
        throw std::runtime_error("Failed while writing point cloud: " + path.string());
    }
}

CloudScore score_clouds(const Points& reconstruction, const Points& ground_truth,
                        const std::vector<double>& thresholds,
                        const std::size_t metric_points, const int threads) {
    // TODO: once occlusion volumes exist decide whether hidden ground truth should disappear
    // TODO: decide whether geometry inside an occluder is ignored or still counted as wrong
    // TODO: once visibility uses multiple cameras decide whether seeing a point once is enough
    const Points recon = deterministic_sample(reconstruction, metric_points);
    const Points gt = deterministic_sample(ground_truth, metric_points);
    if (recon.rows() == 0 || gt.rows() == 0) {
        throw std::runtime_error("Cannot score an empty point cloud");
    }
    const std::size_t largest_query_count = static_cast<std::size_t>(std::max(recon.rows(), gt.rows()));
    const std::size_t worker_threads = resolve_thread_count(threads, largest_query_count);
    // both directions matter
    // a tiny clean cloud can look accurate while missing nearly everything and a huge messy cloud
    // can cover the scene while putting plenty of points where they do not belong
    const std::vector<double> accuracy = nearest_distances(recon, gt, worker_threads);
    const std::vector<double> completeness = nearest_distances(gt, recon, worker_threads);
    double accuracy_sum = 0.0;
    double completeness_sum = 0.0;
    double accuracy_squared = 0.0;
    double completeness_squared = 0.0;
    for (const double value : accuracy) {
        accuracy_sum += value;
        accuracy_squared += value * value;
    }
    for (const double value : completeness) {
        completeness_sum += value;
        completeness_squared += value * value;
    }
    const double accuracy_mean = accuracy_sum / static_cast<double>(accuracy.size());
    const double completeness_mean = completeness_sum / static_cast<double>(completeness.size());
    std::vector<ThresholdScore> threshold_metrics;
    threshold_metrics.reserve(thresholds.size());
    for (const double threshold : thresholds) {
        const std::size_t precision_count = count_at_or_below(accuracy, threshold);
        const std::size_t recall_count = count_at_or_below(completeness, threshold);
        const double precision =
            static_cast<double>(precision_count) / static_cast<double>(accuracy.size());
        const double recall =
            static_cast<double>(recall_count) / static_cast<double>(completeness.size());
        double f1 = 0.0;
        if (precision + recall > 0.0) {
            f1 = 2.0 * precision * recall / (precision + recall);
        }
        threshold_metrics.push_back({
            .threshold = threshold,
            .precision = precision,
            .recall = recall,
            .f1 = f1,
        });
    }
    return {
        .reconstruction_points_scored = static_cast<std::size_t>(recon.rows()),
        .ground_truth_points_scored = static_cast<std::size_t>(gt.rows()),
        .worker_threads = worker_threads,
        .recon_to_ground_truth = describe(accuracy),
        .ground_truth_to_recon = describe(completeness),
        .chamfer_mean_unsquared = 0.5 * (accuracy_mean + completeness_mean),
        .chamfer_mean_squared =
            0.5 * (accuracy_squared / static_cast<double>(accuracy.size()) +
                   completeness_squared / static_cast<double>(completeness.size())),
        .threshold_metrics = std::move(threshold_metrics),
    };
}

nlohmann::json to_json(const CloudScore& score) {
    const nlohmann::json accuracy = {
        {"mean", score.recon_to_ground_truth.mean},
        {"median", score.recon_to_ground_truth.median},
        {"rmse", score.recon_to_ground_truth.rmse},
        {"p90", score.recon_to_ground_truth.p90},
        {"p95", score.recon_to_ground_truth.p95},
    };
    const nlohmann::json completeness = {
        {"mean", score.ground_truth_to_recon.mean},
        {"median", score.ground_truth_to_recon.median},
        {"rmse", score.ground_truth_to_recon.rmse},
        {"p90", score.ground_truth_to_recon.p90},
        {"p95", score.ground_truth_to_recon.p95},
    };
    nlohmann::json thresholds = nlohmann::json::object();
    for (const ThresholdScore& metric : score.threshold_metrics) {
        thresholds[threshold_label(metric.threshold)] = {
            {"threshold", metric.threshold},
            {"precision", metric.precision},
            {"recall", metric.recall},
            {"f1", metric.f1},
        };
    }
    return {
        {"reconstruction_points_scored", score.reconstruction_points_scored},
        {"ground_truth_points_scored", score.ground_truth_points_scored},
        {"worker_threads", score.worker_threads},
        {"accuracy_recon_to_gt", accuracy},
        {"completeness_gt_to_recon", completeness},
        {"chamfer_mean_unsquared", score.chamfer_mean_unsquared},
        {"chamfer_mean_squared", score.chamfer_mean_squared},
        {"threshold_metrics", thresholds},
    };
}

}  // namespace eth3d_score
