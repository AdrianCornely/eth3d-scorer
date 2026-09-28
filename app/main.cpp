#include "eth3d_score/config.hpp"
#include "eth3d_score/scorer.hpp"

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args;
        if (argc > 0) {
            args.reserve(static_cast<std::size_t>(argc - 1));
        }
        for (int i = 1; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }

        const char* executable = "eth3d-scorer";
        if (argc > 0) {
            executable = argv[0];
        }

        const eth3d_score::ParseResult parsed = eth3d_score::parse_command_line(args);
        if (parsed.help_requested) {
            std::cout << eth3d_score::usage(executable);
            return 0;
        }
        if (!parsed.config) {
            std::cerr << "error: " << parsed.error << "\n\n"
                      << eth3d_score::usage(executable);
            return 2;
        }

        const std::vector<std::string> errors = eth3d_score::validate(*parsed.config);
        if (!errors.empty()) {
            for (const std::string& error : errors) {
                std::cerr << "error: " << error << '\n';
            }
            return 2;
        }

        const eth3d_score::ScoreSummary summary = eth3d_score::score(*parsed.config);
        std::filesystem::path output_directory = parsed.config->out.parent_path();
        if (output_directory.empty()) {
            output_directory = ".";
        }
        std::filesystem::create_directories(output_directory);
        std::ofstream output(parsed.config->out);
        if (!output) {
            std::cerr << "error: cannot open output JSON: "
                      << parsed.config->out << '\n';
            return 1;
        }
        output << eth3d_score::to_json(summary).dump(2) << '\n';
        if (!output) {
            std::cerr << "error: failed while writing output JSON: "
                      << parsed.config->out << '\n';
            return 1;
        }
        std::cout << "Saved metrics: " << parsed.config->out << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}

