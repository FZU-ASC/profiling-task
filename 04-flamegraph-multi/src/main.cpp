#include "pipeline.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#if !defined(__linux__) && !defined(__APPLE__)
#error "Run this experiment on Linux/macOS; Windows users should use WSL2."
#endif

int positive(const std::string& value) {
    std::size_t used = 0;
    int number = std::stoi(value, &used);
    if (used != value.size() || number <= 0) throw std::invalid_argument("expected positive integer");
    return number;
}

int main(int argc, char** argv) {
    try {
        Config config;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "signal_pipeline [--records 1600] [--length 256] [--bins 24] "
                             "[--queries 64] [--rounds 5]\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::invalid_argument("missing argument value");
            const int value = positive(argv[++i]);
            if (key == "--records") config.records = value;
            else if (key == "--length") config.length = value;
            else if (key == "--bins") config.bins = value;
            else if (key == "--queries") config.queries = value;
            else if (key == "--rounds") config.rounds = value;
            else throw std::invalid_argument("unknown option: " + key);
        }
        if (config.length < 8 || config.bins > config.length / 2)
            throw std::invalid_argument("require length >= 8 and bins <= length / 2");
        const auto start = std::chrono::steady_clock::now();
        const auto result = run_pipeline(config);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!std::isfinite(result.energy) || !std::isfinite(result.distance))
            throw std::runtime_error("non-finite result");
        std::cout << std::setprecision(17) << "{\"workload\":\"multi_signal_pipeline\",\"records\":"
                  << config.records << ",\"length\":" << config.length << ",\"bins\":" << config.bins
                  << ",\"queries\":" << config.queries << ",\"rounds\":" << config.rounds
                  << ",\"energy\":" << result.energy << ",\"distance\":" << result.distance
                  << ",\"wall_seconds\":" << seconds << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
