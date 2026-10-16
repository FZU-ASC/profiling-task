#include "pipeline.hpp"
#include <cmath>
#include <random>

Dataset generate_dataset(const Config& config) {
    std::mt19937 generator(42);
    std::uniform_real_distribution<double> noise(-0.2, 0.2);
    Dataset records(config.records, Signal(config.length));
    for (int row = 0; row < config.records; ++row) {
        const double frequency = 1.0 + row % 19;
        for (int sample = 0; sample < config.length; ++sample) {
            double phase = 6.283185307179586 * sample / config.length;
            records[row][sample] = std::sin(frequency * phase) +
                                  0.3 * std::cos(3 * phase) + noise(generator);
        }
    }
    return records;
}
