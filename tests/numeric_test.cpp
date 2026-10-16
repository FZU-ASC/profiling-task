#include "pipeline.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        const auto normalized = normalize_signal(Signal(16, 5.0));
        for (double value : normalized) require(value == 0, "constant normalization");
        const auto smoothed = smooth_signal(Signal(16, 3.0));
        for (double value : smoothed) require(value == 3, "constant smoothing, including edges");
        Signal cosine(32);
        for (std::size_t i = 0; i < cosine.size(); ++i)
            cosine[i] = std::cos(6.283185307179586 * 2 * i / cosine.size());
        const auto spectrum = spectral_features(cosine, 4);
        require(std::abs(spectrum[1] - 0.25) < 1e-12, "known DFT bin power");
        require(spectrum[0] < 1e-24 && spectrum[2] < 1e-24, "other DFT bins");
        require(std::abs(time_features(cosine)[0] - 0.5) < 1e-12, "known time-domain energy");
        Features values{{1, 2}, {3, 4}, {5, 6}};
        const auto index = build_index(values);
        require(query_neighborhood(values, index, {3, 4}) == 0, "exact match in neighborhood");
        require(squared_distance({1, 2}, {3, 4}) == 8, "squared Euclidean distance");
        std::cout << "PASS: normalization, smoothing, DFT, energy, and neighborhood distance\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
