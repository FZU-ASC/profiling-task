#include "pipeline.hpp"
#include <cmath>
#include <numeric>

Feature time_features(const Signal& signal) {
    const double mean = std::accumulate(signal.begin(), signal.end(), 0.0) / signal.size();
    double energy = 0;
    for (double value : signal) energy += value * value;
    return {energy / signal.size(), mean};
}

Feature spectral_features(const Signal& signal, int bins) {
    // Intentionally direct DFT, rather than an FFT: a visible numeric hot path.
    Feature output(bins);
    for (int bin = 1; bin <= bins; ++bin) {
        double real = 0, imaginary = 0;
        for (std::size_t sample = 0; sample < signal.size(); ++sample) {
            const double phase = 6.283185307179586 * bin * sample / signal.size();
            real += signal[sample] * std::cos(phase);
            imaginary -= signal[sample] * std::sin(phase);
        }
        output[bin - 1] = (real * real + imaginary * imaginary) /
                          (signal.size() * signal.size());
    }
    return output;
}

Features extract_features(const Dataset& records, int bins) {
    Features output;
    output.reserve(records.size());
    for (const auto& record : records) {
        auto feature = time_features(record);
        auto spectrum = spectral_features(record, bins);
        feature.insert(feature.end(), spectrum.begin(), spectrum.end());
        output.push_back(std::move(feature));
    }
    return output;
}
