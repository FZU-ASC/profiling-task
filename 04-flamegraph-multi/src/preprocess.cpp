#include "pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

Signal normalize_signal(const Signal& signal) {
    const double mean = std::accumulate(signal.begin(), signal.end(), 0.0) / signal.size();
    double variance = 0;
    for (double value : signal) variance += (value - mean) * (value - mean);
    const double scale = std::sqrt(variance / signal.size() + 1e-8);
    Signal output(signal.size());
    for (std::size_t i = 0; i < signal.size(); ++i) output[i] = (signal[i] - mean) / scale;
    return output;
}

Signal smooth_signal(const Signal& signal) {
    Signal output(signal.size());
    for (int i = 0; i < static_cast<int>(signal.size()); ++i) {
        double sum = 0;
        int count = 0;
        for (int offset = -2; offset <= 2; ++offset) {
            int position = i + offset;
            if (position >= 0 && position < static_cast<int>(signal.size())) {
                sum += signal[position];
                ++count;
            }
        }
        output[i] = sum / count;
    }
    return output;
}

Dataset preprocess_records(const Dataset& records) {
    Dataset output;
    output.reserve(records.size());
    for (const auto& record : records) output.push_back(smooth_signal(normalize_signal(record)));
    return output;
}
