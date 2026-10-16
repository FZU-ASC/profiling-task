#include "pipeline.hpp"
#include <algorithm>
#include <limits>
#include <numeric>

std::vector<std::size_t> build_index(const Features& features) {
    std::vector<std::size_t> index(features.size());
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [&](std::size_t a, std::size_t b) {
        return features[a][0] < features[b][0];
    });
    return index;
}

double squared_distance(const Feature& left, const Feature& right) {
    double distance = 0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        double delta = left[i] - right[i];
        distance += delta * delta;
    }
    return distance;
}

double query_neighborhood(const Features& features, const std::vector<std::size_t>& index,
                          const Feature& query) {
    auto found = std::lower_bound(index.begin(), index.end(), query[0],
        [&](std::size_t row, double value) { return features[row][0] < value; });
    const auto position = static_cast<std::ptrdiff_t>(found - index.begin());
    const auto start = std::max<std::ptrdiff_t>(0, position - 32);
    const auto end = std::min<std::ptrdiff_t>(index.size(), position + 32);
    double best = std::numeric_limits<double>::infinity();
    for (auto i = start; i < end; ++i) {
        best = std::min(best, squared_distance(features[index[i]], query));
    }
    return best;
}

double query_batch(const Features& features, const std::vector<std::size_t>& index, int count) {
    double sum = 0;
    for (int i = 0; i < count; ++i) {
        Feature query = features[(static_cast<std::size_t>(i) * 17) % features.size()];
        query[0] += 0.001;  // Avoid an identical query; this is approximate neighborhood search.
        sum += query_neighborhood(features, index, query);
    }
    return sum;
}
