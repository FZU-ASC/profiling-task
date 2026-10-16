#include "pipeline.hpp"

Result aggregate_results(const Features& features, double distance) {
    double energy = 0;
    for (const auto& feature : features) energy += feature[0];
    return {energy, distance};
}

Result run_round(const Dataset& data, const Config& config) {
    const auto prepared = preprocess_records(data);
    const auto features = extract_features(prepared, config.bins);
    const auto index = build_index(features);
    return aggregate_results(features, query_batch(features, index, config.queries));
}

Result run_pipeline(const Config& config) {
    const auto data = generate_dataset(config);
    Result total{0, 0};
    for (int round = 0; round < config.rounds; ++round) {
        const auto result = run_round(data, config);
        total.energy += result.energy;
        total.distance += result.distance;
    }
    return total;
}
