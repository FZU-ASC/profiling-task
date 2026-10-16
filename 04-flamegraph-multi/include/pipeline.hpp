#pragma once
#include <cstddef>
#include <vector>

// Retain teaching call boundaries in optimized builds (GCC / Clang).
#define PROFILE_NOINLINE __attribute__((noinline))

struct Config {
    int records = 1600;
    int length = 256;
    int bins = 24;
    int queries = 64;
    int rounds = 5;
};
using Signal = std::vector<double>;
using Feature = std::vector<double>;
using Dataset = std::vector<Signal>;
using Features = std::vector<Feature>;
struct Result { double energy; double distance; };

PROFILE_NOINLINE Dataset generate_dataset(const Config& config);
PROFILE_NOINLINE Signal normalize_signal(const Signal& signal);
PROFILE_NOINLINE Signal smooth_signal(const Signal& signal);
PROFILE_NOINLINE Dataset preprocess_records(const Dataset& records);
PROFILE_NOINLINE Feature time_features(const Signal& signal);
PROFILE_NOINLINE Feature spectral_features(const Signal& signal, int bins);
PROFILE_NOINLINE Features extract_features(const Dataset& records, int bins);
PROFILE_NOINLINE std::vector<std::size_t> build_index(const Features& features);
PROFILE_NOINLINE double squared_distance(const Feature& left, const Feature& right);
PROFILE_NOINLINE double query_neighborhood(const Features& features,
    const std::vector<std::size_t>& index, const Feature& query);
PROFILE_NOINLINE double query_batch(const Features& features,
    const std::vector<std::size_t>& index, int count);
PROFILE_NOINLINE Result aggregate_results(const Features& features, double distance);
PROFILE_NOINLINE Result run_round(const Dataset& data, const Config& config);
PROFILE_NOINLINE Result run_pipeline(const Config& config);
