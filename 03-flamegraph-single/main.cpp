// A single-file dense matrix multiplication workload. No downloads or dependencies.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(__linux__) && !defined(__APPLE__)
#error "Run this experiment on Linux/macOS; Windows users should use WSL2."
#endif

__attribute__((noinline))
void multiply(const std::vector<double>& a, const std::vector<double>& b,
              std::vector<double>& c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

__attribute__((noinline))
double matrix_checksum(const std::vector<double>& values) {
    double result = 0;
    for (double value : values) result += value;
    return result;
}

int positive(const std::string& value) {
    std::size_t used = 0;
    int number = std::stoi(value, &used);
    if (used != value.size() || number <= 0) throw std::invalid_argument("expected positive integer");
    return number;
}

int main(int argc, char** argv) {
    try {
        int n = 256, rounds = 80;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << "matrix_demo [--size 256] [--rounds 80]\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::invalid_argument("missing argument value");
            int value = positive(argv[++i]);
            if (key == "--size") n = value;
            else if (key == "--rounds") rounds = value;
            else throw std::invalid_argument("unknown option: " + key);
        }
        const std::size_t count = static_cast<std::size_t>(n) * n;
        std::vector<double> a(count), b(count), c(count);
        for (std::size_t i = 0; i < count; ++i) {
            a[i] = static_cast<double>(i % 17) / 17.0;
            b[i] = static_cast<double>(i % 13) / 13.0;
        }
        auto start = std::chrono::steady_clock::now();
        double checksum = 0;
        for (int round = 0; round < rounds; ++round) {
            multiply(a, b, c, n);
            checksum += matrix_checksum(c);
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!std::isfinite(checksum)) throw std::runtime_error("non-finite result");
        std::cout << std::setprecision(17) << "{\"workload\":\"single_matrix_multiply\",\"size\":"
                  << n << ",\"rounds\":" << rounds << ",\"checksum\":" << checksum
                  << ",\"wall_seconds\":" << seconds << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
