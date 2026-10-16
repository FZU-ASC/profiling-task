#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$(uname -s)" in
    Linux|Darwin) ;;
    *) echo "Run on Linux/macOS; Windows users should enter WSL2." >&2; exit 1 ;;
esac
case "${1:-}" in
    single) target=matrix_demo; name=flamegraph-single ;;
    multi) target=signal_pipeline; name=flamegraph-multi ;;
    *) echo "Usage: bash scripts/profile_cpp.sh single|multi [program arguments]" >&2; exit 1 ;;
esac
shift
flamegraph_bin="${FLAMEGRAPH:-flamegraph}"
if ! command -v "$flamegraph_bin" >/dev/null 2>&1; then
    echo "flamegraph not found. Install: cargo install flamegraph --locked" >&2
    exit 1
fi
make -C "$repo_root" "build/$target"
output_dir="${OUTPUT_DIR:-$repo_root/results/$name}"
mkdir -p "$output_dir"
cd "$output_dir"
{
    uname -a
    "${CXX:-c++}" --version
    "$flamegraph_bin" --version
    git -C "$repo_root" rev-parse HEAD 2>/dev/null || true
    printf 'program:'
    printf ' %q' "$repo_root/build/$target" "$@"
    printf '\n'
} > environment.txt
if [[ "$(uname -s)" == "Linux" ]]; then
    # Software CPU-clock sampling also works on WSL kernels without a hardware PMU.
    "${PERF:-perf}" --version >> environment.txt
    "$flamegraph_bin" -c "record -F 997 -e cpu-clock -g --call-graph dwarf" \
        -o flamegraph.svg -- "$repo_root/build/$target" "$@" | tee program-output.json
else
    "$flamegraph_bin" -o flamegraph.svg -- "$repo_root/build/$target" "$@" | tee program-output.json
fi
printf 'Saved results to %s\n' "$output_dir"
