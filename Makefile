CXX ?= c++
CXXFLAGS ?= -O2 -g -std=c++17 -fno-omit-frame-pointer -fno-optimize-sibling-calls -Wall -Wextra -Wpedantic
MULTI_SOURCES := $(wildcard 04-flamegraph-multi/src/*.cpp)

.PHONY: all smoke
all: build/matrix_demo build/signal_pipeline

build:
	mkdir -p build

build/matrix_demo: 03-flamegraph-single/main.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/signal_pipeline: $(MULTI_SOURCES) 04-flamegraph-multi/include/pipeline.hpp | build
	$(CXX) $(CXXFLAGS) -I04-flamegraph-multi/include $(MULTI_SOURCES) -o $@

build/numeric_test: tests/numeric_test.cpp $(filter-out %/main.cpp,$(MULTI_SOURCES)) 04-flamegraph-multi/include/pipeline.hpp | build
	$(CXX) $(CXXFLAGS) -I04-flamegraph-multi/include tests/numeric_test.cpp $(filter-out %/main.cpp,$(MULTI_SOURCES)) -o $@

smoke: all
	python3 scripts/smoke_test.py
