#pragma once
// cuBQL types, per-observation resources, constants and checked CUDA calls.
// Exact full-cloud GPU point NN for the selected angular detector.
// cuBQL revision: 20f9db19cdb160e2b83d8f05dfdb0437fe11fe24 (v1.3.1-dev)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/math/conservativeDistances.h"
#include "cuBQL/traversal/shrinkingRadiusQuery.h"

namespace {

using Clock = std::chrono::steady_clock;
using Vec3 = cuBQL::vec3f;
using Box3 = cuBQL::box3f;
using BVH = cuBQL::WideBVH<float, 3, 8>;

constexpr int kBlockSize = 256;
constexpr int kSources = 512;
constexpr int kMaxCandidates = 128;
constexpr int kInitialSketch = FAST_SYMMETRY_INITIAL_SKETCH;
constexpr const char *kRevision = "20f9db19cdb160e2b83d8f05dfdb0437fe11fe24";

struct Timings {
  float setup_ms;
  float boxes_ms;
  float build_ms;
  float input_ms;
  float reflection_ms;
  float traversal_ms;
  float percentile_ms;
  float output_ms;
  float total_ms;
};

struct Context {
  int point_count = 0;
  Vec3 *points = nullptr;
  bool owns_points = true;
  Box3 *boxes = nullptr;
  BVH bvh{};
  int *source_indices = nullptr;
  Vec3 *normals = nullptr;
  float *offsets = nullptr;
  Vec3 *queries = nullptr;
  float *distances = nullptr;
  float *p90 = nullptr;
  float *p99 = nullptr;
  cudaStream_t stream = nullptr;
  bool owns_stream = false;
  cuBQL::AsyncGpuMemoryResource memory;
};

thread_local char error_message[1024] = "";

inline float elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<float, std::milli>(end - begin).count();
}

inline bool checked(cudaError_t error, const char *where) {
  if (error == cudaSuccess) return true;
  std::snprintf(error_message, sizeof(error_message), "%s: %s", where,
                cudaGetErrorString(error));
  return false;
}

}  // namespace
