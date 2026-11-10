#pragma once
// Reflection, exact NN traversal and NumPy-linear quantiles on the device.
#include "context.cuh"

namespace {
__global__ void make_boxes(Box3 *boxes, const Vec3 *points, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) boxes[index] = Box3().including(points[index]);
}

__global__ void reflect_sources(Vec3 *queries, const Vec3 *points,
                                const int *source_indices, const Vec3 *normals,
                                const float *offsets, int candidate_count,
                                int source_count) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int count = candidate_count * source_count;
  if (linear >= count) return;
  int candidate = linear / source_count;
  int source = linear - candidate * source_count;
  Vec3 point = points[source_indices[source]];
  Vec3 normal = normals[candidate];
  float signed_distance = dot(point, normal) - offsets[candidate];
  queries[linear] = point - (2.f * signed_distance) * normal;
}

__global__ void exact_nearest(float *normalized_distances, int *nearest_ids,
                              const Vec3 *queries,
                              const int *source_indices, const Vec3 *points,
                              BVH bvh, int point_count, int query_count,
                              float inverse_diagonal, int source_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= query_count) return;
  Vec3 query = queries[index];

  // The original source is a valid full-cloud target and gives a safe finite
  // upper bound. Traversal still visits every node that can improve it.
  int source = source_indices[index % source_count];
  float best_squared = cuBQL::fSqrDistance_rd(points[source], query);
  int best_id = source;
  // A small deterministic sketch only tightens the valid upper bound. The BVH
  // traversal below still certifies against the complete target cloud.
#pragma unroll 4
  for (int sample = 0; sample < kInitialSketch; ++sample) {
    int point_id = static_cast<int>((static_cast<int64_t>(sample) * point_count) /
                                    kInitialSketch);
    float squared = cuBQL::fSqrDistance_rd(points[point_id], query);
    if (squared < best_squared) {
      best_squared = squared;
      best_id = point_id;
    }
  }
  auto consider = [&best_squared, &best_id, points, query](uint32_t point_id) -> float {
    float squared = cuBQL::fSqrDistance_rd(points[point_id], query);
    if (squared < best_squared) {
      best_squared = squared;
      best_id = static_cast<int>(point_id);
    }
    return best_squared;
  };
  cuBQL::shrinkingRadiusQuery::forEachPrim(consider, bvh, query, best_squared);
  normalized_distances[index] = sqrtf(best_squared) * inverse_diagonal;
  if (nearest_ids) nearest_ids[index] = best_id;
}

__global__ void linear_p90_p99(float *p90, float *p99,
                               const float *distances, int candidate_count) {
  using Sort = cub::BlockRadixSort<float, kBlockSize, 2>;
  __shared__ typename Sort::TempStorage storage;
  __shared__ float brackets[4];
  int candidate = blockIdx.x;
  if (candidate >= candidate_count) return;
  int base = candidate * kSources + threadIdx.x * 2;
  float values[2] = {distances[base], distances[base + 1]};
  Sort(storage).Sort(values);
  // NumPy linear quantiles over 512 values:
  // p90 rank 459.9; p99 rank 505.89.
  if (threadIdx.x == 229) brackets[0] = values[1];  // rank 459
  if (threadIdx.x == 230) brackets[1] = values[0];  // rank 460
  if (threadIdx.x == 252) brackets[2] = values[1];  // rank 505
  if (threadIdx.x == 253) brackets[3] = values[0];  // rank 506
  __syncthreads();
  if (threadIdx.x == 0) {
    p90[candidate] = .1f * brackets[0] + .9f * brackets[1];
    p99[candidate] = .11f * brackets[2] + .89f * brackets[3];
  }
}

}  // namespace
