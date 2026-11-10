// Host resource lifetime and the small C ABI consumed by the Python wrapper.
#include "context.cuh"
#include "kernels.cuh"

namespace {
inline void release(Context *context) {
  if (!context) return;
  if (context->bvh.nodes || context->bvh.primIDs)
    cuBQL::cuda::free(context->bvh, context->stream, context->memory);
#define FREE(member)                                                           \
  do {                                                                         \
    if (context->member) cudaFreeAsync(context->member, context->stream);       \
  } while (0)
  FREE(p99);
  FREE(p90);
  FREE(distances);
  FREE(queries);
  FREE(offsets);
  FREE(normals);
  FREE(source_indices);
  FREE(boxes);
  if (context->owns_points) FREE(points);
#undef FREE
  if (context->stream) cudaStreamSynchronize(context->stream);
  if (context->owns_stream) cudaStreamDestroy(context->stream);
  delete context;
}

}  // namespace

extern "C" {

const char *spatial_cubql_revision() { return kRevision; }
const char *spatial_cubql_last_error() { return error_message; }

static void *build_common(const float *input_points, int point_count,
                          int input_dimension, bool input_is_device,
                          uintptr_t stream_handle, Timings *timings) {
  error_message[0] = '\0';
  if (!input_points || point_count <= 0 || !timings ||
      input_dimension != 3) {
    std::snprintf(error_message, sizeof(error_message), "invalid build arguments");
    return nullptr;
  }
  std::memset(timings, 0, sizeof(*timings));
  auto total_begin = Clock::now();
  Context *context = new (std::nothrow) Context;
  if (!context) {
    std::snprintf(error_message, sizeof(error_message), "host allocation failed");
    return nullptr;
  }
  context->point_count = point_count;
  context->stream = reinterpret_cast<cudaStream_t>(stream_handle);
  if (!context->stream) {
    if (!checked(cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking),
                 "create verifier stream")) {
      delete context;
      return nullptr;
    }
    context->owns_stream = true;
  }

  auto setup_begin = Clock::now();
#define ALLOC(member, count)                                                   \
  if (!checked(cudaMallocAsync(reinterpret_cast<void **>(&context->member),    \
                               (count) * sizeof(*context->member),             \
                               context->stream),                               \
               "cudaMallocAsync(" #member ")")) {                            \
    release(context);                                                          \
    return nullptr;                                                            \
  }
  if (input_is_device && input_dimension == 3) {
    context->points = const_cast<Vec3 *>(reinterpret_cast<const Vec3 *>(input_points));
    context->owns_points = false;
  } else {
    ALLOC(points, point_count)
  }
  ALLOC(boxes, point_count)
  ALLOC(source_indices, kSources)
  ALLOC(normals, kMaxCandidates)
  ALLOC(offsets, kMaxCandidates)
  ALLOC(queries, kMaxCandidates * kSources)
  ALLOC(distances, kMaxCandidates * kSources)
  ALLOC(p90, kMaxCandidates)
  ALLOC(p99, kMaxCandidates)
#undef ALLOC
  timings->setup_ms = elapsed_ms(setup_begin, Clock::now());

  auto boxes_begin = Clock::now();
  make_boxes<<<(point_count + kBlockSize - 1) / kBlockSize, kBlockSize, 0,
               context->stream>>>(
      context->boxes, context->points, point_count);
  timings->boxes_ms = elapsed_ms(boxes_begin, Clock::now());

  auto build_begin = Clock::now();
  cuBQL::gpuBuilder(context->bvh, context->boxes,
                    static_cast<uint32_t>(point_count),
                    cuBQL::BuildConfig(FAST_SYMMETRY_LEAF_THRESHOLD),
                    context->stream, context->memory);
  if (!checked(cudaStreamSynchronize(context->stream), "gpuBuilder")) {
    release(context);
    return nullptr;
  }
  timings->build_ms = elapsed_ms(build_begin, Clock::now());
  timings->total_ms = elapsed_ms(total_begin, Clock::now());
  return context;
}

void *spatial_cubql_build_device(uintptr_t device_points, int point_count,
                                 int dimension, uintptr_t stream_handle,
                                 Timings *timings) {
  return build_common(reinterpret_cast<const float *>(device_points), point_count,
                      dimension, true, stream_handle, timings);
}

// Exact score-only verifier path. All operations are ordered on one stream,
// and only the two linear quantiles leave the device. This removes redundant
// host synchronizations and the candidate-by-source distance copy.
int spatial_cubql_query_quantiles(void *opaque, const int *host_source_indices,
                                  const float *host_normals,
                                  const float *host_offsets,
                                  int candidate_count, float diagonal,
                                  float *host_p99, float *host_p90,
                                  Timings *timings) {
  error_message[0] = '\0';
  Context *context = reinterpret_cast<Context *>(opaque);
  if (!context || !host_source_indices || !host_normals || !host_offsets ||
      !host_p99 || !host_p90 || !timings || candidate_count <= 0 ||
      candidate_count > kMaxCandidates || !(diagonal > 0.f)) {
    std::snprintf(error_message, sizeof(error_message), "invalid query arguments");
    return -1;
  }
  for (int i = 0; i < kSources; ++i) {
    if (host_source_indices[i] < 0 || host_source_indices[i] >= context->point_count) {
      std::snprintf(error_message, sizeof(error_message),
                    "source index %d out of range", i);
      return -1;
    }
  }
  std::memset(timings, 0, sizeof(*timings));
  auto total_begin = Clock::now();
  if (!checked(cudaMemcpyAsync(context->source_indices, host_source_indices,
                               kSources * sizeof(int), cudaMemcpyHostToDevice,
                               context->stream), "copy source indices") ||
      !checked(cudaMemcpyAsync(context->normals, host_normals,
                               candidate_count * sizeof(Vec3),
                               cudaMemcpyHostToDevice, context->stream),
               "copy normals") ||
      !checked(cudaMemcpyAsync(context->offsets, host_offsets,
                               candidate_count * sizeof(float),
                               cudaMemcpyHostToDevice, context->stream),
               "copy offsets"))
    return -1;

  int query_count = candidate_count * kSources;
  reflect_sources<<<(query_count + kBlockSize - 1) / kBlockSize, kBlockSize, 0,
                      context->stream>>>(
      context->queries, context->points, context->source_indices,
      context->normals, context->offsets, candidate_count, kSources);
  exact_nearest<<<(query_count + kBlockSize - 1) / kBlockSize, kBlockSize, 0,
                    context->stream>>>(
      context->distances, nullptr, context->queries,
      context->source_indices, context->points, context->bvh,
      context->point_count, query_count, 1.f / diagonal, kSources);
  linear_p90_p99<<<candidate_count, kBlockSize, 0, context->stream>>>(
      context->p90, context->p99, context->distances, candidate_count);
  if (!checked(cudaMemcpyAsync(host_p99, context->p99,
                               candidate_count * sizeof(float),
                               cudaMemcpyDeviceToHost, context->stream),
               "copy p99") ||
      !checked(cudaMemcpyAsync(host_p90, context->p90,
                               candidate_count * sizeof(float),
                               cudaMemcpyDeviceToHost, context->stream),
               "copy p90") ||
      !checked(cudaStreamSynchronize(context->stream), "exact quantile query"))
    return -1;
  timings->total_ms = elapsed_ms(total_begin, Clock::now());
  return 0;
}

void spatial_cubql_destroy(void *opaque) {
  release(reinterpret_cast<Context *>(opaque));
}

}  // extern "C"
