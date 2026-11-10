#pragma once
// Stateless JAX-compatible Threefry streams. These integer operations are frozen.
#include <cstdint>
#include <algorithm>

struct Pair32 { uint32_t x; uint32_t y; };

static inline uint32_t rotate_left(uint32_t x, unsigned r) {
  return (x << r) | (x >> (32 - r));
}

static inline Pair32 threefry_2x32(Pair32 key, uint32_t x0, uint32_t x1) {
  const uint32_t keys[3] = {key.x, key.y, key.x ^ key.y ^ 0x1BD11BDAu};
  static constexpr unsigned rotation[8] = {13, 15, 26, 6, 17, 29, 16, 24};
  x0 += keys[0];
  x1 += keys[1];
  for (unsigned block = 0; block < 5; ++block) {
    for (unsigned round = 0; round < 4; ++round) {
      x0 += x1;
      x1 = rotate_left(x1, rotation[(block & 1u) * 4u + round]);
      x1 = x0 ^ x1;
    }
    x0 += keys[(block + 1) % 3];
    x1 += keys[(block + 2) % 3] + block + 1;
  }
  return {x0, x1};
}

static inline Pair32 split_key(Pair32 key, uint32_t index) {
  return threefry_2x32(key, 0u, index);
}

static inline uint32_t random_large_span(
    Pair32 low_key, uint32_t index, uint32_t span) {
  const Pair32 value = threefry_2x32(low_key, 0u, index);
  return (value.x ^ value.y) % span;
}


// Pair indices are ordered and distinct. Each witness independently excludes
// both endpoints; witnesses may equal one another, as in the production stream.
struct SampledPair {
  int32_t first, second;
  int32_t witness[2];
};

static inline SampledPair sample_pair(
    uint32_t pair, int point_count, Pair32 first_low,
    Pair32 second_low, Pair32 witness_low) {
  const int32_t first = static_cast<int32_t>(
      random_large_span(first_low, pair, point_count));
  const int32_t compressed = static_cast<int32_t>(
      random_large_span(second_low, pair, point_count - 1));
  const int32_t second = compressed + (compressed >= first);
  const int32_t lo = std::min(first, second);
  const int32_t hi = std::max(first, second);
  SampledPair sampled{first, second, {0, 0}};
  for (int w = 0; w < 2; ++w) {
    int32_t value = static_cast<int32_t>(random_large_span(
        witness_low, 2u * pair + w, point_count - 2));
    value += value >= lo;
    value += value >= hi;
    sampled.witness[w] = value;
  }
  return sampled;
}
