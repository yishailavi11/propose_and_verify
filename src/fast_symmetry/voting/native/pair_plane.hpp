#pragma once
#include <cmath>

// A pair-induced reflection equation in normalized coordinates.
struct PairPlane { float nx, ny, nz, offset; };

static inline PairPlane make_pair_plane(const float* points, int first, int second) {
  const float ax = points[3 * first];
  const float ay = points[3 * first + 1];
  const float az = points[3 * first + 2];
  const float bx = points[3 * second];
  const float by = points[3 * second + 1];
  const float bz = points[3 * second + 2];
  const float dx = bx - ax;
  const float dy = by - ay;
  const float dz = bz - az;
  const float length = std::sqrt(dx * dx + dy * dy + dz * dz + 1e-16f);
  const float nx = dx / length;
  const float ny = dy / length;
  const float nz = dz / length;
  const float offset = 0.5f * (ax + bx) * nx
      + 0.5f * (ay + by) * ny + 0.5f * (az + bz) * nz;
  return {nx, ny, nz, offset};
}
