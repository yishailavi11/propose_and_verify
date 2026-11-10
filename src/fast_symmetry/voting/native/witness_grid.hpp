#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Occupied voxel centers approximate the surface only for witness weighting.
// Final plane verification still uses exact nearest neighbors in the full cloud.
struct WitnessGrid {
  float lower[3], upper[3], step[3];
  std::vector<uint8_t> occupancy;
};

static inline WitnessGrid build_witness_grid(
    const float* points, int point_count, int resolution) {
  WitnessGrid grid;
  float* lower = grid.lower;
  float* upper = grid.upper;
  for (int axis = 0; axis < 3; ++axis) lower[axis] = upper[axis] = points[axis];
  for (int i = 1; i < point_count; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float value = points[3 * i + axis];
      lower[axis] = std::min(lower[axis], value);
      upper[axis] = std::max(upper[axis], value);
    }
  }
  float* step = grid.step;
  for (int axis = 0; axis < 3; ++axis) {
    step[axis] = std::max((upper[axis] - lower[axis]) / resolution, 1e-7f);
  }
  const int cell_count = resolution * resolution * resolution;
  grid.occupancy.assign(cell_count, 0);
  auto& occupancy = grid.occupancy;
  for (int i = 0; i < point_count; ++i) {
    int cell[3];
    for (int axis = 0; axis < 3; ++axis) {
      cell[axis] = std::max(0, std::min(resolution - 1,
          static_cast<int>(std::floor(
              (points[3 * i + axis] - lower[axis]) / step[axis]))));
    }
    occupancy[(cell[0] * resolution + cell[1]) * resolution + cell[2]] = 1;
  }

  return grid;
}

// Reflect one witness through its pair's plane and inspect 27 adjacent cells.
// The nearest occupied-center residual becomes a continuous Gaussian weight.
static inline float witness_compatibility(
    const float* points, int index, float nx, float ny, float nz, float offset,
    const WitnessGrid& grid, int resolution, float inverse_two_tau_squared) {
  const float* lower = grid.lower;
  const float* step = grid.step;
  const auto& occupancy = grid.occupancy;
  const float sx = points[3 * index];
  const float sy = points[3 * index + 1];
  const float sz = points[3 * index + 2];
  const float signed_distance = sx * nx + sy * ny + sz * nz - offset;
  const float qx = sx - 2.0f * signed_distance * nx;
  const float qy = sy - 2.0f * signed_distance * ny;
  const float qz = sz - 2.0f * signed_distance * nz;
  const int gx = static_cast<int>(std::floor((qx - lower[0]) / step[0]));
  const int gy = static_cast<int>(std::floor((qy - lower[1]) / step[1]));
  const int gz = static_cast<int>(std::floor((qz - lower[2]) / step[2]));
  float best = std::numeric_limits<float>::infinity();
  for (int ox = -1; ox <= 1; ++ox) {
    const int x = gx + ox;
    if (x < 0 || x >= resolution) continue;
    for (int oy = -1; oy <= 1; ++oy) {
      const int y = gy + oy;
      if (y < 0 || y >= resolution) continue;
      for (int oz = -1; oz <= 1; ++oz) {
        const int z = gz + oz;
        if (z < 0 || z >= resolution) continue;
        if (!occupancy[(x * resolution + y) * resolution + z]) continue;
        const float ex = qx - (lower[0] + (x + 0.5f) * step[0]);
        const float ey = qy - (lower[1] + (y + 0.5f) * step[1]);
        const float ez = qz - (lower[2] + (z + 0.5f) * step[2]);
        best = std::min(best, ex * ex + ey * ey + ez * ez);
      }
    }
  }
  return std::isfinite(best)
      ? std::exp(-best * inverse_two_tau_squared) : 0.0f;
}
