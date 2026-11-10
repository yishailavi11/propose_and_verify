#pragma once
#include <algorithm>
#include <cmath>

// Deposit one weighted pair into four GL-latitude/longitude neighbors.
// Retain the original positive vector scale before normalization: removing it
// changes float32 rounding, although the searched coordinate is angular only.
static inline void deposit_angular_vote(
    float nx, float ny, float nz, float offset, float raw,
    const float* mu_nodes, int n_mu, int n_phi, float* local) {
  const float two_pi = 6.2831853071795864769f;
  const float sign = offset < 0.0f ? -1.0f : 1.0f;
  const float scale = std::fabs(offset) + 0.5f;
  const float vx = nx * sign * scale;
  const float vy = ny * sign * scale;
  const float vz = nz * sign * scale;
  const float radius = std::sqrt(vx * vx + vy * vy + vz * vz);
  const float mu = std::max(-1.0f, std::min(1.0f, vz / radius));
  const float* position = std::upper_bound(mu_nodes, mu_nodes + n_mu, mu);
  const int q1 = std::max(1, std::min(
      n_mu - 1, static_cast<int>(position - mu_nodes)));
  const int q0 = q1 - 1;
  const float wq1 = std::max(0.0f, std::min(
      1.0f, (mu - mu_nodes[q0]) / (mu_nodes[q1] - mu_nodes[q0])));
  const float wq0 = 1.0f - wq1;
  float phi = std::atan2(vy, vx);
  if (phi < 0.0f) phi += two_pi;
  const float longitude = phi * static_cast<float>(n_phi) / two_pi;
  const int p0 = std::min(n_phi - 1, static_cast<int>(std::floor(longitude)));
  const int p1 = (p0 + 1) % n_phi;
  const float wp1 = longitude - std::floor(longitude);
  const float wp0 = 1.0f - wp1;
  const int qq[2] = {q0, q1};
  const int pp[2] = {p0, p1};
  const float qw[2] = {wq0, wq1};
  const float pw[2] = {wp0, wp1};
  for (int iq = 0; iq < 2; ++iq) {
    for (int ip = 0; ip < 2; ++ip) {
      const int flat = qq[iq] * n_phi + pp[ip];
      local[flat] += raw * qw[iq] * pw[ip];
    }
  }
}
