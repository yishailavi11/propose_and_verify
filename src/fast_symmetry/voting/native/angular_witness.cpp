// Selected FAST CPU frontend. Follow the helpers in random.hpp,
// pair_plane.hpp, witness_grid.hpp and angular_cic.hpp for the numerical stages.
#include <omp.h>
#include "random.hpp"
#include "pair_plane.hpp"
#include "witness_grid.hpp"
#include "angular_cic.hpp"

// Sum private histograms in the frozen thread order, then normalize total mass.
static void reduce_histograms(
    const std::vector<float>& private_histogram,
    const std::vector<double>& private_mass,
    int thread_count, int histogram_size, float* histogram) {
  double total_mass = 0.0;
  for (int thread = 0; thread < thread_count; ++thread) {
    total_mass += private_mass[thread];
  }
  const float inverse_mass = total_mass > 1e-30
      ? static_cast<float>(1.0 / total_mass) : 0.0f;
#pragma omp parallel for schedule(static)
  for (int bin = 0; bin < histogram_size; ++bin) {
    float value = 0.0f;
    for (int thread = 0; thread < thread_count; ++thread) {
      value += private_histogram[
          static_cast<size_t>(thread) * histogram_size + bin];
    }
    histogram[bin] = value * inverse_mass;
  }
}

// CPU frontend C ABI. Python allocates output/audit buffers and checks status.
extern "C" int witness_histogram_all_fused(
    const float* points,
    int point_count,
    uint32_t seed,
    int pair_count,
    int resolution,
    float cutoff,
    float tau,
    const float* mu_nodes,
    int n_mu,
    int n_phi,
    float* histogram,
    int32_t* audit_first,
    int32_t* audit_second,
    int32_t* audit_witnesses) {
  if (!points || !mu_nodes || !histogram || point_count < 4 || pair_count <= 0 ||
      resolution <= 0 || tau <= 0 || n_mu < 2 || n_phi <= 0) return 1;

  const WitnessGrid grid = build_witness_grid(points, point_count, resolution);
  const Pair32 seed_key = {0u, seed};
  const Pair32 first_low = split_key(split_key(seed_key, 1u), 1u);
  const Pair32 second_low = split_key(split_key(seed_key, 2u), 1u);
  const Pair32 witness_folded = threefry_2x32(seed_key, 0u, 0x57544E53u);
  const Pair32 witness_low = split_key(witness_folded, 1u);
  const int histogram_size = n_mu * n_phi;
  // Leave one of the eight allocated logical CPUs for the concurrently built
  // exact KD-tree instead of oversubscribing the allocation with 9 workers.
  const int thread_count = 7;
  std::vector<float> private_histogram(
      static_cast<size_t>(thread_count) * histogram_size, 0.0f);
  std::vector<double> private_mass(thread_count, 0.0);
  const float inverse_two_tau_squared = 1.0f / (2.0f * tau * tau);

  omp_set_dynamic(0);
  omp_set_num_threads(thread_count);
#pragma omp parallel
  {
    const int thread = omp_get_thread_num();
    float* local = private_histogram.data() +
        static_cast<size_t>(thread) * histogram_size;
    double mass_sum = 0.0;
#pragma omp for schedule(static)
    for (int pair = 0; pair < pair_count; ++pair) {
      const auto sampled = sample_pair(pair, point_count, first_low, second_low, witness_low);
      if (audit_first) audit_first[pair] = sampled.first;
      if (audit_second) audit_second[pair] = sampled.second;
      if (audit_witnesses) {
        audit_witnesses[2 * pair] = sampled.witness[0];
        audit_witnesses[2 * pair + 1] = sampled.witness[1];
      }
      const auto plane = make_pair_plane(points, sampled.first, sampled.second);
      if (std::fabs(plane.offset) > cutoff) continue;
      float compatibility[2];
      for (int w = 0; w < 2; ++w) {
        compatibility[w] = witness_compatibility(
            points, sampled.witness[w], plane.nx, plane.ny, plane.nz, plane.offset,
            grid, resolution, inverse_two_tau_squared);
      }
      const float mean = 0.5f * (compatibility[0] + compatibility[1]);
      const float raw = mean * mean;
      if (!(raw > 0.0f)) continue;
      mass_sum += raw;
      deposit_angular_vote(plane.nx, plane.ny, plane.nz, plane.offset,
                           raw, mu_nodes, n_mu, n_phi, local);
    }
    private_mass[thread] = mass_sum;
  }
  reduce_histograms(private_histogram, private_mass, thread_count, histogram_size, histogram);
  omp_set_num_threads(1);
  return 0;
}
