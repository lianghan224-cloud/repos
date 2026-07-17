#ifndef SPLIT_DATASET_HPP_
#define SPLIT_DATASET_HPP_

#include <stdint.h>

#include "sptensor.hpp"

struct SplitConfig {
  double val_rate;
  uint32_t val_seed;
  double sampling_rate;
  int sampling_mode;  // 0=uniform, 1=value-biased
  double sampling_alpha;
  double sampling_eps;
  uint32_t omega_seed;
};

// Build three disjoint sets from valid entries in input tensor:
// V (validation), Omega (train), U (holdout).
//
// Returns 0 on success, non-zero on failure.
int SplitSparseTensorThreeSets(
    const SparseTensor* input,
    const SplitConfig& cfg,
    SparseTensor** out_omega,
    SparseTensor** out_v,
    SparseTensor** out_u);

#endif  // SPLIT_DATASET_HPP_
