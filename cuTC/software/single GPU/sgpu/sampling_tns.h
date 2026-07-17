#ifndef SAMPLING_TNS_H
#define SAMPLING_TNS_H

#include <stdint.h>
#include "sptensor.h"

// mode: 0=uniform, 1=value-biased
sptensor_t* tt_sample_uniform(const sptensor_t* T, double sampling_rate, uint32_t seed);
sptensor_t* tt_sample_value_biased(const sptensor_t* T, double sampling_rate,
                                   double alpha, double eps, uint32_t seed);

// Build three disjoint sets from valid entries:
// V (validation), Omega (train), U (holdout)
// following the provided Bernoulli split logic.
int tt_split_three_sets(
    const sptensor_t* train,
    const sptensor_t* validation,
    const sptensor_t* test,
    double val_rate,
    uint32_t val_seed,
    double sampling_rate,
    int sampling_mode,        // 0=uniform, 1=value-biased
    double sampling_alpha,
    double sampling_eps,
    uint32_t omega_seed,
    sptensor_t** out_omega,
    sptensor_t** out_v,
    sptensor_t** out_u
);

#endif
