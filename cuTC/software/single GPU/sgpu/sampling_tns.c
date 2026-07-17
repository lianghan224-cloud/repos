#include "sampling_tns.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// ===== deterministic RNG (MT19937 to match provided pseudocode) =====
typedef struct {
    uint32_t mt[624];
    int idx;
} mt19937_state_t;

static void mt19937_seed(mt19937_state_t* st, uint32_t seed) {
    st->mt[0] = seed;
    for (int i = 1; i < 624; ++i) {
        st->mt[i] = 1812433253U * (st->mt[i - 1] ^ (st->mt[i - 1] >> 30)) + (uint32_t)i;
    }
    st->idx = 624;
}

static void mt19937_twist(mt19937_state_t* st) {
    for (int i = 0; i < 624; ++i) {
        uint32_t y = (st->mt[i] & 0x80000000U) + (st->mt[(i + 1) % 624] & 0x7fffffffU);
        st->mt[i] = st->mt[(i + 397) % 624] ^ (y >> 1);
        if (y & 1U) {
            st->mt[i] ^= 0x9908b0dfU;
        }
    }
    st->idx = 0;
}

static uint32_t mt19937_u32(mt19937_state_t* st) {
    if (st->idx >= 624) {
        mt19937_twist(st);
    }
    uint32_t y = st->mt[st->idx++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

// Equivalent form of uniform_real_distribution<float>(0,1)
static inline float mt19937_uniform01f(mt19937_state_t* st) {
    return (float)((double)mt19937_u32(st) * (1.0 / 4294967296.0)); // 2^32
}

// ===== deterministic RNG (splitmix64) =====
static inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// uniform [0,1)
static inline double rng01(uint64_t* state) {
    // use top 53 bits -> double in [0,1)
    const uint64_t r = splitmix64_next(state);
    const uint64_t x = (r >> 11); // 53 bits
    return (double)x * (1.0 / 9007199254740992.0); // 2^53
}

// Allocate a new sptensor_t with same layout used by your codebase
static sptensor_t* alloc_like(const sptensor_t* T, idx_t nnz_new) {
    idx_t alloc_nnz = (nnz_new > 0) ? nnz_new : 1;
    sptensor_t* S = (sptensor_t*)malloc(sizeof(sptensor_t));
    if (!S) return NULL;

    S->nmodes = T->nmodes;
    S->nnz = nnz_new;

    S->dims = (idx_t*)malloc(T->nmodes * sizeof(idx_t));
    S->ind  = (idx_t**)malloc(T->nmodes * sizeof(idx_t*));
    S->vals = (double*)malloc(alloc_nnz * sizeof(double));

    if (!S->dims || !S->ind || !S->vals) {
        free(S->dims); free(S->ind); free(S->vals); free(S);
        return NULL;
    }

    for (idx_t m = 0; m < T->nmodes; ++m) {
        S->dims[m] = T->dims[m];
        S->ind[m] = (idx_t*)malloc(alloc_nnz * sizeof(idx_t));
        if (!S->ind[m]) {
            for (idx_t mm = 0; mm < m; ++mm) free(S->ind[mm]);
            free(S->dims); free(S->ind); free(S->vals); free(S);
            return NULL;
        }
    }
    return S;
}

static inline double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

static int same_shape(const sptensor_t* A, const sptensor_t* B) {
    if (!A || !B) return 1;
    if (A->nmodes != B->nmodes) return 0;
    for (idx_t m = 0; m < A->nmodes; ++m) {
        if (A->dims[m] != B->dims[m]) return 0;
    }
    return 1;
}

static void copy_nnz_at(
    sptensor_t* dst, idx_t dst_pos,
    const sptensor_t* src, idx_t src_pos)
{
    for (idx_t m = 0; m < src->nmodes; ++m) {
        dst->ind[m][dst_pos] = src->ind[m][src_pos];
    }
    dst->vals[dst_pos] = src->vals[src_pos];
}

static sptensor_t* concat_three_sets(
    const sptensor_t* train,
    const sptensor_t* validation,
    const sptensor_t* test)
{
    const sptensor_t* ref = train ? train : (validation ? validation : test);
    if (!ref) return NULL;
    if (!same_shape(ref, train) || !same_shape(ref, validation) || !same_shape(ref, test)) {
        return NULL;
    }

    idx_t n_train = train ? train->nnz : 0;
    idx_t n_val = validation ? validation->nnz : 0;
    idx_t n_test = test ? test->nnz : 0;
    idx_t n_total = n_train + n_val + n_test;

    sptensor_t* full = alloc_like(ref, n_total);
    if (!full) return NULL;

    idx_t pos = 0;
    if (train) {
        for (idx_t x = 0; x < train->nnz; ++x) {
            copy_nnz_at(full, pos++, train, x);
        }
    }
    if (validation) {
        for (idx_t x = 0; x < validation->nnz; ++x) {
            copy_nnz_at(full, pos++, validation, x);
        }
    }
    if (test) {
        for (idx_t x = 0; x < test->nnz; ++x) {
            copy_nnz_at(full, pos++, test, x);
        }
    }

    return full;
}

sptensor_t* tt_sample_uniform(const sptensor_t* T, double sampling_rate, uint32_t seed) {
    if (!T) return NULL;
    if (sampling_rate <= 0.0) {
        // return empty tensor
        sptensor_t* S = alloc_like(T, 0);
        if (S) printf("[sample] uniform: rate<=0 -> nnz=0\n");
        return S;
    }
    if (sampling_rate >= 1.0) {
        // no sampling, just return a copy (to keep ownership simple)
        sptensor_t* S = alloc_like(T, T->nnz);
        if (!S) return NULL;
        for (idx_t m = 0; m < T->nmodes; ++m) {
            for (idx_t x = 0; x < T->nnz; ++x) S->ind[m][x] = T->ind[m][x];
        }
        for (idx_t x = 0; x < T->nnz; ++x) S->vals[x] = T->vals[x];
        printf("[sample] uniform: rate>=1 -> nnz=%llu (copy)\n", (unsigned long long)T->nnz);
        return S;
    }

    // One-pass: allocate max, fill, then shrink (simpler)
    sptensor_t* S = alloc_like(T, T->nnz);
    if (!S) return NULL;

    uint64_t st = ((uint64_t)seed << 1) ^ 0x123456789abcdef0ULL;

    idx_t out = 0;
    idx_t in_nnz = T->nnz;
    for (idx_t x = 0; x < in_nnz; ++x) {
        if (rng01(&st) < sampling_rate) {
            for (idx_t m = 0; m < T->nmodes; ++m) {
                S->ind[m][out] = T->ind[m][x];
            }
            S->vals[out] = T->vals[x];
            out++;
        }
    }

    // shrink arrays
    S->nnz = out;
    S->vals = (double*)realloc(S->vals, out * sizeof(double));
    for (idx_t m = 0; m < T->nmodes; ++m) {
        S->ind[m] = (idx_t*)realloc(S->ind[m], out * sizeof(idx_t));
    }

    printf("[sample] uniform: in=%llu out=%llu (%.2f%%) rate=%.4f seed=%u\n",
           (unsigned long long)in_nnz, (unsigned long long)out,
           (in_nnz ? 100.0 * (double)out / (double)in_nnz : 0.0),
           sampling_rate, seed);
    return S;
}

sptensor_t* tt_sample_value_biased(const sptensor_t* T, double sampling_rate,
                                   double alpha, double eps, uint32_t seed)
{
    if (!T) return NULL;
    if (sampling_rate <= 0.0) {
        sptensor_t* S = alloc_like(T, 0);
        if (S) printf("[sample] value-biased: rate<=0 -> nnz=0\n");
        return S;
    }
    if (sampling_rate >= 1.0) {
        sptensor_t* S = alloc_like(T, T->nnz);
        if (!S) return NULL;
        for (idx_t m = 0; m < T->nmodes; ++m) {
            for (idx_t x = 0; x < T->nnz; ++x) S->ind[m][x] = T->ind[m][x];
        }
        for (idx_t x = 0; x < T->nnz; ++x) S->vals[x] = T->vals[x];
        printf("[sample] value-biased: rate>=1 -> nnz=%llu (copy)\n", (unsigned long long)T->nnz);
        return S;
    }

    const idx_t in_nnz = T->nnz;

    // compute avg_w over all nnz (pool is all nnz in this cuTC sparse setting)
    double sum_w = 0.0;
    for (idx_t x = 0; x < in_nnz; ++x) {
        double v = fabs(T->vals[x]);
        double w = eps + pow(v, alpha);
        sum_w += w;
    }
    double avg_w = (in_nnz > 0) ? (sum_w / (double)in_nnz) : 1.0;
    if (!(avg_w > 0.0)) avg_w = 1.0;

    sptensor_t* S = alloc_like(T, in_nnz);
    if (!S) return NULL;

    uint64_t st = ((uint64_t)seed << 1) ^ 0x0fedcba987654321ULL;

    idx_t out = 0;
    double p_min = 1e300, p_max = -1e300;

    for (idx_t x = 0; x < in_nnz; ++x) {
        double v = fabs(T->vals[x]);
        double w = eps + pow(v, alpha);
        double p = sampling_rate * (w / avg_w);
        if (p > 1.0) p = 1.0;
        if (p < 0.0) p = 0.0;
        if (p < p_min) p_min = p;
        if (p > p_max) p_max = p;

        if (rng01(&st) < p) {
            for (idx_t m = 0; m < T->nmodes; ++m) {
                S->ind[m][out] = T->ind[m][x];
            }
            S->vals[out] = T->vals[x];
            out++;
        }
    }

    S->nnz = out;
    S->vals = (double*)realloc(S->vals, out * sizeof(double));
    for (idx_t m = 0; m < T->nmodes; ++m) {
        S->ind[m] = (idx_t*)realloc(S->ind[m], out * sizeof(idx_t));
    }

    printf("[sample] value-biased: in=%llu out=%llu (%.2f%%) rate=%.4f alpha=%.3f eps=%.1e p=[%.6f,%.6f] seed=%u\n",
           (unsigned long long)in_nnz, (unsigned long long)out,
           (in_nnz ? 100.0 * (double)out / (double)in_nnz : 0.0),
           sampling_rate, alpha, eps, p_min, p_max, seed);
    return S;
}

int tt_split_three_sets(
    const sptensor_t* train,
    const sptensor_t* validation,
    const sptensor_t* test,
    double val_rate,
    uint32_t val_seed,
    double sampling_rate,
    int sampling_mode,
    double sampling_alpha,
    double sampling_eps,
    uint32_t omega_seed,
    sptensor_t** out_omega,
    sptensor_t** out_v,
    sptensor_t** out_u)
{
    if (!out_omega || !out_v || !out_u) return -1;
    *out_omega = NULL;
    *out_v = NULL;
    *out_u = NULL;

    sptensor_t* full = concat_three_sets(train, validation, test);
    if (!full) {
        printf("[split3] failed: unable to build valid-set union (shape mismatch or OOM)\n");
        return -1;
    }

    const idx_t N = full->nnz;
    val_rate = clamp01(val_rate);
    sampling_rate = clamp01(sampling_rate);

    // No valid points.
    if (N == 0) {
        *out_omega = alloc_like(full, 0);
        *out_v = alloc_like(full, 0);
        *out_u = alloc_like(full, 0);
        tt_free(full);
        if (!(*out_omega) || !(*out_v) || !(*out_u)) {
            if (*out_omega) tt_free(*out_omega);
            if (*out_v) tt_free(*out_v);
            if (*out_u) tt_free(*out_u);
            *out_omega = NULL;
            *out_v = NULL;
            *out_u = NULL;
            return -1;
        }
        printf("[split3] valid=0 V=0 Omega=0 U=0\n");
        return 0;
    }

    unsigned char* val_mask = (unsigned char*)calloc((size_t)N, sizeof(unsigned char));
    unsigned char* omega_mask = (unsigned char*)calloc((size_t)N, sizeof(unsigned char));
    if (!val_mask || !omega_mask) {
        free(val_mask);
        free(omega_mask);
        tt_free(full);
        return -1;
    }

    // Step 1: sample V from valid with val_seed.
    mt19937_state_t rngV;
    mt19937_seed(&rngV, val_seed);
    for (idx_t x = 0; x < N; ++x) {
        if (mt19937_uniform01f(&rngV) < (float)val_rate) {
            val_mask[x] = 1;
        }
    }

    // Step 3: sample Omega from train_pool = valid \ V with omega_seed.
    mt19937_state_t rngO;
    mt19937_seed(&rngO, omega_seed);

    if (sampling_mode == 0) {
        for (idx_t x = 0; x < N; ++x) {
            if (val_mask[x]) continue;
            if (mt19937_uniform01f(&rngO) < (float)sampling_rate) {
                omega_mask[x] = 1;
            }
        }
    } else {
        double sum_w = 0.0;
        idx_t pool_cnt = 0;
        for (idx_t x = 0; x < N; ++x) {
            if (val_mask[x]) continue;
            double w = sampling_eps + pow(fabs(full->vals[x]), sampling_alpha);
            sum_w += w;
            pool_cnt++;
        }
        double avg_w = (pool_cnt > 0) ? (sum_w / (double)pool_cnt) : 1.0;
        if (!(avg_w > 0.0)) avg_w = 1.0;

        for (idx_t x = 0; x < N; ++x) {
            if (val_mask[x]) continue;
            double w = sampling_eps + pow(fabs(full->vals[x]), sampling_alpha);
            double p = sampling_rate * (w / avg_w);
            if (p > 1.0) p = 1.0;
            if (p < 0.0) p = 0.0;
            if (mt19937_uniform01f(&rngO) < (float)p) {
                omega_mask[x] = 1;
            }
        }
    }

    // Step 4: construct V/Omega/U.
    idx_t cnt_v = 0, cnt_o = 0, cnt_u = 0;
    for (idx_t x = 0; x < N; ++x) {
        if (val_mask[x]) {
            cnt_v++;
        } else if (omega_mask[x]) {
            cnt_o++;
        } else {
            cnt_u++;
        }
    }

    sptensor_t* V = alloc_like(full, cnt_v);
    sptensor_t* O = alloc_like(full, cnt_o);
    sptensor_t* U = alloc_like(full, cnt_u);
    if (!V || !O || !U) {
        if (V) tt_free(V);
        if (O) tt_free(O);
        if (U) tt_free(U);
        free(val_mask);
        free(omega_mask);
        tt_free(full);
        return -1;
    }

    idx_t iv = 0, io = 0, iu = 0;
    for (idx_t x = 0; x < N; ++x) {
        if (val_mask[x]) {
            copy_nnz_at(V, iv++, full, x);
        } else if (omega_mask[x]) {
            copy_nnz_at(O, io++, full, x);
        } else {
            copy_nnz_at(U, iu++, full, x);
        }
    }

    free(val_mask);
    free(omega_mask);
    tt_free(full);

    *out_omega = O;
    *out_v = V;
    *out_u = U;

    printf("[split3] val_rate=%.6f val_seed=%u sampling_rate=%.6f mode=%d alpha=%.6f eps=%.1e omega_seed=%u\n",
           val_rate, val_seed, sampling_rate, sampling_mode, sampling_alpha, sampling_eps, omega_seed);
    printf("[split3] valid=%llu V=%llu Omega=%llu U=%llu\n",
           (unsigned long long)N,
           (unsigned long long)cnt_v,
           (unsigned long long)cnt_o,
           (unsigned long long)cnt_u);
    printf("[split3] ratio(valid): V=%.6f Omega=%.6f U=%.6f\n",
           (double)cnt_v / (double)N,
           (double)cnt_o / (double)N,
           (double)cnt_u / (double)N);

    return 0;
}
