#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <random>
#include <vector>

#include "split_dataset.hpp"

namespace {

static inline double clamp01(double x) {
  if (x < 0.0) return 0.0;
  if (x > 1.0) return 1.0;
  return x;
}

static SparseTensor* AllocateLike(const SparseTensor* src, IType nnz_new) {
  SparseTensor* out = (SparseTensor*)AlignedMalloc(sizeof(SparseTensor));
  if (!out) return NULL;

  out->nmodes = src->nmodes;
  out->nnz = nnz_new;
  out->dims = (IType*)AlignedMalloc(sizeof(IType) * src->nmodes);
  out->cidx = (IType**)AlignedMalloc(sizeof(IType*) * src->nmodes);
  out->vals = (FType*)AlignedMalloc(sizeof(FType) * std::max<IType>(nnz_new, 1));

  if (!out->dims || !out->cidx || !out->vals) {
    if (out->dims) AlignedFree(out->dims);
    if (out->cidx) AlignedFree(out->cidx);
    if (out->vals) AlignedFree(out->vals);
    AlignedFree(out);
    return NULL;
  }

  for (int m = 0; m < src->nmodes; ++m) {
    out->dims[m] = src->dims[m];
    out->cidx[m] = (IType*)AlignedMalloc(sizeof(IType) * std::max<IType>(nnz_new, 1));
    if (!out->cidx[m]) {
      for (int mm = 0; mm < m; ++mm) AlignedFree(out->cidx[mm]);
      AlignedFree(out->vals);
      AlignedFree(out->cidx);
      AlignedFree(out->dims);
      AlignedFree(out);
      return NULL;
    }
  }
  return out;
}

static inline void CopyEntry(SparseTensor* dst, IType dst_pos, const SparseTensor* src,
                             IType src_pos) {
  for (int m = 0; m < src->nmodes; ++m) {
    dst->cidx[m][dst_pos] = src->cidx[m][src_pos];
  }
  dst->vals[dst_pos] = src->vals[src_pos];
}

}  // namespace

int SplitSparseTensorThreeSets(const SparseTensor* input, const SplitConfig& cfg,
                               SparseTensor** out_omega, SparseTensor** out_v,
                               SparseTensor** out_u) {
  if (!input || !out_omega || !out_v || !out_u) return -1;

  *out_omega = NULL;
  *out_v = NULL;
  *out_u = NULL;

  const IType N = input->nnz;
  const double val_rate = clamp01(cfg.val_rate);
  const double sampling_rate = clamp01(cfg.sampling_rate);
  const int sampling_mode = cfg.sampling_mode;
  const double sampling_alpha = cfg.sampling_alpha;
  const double sampling_eps = cfg.sampling_eps;

  if (N == 0) {
    SparseTensor* empty_omega = AllocateLike(input, 0);
    SparseTensor* empty_v = AllocateLike(input, 0);
    SparseTensor* empty_u = AllocateLike(input, 0);
    if (!empty_omega || !empty_v || !empty_u) {
      if (empty_omega) DestroySparseTensor(empty_omega);
      if (empty_v) DestroySparseTensor(empty_v);
      if (empty_u) DestroySparseTensor(empty_u);
      return -1;
    }
    *out_omega = empty_omega;
    *out_v = empty_v;
    *out_u = empty_u;
    printf("[split3] valid=0 V=0 Omega=0 U=0\n");
    return 0;
  }

  std::vector<uint8_t> val_mask((size_t)N, 0);
  std::vector<uint8_t> train_mask((size_t)N, 0);

  // Step 1: sample V
  std::mt19937 rngV(cfg.val_seed);
  std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
  for (IType idx = 0; idx < N; ++idx) {
    if (dist01(rngV) < val_rate) {
      val_mask[(size_t)idx] = 1;
    }
  }

  // Step 3: sample Omega from train_pool = valid \ V
  std::mt19937 rngO(cfg.omega_seed);
  std::uniform_real_distribution<float> dist02(0.0f, 1.0f);
  if (sampling_mode == 0) {
    for (IType idx = 0; idx < N; ++idx) {
      if (val_mask[(size_t)idx]) continue;
      if (dist02(rngO) < sampling_rate) {
        train_mask[(size_t)idx] = 1;
      }
    }
  } else {
    double sum_w = 0.0;
    IType pool_cnt = 0;
    for (IType idx = 0; idx < N; ++idx) {
      if (val_mask[(size_t)idx]) continue;
      const double w = sampling_eps + pow(fabs((double)input->vals[idx]), sampling_alpha);
      sum_w += w;
      ++pool_cnt;
    }
    double avg_w = (pool_cnt > 0) ? (sum_w / (double)pool_cnt) : 1.0;
    if (avg_w <= 0.0) avg_w = 1.0;

    for (IType idx = 0; idx < N; ++idx) {
      if (val_mask[(size_t)idx]) continue;
      const double w = sampling_eps + pow(fabs((double)input->vals[idx]), sampling_alpha);
      float p = (float)(sampling_rate * (w / avg_w));
      if (p > 1.0f) p = 1.0f;
      if (p < 0.0f) p = 0.0f;
      if (dist02(rngO) < p) {
        train_mask[(size_t)idx] = 1;
      }
    }
  }

  // Step 4: build V / Omega / U
  IType cnt_v = 0;
  IType cnt_o = 0;
  IType cnt_u = 0;
  for (IType idx = 0; idx < N; ++idx) {
    if (val_mask[(size_t)idx]) {
      ++cnt_v;
    } else if (train_mask[(size_t)idx]) {
      ++cnt_o;
    } else {
      ++cnt_u;
    }
  }

  SparseTensor* V = AllocateLike(input, cnt_v);
  SparseTensor* O = AllocateLike(input, cnt_o);
  SparseTensor* U = AllocateLike(input, cnt_u);
  if (!V || !O || !U) {
    if (V) DestroySparseTensor(V);
    if (O) DestroySparseTensor(O);
    if (U) DestroySparseTensor(U);
    return -1;
  }

  IType iv = 0;
  IType io = 0;
  IType iu = 0;
  for (IType idx = 0; idx < N; ++idx) {
    if (val_mask[(size_t)idx]) {
      CopyEntry(V, iv++, input, idx);
    } else if (train_mask[(size_t)idx]) {
      CopyEntry(O, io++, input, idx);
    } else {
      CopyEntry(U, iu++, input, idx);
    }
  }

  assert(iv == cnt_v && io == cnt_o && iu == cnt_u);

  printf(
      "[split3] val_rate=%.6f val_seed=%u sampling_rate=%.6f mode=%d alpha=%.6f "
      "eps=%.1e omega_seed=%u\n",
      val_rate, cfg.val_seed, sampling_rate, sampling_mode, sampling_alpha, sampling_eps,
      cfg.omega_seed);
  printf("[split3] valid=%llu V=%llu Omega=%llu U=%llu\n",
         (unsigned long long)N, (unsigned long long)cnt_v, (unsigned long long)cnt_o,
         (unsigned long long)cnt_u);
  printf("[split3] ratio(valid): V=%.6f Omega=%.6f U=%.6f\n",
         (double)cnt_v / (double)N, (double)cnt_o / (double)N, (double)cnt_u / (double)N);

  *out_omega = O;
  *out_v = V;
  *out_u = U;
  return 0;
}
