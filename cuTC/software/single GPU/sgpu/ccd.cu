extern "C"
{
#include "completion.h"
#include "ciss.h"
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
}

#include "ccd.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
//#include "loss.h"

static inline long long p_walltime_us_now()
{
  struct timeval now;
  gettimeofday(&now, NULL);
  return 1000000LL * (long long)now.tv_sec + (long long)now.tv_usec;
}

static inline long long p_cuda_elapsed_us(cudaEvent_t start_event, cudaEvent_t stop_event)
{
  float millis = 0.0f;
  HANDLE_ERROR(cudaEventSynchronize(stop_event));
  HANDLE_ERROR(cudaEventElapsedTime(&millis, start_event, stop_event));
  if(millis < 0.0f) {
    return 0;
  }
  return (long long)(millis * 1000.0f + 0.5f);
}

static int p_debug_ccd_stats_enabled(void)
{
  const char* env = getenv("CUTC_CCD_DEBUG_STATS");
  return env != NULL && env[0] != '\0' && env[0] != '0';
}

static void p_debug_matrix_stats(ordi_matrix ** mats, idx_t nmodes, idx_t epoch)
{
  if(!p_debug_ccd_stats_enabled()) {
    return;
  }

  for(idx_t m = 0; m < nmodes; ++m) {
    idx_t n = mats[m]->I * DEFAULT_NFACTORS;
    double norm = 0.0;
    double minv = 0.0;
    double maxv = 0.0;
    idx_t nonzero = 0;
    idx_t finite = 0;
    for(idx_t i = 0; i < n; ++i) {
      double v = mats[m]->values[i];
      if(isfinite(v)) {
        finite++;
      }
      if(i == 0 || v < minv) {
        minv = v;
      }
      if(i == 0 || v > maxv) {
        maxv = v;
      }
      if(v != 0.0) {
        nonzero++;
      }
      norm += v * v;
    }
    printf("[ccd-debug] epoch:%llu mode:%llu rows:%llu nonzero:%llu finite:%llu norm:%0.6e min:%0.6e max:%0.6e\n",
           (unsigned long long)epoch,
           (unsigned long long)m,
           (unsigned long long)mats[m]->I,
           (unsigned long long)nonzero,
           (unsigned long long)finite,
           sqrt(norm),
           minv,
           maxv);
  }
}



/**
* @brief Transpose a model's factor matrices.
*
* @param model The model to transpose.
*/
static void p_transpose_model(
    idx_t nmodes,
    ordi_matrix ** mats)
{
  double * buf = mats[MAX_NMODES]->values;

  for(idx_t m=0; m < nmodes; ++m) {
    idx_t const nrows = mats[m]->I;
    double * factor = mats[m]->values;
    idx_t const ncols = mats[m]->J;
    for(idx_t j=0; j < ncols; ++j) {
      for(idx_t i=0; i < nrows; ++i) {
        buf[i + (j*nrows)] = factor[j + (i*ncols)];
      }
    }

    memcpy(factor, buf, nrows * ncols * sizeof(*factor));
  }

}

//gpu kernels
/**
 * @brief Compute the loss
 * @version Now contains the segment scan
*/
__global__ void update_residual_gpu(ciss_t * d_traina,
                                    idx_t tilenum,
                                    double* loss)
{
  //__shared__ double accum[DEFAULT_BLOCKSIZE];

  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  double * entries = d_traina->entries;

  double localloss = 0;
  if(tileid < tilenum)
  {
    idx_t localtile = tileid * DEFAULT_T_TILE_LENGTH * DEFAULT_T_TILE_WIDTH + DEFAULT_T_TILE_WIDTH;
    for(idx_t i = 0; i<DEFAULT_T_TILE_LENGTH; i++)
    {
      if(entries[localtile] < 0 && entries[localtile+1]<0) break;
      localloss+= entries[localtile + 2];
      localtile+= DEFAULT_T_TILE_WIDTH;
    }
    atomicAdd(loss, localloss);
  }


}


/**
 * @brief Compute the nomin and denomin of the fraction with warp shuffle
 * @version Now reduces the atomic operation
*/
__global__ void update_frac_gpu_as(ciss_t * d_traina,
                                   ordi_matrix * d_factora,
                                   ordi_matrix * d_factorb,
                                   ordi_matrix * d_factorc,
                                   double * d_nominbuffer,
                                   double * d_denominbuffer,
                                   idx_t tilenum)
{
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t warpid = tid / ((idx_t)CCD_WARPSIZE);
  idx_t laneid = tid % ((idx_t)CCD_WARPSIZE);
  idx_t tileid = bid * ((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)CCD_WARPSIZE) + warpid;
  double * entries = d_traina -> entries;
  idx_t localtile = tileid * ((DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH);
  idx_t global_nnz = tileid * DEFAULT_T_TILE_LENGTH + laneid;
  int valid = (tileid < tilenum) && (global_nnz < d_traina->nnz);
  idx_t myfid = 0;
  idx_t tmpi = 0;
  double residual = 0.0;
  double __align__(256)  localmbuffer[DEFAULT_NFACTORS];

  for(int m = 0; m < DEFAULT_NFACTORS; m++) {
    localmbuffer[m] = 0.0;
  }

  if(valid) {
    idx_t f_id = (idx_t)(entries[localtile] * (-1));
    idx_t bitmap = (idx_t)(entries[localtile+2]);

    bitmap = __brevll(bitmap);
    while((bitmap & 1) == 0) {bitmap = bitmap >> 1;}
    bitmap = bitmap >> 1;
    myfid = f_id + laneid - __popcll((bitmap << (63-laneid))) + 1;

    double other0 = entries[localtile + (laneid + 1) * DEFAULT_T_TILE_WIDTH];
    double other1 = entries[localtile + (laneid + 1) * DEFAULT_T_TILE_WIDTH + 1];
    residual = entries[localtile + (laneid + 1) * DEFAULT_T_TILE_WIDTH + 2];

    tmpi = d_traina->directory[myfid] - 1;
    idx_t b = (idx_t)other0 - 1;
    idx_t c = (idx_t)other1 - 1;

    for(int m = 0; m < DEFAULT_NFACTORS; m++) {
      localmbuffer[m] = d_factorb->values[b * DEFAULT_NFACTORS + m] *
                        d_factorc->values[c * DEFAULT_NFACTORS + m];
      residual -= d_factora->values[tmpi * DEFAULT_NFACTORS + m] * localmbuffer[m];
    }
  }

  unsigned int const mask = 0xffffffffU;
  int next_valid = __shfl_down_sync(mask, valid, 1, (int)CCD_WARPSIZE);
  idx_t next_fid = __shfl_down_sync(mask, myfid, 1, (int)CCD_WARPSIZE);
  int segment_end = valid && (laneid == CCD_WARPSIZE - 1 || !next_valid || next_fid != myfid);

  if(segment_end) {
    for(int m = 0; m < DEFAULT_NFACTORS; m++) {
      double denomin_sum = 0.0;
      double nomin_sum = 0.0;
      for(int lane = 0; lane < CCD_WARPSIZE; lane++) {
        int src_valid = __shfl_sync(mask, valid, lane, (int)CCD_WARPSIZE);
        idx_t src_fid = __shfl_sync(mask, myfid, lane, (int)CCD_WARPSIZE);
        double src_prod = __shfl_sync(mask, localmbuffer[m], lane, (int)CCD_WARPSIZE);
        double src_residual = __shfl_sync(mask, residual, lane, (int)CCD_WARPSIZE);
        double src_current = __shfl_sync(
            mask, d_factora->values[tmpi * DEFAULT_NFACTORS + m], lane, (int)CCD_WARPSIZE);
        if(src_valid && src_fid == myfid) {
          denomin_sum += src_prod * src_prod;
          nomin_sum += (src_residual + src_current * src_prod) * src_prod;
        }
      }
      atomicAdd(&(d_denominbuffer[myfid * DEFAULT_NFACTORS + m]), denomin_sum);
      atomicAdd(&(d_nominbuffer[myfid * DEFAULT_NFACTORS + m]), nomin_sum);
    }
  }

}

/**
 * @brief Compute the nomin and denomin of the fraction
 * @version Now contains the atomic operation
*/
__global__ void update_frac_gpu(ciss_t * d_traina,
                                ordi_matrix * d_factora,
                                ordi_matrix * d_factorb,
                                ordi_matrix * d_factorc,
                                double * d_nominbuffer,
                                double * d_denominbuffer,
                                idx_t tilenum)
{
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  double * entries = d_traina -> entries;
  idx_t localtile = tileid*((DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH);
  double __align__(64) localloss[2] = {0, 0};
  double __align__(256) localtbuffer[6];
  double __align__(256) localmbuffer[2 * DEFAULT_NFACTORS];
  idx_t b, c;

  if(tileid < tilenum)
  {
    //get the indices and value
    idx_t f_id = (idx_t)(entries[localtile] * (-1));
    idx_t l_id = (idx_t)(entries[localtile+1] * (-1));
    idx_t bitmap = (idx_t)(entries[localtile+2]);
    bitmap = __brevll(bitmap);
    while((bitmap & 1) == 0) {bitmap = bitmap >> 1;}
    bitmap = bitmap >> 1;
    localtile += DEFAULT_T_TILE_WIDTH;

    for(idx_t j = 0; j < DEFAULT_T_TILE_LENGTH/2; j++)
    {
      //unroll loop and load
      localtbuffer[0] = entries[localtile];
      localtbuffer[1] = entries[localtile + 1];
      localtbuffer[2] = entries[localtile + 2];
      localtbuffer[3] = entries[localtile + 3];
      localtbuffer[4] = entries[localtile + 4];
      localtbuffer[5] = entries[localtile + 5];

      //for the first
      f_id += (!(bitmap & 1));
      bitmap = bitmap >> 1;
      idx_t tmpi = d_traina->directory[f_id] - 1;
      b = (idx_t)localtbuffer[0] - 1;
      c = (idx_t)localtbuffer[1] - 1;
      localloss[0] = localtbuffer[2];
      if(localtbuffer[0] == -1 && localtbuffer[1] == -1) break;
      //load the factor matrices
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        ((double2*)localmbuffer)[i] = ((double2*)d_factorb->values)[(b * DEFAULT_NFACTORS) / 2 + i];
        ((double2*)localmbuffer)[i + DEFAULT_NFACTORS / 2] = ((double2*)d_factorc->values)[(c * DEFAULT_NFACTORS)/2 + i];
      }
      //compute the loss and denomin
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        localloss[0] -= (d_factora->values)[(tmpi * DEFAULT_NFACTORS) + i] * localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS];
        double denomin = (localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS]) * (localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS]);
        atomicAdd(&(d_denominbuffer[f_id * DEFAULT_NFACTORS + i]), denomin);
      }
      //compute the nomin
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        double prod = localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS];
        double nomin = (localloss[0] + (d_factora->values)[(tmpi * DEFAULT_NFACTORS) + i] * prod) * prod;
        atomicAdd(&(d_nominbuffer[f_id * DEFAULT_NFACTORS + i]), nomin);
      }

      //for the second
      f_id += (!(bitmap & 1));
      bitmap = bitmap >> 1;
      b = (idx_t)localtbuffer[3] -1 ;
      c = (idx_t)localtbuffer[4] - 1;
      tmpi = d_traina->directory[f_id] - 1;
      localloss[1] = localtbuffer[5];
      if(localtbuffer[3] == -1 && localtbuffer[4] == -1) break;
      //load the factor matrices
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        ((double2*)localmbuffer)[i] = ((double2*)d_factorb->values)[(b * DEFAULT_NFACTORS) / 2 + i];
        ((double2*)localmbuffer)[i + DEFAULT_NFACTORS / 2] = ((double2*)d_factorc->values)[(c * DEFAULT_NFACTORS)/2 + i];
      }
      //compute the loss and denomin
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        localloss[1] -= (d_factora->values)[(tmpi * DEFAULT_NFACTORS) + i]* localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS];
        double denomin = (localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS]) * (localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS]);
        atomicAdd(&(d_denominbuffer[f_id * DEFAULT_NFACTORS + i]), denomin);
      }
      //compute the nomin
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        double prod = localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS];
        double nomin = (localloss[1] + (d_factora->values)[(tmpi * DEFAULT_NFACTORS) + i] * prod) * prod;
        atomicAdd(&(d_nominbuffer[f_id * DEFAULT_NFACTORS + i]), nomin);
      }
      localtile += 2 * DEFAULT_T_TILE_WIDTH;
    }

  }

}


/**
 * @brief Finally update the column for factor matrices
 * @version preliminary
*/
__global__ void update_ccd_gpu(ciss_t * d_traina,
                               ordi_matrix * d_factora,
                               double * d_nominbuffer,
                               double * d_denominbuffer,
                               idx_t  dlength,
                               double regularization_index)
{
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  double __align__(256) nomin[DEFAULT_NFACTORS];
  double __align__(256) denomin[DEFAULT_NFACTORS];
  double * value = d_factora->values;

  if(tileid < dlength)
  {
    idx_t localtile = tileid * DEFAULT_NFACTORS;
    idx_t localid = d_traina->directory[tileid] - 1;
    for(idx_t i = 0; i<DEFAULT_NFACTORS;i++)
    {
      //((double2*)nomin)[i] = ((double2*)d_nominbuffer)[(localtile)/2 + i];
      //((double2*)denomin)[i] = ((double2*)d_denominbuffer)[localtile/2 + i];
      nomin[i] = d_nominbuffer[localtile + i];
      denomin[i] = d_denominbuffer[localtile + i];
    }
    for(idx_t i = 0; i<DEFAULT_NFACTORS;i++)
    {
      value[localid * DEFAULT_NFACTORS + i] = (nomin[i])/(regularization_index+denomin[i]);
    }
  }
}

__global__ void update_frac_raw_gpu(
    idx_t nnz,
    idx_t const * ind0,
    idx_t const * ind1,
    idx_t const * ind2,
    double const * vals,
    int mode,
    ordi_matrix * d_factora,
    ordi_matrix * d_factorb,
    ordi_matrix * d_factorc,
    double * d_nominbuffer,
    double * d_denominbuffer)
{
  idx_t x = ((idx_t)blockIdx.x) * DEFAULT_BLOCKSIZE + threadIdx.x;
  if(x >= nnz) {
    return;
  }

  idx_t i = ind0[x] - 1;
  idx_t j = ind1[x] - 1;
  idx_t k = ind2[x] - 1;
  double * A = d_factora->values + i * DEFAULT_NFACTORS;
  double * B = d_factorb->values + j * DEFAULT_NFACTORS;
  double * C = d_factorc->values + k * DEFAULT_NFACTORS;

  double pred = 0.0;
  for(int f = 0; f < DEFAULT_NFACTORS; ++f) {
    pred += A[f] * B[f] * C[f];
  }
  double residual = vals[x] - pred;

  idx_t row = i;
  if(mode == 1) {
    row = j;
  } else if(mode == 2) {
    row = k;
  }

  for(int f = 0; f < DEFAULT_NFACTORS; ++f) {
    double prod = B[f] * C[f];
    double current = A[f];
    if(mode == 1) {
      prod = C[f] * A[f];
      current = B[f];
    } else if(mode == 2) {
      prod = A[f] * B[f];
      current = C[f];
    }
    idx_t pos = row * DEFAULT_NFACTORS + f;
    atomicAdd(d_denominbuffer + pos, prod * prod);
    atomicAdd(d_nominbuffer + pos, (residual + current * prod) * prod);
  }
}

__global__ void update_ccd_raw_gpu(
    ordi_matrix * d_factor,
    double const * d_nominbuffer,
    double const * d_denominbuffer,
    idx_t nrows,
    double regularization_index)
{
  idx_t row = ((idx_t)blockIdx.x) * DEFAULT_BLOCKSIZE + threadIdx.x;
  if(row >= nrows) {
    return;
  }

  double * value = d_factor->values + row * DEFAULT_NFACTORS;
  idx_t localtile = row * DEFAULT_NFACTORS;
  for(int f = 0; f < DEFAULT_NFACTORS; ++f) {
    double denom = d_denominbuffer[localtile + f];
    double nomin = d_nominbuffer[localtile + f];
    if(denom > 0.0 && isfinite(denom) && isfinite(nomin)) {
      value[f] = nomin / (regularization_index + denom);
    } else {
      value[f] = 0.0;
    }
  }
}

static void p_tc_ccd_raw(
            sptensor_t * traina,
            sptensor_t * validation,
            sptensor_t * test,
            ordi_matrix ** mats,
            ordi_matrix ** best_mats,
            int algorithm_index,
            long long project_start_us,
            double regularization_index,
            double * best_rmse,
            double * tolerance,
            idx_t * nbadepochs,
            idx_t * bestepochs,
            idx_t * max_badepochs)
{
    idx_t const nmodes = traina->nmodes;
    struct timeval start;
    struct timeval end;
    idx_t diff;

    cudaSetDevice(0);

    idx_t nnz = traina->nnz;
    idx_t blocknum_m = nnz / ((idx_t)DEFAULT_BLOCKSIZE) + 1;
    idx_t max_rows = SS_MAX(SS_MAX(mats[0]->I, mats[1]->I), mats[2]->I);

    idx_t * d_ind0 = NULL, * d_ind1 = NULL, * d_ind2 = NULL;
    double * d_vals = NULL;
    double * d_nominbuffer = NULL, * d_denominbuffer = NULL;
    HANDLE_ERROR(cudaMalloc((void**)&d_ind0, nnz * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_ind1, nnz * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_ind2, nnz * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_vals, nnz * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_ind0, traina->ind[0], nnz * sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_ind1, traina->ind[1], nnz * sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_ind2, traina->ind[2], nnz * sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_vals, traina->vals, nnz * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)&d_nominbuffer, max_rows * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&d_denominbuffer, max_rows * DEFAULT_NFACTORS * sizeof(double)));

    ordi_matrix * d_factora = NULL, * d_factorb = NULL, * d_factorc = NULL;
    double * d_value_a = NULL, * d_value_b = NULL, * d_value_c = NULL;
    double * h_values = NULL;

    HANDLE_ERROR(cudaMalloc((void**)&d_factora, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    h_values = mats[0]->values;
    mats[0]->values = d_value_a;
    HANDLE_ERROR(cudaMemcpy(d_factora, mats[0], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[0]->values = h_values;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorb, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_b, mats[1]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    h_values = mats[1]->values;
    mats[1]->values = d_value_b;
    HANDLE_ERROR(cudaMemcpy(d_factorb, mats[1], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[1]->values = h_values;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorc, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_c, mats[2]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    h_values = mats[2]->values;
    mats[2]->values = d_value_c;
    HANDLE_ERROR(cudaMemcpy(d_factorc, mats[2], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[2]->values = h_values;

    double loss = tc_loss_sq(traina, mats, algorithm_index);
    double frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    tc_converge(traina, validation, test, mats, best_mats, algorithm_index,
                loss, frobsq, 0, nmodes, best_rmse, tolerance, nbadepochs,
                bestepochs, max_badepochs);
    p_debug_matrix_stats(mats, nmodes, 0);

    idx_t max_iterate = DEFAULT_MAX_ITERATE;
    const char* env_max_iterate = getenv("CUTC_MAX_ITERATE");
    if(env_max_iterate != NULL && env_max_iterate[0] != '\0') {
      char* endptr = NULL;
      unsigned long long parsed = strtoull(env_max_iterate, &endptr, 10);
      if(endptr != env_max_iterate && parsed > 0ULL) {
        max_iterate = (idx_t)parsed;
      }
    }

    long long total_gpu_time_us = 0;
    long long total_time_us = 0;
    cudaEvent_t train_start, train_stop;
    HANDLE_ERROR(cudaEventCreate(&train_start));
    HANDLE_ERROR(cudaEventCreate(&train_stop));
    for(idx_t e = 1; e < max_iterate + 1; ++e) {
      long long epoch_gpu_us = 0;
      long long epoch_total_us = 0;
      gettimeofday(&start, NULL);

      for(idx_t mode = 0; mode < nmodes; ++mode) {
        idx_t nrows = mats[mode]->I;
        idx_t blocknum_u = nrows / DEFAULT_BLOCKSIZE + 1;
        HANDLE_ERROR(cudaMemset(d_nominbuffer, 0, max_rows * DEFAULT_NFACTORS * sizeof(double)));
        HANDLE_ERROR(cudaMemset(d_denominbuffer, 0, max_rows * DEFAULT_NFACTORS * sizeof(double)));
        HANDLE_ERROR(cudaEventRecord(train_start, 0));
        update_frac_raw_gpu<<<blocknum_m, DEFAULT_BLOCKSIZE, 0>>>(
            nnz, d_ind0, d_ind1, d_ind2, d_vals, (int)mode,
            d_factora, d_factorb, d_factorc, d_nominbuffer, d_denominbuffer);
        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaEventRecord(train_stop, 0));
        epoch_gpu_us += p_cuda_elapsed_us(train_start, train_stop);

        HANDLE_ERROR(cudaEventRecord(train_start, 0));
        if(mode == 0) {
          update_ccd_raw_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE, 0>>>(
              d_factora, d_nominbuffer, d_denominbuffer, nrows, regularization_index);
        } else if(mode == 1) {
          update_ccd_raw_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE, 0>>>(
              d_factorb, d_nominbuffer, d_denominbuffer, nrows, regularization_index);
        } else {
          update_ccd_raw_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE, 0>>>(
              d_factorc, d_nominbuffer, d_denominbuffer, nrows, regularization_index);
        }
        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaEventRecord(train_stop, 0));
        epoch_gpu_us += p_cuda_elapsed_us(train_start, train_stop);
      }

      total_gpu_time_us += epoch_gpu_us;

      HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_b, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_c, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaDeviceSynchronize());
      p_debug_matrix_stats(mats, nmodes, e);

      gettimeofday(&end, NULL);
      diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
      epoch_total_us = diff;

      loss = tc_loss_sq(traina, mats, algorithm_index);
      frobsq = tc_frob_sq(nmodes, regularization_index, mats);
      bool converged = tc_converge(traina, validation, test, mats, best_mats,
          algorithm_index, loss, frobsq, e, nmodes, best_rmse, tolerance,
          nbadepochs, bestepochs, max_badepochs);
      total_time_us = p_walltime_us_now() - project_start_us;
      if(total_time_us < 0) {
        total_time_us = 0;
      }
      printf("epoch:%d   time-us:%lld   gpu-us:%lld   total-us:%lld   total-gpu-us:%lld\n",
          (int)e, epoch_total_us, epoch_gpu_us, total_time_us, total_gpu_time_us);
      if(converged) {
        break;
      }
    }
    HANDLE_ERROR(cudaEventDestroy(train_start));
    HANDLE_ERROR(cudaEventDestroy(train_stop));

    cudaFree(d_ind0);
    cudaFree(d_ind1);
    cudaFree(d_ind2);
    cudaFree(d_vals);
    cudaFree(d_nominbuffer);
    cudaFree(d_denominbuffer);
    cudaFree(d_value_a);
    cudaFree(d_value_b);
    cudaFree(d_value_c);
    cudaFree(d_factora);
    cudaFree(d_factorb);
    cudaFree(d_factorc);
}


/**
 * @brief The main function for tensor completion in ccd
 * @param train The tensor for generating factor matrices
 * @param validation The tensor for validation(RMSE)
 * @param test The tensor for testing the quality
 * @param regularization_index Lambda
*/
extern "C"{
void tc_ccd(sptensor_t * traina,
            sptensor_t * trainb,
            sptensor_t * trainc,
            sptensor_t * validation,
            sptensor_t * test,
            ordi_matrix ** mats,
            ordi_matrix ** best_mats,
            int algorithm_index,
            long long project_start_us,
            double regularization_index,
            double * best_rmse,
            double * tolerance,
            idx_t * nbadepochs,
            idx_t * bestepochs,
            idx_t * max_badepochs)
{
    idx_t const nmodes = traina->nmodes;
    const char* env_ciss = getenv("CUTC_CCD_CISS");
    if(env_ciss == NULL || env_ciss[0] == '\0' || env_ciss[0] == '0') {
      p_tc_ccd_raw(traina, validation, test, mats, best_mats,
                   algorithm_index, project_start_us, regularization_index,
                   best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);
      return;
    }
    //pay attention to this
    //p_transpose_model(nmodes, mats);

    //for the residual
    //sptensor_t * rtensor = tt_copy(train);
    int const rank = 0;

    #ifdef CUDA_LOSS
    //to be done
    #else
    /* initialize residual, to be done in gpu */
    //for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    //    p_update_residual(rtensor, mats, DEFAULT_NFACTORS, f, -1);
    //}
    #endif

    //initialize the devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(0);
    //prepare the tensor in TB-COO
    ciss_t * h_traina = ciss_alloc(traina, 1);
    ciss_t * h_trainb = ciss_alloc(trainb, 2);
    ciss_t * h_trainc = ciss_alloc(trainc, 3);
    #ifdef CCD_DEBUG
    printf("finish the allocation\n");
    printf("mode2:dimension in train %d, dimension in mats %d\n", traina->dims[1],mats[1]->I);
    #endif
    struct timeval start;
    struct timeval end;
    idx_t diff;

    //malloc and copy the tensors + matrices to gpu
    ciss_t * d_traina, * d_trainb, * d_trainc;
    idx_t * d_directory_a, * d_directory_b, * d_directory_c;
    idx_t * d_dims_a, * d_dims_b, * d_dims_c;
    idx_t * d_itemp1, *d_itemp2;
    double * d_entries_a , * d_entries_b, * d_entries_c;
    double * d_ftemp, * d_nominbuffer;
    //copy tensor for mode-1
    HANDLE_ERROR(cudaMalloc((void**)&d_traina, sizeof(ciss_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_directory_a, h_traina->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_entries_a, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&d_dims_a, nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_a, h_traina->directory, h_traina->dlength*sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_a, h_traina->entries, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_a, h_traina->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_traina->directory;
    d_itemp2 = h_traina->dims;
    d_ftemp = h_traina->entries;
    h_traina->directory = d_directory_a;
    h_traina->dims = d_dims_a;
    h_traina->entries = d_entries_a;
    HANDLE_ERROR(cudaMemcpy(d_traina, h_traina, sizeof(ciss_t), cudaMemcpyHostToDevice));
    h_traina->directory = d_itemp1;
    h_traina->dims = d_itemp2;
    h_traina->entries = d_ftemp;
    //copy tensor for mode-2
    HANDLE_ERROR(cudaMalloc((void**)&d_trainb, sizeof(ciss_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_directory_b, h_trainb->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_entries_b, h_trainb->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&d_dims_b, nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_b, h_trainb->directory, h_trainb->dlength*sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_b, h_trainb->entries, h_trainb->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_b, h_trainb->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_trainb->directory;
    d_itemp2 = h_trainb->dims;
    d_ftemp = h_trainb->entries;
    h_trainb->directory = d_directory_b;
    h_trainb->dims = d_dims_b;
    h_trainb->entries = d_entries_b;
    HANDLE_ERROR(cudaMemcpy(d_trainb, h_trainb, sizeof(ciss_t), cudaMemcpyHostToDevice));
    h_trainb->directory = d_itemp1;
    h_trainb->dims = d_itemp2;
    h_trainb->entries = d_ftemp;
    //copy tensor for mode-3
    HANDLE_ERROR(cudaMalloc((void**)&d_trainc, sizeof(ciss_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_directory_c, h_trainc->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_entries_c, h_trainc->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&d_dims_c, nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_c, h_trainc->directory, h_trainc->dlength*sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_c, h_trainc->entries, h_trainc->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_c, h_trainc->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_trainc->directory;
    d_itemp2 = h_trainc->dims;
    d_ftemp = h_trainc->entries;
    h_trainc->directory = d_directory_c;
    h_trainc->dims = d_dims_c;
    h_trainc->entries = d_entries_c;
    HANDLE_ERROR(cudaMemcpy(d_trainc, h_trainc, sizeof(ciss_t), cudaMemcpyHostToDevice));
    h_trainc->directory = d_itemp1;
    h_trainc->dims = d_itemp2;
    h_trainc->entries = d_ftemp;

    //buffer for nomin and denomin
    idx_t maxdlength = SS_MAX(SS_MAX(h_traina->dlength, h_trainb->dlength),h_trainc->dlength);
    double * h_nominbuffer = (double *)malloc(maxdlength * DEFAULT_NFACTORS *  sizeof(double));
    double * h_denominbuffer = (double *)malloc(DEFAULT_NFACTORS * maxdlength * sizeof(double));
    HANDLE_ERROR(cudaMalloc((void**)&d_nominbuffer, DEFAULT_NFACTORS * maxdlength * sizeof(double)));
    double* d_denominbuffer;
    HANDLE_ERROR(cudaMalloc((void**)&d_denominbuffer, DEFAULT_NFACTORS *  maxdlength * sizeof(double)));

    //copy the factor matrices
    ordi_matrix * d_factora, * d_factorb, * d_factorc;
    double * d_value_a, * d_value_b, * d_value_c;
    HANDLE_ERROR(cudaMalloc((void**)&d_factora, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_a, (mats[0]->I) * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values, (mats[0]->I) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[0]->values;
    mats[0]->values = d_value_a;
    HANDLE_ERROR(cudaMemcpy(d_factora, mats[0], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[0]->values = d_ftemp;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorb, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_b, (mats[1]->I) * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values, (mats[1]->I) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[1]->values;
    mats[1]->values = d_value_b;
    HANDLE_ERROR(cudaMemcpy(d_factorb, mats[1], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[1]->values = d_ftemp;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorc, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_c, (mats[2]->I) * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values, (mats[2]->I) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[2]->values;
    mats[2]->values = d_value_c;
    HANDLE_ERROR(cudaMemcpy(d_factorc, mats[2], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[2]->values = d_ftemp;



    #ifdef CUDA_LOSS //to be done
    sptensor_gpu_t * d_test, * d_validate;
    #else
    double loss = tc_loss_sq(traina, mats, algorithm_index);
    double frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    tc_converge(traina, validation, test, mats, best_mats, algorithm_index, loss, frobsq, 0, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);
    p_debug_matrix_stats(mats, nmodes, 0);
    #endif

    //double * nomin = (double*)malloc(argmax_elem(spnewtensor->dims, nmodes)*sizeof(double));
    //double * dnomin = (double*)malloc(argmax_elem(spnewtensor->dims, nmodes)*sizeof(double));

    //step into the kernel
    idx_t nnz = traina->nnz;
    idx_t tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
    idx_t blocknum_m = tilenum/((idx_t)DEFAULT_BLOCKSIZE) + 1;

    idx_t mode_n, mode_i;
    long long total_gpu_time_us = 0;
    long long total_time_us = 0;
    idx_t max_iterate = DEFAULT_MAX_ITERATE;
    const char* env_max_iterate = getenv("CUTC_MAX_ITERATE");
    if(env_max_iterate != NULL && env_max_iterate[0] != '\0') {
      char* endptr = NULL;
      unsigned long long parsed = strtoull(env_max_iterate, &endptr, 10);
      if(endptr != env_max_iterate && parsed > 0ULL) {
        max_iterate = (idx_t)parsed;
      }
    }
    cudaEvent_t train_start, train_stop;
    HANDLE_ERROR(cudaEventCreate(&train_start));
    HANDLE_ERROR(cudaEventCreate(&train_stop));
    /* foreach epoch */
    for(idx_t e=1; e < max_iterate+1; ++e) {
       long long epoch_gpu_us = 0;
       long long epoch_total_us = 0;
       HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values,mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
       HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values,mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
       HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values,mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
       loss = 0;
       srand(time(0));
       mode_i = rand()%3;
       gettimeofday(&start,NULL);
      /*
        for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
        /* add current component to residual
        p_update_residual(rtensor, mats, DEFAULT_NFACTORS, f, 1);

        for(idx_t inner=0; inner < NUM_INNER; ++inner) {

          /* compute new column 'f' for each factor
          for(idx_t m=0; m < nmodes; ++m) {
            memcpy(nomin, 0, rtensor->dims[m] * sizeof(double));
            memcpy(dnomin, 0, rtensor->dims[m] * sizeof(double));
            p_ccd_update(rtensor, m, f, nmodes, DEFAULT_NFACTORS, mats, nomin, dnomin);

            /* numerator/denominator are now computed; update factor column
            static inline void p_compute_newcol(rtensor, mats, nmoin, denomin, regularization_index, m, f);

          } /* foreach mode
        } /* foreach inner iteration

        /* subtract new rank-1 factor from residual
        update_residual_gpu(rtensor, mats, DEFAULT_NFACTORS, f, -1);

      } /* foreach factor */
      //the GPU version, update the factors all at once
      //HANDLE_ERROR(cudaMemset(d_nominbuffer, 0, DEFAULT_NFACTORS * maxdlength * sizeof(double)));
      //HANDLE_ERROR(cudaMemset(d_denominbuffer, 0, DEFAULT_NFACTORS * maxdlength*sizeof(double)));
      for(idx_t m = 0; m < nmodes; m++)
      {
        mode_n = (mode_i + m)%3;
        HANDLE_ERROR(cudaMemset(d_nominbuffer, 0, DEFAULT_NFACTORS * maxdlength * sizeof(double)));
        HANDLE_ERROR(cudaMemset(d_denominbuffer, 0, DEFAULT_NFACTORS * maxdlength*sizeof(double)));
        switch(mode_n)
        {
	          case 0: //for the first mode
	          {
	            idx_t blocknum_u = h_traina->dlength / DEFAULT_BLOCKSIZE + 1;
	            HANDLE_ERROR(cudaEventRecord(train_start, 0));
	            update_frac_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_traina, d_factora, d_factorb, d_factorc, d_nominbuffer, d_denominbuffer, tilenum);
	            update_ccd_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_traina, d_factora, d_nominbuffer,d_denominbuffer, h_traina->dlength, regularization_index);
	            HANDLE_ERROR(cudaGetLastError());
	            HANDLE_ERROR(cudaEventRecord(train_stop, 0));
	            epoch_gpu_us += p_cuda_elapsed_us(train_start, train_stop);
	            break;
	          }

	          case 1: //for the second mode
	          {
	            idx_t blocknum_u = h_trainb->dlength / DEFAULT_BLOCKSIZE + 1;
	            HANDLE_ERROR(cudaEventRecord(train_start, 0));
	            update_frac_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainb, d_factorb, d_factorc, d_factora, d_nominbuffer, d_denominbuffer, tilenum);
	            update_ccd_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_trainb, d_factorb, d_nominbuffer,d_denominbuffer, h_trainb->dlength, regularization_index);
	            HANDLE_ERROR(cudaGetLastError());
	            HANDLE_ERROR(cudaEventRecord(train_stop, 0));
	            epoch_gpu_us += p_cuda_elapsed_us(train_start, train_stop);
	            break;
	          }

        //for the third mode
          default:
	          {
	            idx_t blocknum_u = h_trainc->dlength / DEFAULT_BLOCKSIZE + 1;
	            HANDLE_ERROR(cudaEventRecord(train_start, 0));
	            update_frac_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainc, d_factorc, d_factora, d_factorb, d_nominbuffer, d_denominbuffer, tilenum);
	            update_ccd_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_trainc, d_factorc, d_nominbuffer,d_denominbuffer, h_trainc->dlength, regularization_index);
	            HANDLE_ERROR(cudaGetLastError());
	            HANDLE_ERROR(cudaEventRecord(train_stop, 0));
	            epoch_gpu_us += p_cuda_elapsed_us(train_start, train_stop);
	            break;
	          }
        }

	      }

	        gettimeofday(&end,NULL);
	        diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
	        total_gpu_time_us += epoch_gpu_us;

        HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_b, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_c, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaDeviceSynchronize());
        p_debug_matrix_stats(mats, nmodes, e);
        #ifdef CCD_DEBUG
        matrix_display(mats[0]);
        matrix_display(mats[1]);
        matrix_display(mats[2]);
        #endif
        gettimeofday(&end,NULL);
        diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
        epoch_total_us = diff;


   /* compute RMSE and adjust learning rate */
    //the first element is used to store the final loss
    //idx_t blocknum_u = h_traina->dlength / DEFAULT_BLOCKSIZE + 1;
    //HANDLE_ERROR(cudaMemset(d_nominbuffer, 0, sizeof(double)));
    //update_residual_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_traina, tilenum, d_nominbuffer);
    //HANDLE_ERROR(cudaMemcpy(&loss, d_nominbuffer, sizeof(double), cudaMemcpyDeviceToHost));
    //frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    loss = tc_loss_sq(traina, mats, algorithm_index);
    frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    double obj = loss + frobsq;
    bool converged = tc_converge(traina, validation, test, mats, best_mats, algorithm_index, loss, frobsq, e, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);
    total_time_us = p_walltime_us_now() - project_start_us;
    if(total_time_us < 0) {
      total_time_us = 0;
    }
    printf("epoch:%d   time-us:%lld   gpu-us:%lld   total-us:%lld   total-gpu-us:%lld\n",
       (int)e, epoch_total_us, epoch_gpu_us, total_time_us, total_gpu_time_us);

    if(converged) {
      break;
    }

	  } /* foreach epoch */
    HANDLE_ERROR(cudaEventDestroy(train_start));
    HANDLE_ERROR(cudaEventDestroy(train_stop));

	  /* print times */
  //p_transpose_model(mats);
  //p_transpose_model(ws->best_model);

  /* cleanup */
  //tt_free(rtensor);
  //free(nomin);
  //free(denomin);
  cudaFree(d_directory_a);
  cudaFree(d_dims_a);
  cudaFree(d_entries_a);
  cudaFree(d_directory_b);
  cudaFree(d_dims_b);
  cudaFree(d_entries_b);
  cudaFree(d_directory_c);
  cudaFree(d_dims_c);
  cudaFree(d_entries_c);
  cudaFree(d_nominbuffer);
  cudaFree(d_denominbuffer);
  cudaFree(d_value_a);
  cudaFree(d_value_b);
  cudaFree(d_value_c);
  cudaFree(d_traina);
  cudaFree(d_trainb);
  cudaFree(d_trainc);
  cudaFree(d_factora);
  cudaFree(d_factorb);
  cudaFree(d_factorc);

  ciss_free(h_traina);
  ciss_free(h_trainb);
  ciss_free(h_trainc);
  cudaDeviceReset();
}


}
