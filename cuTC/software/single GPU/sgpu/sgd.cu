
extern "C"
{
#include "completion.h"
#include "ciss.h"
#include "base.h"
#include "matrixprocess.h"
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
}

#include "sgd.cuh"
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


//the gpu kernel
__global__ void p_update_sgd_gpu(ciss_t * d_traina, 
                                 ordi_matrix * d_factora,
                                 ordi_matrix * d_factorb, 
                                 ordi_matrix * d_factorc, 
                                 double learning_rate, 
                                 double regularization_index,
                                 idx_t tilenum)
{
  //get thread and block index
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  double * entries = d_traina->entries;
  idx_t localtile = tileid*((DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH);
  //buffer for matrices
  double __align__(256) mabuffer[DEFAULT_NFACTORS];
  double __align__(256) mbbuffer[DEFAULT_NFACTORS];
  double __align__(256) mcbuffer[DEFAULT_NFACTORS];
  double __align__(256) localtbuffer[6];
  idx_t a,b,c, localcounter;
  double localvalue;

  if(tileid < tilenum)
  {
    //get the indices and value
    idx_t f_id = (idx_t)(entries[localtile] * (-1));
    idx_t l_id = (idx_t)(entries[localtile+1] * (-1));
    idx_t bitmap = (idx_t)(entries[localtile+2]);
    bitmap = __brevll(bitmap);
    while((bitmap & 1) == 0) {bitmap = (bitmap >> 1);}
    bitmap = (bitmap >> 1);
    localtile += DEFAULT_T_TILE_WIDTH;

    for(idx_t j = 0; j < DEFAULT_T_TILE_LENGTH/2; j++)
    {
      //unroll loop and load
      localtbuffer[0] = entries[localtile];
      localtbuffer[1] = entries[localtile + 1];
      localtbuffer[2] = entries[localtile + 2];
      

      if(localtbuffer[0] == -1 && localtbuffer[1] == -1) break;
      //for the first
      f_id += (!(bitmap & 1));
      bitmap = bitmap >> 1;
      a = d_traina->directory[f_id] - 1;
      localcounter = d_traina->dcounter[f_id + 1] - d_traina->dcounter[f_id];
      b = (idx_t)localtbuffer[0] - 1;
      c = (idx_t)localtbuffer[1] - 1;
      localvalue = localtbuffer[2];
      #ifdef SGD_DEBUG
      printf("now a b c in tile %ld are %ld %ld %ld\n", tileid, a, b, c);
      #endif
      //if(localtbuffer[0] == -1 && localtbuffer[1] == -1) break;
      for(idx_t i = 0; i< DEFAULT_NFACTORS; i++)
      {
        //((double2*)mabuffer)[i] = ((double2*)d_factora->values)[a * DEFAULT_NFACTORS/2 + i];
        //((double2*)mbbuffer)[i] = ((double2*)d_factorb->values)[b * DEFAULT_NFACTORS/2 + i];
        //((double2*)mcbuffer)[i] = ((double2*)d_factorc->values)[c * DEFAULT_NFACTORS/2 + i];
        mabuffer[i] = (d_factora->values)[a * DEFAULT_NFACTORS + i];
        mbbuffer[i] = (d_factorb->values)[b * DEFAULT_NFACTORS + i];
        mcbuffer[i] = (d_factorc->values)[c * DEFAULT_NFACTORS + i];

      }
      /* predict value */
      double predicted = 0;
      for(idx_t f=0; f < DEFAULT_NFACTORS; f++) {
        predicted += mabuffer[f] * mbbuffer[f] * mcbuffer[f];
      }
      predicted = localvalue - predicted;
      /* update rows */
      for(idx_t f=0; f < DEFAULT_NFACTORS; f++) {
        double moda = (predicted * mbbuffer[f] * mcbuffer[f]) - (regularization_index * mabuffer[f]);
        double modb = (predicted * mabuffer[f] * mcbuffer[f]) - (regularization_index * mbbuffer[f]);
        double modc = (predicted * mbbuffer[f] * mabuffer[f]) - (regularization_index * mcbuffer[f]);
        atomicAdd(&(d_factora->values[a * DEFAULT_NFACTORS + f]), learning_rate*moda * (double)SGD_MODIFICATIONA);
        atomicAdd(&(d_factorb->values[b * DEFAULT_NFACTORS + f]), learning_rate*modb * (double)SGD_MODIFICATIONB);
        atomicAdd(&(d_factorc->values[c * DEFAULT_NFACTORS + f]), learning_rate*modc * (double)SGD_MODIFICATIONC);
     }

     //for the second
     localtbuffer[3] = entries[localtile + 3];
     localtbuffer[4] = entries[localtile + 4];
     localtbuffer[5] = entries[localtile + 5];
     f_id += (!(bitmap & 1));
     bitmap = bitmap >> 1;
     a = d_traina->directory[f_id] - 1;
     localcounter = d_traina->dcounter[f_id + 1] - d_traina->dcounter[f_id];
     b = (idx_t)localtbuffer[3] - 1;
     c = (idx_t)localtbuffer[4] - 1;
     #ifdef SGD_DEBUG
     printf("now a b c in tile %ld are %ld %ld %ld\n", tileid, a, b, c);
     #endif
     localvalue = localtbuffer[5];
     if(localtbuffer[3] == -1 && localtbuffer[4] == -1) break;
     for(idx_t i = 0; i< DEFAULT_NFACTORS; i++)
     {
      mabuffer[i] = (d_factora->values)[a * DEFAULT_NFACTORS + i];
      mbbuffer[i] = (d_factorb->values)[b * DEFAULT_NFACTORS + i];
      mcbuffer[i] = (d_factorc->values)[c * DEFAULT_NFACTORS + i];
     }
     /* predict value */
     predicted = 0;
     for(idx_t f=0; f < DEFAULT_NFACTORS; f++) {
       predicted += mabuffer[f] * mbbuffer[f] * mcbuffer[f];
     }
     predicted = localvalue - predicted;
     /* update rows */
     for(idx_t f=0; f < DEFAULT_NFACTORS; f++) {
       double moda = (predicted * mbbuffer[f] * mcbuffer[f]) - (regularization_index * mabuffer[f]);
       double modb = (predicted * mabuffer[f] * mcbuffer[f]) - (regularization_index * mbbuffer[f]);
       double modc = (predicted * mbbuffer[f] * mabuffer[f]) - (regularization_index * mcbuffer[f]);
       atomicAdd(&(d_factora->values[a * DEFAULT_NFACTORS + f]), learning_rate*moda * (double)SGD_MODIFICATIONA);
       atomicAdd(&(d_factorb->values[b * DEFAULT_NFACTORS + f]), learning_rate*modb * (double)SGD_MODIFICATIONB);
       atomicAdd(&(d_factorc->values[c * DEFAULT_NFACTORS + f]), learning_rate*modc * (double)SGD_MODIFICATIONC);
    }
    localtile +=  2 * DEFAULT_T_TILE_WIDTH;
}

}


}



/**
 * @brief The main function for tensor completion in sgd
 * @param train The tensor for generating factor matrices
 * @param validation The tensor for validation(RMSE)
 * @param test The tensor for testing the quality
 * @param regularization_index Lambda
*/
extern "C"{
void tc_sgd(sptensor_t * traina,
            sptensor_t * trainb,
            sptensor_t * trainc, 
            sptensor_t * validation,
            sptensor_t * test,
            ordi_matrix ** mats, 
            ordi_matrix ** best_mats,
            int algorithm_index,
            long long project_start_us,
            double regularization_index, 
            double learning_rate,
            double * best_rmse, 
            double * tolerance, 
            idx_t * nbadepochs, 
            idx_t * bestepochs, 
            idx_t * max_badepochs)
{
    if(project_start_us <= 0) {
      project_start_us = p_walltime_us_now();
    }

    //only in sgd
    idx_t steps_size = 1000;
    idx_t nmodes = traina->nmodes;

    //initialize the devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(0);
    //prepare the tensor in TB-COO
    ciss_t * h_traina = ciss_alloc(traina, 1);
    #ifdef CISS_DEBUG
    ciss_display(h_traina);
    #endif
    //ciss_t * h_trainb = ciss_alloc(train, 1);
    //ciss_t * h_trainc = ciss_alloc(train, 2);
    struct timeval start;
    struct timeval end;
    long long diff;
    
    //malloc and copy the tensors + matrices to gpu
    ciss_t * d_traina;
    idx_t * d_directory_a, * d_counter_a;
    idx_t * d_dims_a; 
    idx_t * d_itemp1, *d_itemp2, *d_itemp3;
    double * d_entries_a; 
    double * d_ftemp;
    //copy tensor for mode-1
    HANDLE_ERROR(cudaMalloc((void**)&d_traina, sizeof(ciss_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_directory_a, h_traina->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_counter_a), (h_traina->dlength + 1) * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_entries_a, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&d_dims_a, nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_a, h_traina->directory, h_traina->dlength*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_counter_a, h_traina->dcounter, (h_traina->dlength + 1)*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_a, h_traina->entries, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_a, h_traina->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_traina->directory;
    d_itemp2 = h_traina->dims;
    d_itemp3 = h_traina->dcounter;
    d_ftemp = h_traina->entries;
    h_traina->directory = d_directory_a;
    h_traina->dcounter = d_counter_a;
    h_traina->dims = d_dims_a;
    h_traina->entries = d_entries_a;
    HANDLE_ERROR(cudaMemcpy(d_traina, h_traina, sizeof(ciss_t), cudaMemcpyHostToDevice));
    h_traina->directory = d_itemp1;
    h_traina->dims = d_itemp2;
    h_traina->dcounter = d_itemp3;
    h_traina->entries = d_ftemp;
    
    //buffer for HTH
    //idx_t maxdlength = SS_MAX(SS_MAX(h_traina->dlength, h_trainb->dlength),h_trainc->dlength);
    //double * h_hbuffer = (double *)malloc(DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double));
    //double * h_invbuffer = (double *)malloc(DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double));
    //HANDLE_ERROR(cudaMalloc((void**)&d_hbuffer, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)));
    //double* d_invbuffer; //for inverse
    //HANDLE_ERROR(cudaMalloc((void**)&d_invbuffer, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)));

    //copy the factor matrices
    ordi_matrix * d_factora, * d_factorb, * d_factorc;
    double * d_value_a, * d_value_b, * d_value_c;
    HANDLE_ERROR(cudaMalloc((void**)&d_factora, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[0]->values;
    mats[0]->values = d_value_a;
    HANDLE_ERROR(cudaMemcpy(d_factora, mats[0], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[0]->values = d_ftemp;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorb, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_b, mats[1]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[1]->values;
    mats[1]->values = d_value_b;
    HANDLE_ERROR(cudaMemcpy(d_factorb, mats[1], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[1]->values = d_ftemp;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorc, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_c, mats[2]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[2]->values;
    mats[2]->values = d_value_c;
    HANDLE_ERROR(cudaMemcpy(d_factorc, mats[2], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[2]->values = d_ftemp;

    #ifdef CUDA_LOSS
    //to be done
    #else
    double loss = tc_loss_sq(traina, mats, algorithm_index);
    double frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    tc_converge(traina, validation, test, mats, best_mats, algorithm_index, loss, frobsq, 0, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);
    #endif

    /* for bold driver */
    double obj = loss + frobsq;
    double prev_obj = obj;

    long long total_time_us = 0;
    long long total_gpu_time_us = 0;
    cudaEvent_t train_start, train_stop;
    HANDLE_ERROR(cudaEventCreate(&train_start));
    HANDLE_ERROR(cudaEventCreate(&train_stop));


    //step into the kernel
    idx_t nnz = traina->nnz;
    idx_t tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
    idx_t blocknum_m = tilenum/DEFAULT_BLOCKSIZE + 1;
    #ifdef SGD_DEBUG
    printf("nnz %d tilenum %d\n", nnz, tilenum);
    #endif

    idx_t max_iterate = DEFAULT_MAX_ITERATE;
    const char* env_max_iterate = getenv("CUTC_MAX_ITERATE");
    if(env_max_iterate != NULL && env_max_iterate[0] != '\0') {
      char* endptr = NULL;
      unsigned long long parsed = strtoull(env_max_iterate, &endptr, 10);
      if(endptr != env_max_iterate && parsed > 0ULL) {
        max_iterate = (idx_t)parsed;
      }
    }

    /* foreach epoch */
  for(idx_t e=1; e < max_iterate + 1; ++e) {
      long long epoch_gpu_us = 0;
      long long epoch_total_us = 0;


    /* update model from all training observations */
    gettimeofday(&start,NULL);
    HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaDeviceSynchronize()); 

    HANDLE_ERROR(cudaEventRecord(train_start, 0));
    p_update_sgd_gpu<<<blocknum_m, DEFAULT_BLOCKSIZE, 0>>>(d_traina, d_factora, d_factorb, d_factorc, learning_rate, regularization_index, tilenum);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaEventRecord(train_stop, 0));
    epoch_gpu_us = p_cuda_elapsed_us(train_start, train_stop);

    gettimeofday(&end,NULL);
    diff = 1000000LL*(end.tv_sec-start.tv_sec) + (end.tv_usec - start.tv_usec);
    (void)diff;

    
    HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_b, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_c, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaDeviceSynchronize()); 
    #ifdef SGD_DEBUG
    printf("start display matrices\n");
    matrix_display(mats[0]);
    matrix_display(mats[1]);
    matrix_display(mats[2]);
    #endif
    gettimeofday(&end,NULL);
    diff = 1000000LL*(end.tv_sec-start.tv_sec) + (end.tv_usec - start.tv_usec);
    epoch_total_us = diff;     // 从start到D2H结束的总耗时
    // 累加每一轮 GPU 训练 kernel 时间；H2D/D2H/评估不计入 train_gpu。
    total_gpu_time_us += epoch_gpu_us;

    /* compute RMSE and adjust learning rate */
    loss = tc_loss_sq(traina, mats, algorithm_index);
    frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    obj = loss + frobsq;
    bool converged = tc_converge(traina, validation, test, mats, best_mats, algorithm_index, loss, frobsq, e, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);

    // total-us 改为“从程序启动到当前日志点”的累计时间。
    total_time_us = p_walltime_us_now() - project_start_us;
    if(total_time_us < 0) {
      total_time_us = 0;
    }
    printf("epoch:%d   time-us:%lld   gpu-us:%lld   total-us:%lld   total-gpu-us:%lld\n",
       (int)e, epoch_total_us, epoch_gpu_us, total_time_us, total_gpu_time_us);

    if(converged) {
      break;
    }

    /* bold driver */
    if(e > 1) {
      if(obj < prev_obj) {
        learning_rate *= 1.05;
      } else {
        learning_rate *= 0.50;
      }
    }

    prev_obj = obj;
			  }

  HANDLE_ERROR(cudaEventDestroy(train_start));
  HANDLE_ERROR(cudaEventDestroy(train_stop));

  // Restore the best checkpoint before final reporting so downstream
  // evaluation/log inspection can use the true best-epoch model.
  for(idx_t m = 0; m < nmodes; ++m) {
    matrix_copy(best_mats[m], mats[m]);
  }

  double best_train_rmse = 0.0, best_train_mae = 0.0, best_train_er = 0.0;
  if(traina != NULL && traina->nnz > 0) {
    double const best_train_loss = tc_loss_sq(traina, mats, algorithm_index);
    best_train_rmse = sqrt(best_train_loss / traina->nnz);
    best_train_mae = tc_mae(traina, mats, algorithm_index);
    best_train_er = tc_er(traina, mats, algorithm_index);
  }

  double best_val_rmse = 0.0, best_val_mae = 0.0, best_val_er = 0.0;
  if(validation != NULL && validation->nnz > 0) {
    best_val_rmse = tc_rmse(validation, mats, algorithm_index);
    best_val_mae = tc_mae(validation, mats, algorithm_index);
    best_val_er = tc_er(validation, mats, algorithm_index);
  }

  double best_u_rmse = 0.0, best_u_mae = 0.0, best_u_er = 0.0;
  if(test != NULL && test->nnz > 0) {
    best_u_rmse = tc_rmse(test, mats, algorithm_index);
    best_u_mae = tc_mae(test, mats, algorithm_index);
    best_u_er = tc_er(test, mats, algorithm_index);
  }
  printf("[best-model] epoch:%d   "
         "RMSE-tr: %0.5e   RMSE-vl: %0.5e   RMSE-u: %0.5e   "
         "MAE-tr: %0.5e   MAE-vl: %0.5e   MAE-u: %0.5e   "
         "ER-tr: %0.5e   ER-vl: %0.5e   ER-u: %0.5e\n",
         (int)(*bestepochs),
         best_train_rmse, best_val_rmse, best_u_rmse,
         best_train_mae, best_val_mae, best_u_mae,
         best_train_er, best_val_er, best_u_er);

  total_time_us = p_walltime_us_now() - project_start_us;
  if(total_time_us < 0) {
    total_time_us = 0;
  }
  printf("TOTAL_TIME_us:%lld (%.3f s)   TOTAL_GPU_us:%lld (%.3f s)\n",
       total_time_us, (double)total_time_us / 1e6,
       total_gpu_time_us, (double)total_gpu_time_us / 1e6);

  //free the cudabuffer
  cudaFree(d_directory_a);
  cudaFree(d_dims_a);
  cudaFree(d_entries_a);
  //cudaFree(d_hbuffer);
  cudaFree(d_value_a);
  cudaFree(d_value_b);
  cudaFree(d_value_c);
  cudaFree(d_traina);
  cudaFree(d_factora);
  cudaFree(d_factorb);
  cudaFree(d_factorc);

  ciss_free(h_traina);
  cudaDeviceReset();

  
}


}
