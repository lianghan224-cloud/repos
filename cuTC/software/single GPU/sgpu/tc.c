//include
#include "base.h"
#include "matrixprocess.h"
#include "sptensor.h"
#include "io.h"
#include "completion.h"
#include "sampling_tns.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

/*
The code number of algorithms:
0 ALS | 1 SGD | 2 CCD+
*/

static int parse_algorithm_index(void)
{
  int algorithm_index = 1;
  const char* env_alg = getenv("CUTC_ALGORITHM");
  if(env_alg == NULL || env_alg[0] == '\0') {
    env_alg = getenv("CUTC_ALGORITHM_INDEX");
  }
  if(env_alg == NULL || env_alg[0] == '\0') {
    return algorithm_index;
  }

  if(strcmp(env_alg, "0") == 0 || strcmp(env_alg, "als") == 0 || strcmp(env_alg, "ALS") == 0) {
    algorithm_index = 0;
  } else if(strcmp(env_alg, "1") == 0 || strcmp(env_alg, "sgd") == 0 || strcmp(env_alg, "SGD") == 0) {
    algorithm_index = 1;
  } else if(strcmp(env_alg, "2") == 0 || strcmp(env_alg, "ccd") == 0 || strcmp(env_alg, "CCD") == 0 ||
            strcmp(env_alg, "ccd+") == 0 || strcmp(env_alg, "CCD+") == 0) {
    algorithm_index = 2;
  } else {
    fprintf(stderr, "[tc] unknown CUTC_ALGORITHM='%s'; using SGD\n", env_alg);
  }
  return algorithm_index;
}

int main(int argc, char** argv)
{
  struct timeval project_start;
  gettimeofday(&project_start, NULL);
  long long project_start_us =
      1000000LL * (long long)project_start.tv_sec + (long long)project_start.tv_usec;

  int flag;
  idx_t nmodes = 3;
  int j = 0;
  int algorithm_index = parse_algorithm_index();
  double cost;
  clock_t start;
  clock_t end;

  #ifdef DEBUG
  printf("%d\n",argc);
  #endif

  if(argc != 2 && argc != 6) {
    fprintf(stderr,
            "Usage:\n"
            "  %s <full_tensor.tns>\n"
            "  %s <tr.tns> <tr2.tns> <tr3.tns> <v.tns> <te.tns>\n",
            argv[0], argv[0]);
    return -1;
  }

  //load tensor(s)
  sptensor_t* traina, * trainb, * trainc;
  sptensor_t* validation;
  sptensor_t* test;
  traina = NULL;
  trainb = NULL;
  trainc = NULL;
  validation = NULL;
  test = NULL;
  #ifdef CISS_DEBUG
  printf("%s\n",argv[1]); 
  #endif

  int presplit_input = 0;
  const char* env_presplit = getenv("CUTC_PRESPLIT");
  if(env_presplit != NULL && env_presplit[0] != '\0' &&
     strcmp(env_presplit, "0") != 0 && strcmp(env_presplit, "false") != 0 &&
     strcmp(env_presplit, "FALSE") != 0) {
    presplit_input = 1;
  }

  if(argc == 6) {
    #ifdef CISS_DEBUG
    printf("%s\n",argv[2]);
    printf("%s\n",argv[3]);  
    #endif
    traina = tt_read(argv[1]);
    trainb = tt_read(argv[2]);
    trainc = tt_read(argv[3]);
    validation = tt_read(argv[4]); 
    test = tt_read(argv[5]);
  } else {
    // single-file mode: pass full valid set, then split into Omega/V/U inside tc
    traina = tt_read(argv[1]);
  }

  if(!traina || (argc == 6 && (!trainb || !trainc || !validation || !test))) {
    printf("[tc] failed to read input tensor file(s)\n");
    if(traina) tt_free(traina);
    if(trainb) tt_free(trainb);
    if(trainc) tt_free(trainc);
    if(validation) tt_free(validation);
    if(test) tt_free(test);
    return -1;
  }

  printf("[tc] algorithm=%d\n", algorithm_index);

  // ===== three-set split + sampling control (SGD/CCD path) =====
  {
    const double val_rate = 0.05;         // default validation split ratio
    const uint32_t val_seed = 20250101;   // fixed
    double sampling_rate = 0.1;           // default; can override by env
    const int sampling_mode = 0;          // 0=uniform, 1=value-biased
    const double sampling_alpha = 1.0;
    const double sampling_eps = 1e-6;
    const uint32_t omega_seed = 2025;     // fixed

    {
      const char* env_rate = getenv("CUTC_SAMPLING_RATE");
      if(env_rate != NULL && env_rate[0] != '\0') {
        char* endptr = NULL;
        double parsed = strtod(env_rate, &endptr);
        if(endptr != env_rate) {
          if(parsed < 0.0) parsed = 0.0;
          if(parsed > 1.0) parsed = 1.0;
          sampling_rate = parsed;
        }
      }
    }

    if (presplit_input && argc == 6) {
      printf("[split3] using pre-split input tensors: train=%llu validation=%llu test=%llu\n",
             (unsigned long long)traina->nnz,
             (unsigned long long)validation->nnz,
             (unsigned long long)test->nnz);
    } else if (algorithm_index == 0 || algorithm_index == 1 || algorithm_index == 2) {
      sptensor_t* omega = NULL;
      sptensor_t* vset = NULL;
      sptensor_t* uset = NULL;

      int split_ok = tt_split_three_sets(
          traina,
          validation,
          test,
          val_rate,
          val_seed,
          sampling_rate,
          sampling_mode,
          sampling_alpha,
          sampling_eps,
          omega_seed,
          &omega,
          &vset,
          &uset);

      if (split_ok != 0 || !omega || !vset || !uset) {
        printf("[split3] failed (out of memory or invalid input shape)\n");
        if (omega) tt_free(omega);
        if (vset) tt_free(vset);
        if (uset) tt_free(uset);
        tt_free(traina);
        if (trainb) tt_free(trainb);
        if (trainc) tt_free(trainc);
        if (validation) tt_free(validation);
        if (test) tt_free(test);
        return -1;
      }

      tt_free(traina);
      if(validation) tt_free(validation);
      if(test) tt_free(test);
      traina = omega;       // Omega
      validation = vset;    // V
      test = uset;          // U

      if (algorithm_index == 0 || algorithm_index == 2) {
        trainb = tt_copy(traina);
        trainc = tt_copy(traina);
        if (!trainb || !trainc) {
          printf("[split3] failed to copy Omega for ALS/CCD modes\n");
          tt_free(traina);
          if (trainb) tt_free(trainb);
          if (trainc) tt_free(trainc);
          tt_free(validation);
          tt_free(test);
          return -1;
        }
      }

      printf("[split3] remap done: train=Omega, validation=V, test=U\n");
    }
  }
  // ============================================================
  
  #ifdef DEBUG
  printf("finish tensor reading\n");
  #endif

  tc_main_ciss(
      traina, trainb, trainc, validation, test, algorithm_index, project_start_us);
  
  tt_free(traina); 
  if(trainb) tt_free(trainb);
  if(trainc) tt_free(trainc);
  if(validation) tt_free(validation);
  if(test) tt_free(test);
  
  
  return 0;
}
