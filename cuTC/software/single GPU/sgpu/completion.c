//includes
#include "als.cuh"
#include "sgd.cuh"
#include "ccd.cuh"
#include "sptensor.h"
#include "completion.h"
#include "matrixprocess.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

//all the tc_* in completion can be paralleled

/******************************************************************************
* PRIVATE FUNCTIONS
*****************************************************************************/

/**
* @brief Predict a value for a three-way tensor.
*/
static inline double p_predict_val3(
    idx_t nmodes,
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index)
{
  double est = 0;
  assert(nmodes == 3);
  idx_t const i = test->ind[0][index] - 1;
  idx_t const j = test->ind[1][index] - 1;
  idx_t const k = test->ind[2][index] - 1;

  double * A = mats[0]->values + (i * DEFAULT_NFACTORS);
  double * B = mats[1]->values + (j * DEFAULT_NFACTORS);
  double * C = mats[2]->values + (k * DEFAULT_NFACTORS);

  for(idx_t f = 0; f < DEFAULT_NFACTORS; ++f) {
    est += A[f] * B[f] * C[f];
  }

  return est;
}


/**
* @brief Predict a value for a three-way tensor when the model uses column-major matrices.
*/
static inline double p_predict_val3_col(
    idx_t nmodes,
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index)
{
  double est = 0;

  assert(nmodes == 3);

  idx_t const i = (test->ind)[0][index] - 1;
  idx_t const j = (test->ind)[1][index] - 1;
  idx_t const k = (test->ind)[2][index] - 1;

  assert(i < mats[0]->I);
  assert(j < mats[1]->I);
  assert(k < mats[2]->I);

  idx_t const I = test->dims[0];
  idx_t const J = test->dims[1];
  idx_t const K = test->dims[2];

  double * A = mats[0]->values;
  double * B = mats[1]->values;
  double * C = mats[2]->values;

  for(idx_t f = 0; f < DEFAULT_NFACTORS; ++f) {
    est += A[i + (f * I)] * B[j + (f * J)] * C[k + (f * K)];
  }

  return est;
}


/**
* @brief Print progress (RMSE/MAE/ER)
*/
static void p_print_progress(
    idx_t const epoch,
    double const loss,
    double const rmse_tr,
    double const rmse_vl,
    double const rmse_u,
    double const mae_tr,
    double const mae_vl,
    double const mae_u,
    double const er_tr,
    double const er_vl,
    double const er_u
    )
{
  printf("epoch:%d   loss: %0.5e   "
         "RMSE-tr: %0.5e   RMSE-vl: %0.5e   RMSE-u: %0.5e   "
         "MAE-tr: %0.5e   MAE-vl: %0.5e   MAE-u: %0.5e   "
         "ER-tr: %0.5e   ER-vl: %0.5e   ER-u: %0.5e \n",
         (int)epoch, loss,
         rmse_tr, rmse_vl, rmse_u,
         mae_tr, mae_vl, mae_u,
         er_tr, er_vl, er_u);
}


/******************************************************************************
* PUBLIC FUNCTIONS
*****************************************************************************/

double tc_rmse(
    sptensor_t * test,
    ordi_matrix ** mats,
    int algorithm_index
    )
{
  return sqrt(tc_loss_sq(test, mats, algorithm_index) / test->nnz);
}

double tc_mae(
    sptensor_t * test,
    ordi_matrix ** mats,
    int algorithm_index
    )
{
  double loss_obj = 0.0;
  double * test_vals = test->vals;

  for(idx_t x = 0; x < test->nnz; ++x) {
    double predicted = tc_predict_val(test, mats, x);
    loss_obj += fabs(test_vals[x] - predicted);
  }

  return loss_obj / test->nnz;
}

/**
 * @brief ER = sqrt( Σ(t-ŷ)^2 / Σ(t^2) )
 */
double tc_er(
    sptensor_t * test,
    ordi_matrix ** mats,
    int algorithm_index
    )
{
  if(test->nnz == 0) return 0.0;

  double st2 = 0.0;
  for(idx_t x = 0; x < test->nnz; ++x) {
    double t = test->vals[x];
    st2 += t * t;
  }
  if(st2 <= 0.0) return 0.0;

  double sse = tc_loss_sq(test, mats, algorithm_index);
  return sqrt(sse / st2);
}

double tc_loss_sq(
    sptensor_t * test,
    ordi_matrix ** mats,
    int algorithm_index
    )
{
  double loss_obj = 0.0;
  double * test_vals = test->vals;

  for(idx_t x = 0; x < test->nnz; ++x) {
    double err = test_vals[x] - tc_predict_val(test, mats, x);
    loss_obj += err * err;
  }

  return loss_obj;
}

double tc_frob_sq(
    idx_t nmodes,
    double regularization_index,
    ordi_matrix ** mats)
{
  double reg_obj = 0.0;

  for(idx_t m = 0; m < nmodes; ++m) {
    double accum = 0.0;
    idx_t const nrows = mats[m]->I;
    double * mat = mats[m]->values;
    for(idx_t x = 0; x < nrows * DEFAULT_NFACTORS; ++x) {
      accum += mat[x] * mat[x];
    }
    reg_obj += regularization_index * accum;
  }

  return reg_obj;
}

double tc_predict_val(
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index
    )
{
  if(test->nmodes == 3) {
    return p_predict_val3(test->nmodes, test, mats, index);
  }
  return 0.0;
}

double tc_predict_val_col(
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index
    )
{
  if(test->nmodes == 3) {
    return p_predict_val3_col(test->nmodes, test, mats, index);
  }
  return 0.0;
}

bool tc_converge(
    sptensor_t * train,
    sptensor_t * validate,
    sptensor_t * holdout,
    ordi_matrix ** mats,
    ordi_matrix ** best_mats,
    int algorithm_index,
    double const loss,
    double const frobsq,
    idx_t const epoch,
    idx_t nmodes,
    double * best_rmse,
    double * tolerance,
    idx_t * nbadepochs,
    idx_t * bestepochs,
    idx_t * max_badepochs
    )
{
  (void)frobsq;
  (void)nmodes;

  /* train metrics */
  double const train_rmse = (train->nnz > 0) ? sqrt(loss / train->nnz) : 0.0;

  double train_st2 = 0.0;
  for(idx_t x = 0; x < train->nnz; ++x) {
    double t = train->vals[x];
    train_st2 += t * t;
  }
  double const train_er = (train_st2 > 0.0) ? sqrt(loss / train_st2) : 0.0;
  double const train_mae = (train->nnz > 0) ? tc_mae(train, mats, algorithm_index) : 0.0;

  /* validation metrics */
  double val_rmse = 0.0, val_mae = 0.0, val_er = 0.0;
  double converge_rmse = train_rmse;

  if(validate != NULL && validate->nnz > 0) {
    double const val_loss = tc_loss_sq(validate, mats, algorithm_index);
    val_rmse = sqrt(val_loss / validate->nnz);

    double val_st2 = 0.0;
    for(idx_t x = 0; x < validate->nnz; ++x) {
      double t = validate->vals[x];
      val_st2 += t * t;
    }
    val_er = (val_st2 > 0.0) ? sqrt(val_loss / val_st2) : 0.0;

    val_mae = tc_mae(validate, mats, algorithm_index);
    converge_rmse = val_rmse;
  }

  /* holdout (U) metrics */
  double u_rmse = 0.0, u_mae = 0.0, u_er = 0.0;
  if(holdout != NULL && holdout->nnz > 0) {
    double const u_loss = tc_loss_sq(holdout, mats, algorithm_index);
    u_rmse = sqrt(u_loss / holdout->nnz);

    double u_st2 = 0.0;
    for(idx_t x = 0; x < holdout->nnz; ++x) {
      double t = holdout->vals[x];
      u_st2 += t * t;
    }
    u_er = (u_st2 > 0.0) ? sqrt(u_loss / u_st2) : 0.0;
    u_mae = tc_mae(holdout, mats, algorithm_index);
  }

  p_print_progress(
      epoch, loss,
      train_rmse, val_rmse, u_rmse,
      train_mae, val_mae, u_mae,
      train_er, val_er, u_er);

  bool converged = false;

  /* Track exact best checkpoint (for restore), independent from tolerance-
   * based patience tracking used by early stopping. */
  double checkpoint_rmse = 0.0;
  if(validate != NULL && validate->nnz > 0) {
    checkpoint_rmse = tc_rmse(validate, best_mats, algorithm_index);
  } else {
    checkpoint_rmse = tc_rmse(train, best_mats, algorithm_index);
  }
  if(converge_rmse < checkpoint_rmse) {
    *bestepochs = epoch;
    for(idx_t m = 0; m < train->nmodes; ++m) {
      matrix_copy(mats[m], best_mats[m]);
    }
  }

  if(converge_rmse - *(best_rmse) < -*(tolerance)) {
    *nbadepochs = 0;
    *best_rmse = converge_rmse;
  } else {
    *nbadepochs = *nbadepochs + 1;
    if(*nbadepochs >= *max_badepochs) {
      converged = true;
    }
  }

  return converged;
}


/**
* @brief The main function for tensor completion under TB-COO(in ciss.h)
*/
void tc_main_ciss(sptensor_t* traina, sptensor_t* trainb, sptensor_t* trainc,
                  sptensor_t* validation, sptensor_t* test, int algorithm_index,
                  long long project_start_us)
{
  double regularization_index = 0;
  srand(time(NULL));

  idx_t nmodes = traina->nmodes;

  ordi_matrix ** mats = (ordi_matrix**)malloc((MAX_NMODES) * sizeof(ordi_matrix*));
  ordi_matrix ** best_mats = (ordi_matrix**)malloc(nmodes * sizeof(ordi_matrix*));

  idx_t maxdim = traina->dims[argmax_elem(traina->dims, nmodes)];

  for(idx_t m = 0; m < nmodes; m++) {
    mats[m] = (ordi_matrix*) matrix_randomize(traina->dims[m], (idx_t)DEFAULT_NFACTORS);
    best_mats[m] = (ordi_matrix*) matrix_randomize(traina->dims[m], (idx_t)DEFAULT_NFACTORS);
    matrix_copy(mats[m], best_mats[m]);
  }

  mats[MAX_NMODES-1] = matrix_alloc(maxdim, (idx_t)DEFAULT_NFACTORS);

  double best_rmse = (double)BEST_RMSE;
  double tolerance = (double)TOLERANCE;
  idx_t nbadepochs = NBADEPOCHS;
  idx_t bestepochs = BEST_EPOCH;
  idx_t max_badepochs = MAX_BADEPOCHS;

  const char* env_tol = getenv("CUTC_TOLERANCE");
  if(env_tol != NULL && env_tol[0] != '\0') {
    char* endptr = NULL;
    double parsed = strtod(env_tol, &endptr);
    if(endptr != env_tol && parsed >= 0.0) {
      tolerance = parsed;
    }
  }

  const char* env_bad = getenv("CUTC_MAX_BADEPOCHS");
  if(env_bad != NULL && env_bad[0] != '\0') {
    char* endptr = NULL;
    unsigned long long parsed = strtoull(env_bad, &endptr, 10);
    if(endptr != env_bad && parsed > 0ULL) {
      max_badepochs = (idx_t)parsed;
    }
  }

  switch (algorithm_index)
  {
    case 0:
      regularization_index = (double)ALS_REGULARIZATION;
      {
        const char* env_reg = getenv("CUTC_ALS_REGULARIZATION");
        if(env_reg != NULL && env_reg[0] != '\0') {
          char* endptr = NULL;
          double parsed = strtod(env_reg, &endptr);
          if(endptr != env_reg && parsed >= 0.0) {
            regularization_index = parsed;
          }
        }
      }
      tc_als(traina, trainb, trainc, validation, test,
             mats, best_mats, algorithm_index, project_start_us, regularization_index,
             &best_rmse, &tolerance, &nbadepochs, &bestepochs, &max_badepochs);
      break;

    case 1:
      regularization_index = (double)SGD_REGULARIZATION;
      {
        double learning_rate = (double)LEARN_RATE;
        tc_sgd(traina, trainb, trainc, validation, test,
               mats, best_mats, algorithm_index, project_start_us, regularization_index,
               learning_rate,
               &best_rmse, &tolerance, &nbadepochs, &bestepochs, &max_badepochs);
      }
      break;

    default:
      regularization_index = (double)CCD_REGULARIZATION;
      {
        const char* env_reg = getenv("CUTC_CCD_REGULARIZATION");
        if(env_reg != NULL && env_reg[0] != '\0') {
          char* endptr = NULL;
          double parsed = strtod(env_reg, &endptr);
          if(endptr != env_reg && parsed >= 0.0) {
            regularization_index = parsed;
          }
        }
      }
      tc_ccd(traina, trainb, trainc, validation, test,
             mats, best_mats, algorithm_index, project_start_us, regularization_index,
             &best_rmse, &tolerance, &nbadepochs, &bestepochs, &max_badepochs);
      break;
  }

  for(idx_t m = 0; m < nmodes; m++) {
    matrix_free(mats[m]);
    matrix_free(best_mats[m]);
  }
  matrix_free(mats[MAX_NMODES-1]);

  free(mats);
  free(best_mats);
}
