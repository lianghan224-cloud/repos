#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <omp.h>
#include <time.h>
#include <memory>

#include "poisson_generator.hpp"
#include "common.hpp"
#include "alto.hpp"
#include "alto_dev.hpp"
#include "cpd.hpp"
#include "utils.hpp"
#include "blco.hpp"
#include "cpd_gpu.hpp"
#include "split_dataset.hpp"

#include <unistd.h>
#include <sys/resource.h>
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>

#include <sched.h>
//#include <numaif.h>

#if ALTO_MASK_LENGTH == 64
    typedef unsigned long long LIType;
#elif ALTO_MASK_LENGTH == 128
    typedef unsigned __int128 LIType;
#else
    #pragma message("!WARNING! ALTO_MASK_LENGTH invalid. Using default 64-bit.")
    typedef unsigned long long LIType;
#endif

#define error(msg...) do {						\
	char ____buf[128];						\
	snprintf(____buf, 128, msg);					\
	fprintf(stderr, "[%s:%d]: %s\n", __FILE__, __LINE__, ____buf);	\
	exit(-1);							\
} while (0)

void BenchmarkAlto(SparseTensor* X, int max_iters, IType rank,
                   IType seed, int target_mode, int num_partitions);

void RunAltoCheck(SparseTensor* X, IType rank, IType seed,
                  int target_mode, int num_partitions);

void VerifyResult(const char* name, FType* truth, FType* factor, IType size);

static void MakeSparseTensor(int nmodes, IType* dims, double sparsity,
                             IType rank, SparseTensor** X);

static std::vector<IType> ParseDimensions(char* argv, int* nmodes_);
static void ApplyDimsOverride(SparseTensor* X, const std::vector<IType>& dims);
static void PrintVersion(char* call);
static void Usage(char* call);

static void PrintTensorInfo(IType rank, int max_iters, SparseTensor* X);

#ifdef memtrace
static long PrintNodeMem(int node, const char* tag);
#endif

const struct option long_opt[] = {
    {"help",           0, NULL, 'h'},
    {"version",        0, NULL, 'v'},
    {"input",          1, NULL, 'i'},
    {"output",         1, NULL, 'o'},
    {"bin",            1, NULL, 'b'},
    {"rank",           1, NULL, 'r'},
    {"max-iter",       1, NULL, 'm'},
    {"seed",           1, NULL, 'x'},
    {"dims",           1, NULL, 'd'},
    {"target-mode",    1, NULL, 't'},
    {"sparsity",       1, NULL, 's'},
    {"epsilon",        1, NULL, 'e'},
    {"file",           1, NULL, 'f'},
    {"check",          0, NULL, 'c'},
    {"bench",          0, NULL, 'p'},
    {"kernel-id",	   1, NULL, 'k'},
    {"num-partitions", 1, NULL, 'n'},
    {"device",         1, NULL, 1001},
    {"thread-cf",      1, NULL, 1002},
	{"stream-data",	   0, NULL, 1003},
	{"max-block-size", 1, NULL, 1004},
	{"batch",		   0, NULL, 1005},
    {"val-rate",       1, NULL, 1006},
    {"sampling-rate",  1, NULL, 1007},
    {"sampling-mode",  1, NULL, 1008},
    {"sampling-alpha", 1, NULL, 1009},
    {"sampling-eps",   1, NULL, 1010},
    {"train-file",     1, NULL, 1011},
    {"val-file",       1, NULL, 1012},
    {"test-file",      1, NULL, 1013},
    {NULL,             0, NULL,    0}
};

const char* const short_opt = "hvi:o:b:r:m:x:d:t:s:e:f:cpk:n:b:";
const char* version_info = "0.1.1";

int main(int argc, char** argv)
{
    struct timeval project_start_tv;
    gettimeofday(&project_start_tv, NULL);
    long long project_start_us =
        1000000LL * (long long)project_start_tv.tv_sec + (long long)project_start_tv.tv_usec;

	// Set up timer
	InitTSC();
	uint64_t ticks_start = 0;
	uint64_t ticks_end = 0;
	double t_read = 0.0;
	double t_write = 0.0;
	double t_create = 0.0;
	double t_cpd = 0.0;

	int max_iters = 150;
	IType rank = 16;
	std::vector<IType> dims;
	int nmodes = 0;
	int target_mode = 0;
	std::string text_file;
	std::string text_file_out;
	std::string binary_file;
    std::string train_file;
    std::string val_file;
    std::string test_file;
	double sparsity = 0.1;
	double epsilon = 1e-5;
	int seed = time(NULL);
	int save_to_file = 0;
    bool do_check = false;
    bool do_mttkrp_bench = false;
    int kernel_id = 0;
    int num_partitions = omp_get_max_threads();
    int device = 0;
    IType thread_cf = 2;
	IType max_block_size = 16777216;
	bool stream_data = false;
	bool do_batching = false;
    double val_rate = 0.05;
    double sampling_rate = 0.1;
    int sampling_mode = 0;
    double sampling_alpha = 1.0;
    double sampling_eps = 1e-6;
    const uint32_t val_seed = 20250101u;
    const uint32_t omega_seed = 2025u;

	int c = 0;
	while ((c = getopt_long(argc, argv, short_opt, long_opt, NULL)) != -1) {
		switch (c) {
		case 'h':
			Usage(argv[0]);
			return 0;
		case 'v':
			PrintVersion(argv[0]);
			return 0;
		case 'i':
			text_file = std::string(optarg);
			break;
		case 'o':
			text_file_out = std::string(optarg);
			break;
		case 'b':
			binary_file = std::string(optarg);
			break;
		case 'r':
			rank = (IType)atoll(optarg);
			if (rank < 1) {
				fprintf(stderr, "Invalid -rank: %s.\n", optarg);
			}
			break;
		case 'm':
			max_iters = atoi(optarg);
			if (max_iters < 0) {
				fprintf(stderr, "Invalid -max-iter: %s.\n", optarg);
			}
			break;
		case 'x':
			seed = atoi(optarg);
			if (seed < 0) {
				fprintf(stderr, "Invalid -seed: %s.\n", optarg);
			}
			break;
		case 'd':
			dims = ParseDimensions(optarg, &nmodes);
                        if (dims.empty()) {
				fprintf(stderr, "Invalid -dims: %s.\n", optarg);
			}
			break;
		case 't':
			target_mode = atoi(optarg);
			if (target_mode < -1) {
				fprintf(stderr, "Invalid -target-mode: %s.\n", optarg);
                return -1;
			}
			break;
		case 's':
			sparsity = atof(optarg);
			if (sparsity <= 0.0) {
				fprintf(stderr, "Invalid -sparsity: %s.\n", optarg);
			}
			break;
		case 'e':
			epsilon = atof(optarg);
			if (epsilon < 0.0) {
				fprintf(stderr, "Invalid -epsilon: %s.\n", optarg);
			}
			break;
		case 'f':
			save_to_file = atoi(optarg);
			if (save_to_file < 0) {
				fprintf(stderr, "Invalid -file: %s.\n", optarg);
			}
			break;
        case 'c':
            do_check = true;
            break;
        case 'p':
            do_mttkrp_bench = true;
            break;
        case 'k':
            kernel_id = atoi(optarg);
            if (kernel_id < 0) {
                fprintf(stderr, "Invalid -kernel-id: %s.\n", optarg);
                return -1;
            }
//            if (kernel_id > 0) {
//                do_mttkrp_bench = false;
//                do_check = false;
//            }
            break;
        case 'n':
            num_partitions = atoi(optarg);
            if (num_partitions <= 0) {
                fprintf(stderr, "Invalid -num-partitions: %s.\n", optarg);
                return -1;
            }
            break;
        case 1001:
            device = atoi(optarg);
            if (device < 0) {
                fprintf(stderr, "Invalid -device: %s.\n", optarg);
                return -1;
            }
            break;
        case 1002:
            thread_cf = atoi(optarg);
            if (thread_cf <= 0) {
                fprintf(stderr, "Invalid -thread-cf: %s.\n", optarg);
                return -1;
            }
            break;
		case 1003:
			stream_data = true;
			break;
		case 1004:
            max_block_size = atoi(optarg);
            if (max_block_size <= 0) {
                fprintf(stderr, "Invalid -max-block-size: %s.\n", optarg);
                return -1;
            }
            break;
		case 1005:
			do_batching = true;
			break;
        case 1006:
            val_rate = atof(optarg);
            if (val_rate < 0.0 || val_rate > 1.0) {
                fprintf(stderr, "Invalid --val-rate: %s.\n", optarg);
                return -1;
            }
            break;
        case 1007:
            sampling_rate = atof(optarg);
            if (sampling_rate < 0.0 || sampling_rate > 1.0) {
                fprintf(stderr, "Invalid --sampling-rate: %s.\n", optarg);
                return -1;
            }
            break;
        case 1008:
            sampling_mode = atoi(optarg);
            if (sampling_mode != 0 && sampling_mode != 1) {
                fprintf(stderr, "Invalid --sampling-mode: %s. Must be 0 or 1.\n", optarg);
                return -1;
            }
            break;
        case 1009:
            sampling_alpha = atof(optarg);
            if (sampling_alpha < 0.0) {
                fprintf(stderr, "Invalid --sampling-alpha: %s.\n", optarg);
                return -1;
            }
            break;
        case 1010:
            sampling_eps = atof(optarg);
            if (sampling_eps < 0.0) {
                fprintf(stderr, "Invalid --sampling-eps: %s.\n", optarg);
                return -1;
            }
            break;
        case 1011:
            train_file = std::string(optarg);
            break;
        case 1012:
            val_file = std::string(optarg);
            break;
        case 1013:
            test_file = std::string(optarg);
            break;
		case ':':
			fprintf(stderr, "Option -%c requires an argument.\n", optopt);
			return -1;
		case '?':
			fprintf(stderr, "Unknown option `-%c'.\n", optopt);
			return -1;
		default:
			Usage(argv[0]);
			return -1;
		} // switch (c)
	} // while ((c = getopt_long(...)))

#ifdef memtrace
	printf("Before any initialization\n");
	long long pre_n0, pre_n1;
	pre_n0 = PrintNodeMem(0, "Active:");
	pre_n1 = PrintNodeMem(1, "Active:");
#endif
	SparseTensor* X = NULL;
    SparseTensor* X_presplit_val = NULL;
    SparseTensor* X_presplit_holdout = NULL;
    const bool use_presplit = !train_file.empty() || !val_file.empty() || !test_file.empty();
    if (use_presplit) {
        if (train_file.empty() || val_file.empty() || test_file.empty()) {
            fprintf(stderr, "--train-file, --val-file, and --test-file must be provided together.\n");
            return -1;
        }
        BEGIN_TIMER(&ticks_start);
        ImportSparseTensor(train_file.c_str(), TEXT_FORMAT, &X);
        ImportSparseTensor(val_file.c_str(), TEXT_FORMAT, &X_presplit_val);
        ImportSparseTensor(test_file.c_str(), TEXT_FORMAT, &X_presplit_holdout);
        ApplyDimsOverride(X, dims);
        ApplyDimsOverride(X_presplit_val, dims);
        ApplyDimsOverride(X_presplit_holdout, dims);
        END_TIMER(&ticks_end);
        ELAPSED_TIME(ticks_start, ticks_end, &t_read);
        PRINT_TIMER("Reading pre-split text files", t_read);
        printf("[split3] using pre-split input tensors: train=%llu validation=%llu test=%llu\n",
               (unsigned long long)X->nnz,
               (unsigned long long)X_presplit_val->nnz,
               (unsigned long long)X_presplit_holdout->nnz);
    }
	else if (!binary_file.empty()) {
		BEGIN_TIMER(&ticks_start);
		ImportSparseTensor(binary_file.c_str(), BINARY_FORMAT, &X);
        ApplyDimsOverride(X, dims);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_read);
		PRINT_TIMER("Reading binary file", t_read);

		if (save_to_file) {
			BEGIN_TIMER(&ticks_start);
			ExportSparseTensor(NULL, TEXT_FORMAT, X);
			END_TIMER(&ticks_end);
			ELAPSED_TIME(ticks_start, ticks_end, &t_write);
			PRINT_TIMER("Writing to text file", t_write);
		}
	}
	else if (!text_file.empty()) {
		BEGIN_TIMER(&ticks_start);
		ImportSparseTensor(text_file.c_str(), TEXT_FORMAT, &X);
        ApplyDimsOverride(X, dims);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_read);
		PRINT_TIMER("Reading text file", t_read);

		if (save_to_file) {
			BEGIN_TIMER(&ticks_start);
			ExportSparseTensor(NULL, BINARY_FORMAT, X);
			END_TIMER(&ticks_end);
			ELAPSED_TIME(ticks_start, ticks_end, &t_write);
			PRINT_TIMER("Writing to binary file", t_write);
		}
	}
	else if (dims.empty()) {
		fprintf(stderr, "No dims specified... exiting\n");
		Usage(argv[0]);
		exit(-1);
	}
	else {
		BEGIN_TIMER(&ticks_start);
		MakeSparseTensor(nmodes, &dims[0], sparsity, rank, &X);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_create);
		PRINT_TIMER("Creating a new tensor", t_create);

		if (save_to_file) {
			BEGIN_TIMER(&ticks_start);
			ExportSparseTensor(NULL, TEXT_FORMAT, X);
			ExportSparseTensor(NULL, BINARY_FORMAT, X);
			END_TIMER(&ticks_end);
			ELAPSED_TIME(ticks_start, ticks_end, &t_write);
			PRINT_TIMER("Wrinting to text/binary files", t_write);

		}
	}

    // GPU driver
    if (kernel_id) {
        if (do_check) max_iters = 1;
        check_cuda(cudaSetDevice(device), "cudaSetDevice");

        SparseTensor* X_val = NULL;
        SparseTensor* X_holdout = NULL;

        if (use_presplit) {
            X_val = X_presplit_val;
            X_holdout = X_presplit_holdout;
            X_presplit_val = NULL;
            X_presplit_holdout = NULL;
        } else if (!do_check && !do_mttkrp_bench) {
            SparseTensor* X_train = NULL;
            SplitConfig split_cfg;
            split_cfg.val_rate = val_rate;
            split_cfg.val_seed = val_seed;
            split_cfg.sampling_rate = sampling_rate;
            split_cfg.sampling_mode = sampling_mode;
            split_cfg.sampling_alpha = sampling_alpha;
            split_cfg.sampling_eps = sampling_eps;
            split_cfg.omega_seed = omega_seed;

            if (SplitSparseTensorThreeSets(X, split_cfg, &X_train, &X_val, &X_holdout) != 0 ||
                !X_train || !X_val || !X_holdout) {
                fprintf(stderr, "[split3] failed to split input tensor\n");
                if (X_train) DestroySparseTensor(X_train);
                if (X_val) DestroySparseTensor(X_val);
                if (X_holdout) DestroySparseTensor(X_holdout);
                DestroySparseTensor(X);
                return -1;
            }

            DestroySparseTensor(X);
            X = X_train;
            printf("[split3] remap done: train=Omega, validation=V, test=U\n");
        }

		if (!stream_data) max_block_size = X->nnz;
		blcotensor* tensor = gen_blcotensor_host<LIType>(X, max_block_size);

        //setup
        PrintTensorInfo(rank, max_iters, X);
        KruskalModel* M;
        CreateKruskalModel(X->nmodes, X->dims, rank, &M);
        KruskalModelRandomInit(M, (unsigned int)seed);

        if (do_check) {
			if (!do_mttkrp_bench) printf("Warning: check not implemented for CPD. Checking MTTKRP instead\n");

            // Create factors for ground truth
            FType **truth = (FType **) AlignedMalloc(X->nmodes * sizeof(FType*));
            FType **factors = (FType **) AlignedMalloc(X->nmodes * sizeof(FType*));
            assert(truth);
            assert(factors);
            for (int m = 0; m < X->nmodes; m++) {
                truth[m] = (FType *) AlignedMalloc(X->dims[m] * rank * sizeof(FType));
                factors[m] = (FType *) AlignedMalloc(X->dims[m] * rank * sizeof(FType));
                assert(truth[m]);
                assert(factors[m]);
            }
            // Initialize factors
            for (int m = 0; m < X->nmodes; ++m) {
                memcpy(factors[m], M->U[m], X->dims[m] * rank * sizeof(FType));
            }
            // Do base mttkrp
            printf("===Do base run===\n");
            mttkrp(X, M, (IType) target_mode);
            printf("--> MTTKRP sequential base run done.\n");
            // Copy to ground truth and reset
            for (int m = 0; m < X->nmodes; ++m) {
                memcpy(truth[m], M->U[m], X->dims[m] * rank * sizeof(FType));
                memcpy(M->U[m], factors[m], X->dims[m] * rank * sizeof(FType));
            }

            printf("===Do ALTO GPU run===\n");
            mttkrp_alto_dev<IType>(tensor, M, kernel_id, max_iters, target_mode, thread_cf, stream_data, do_batching, num_partitions);
            printf("MTTKRP ALTO GPU run done.\n");

            // Verify ALTO
            printf("===Verify ALTO-DEV MTTKRP=== (target mode: %d)\n", target_mode);
            for (int m = 0; m < X->nmodes; ++m) {
                printf("mode %d: ", m);
                fflush(stdout);
                VerifyResult("mttkrp_alto_dev", truth[m], M->U[m], tensor->modes[m] * rank);
            }
        }//do_check
        else {
            if (do_mttkrp_bench) mttkrp_alto_dev<IType>(tensor, M, kernel_id, max_iters, target_mode, thread_cf, stream_data, do_batching, num_partitions);
			else cpals_blco_dev(tensor, M, max_iters, epsilon, kernel_id, stream_data, do_batching, thread_cf, num_partitions, X, X_val, X_holdout, project_start_us);
			if (text_file_out != "") {
				ExportKruskalModel(M, text_file_out.c_str());
				printf("--> Kruskal model saved to %s\n", text_file_out.c_str());
			}
        }//do_check

		// Cleanup
		DestroySparseTensor(X);
        if (X_val) DestroySparseTensor(X_val);
        if (X_holdout) DestroySparseTensor(X_holdout);
		delete_blcotensor_host(tensor);

        return 0; // Premature quit
    }

    // if check flag is given, only do this
    if (do_check) {
        RunAltoCheck(X, rank, seed, target_mode, omp_get_max_threads());
        return 0;
    }
    if (do_mttkrp_bench) {
#ifdef memtrace
	printf("After initialization\n");
	long long post_n0, post_n1;
	post_n0 = PrintNodeMem(0, "Active:");
	post_n1 = PrintNodeMem(1, "Active:");
	printf("memory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
    	// adjust target mode and number of partitions accordingly
	    BenchmarkAlto(X, max_iters, rank, seed, target_mode, omp_get_max_threads());
#ifdef memtrace
	printf("After compute\n");

	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("memory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	    return 0;
    }

	PrintTensorInfo(rank, max_iters, X);

	// Set up the factor matrices
	KruskalModel* M;
	CreateKruskalModel(X->nmodes, X->dims, rank, &M);
	KruskalModelRandomInit(M, (unsigned int)seed);
	// PrintKruskalModel(M);

	/*BEGIN_TIMER(&ticks_start);
	cpd(X, M, max_iters, epsilon);
	END_TIMER(&ticks_end);
	ELAPSED_TIME(ticks_start, ticks_end, &t_cpd);
	PRINT_TIMER("CPD (COO)", t_cpd);

	ExportKruskalModel(M, text_file_out.c_str());*/

	// Convert COO to ALTO
	AltoTensor<LIType>* AT;
	//int num_partitions = omp_get_max_threads();
	create_alto(X, &AT, num_partitions);

    BEGIN_TIMER(&ticks_start);
    cpd_alto(AT, M, max_iters, epsilon);
    // cpd(X, M, max_iters, epsilon);
    END_TIMER(&ticks_end);
    ELAPSED_TIME(ticks_start, ticks_end, &t_cpd);
    PRINT_TIMER("CPD (ALTO)", t_cpd);

    // Cleanup
	DestroySparseTensor(X);
    if (X_presplit_val) DestroySparseTensor(X_presplit_val);
    if (X_presplit_holdout) DestroySparseTensor(X_presplit_holdout);
	DestroyKruskalModel(M);
	destroy_alto(AT);

	return 0;
}

static void ApplyDimsOverride(SparseTensor* X, const std::vector<IType>& dims)
{
    if (!X || dims.empty()) return;
    if ((int)dims.size() != X->nmodes) {
        fprintf(stderr, "Invalid --dims count for tensor: got %zu expected %d.\n",
                dims.size(), X->nmodes);
        exit(-1);
    }
    for (int m = 0; m < X->nmodes; ++m) {
        X->dims[m] = dims[(size_t)m];
    }
}

void BenchmarkAlto(SparseTensor* X, int max_iters, IType rank,
                   IType seed, int target_mode, int num_partitions)
{
	double wtime_s, wtime;

    PrintTensorInfo(rank, max_iters, X);
#ifdef memtrace
	printf("After function call\n");
	long long pre_n0, pre_n1;
	pre_n0 = node_mem(0, "Active:");
	pre_n1 = node_mem(1, "Active:");
#endif
	// Set up the factor matrices
	KruskalModel* M;
	CreateKruskalModel(X->nmodes, X->dims, rank, &M);
	KruskalModelRandomInit(M, (unsigned int)seed);
	// PrintKruskalModel(M);
#ifdef memtrace
	printf("After KruskalModel\n");
	long long post_n0, post_n1;
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	FType** factors = (FType**)AlignedMalloc(X->nmodes * sizeof(FType*));
	assert(factors);
	for (int m = 0; m < X->nmodes; m++) {
        factors[m] = (FType*)AlignedMalloc(X->dims[m] * rank * sizeof(FType));
		assert(factors[m]);
	}
	// Initialize factors
	for (int m = 0; m < X->nmodes; ++m) {
		ParMemcpy(factors[m], M->U[m], X->dims[m] * rank * sizeof(FType));
	}
#ifdef memtrace
	printf("After factors allocation \n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	// ---------------------------------------------------------------- //
	// Create ALTO tensor from COO
	AltoTensor<LIType>* AT;
    wtime_s = omp_get_wtime();
	create_alto(X, &AT, num_partitions);
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO creation time:   %f\n", wtime);
#ifdef memtrace
	printf("After create_alto \n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
    DestroyKruskalModel(M);
    DestroySparseTensor(X);
#ifdef memtrace
	printf("After deletion of SparseTensor\n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	// set up OpenMP locks
    omp_lock_t* writelocks = NULL;
	//IType max_mode_len = 0;
	//for (IType i = 0; i < M->mode; ++i) {
	//	if (max_mode_len < M->dims[i]) {
	//		max_mode_len = M->dims[i];
	//	}
	//}
	//omp_lock_t* writelocks = (omp_lock_t*)AlignedMalloc(sizeof(omp_lock_t) *
	//	max_mode_len);
	//assert(writelocks);
	//for (IType i = 0; i < max_mode_len; ++i) {
	//	omp_init_lock(&(writelocks[i]));
	//}
	FType** ofibs = NULL;
	create_da_mem(target_mode, rank, AT, &ofibs);

#ifdef memtrace
	printf("After create_da_mem \n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
    // warmup
    mttkrp_alto_par(target_mode, factors, rank, AT, writelocks, ofibs);
	// Do ALTO mttkrp
	wtime_s = omp_get_wtime();
	for (int i = 0; i < max_iters; ++i) {
		if (target_mode == -1) {
			for (int m = 0; m < AT->nmode; ++m) {
				mttkrp_alto_par(m, factors, rank, AT, writelocks, ofibs);
			}
		}
		else {
			mttkrp_alto_par(target_mode, factors, rank, AT, writelocks, ofibs);
		}
	}
	wtime = omp_get_wtime() - wtime_s;
	printf("ALTO runtime:   %f\n", wtime);
#ifdef memtrace
	printf("After mttkrp_alto_par \n");
	post_n0 = node_mem(0, "Active:");
	post_n1 = node_mem(1, "Active:");
	printf("\nmemory N0 %lld B N1 %lld B\n", (post_n0 - pre_n0), (post_n1 - pre_n1));
#endif
	// ---------------------------------------------------------------- //
	// Cleanup
	destroy_da_mem(AT, ofibs, rank, target_mode);
	destroy_alto(AT);
	// ---------------------------------------------------------------- //
}

void RunAltoCheck(SparseTensor* X, IType rank, IType seed,
                  int target_mode, int num_partitions)
{
    printf("===Check for mode %d===\n", target_mode);
    double wtime_s, wtime;

    PrintTensorInfo(rank, 1, X);
    // ---------------------------------------------------------------- //
    // Set up the factor matrices
    printf("===Create factor matrices===\n");
    KruskalModel *M;
    CreateKruskalModel(X->nmodes, X->dims, rank, &M);
    KruskalModelRandomInit(M, (unsigned int) seed);

    // Create factors for ground truth and ALTO
    FType **truth = (FType **) AlignedMalloc(X->nmodes * sizeof(FType*));
    assert(truth);
    FType **factors = (FType **) AlignedMalloc(X->nmodes * sizeof(FType*));
    assert(factors);
    for(int m = 0; m < X->nmodes; m++) {
        truth[m] = (FType *) AlignedMalloc(X->dims[m] * rank * sizeof(FType));
        assert(truth[m]);
        factors[m] = (FType *) AlignedMalloc(X->dims[m] * rank * sizeof(FType));
        assert(factors[m]);
    }
    // Initialize factors
    for(int m = 0; m < X->nmodes; ++m) {
        memcpy(factors[m], M->U[m], X->dims[m] * rank * sizeof(FType));
    }
    // ---------------------------------------------------------------- //
    // Do base mttkrp
    printf("===Do base run===\n");
    if (target_mode == -1) {
        for (int m = 0; m < X->nmodes; ++m) {
            printf("mode %d...", m);
            fflush(stdout);
            mttkrp(X, M, (IType) m);
        }
        printf("\n");
    } else {
        mttkrp(X, M, (IType) target_mode);
    }
    printf("   MTTKRP sequential base run done.\n");
    // Copy to ground truth
    for (int m = 0; m < X->nmodes; ++m) {
        memcpy(truth[m], M->U[m], X->dims[m] * rank * sizeof(FType));
    }
    // ---------------------------------------------------------------- //
    // Create ALTO tensor from COO
    printf("===Create ALTO tensors===\n");
    AltoTensor<LIType> *AT;
    create_alto(X, &AT, num_partitions);

    FType **ofibs = NULL;
	create_da_mem(target_mode, rank, AT, &ofibs);

    printf("===Run ALTO MTTKRP===\n");
    wtime_s = omp_get_wtime();
    if (target_mode == -1) {
        for (int m = 0; m < AT->nmode; ++m) {
            printf("mode %d...", m);
            fflush(stdout);
            mttkrp_alto_par(m, factors, rank, AT, NULL, ofibs);
        }
        printf("\n");
    } else {
        mttkrp_alto_par(target_mode, factors, rank, AT, NULL, ofibs);
    }
    wtime = omp_get_wtime() - wtime_s;
    printf("   ALTO runtime: %f s\n", wtime);
    // ---------------------------------------------------------------- //
    // Verify ALTO
    printf("===Verify ALTO MTTKRP=== (target mode: %d)\n", target_mode);
    for (int m = 0; m < AT->nmode; ++m) {
        printf("mode %d: ", m);
        fflush(stdout);
        VerifyResult("mttkrp_alto", truth[m], factors[m], AT->dims[m] * rank);
    }
    // ---------------------------------------------------------------- //
    // Cleanup
	destroy_da_mem(AT, ofibs, rank, target_mode);
    DestroySparseTensor(X);
    destroy_alto(AT);
    // ---------------------------------------------------------------- //
}

void VerifyResult(const char* name, FType* truth, FType* factor, IType size)
{
	IType cnt = 0;
	if (truth != NULL) {
		for (IType i = 0; i < size; ++i) {
            FType moe = std::max(std::abs(truth[i]) * 1e-6, 1e-6); // Scaled margin of error
			if (fabs(truth[i] - factor[i]) > moe) {
				cnt++;
				//printf("Mismatch at %d: truth %f vs factor %f\n", i, truth[i], factor[i]);
			}
		}
	}
	else {
		fprintf(stderr, "truth is NULL - nothing to compare against\n");
		exit(-1);
	}

	if (cnt == 0) {
		fprintf(stderr, "Results of %s is correct\n", name);
	}
	else {
		fprintf(stderr, "Results of %s is incorrect by %llu\n", name, cnt);
	}
}

static void MakeSparseTensor(int nmodes, IType* dims, double sparsity,
                             IType rank, SparseTensor** X)
{
	IType tmp = 1;
	for (int m = 0; m < nmodes; m++) {
		tmp = tmp * dims[m];
	}
	IType nnz_before = (IType)(sparsity * tmp);

	SparseTensor* X_ = NULL;
	KruskalModel* M_true = NULL;

	// ---------------------------------------------------------------- //
	// Create a Poisson distribution random data
	PoissonGenerator* pg;
	CreatePoissonGenerator(nmodes, dims, &pg);
	IType num_edges = nnz_before;
	PoissonGeneratorRun(pg, num_edges, rank, &M_true, &X_);
	DestroyPoissonGenerator(pg);
	// ---------------------------------------------------------------- //

	DestroyKruskalModel(M_true);
	*X = X_;
}

static std::vector<IType> ParseDimensions(char* argv, int* nmodes_)
{
	std::vector<IType> dims;
	std::string dstr(argv);
	std::string::size_type pos = 0, end = 0;

	while (end != std::string::npos) {
		end = dstr.find(',', pos);

                auto count = end == std::string::npos ? end : end - pos;
		IType d = (IType)stoll(dstr.substr(pos, count));
		if (d >= 0)
			dims.push_back(d);
		pos = end + 1;
	}

	*nmodes_ = dims.size();
	return dims;
}

static void PrintVersion(char* call)
{
	fprintf(stdout, "%s version %s\n", call, version_info);
}

static void Usage(char* call)
{
	fprintf(stderr, "Usage: %s [OPTIONS]\n", call);
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "\t-h or --help         Display this information\n");
	fprintf(stderr, "\t-v or --version      Display version information\n");
	fprintf(stderr, "\t-i or --input        Input tensor file in text\n");
	fprintf(stderr, "\t-o or --output       Output tensor to file\n");
	fprintf(stderr, "\t-b or --bin          Input tensor file in binary\n");
	fprintf(stderr, "\t-f or --file         Save tensor to another format\n");
	fprintf(stderr, "\t-r or --rank         Rank\n");
	fprintf(stderr, "\t-m or --max-iter     Maximum outter iterations\n");
	fprintf(stderr, "\t-t or --target-mode  Target mode of tensor\n");
	fprintf(stderr, "\t-e or --epsilon      Convergence criteria\n");
	fprintf(stderr, "\t-x or --seed         Random value seed\n");
	fprintf(stderr, "\t-d or --dims         Dimension lenghts (I,J,K)\n");
	fprintf(stderr, "\t-s or --sparsity     Sparsity of generate tensor\n");
    fprintf(stderr, "\t-c or --check        Run ALTO (par) validation against cpd MTTKRP\n");
    fprintf(stderr, "\t-p or --bench        Run ALTO (par) MTTKRP benchmark with the given CMD line options\n");
    
	fprintf(stderr, "\t-k or --kernel-id    If nonzero, the ID of the GPU kernel to run (10 for auto)\n");
    fprintf(stderr, "\t-n or --num-partitions Number of partitions\n");
    fprintf(stderr, "\t--device 			CUDA device ordinal\n");
    fprintf(stderr, "\t--thread-cf 			Thread coarsening factor\n");
	fprintf(stderr, "\t--stream-data		If set, stream data to GPU during computation\n");
	fprintf(stderr, "\t--max-block-size		Block memory reservation on GPU\n");
	fprintf(stderr, "\t--batch		If set, level 1 kernel batching will be enabled\n");
    fprintf(stderr, "\t--val-rate             Validation split ratio (default 0.05)\n");
    fprintf(stderr, "\t--sampling-rate        Omega sampling ratio from train_pool (default 0.1)\n");
    fprintf(stderr, "\t--sampling-mode        0=uniform, 1=value-biased (default 0)\n");
    fprintf(stderr, "\t--sampling-alpha       Value-biased alpha (default 1.0)\n");
    fprintf(stderr, "\t--sampling-eps         Value-biased epsilon (default 1e-6)\n");
}

static void PrintTensorInfo(IType rank, int max_iters, SparseTensor* X)
{
	IType* dims = X->dims;
	IType nnz = X->nnz;
	int nmodes = X->nmodes;

	IType tmp = 1;
	for (int i = 0; i < nmodes; i++) {
		tmp *= dims[i];
	}
	double sparsity = ((double)nnz) / tmp;
	fprintf(stderr, "# Modes         = %u\n", nmodes);
	fprintf(stderr, "Rank            = %llu\n", rank);
	fprintf(stderr, "Sparsity        = %f\n", sparsity);
	fprintf(stderr, "Max iters       = %d\n", max_iters);
	fprintf(stderr, "Dimensions      = [%llu", dims[0]);
	for (int i = 1; i < nmodes; i++) {
		fprintf(stderr, " X %llu", dims[i]);
	}
	fprintf(stderr, "]\n");
	fprintf(stderr, "NNZ             = %llu\n", nnz);
}

#ifdef memtrace
static long PrintNodeMem(int node, const char* tag)
{
	// Parse form "Node 0 Active:            43088 kB"
	char name[128], line[1024];
	snprintf(name, 128, "/sys/devices/system/node/node%d/meminfo", node);
	FILE* in = fopen(name, "r");
	assert(in);

	long val = -1;

	while (fgets(line, 1024, in)) {
		char* where = strstr(line, tag);
		printf("%s", line);
		if (!where)
			continue;
		char* ptr = where + strlen(tag);
		while (isspace(*ptr))
			++ptr;
		char* eptr;
		val = strtol(ptr, &eptr, 10);
		if (strstr(eptr, "kB"))
			val *= 1024;
		break;
	}

	// Could keep both files open...
	fclose(in);

	return val;
}
#endif
