//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// logger_configuration.cpp
//
// Identification: src/backend/benchmark/logger/logger_configuration.cpp
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <iomanip>
#include <algorithm>
#include <sys/stat.h>

#include "backend/common/exception.h"
#include "backend/common/logger.h"
#include "backend/benchmark/logger/logger_configuration.h"
#include "backend/benchmark/ycsb/ycsb_configuration.h"
#include "backend/benchmark/tpcc/tpcc_configuration.h"

extern char* peloton_endpoint_address;

extern ReplicationType peloton_replication_mode;

extern CheckpointType peloton_checkpoint_mode;

extern double peloton_hybrid_storage_ratio;

namespace peloton {
namespace benchmark {
namespace logger {

void Usage(FILE* out) {
  fprintf(out,
          "Command line options :  logger <options> \n"
          "   -h --help              :  Print help message \n"
          "   -a --asynchronous-mode :  Asynchronous mode \n"
          "   -e --experiment-type   :  Experiment Type \n"
          "   -f --data-file-size    :  Data file size (MB) \n"
          "   -l --logging-type      :  Logging type \n"
          "   -n --nvm-latency       :  NVM latency \n"
          "   -p --pcommit-latency   :  pcommit latency \n"
          "   -v --flush-mode        :  Flush mode \n"
          "   -w --commit-interval   :  Group commit interval \n"
          "   -y --benchmark-type    :  Benchmark type \n"
          "   -x --replication-port  :  port for follower \n"
          "   -q --long-running-txn  :  long running txn count \n"
          "   -x --hybrid-storage    :  hybrid storage ratio \n");
}

static struct option opts[] = {
    {"asynchronous_mode", optional_argument, NULL, 'a'},
    {"experiment-type", optional_argument, NULL, 'e'},
    {"data-file-size", optional_argument, NULL, 'f'},
    {"replication-type", optional_argument, NULL, 'g'},
    {"logging-type", optional_argument, NULL, 'l'},
    {"nvm-latency", optional_argument, NULL, 'n'},
    {"pcommit-latency", optional_argument, NULL, 'p'},
    {"skew", optional_argument, NULL, 's'},
    {"flush-mode", optional_argument, NULL, 'v'},
    {"commit-interval", optional_argument, NULL, 'w'},
    {"benchmark-type", optional_argument, NULL, 'y'},
    {"remote-endpoint", optional_argument, NULL, 'z'},
    {"long-running-txn", optional_argument, NULL, 'q'},
    {"hybrid-storage", optional_argument, NULL, 'x'},
    {NULL, 0, NULL, 0}};

//{"replication-port", optional_argument, NULL, 'x'},

static void ValidateLoggingType(const configuration& state) {
  if (state.logging_type <= LOGGING_TYPE_INVALID) {
    LOG_ERROR("Invalid logging_type :: %d", state.logging_type);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("Logging_type :: %s",
           LoggingTypeToString(state.logging_type).c_str());
}

std::string BenchmarkTypeToString(BenchmarkType type) {
  switch (type) {
    case BENCHMARK_TYPE_INVALID:
      return "INVALID";

    case BENCHMARK_TYPE_YCSB:
      return "YCSB";
    case BENCHMARK_TYPE_TPCC:
      return "TPCC";

    default:
      LOG_ERROR("Invalid benchmark_type :: %d", type);
      exit(EXIT_FAILURE);
  }
  return "INVALID";
}

std::string ExperimentTypeToString(ExperimentType type) {
  switch (type) {
    case EXPERIMENT_TYPE_INVALID:
      return "INVALID";

    case EXPERIMENT_TYPE_THROUGHPUT:
      return "THROUGHPUT";
    case EXPERIMENT_TYPE_RECOVERY:
      return "RECOVERY";
    case EXPERIMENT_TYPE_STORAGE:
      return "STORAGE";
    case EXPERIMENT_TYPE_LATENCY:
      return "LATENCY";

    default:
      LOG_ERROR("Invalid experiment_type :: %d", type);
      exit(EXIT_FAILURE);
  }

  return "INVALID";
}

std::string AsynchronousTypeToString(AsynchronousType type) {
  switch (type) {
    case ASYNCHRONOUS_TYPE_INVALID:
      return "INVALID";

    case ASYNCHRONOUS_TYPE_SYNC:
      return "SYNC";
    case ASYNCHRONOUS_TYPE_ASYNC:
      return "ASYNC";
    case ASYNCHRONOUS_TYPE_DISABLED:
      return "DISABLED";
    case ASYNCHRONOUS_TYPE_NO_WRITE:
      return "NO_WRITE";

    default:
      LOG_ERROR("Invalid asynchronous_mode :: %d", type);
      exit(EXIT_FAILURE);
  }

  return "INVALID";
}

static void ValidateBenchmarkType(const configuration& state) {
  if (state.benchmark_type <= 0 || state.benchmark_type > 3) {
    LOG_ERROR("Invalid benchmark_type :: %d", state.benchmark_type);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("%s : %s", "benchmark_type",
           BenchmarkTypeToString(state.benchmark_type).c_str());
}

static void ValidateDataFileSize(const configuration& state) {
  if (state.data_file_size <= 0) {
    LOG_ERROR("Invalid data_file_size :: %lu", state.data_file_size);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("data_file_size :: %lu", state.data_file_size);
}

static void ValidateExperimentType(const configuration& state) {
  if (state.experiment_type < 0 || state.experiment_type > 4) {
    LOG_ERROR("Invalid experiment_type :: %d", state.experiment_type);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("%s : %s", "experiment_type",
           ExperimentTypeToString(state.experiment_type).c_str());
}

static void ValidateWaitTimeout(const configuration& state) {
  if (state.wait_timeout < 0) {
    LOG_ERROR("Invalid wait_timeout :: %d", state.wait_timeout);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("wait_timeout :: %d", state.wait_timeout);
}

static void ValidateFlushMode(const configuration& state) {
  if (state.flush_mode <= 0 || state.flush_mode >= 3) {
    LOG_ERROR("Invalid flush_mode :: %d", state.flush_mode);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("flush_mode :: %d", state.flush_mode);
}

static void ValidateAsynchronousMode(const configuration& state) {
  if (state.asynchronous_mode <= ASYNCHRONOUS_TYPE_INVALID ||
      state.asynchronous_mode > ASYNCHRONOUS_TYPE_NO_WRITE) {
    LOG_ERROR("Invalid asynchronous_mode :: %d", state.asynchronous_mode);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("%s : %s", "asynchronous_mode",
           AsynchronousTypeToString(state.asynchronous_mode).c_str());
}

static void ValidateNVMLatency(const configuration& state) {
  if (state.nvm_latency < 0) {
    LOG_ERROR("Invalid nvm_latency :: %d", state.nvm_latency);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("nvm_latency :: %d", state.nvm_latency);
}

static void ValidatePCOMMITLatency(const configuration& state) {
  if (state.pcommit_latency < 0) {
    LOG_ERROR("Invalid pcommit_latency :: %d", state.pcommit_latency);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("pcommit_latency :: %d", state.pcommit_latency);
}

static void ValidateLogFileDir(configuration& state) {
  struct stat data_stat;

  // Assign log file dir based on logging type
  switch (state.logging_type) {
    // Log file on NVM
    case LOGGING_TYPE_NVM_WAL:
    case LOGGING_TYPE_NVM_WBL: {
      int status = stat(NVM_DIR, &data_stat);
      if (status == 0 && S_ISDIR(data_stat.st_mode)) {
        state.log_file_dir = NVM_DIR;
      }
    } break;

    // Log file on SSD
    case LOGGING_TYPE_SSD_WAL:
    case LOGGING_TYPE_SSD_WBL: {
      int status = stat(SSD_DIR, &data_stat);
      if (status == 0 && S_ISDIR(data_stat.st_mode)) {
        state.log_file_dir = SSD_DIR;
      }
    } break;

    // Log file on HDD
    case LOGGING_TYPE_HDD_WAL:
    case LOGGING_TYPE_HDD_WBL: {
      int status = stat(HDD_DIR, &data_stat);
      if (status == 0 && S_ISDIR(data_stat.st_mode)) {
        state.log_file_dir = HDD_DIR;
      }
    } break;

    case LOGGING_TYPE_INVALID:
    default: {
      int status = stat(TMP_DIR, &data_stat);
      if (status == 0 && S_ISDIR(data_stat.st_mode)) {
        state.log_file_dir = TMP_DIR;
      } else {
        throw Exception("Could not find temp directory : " +
                        std::string(TMP_DIR));
      }
    } break;
  }

  LOG_INFO("log_file_dir :: %s", state.log_file_dir.c_str());
}

static void ValidateLongRunningTxnCount(configuration& state){
  LOG_INFO("long_running_txn_count :: %lu", state.long_running_txn_count);
}

static void ValidateHybridStorageRatio(const configuration& state) {
  if (state.hybrid_storage_ratio < 0 || state.hybrid_storage_ratio > 1) {
    LOG_ERROR("Invalid hybrid_storage_ratio :: %lf", state.hybrid_storage_ratio);
    exit(EXIT_FAILURE);
  }

  LOG_INFO("hybrid_storage_ratio :: %lf", state.hybrid_storage_ratio);
}

void ParseArguments(int argc, char* argv[], configuration& state) {
  // Default Logger Values
  state.logging_type = LOGGING_TYPE_NVM_WAL;
  state.log_file_dir = TMP_DIR;
  state.data_file_size = 512;

  state.experiment_type = EXPERIMENT_TYPE_THROUGHPUT;
  state.wait_timeout = 10;
  state.benchmark_type = BENCHMARK_TYPE_YCSB;
  state.flush_mode = 2;
  state.nvm_latency = 0;
  state.pcommit_latency = 0;
  state.asynchronous_mode = ASYNCHRONOUS_TYPE_SYNC;
  state.replication_port = 0;
  state.remote_endpoint = nullptr;
  state.replication_mode = SYNC_REPLICATION;
  state.checkpoint_type = CHECKPOINT_TYPE_INVALID;
  state.long_running_txn_count = 0;
  state.hybrid_storage_ratio = 0;

  // Default YCSB Values
  ycsb::state.scale_factor = 1;
  ycsb::state.duration = 1000;
  ycsb::state.column_count = 1;
  ycsb::state.update_ratio = 0.5;
  ycsb::state.backend_count = 2;
  ycsb::state.transaction_count = 0;
  ycsb::state.ops_count = 1;
  ycsb::state.abort_mode = false;
  ycsb::state.goetz_mode = false;
  ycsb::state.redo_fraction = 0.1;
  ycsb::state.redo_length = 1;

  // Default Values
  tpcc::state.warehouse_count = 2;  // 10
  tpcc::state.duration = 1000;
  tpcc::state.backend_count = 2;
  tpcc::state.transaction_count = 0;

  srand(23);

  // Parse args
  while (1) {
    int idx = 0;
    // logger - a:e:f:g:hl:n:p:v:w:y:q:
    // ycsb   - b:c:d:k:t:u:o:r:p:m:n:
    // tpcc   - b:d:k:t:
    int c = getopt_long(argc, argv, "a:b:c:d:e:f:g:hij:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:",
                        opts, &idx);

    if (c == -1) break;

    switch (c) {
      case 'a':
        state.asynchronous_mode = (AsynchronousType)atoi(optarg);
        break;
      case 'e':
        state.experiment_type = (ExperimentType)atoi(optarg);
        break;
      case 'f':
        state.data_file_size = atoi(optarg);
        break;
      case 'g': {
        int mode = atoi(optarg);
        switch (mode) {
          case 1:
            state.replication_mode = ASYNC_REPLICATION;
            break;
          case 2:
            state.replication_mode = SEMISYNC_REPLICATION;
            break;
          case 3:
          default:
            state.replication_mode = SYNC_REPLICATION;
            break;
        }
        break;
      }
      case 'i':
        state.checkpoint_type = CHECKPOINT_TYPE_NORMAL;
        break;
      case 'l':
        state.logging_type = (LoggingType)atoi(optarg);
        break;
      case 'n':
        state.nvm_latency = atoi(optarg);
        break;
      case 'p':
        state.pcommit_latency = atoi(optarg);
        break;
      case 'v':
        state.flush_mode = atoi(optarg);
        break;
      case 'w':
        state.wait_timeout = atoi(optarg);
        break;
      //case 'x':
      //  state.replication_port = atoi(optarg);
      //  break;
      case 'z':
        state.remote_endpoint = new char[strlen(optarg)];
        memcpy(state.remote_endpoint, optarg, strlen(optarg));
        peloton_endpoint_address = state.remote_endpoint;
        break;
      case 'y':
        state.benchmark_type = (BenchmarkType)atoi(optarg);
        break;
      case 'q':
        state.long_running_txn_count = atol(optarg);
        break;
      case 'x':
        state.hybrid_storage_ratio = atof(optarg);
        break;

      // YCSB
      case 'b':
        ycsb::state.backend_count = atoi(optarg);
        tpcc::state.backend_count = atoi(optarg);
        break;
      case 'c':
        ycsb::state.column_count = atoi(optarg);
        break;
      case 'd':
        ycsb::state.duration = atoi(optarg);
        tpcc::state.duration = atoi(optarg);
        break;
      case 'k':
        ycsb::state.scale_factor = atoi(optarg);
        tpcc::state.warehouse_count = atoi(optarg);
        break;
      case 't':
        ycsb::state.transaction_count = atoi(optarg);
        tpcc::state.transaction_count = atoi(optarg);
        break;
      case 'u':
        ycsb::state.update_ratio = atof(optarg);
        break;
      case 'o':
        ycsb::state.ops_count = atoi(optarg);
        break;
      case 'r':
        ycsb::state.abort_mode = atoi(optarg);
        break;
      case 's':
        ycsb::state.goetz_mode = atoi(optarg);
        break;
      case 'j':
        ycsb::state.redo_fraction = atof(optarg);
        break;
      case 'm':
        ycsb::state.redo_length = atoi(optarg);
        break;

      case 'h':
        Usage(stderr);
        ycsb::Usage(stderr);
        exit(EXIT_FAILURE);
        break;

      default:
        fprintf(stderr, "\nUnknown option: -%c-\n", c);
        Usage(stderr);
        exit(EXIT_FAILURE);
        break;
    }
  }

  if (state.checkpoint_type == CHECKPOINT_TYPE_NORMAL &&
      (state.logging_type == LOGGING_TYPE_NVM_WAL ||
       state.logging_type == LOGGING_TYPE_SSD_WAL ||
       state.logging_type == LOGGING_TYPE_HDD_WAL)) {
    peloton_checkpoint_mode = CHECKPOINT_TYPE_NORMAL;
  }
  peloton_replication_mode = state.replication_mode;

  // Print Logger configuration
  ValidateLoggingType(state);
  ValidateExperimentType(state);
  ValidateAsynchronousMode(state);
  ValidateBenchmarkType(state);
  ValidateDataFileSize(state);
  ValidateLogFileDir(state);
  ValidateWaitTimeout(state);
  ValidateFlushMode(state);
  ValidateNVMLatency(state);
  ValidatePCOMMITLatency(state);
  ValidateLongRunningTxnCount(state);
  ValidateHybridStorageRatio(state);

  // Print YCSB configuration
  if (state.benchmark_type == BENCHMARK_TYPE_YCSB) {
    ycsb::ValidateScaleFactor(ycsb::state);
    ycsb::ValidateColumnCount(ycsb::state);
    ycsb::ValidateUpdateRatio(ycsb::state);
    ycsb::ValidateBackendCount(ycsb::state);
    ycsb::ValidateDuration(ycsb::state);
    ycsb::ValidateTransactionCount(ycsb::state);
    ycsb::ValidateTransactionCount(ycsb::state);
    ycsb::ValidateOpsCount(ycsb::state);
    ycsb::ValidateAbortMode(ycsb::state);
    ycsb::ValidateGoetzMode(ycsb::state);
    ycsb::ValidateRedoFraction(ycsb::state);
    ycsb::ValidateRedoLength(ycsb::state);

  }
  // Print TPCC configuration
  else if (state.benchmark_type == BENCHMARK_TYPE_TPCC) {
    tpcc::ValidateBackendCount(tpcc::state);
    tpcc::ValidateDuration(tpcc::state);
    tpcc::ValidateWarehouseCount(tpcc::state);
    tpcc::ValidateTransactionCount(tpcc::state);

    // Static TPCC parameters
    tpcc::state.item_count = 1000;            // 100000
    tpcc::state.districts_per_warehouse = 2;  // 10
    tpcc::state.customers_per_district = 30;  // 3000
    tpcc::state.new_orders_per_district = 9;  // 900
  }
}

}  // namespace logger
}  // namespace benchmark
}  // namespace peloton
