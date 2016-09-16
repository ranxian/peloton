//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// sdbench.cpp
//
// Identification: src/main/sdbench/sdbench.cpp
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>

#include "benchmark/sdbench/sdbench_configuration.h"
#include "common/logger.h"
#include "benchmark/sdbench/sdbench_workload.h"
#include "benchmark/sdbench/sdbench_loader.h"

#include <google/protobuf/stubs/common.h>

namespace peloton {
namespace benchmark {
namespace sdbench {

configuration state;

// Main Entry Point
void RunBenchmark() {
  // Initialize settings
  peloton_layout_mode = state.layout_mode;

  // Generate sequence
  GenerateSequence(state.column_count);

  // Single run
  if (state.experiment_type == EXPERIMENT_TYPE_INVALID) {
    CreateAndLoadTable((LayoutType)peloton_layout_mode);

    switch (state.operator_type) {
      case OPERATOR_TYPE_DIRECT: {
        std::vector<oid_t> tuple_key_attrs = {2, 4};
        std::vector<oid_t> index_key_attrs = {0, 1};
        RunQuery(tuple_key_attrs, index_key_attrs);
        break;
      }

      case OPERATOR_TYPE_INSERT:
        RunInsertTest();
        break;

      case OPERATOR_TYPE_JOIN: {
        std::vector<oid_t> left_table_tuple_key_attrs = {1, 2};
        std::vector<oid_t> left_table_index_key_attrs = {0, 1};
        std::vector<oid_t> right_table_tuple_key_attrs = {4, 5};
        std::vector<oid_t> right_table_index_key_attrs = {0, 1};
        oid_t left_table_join_column = 3;
        oid_t right_table_join_column = 6;
        RunJoinTest(left_table_tuple_key_attrs, left_table_index_key_attrs,
                    right_table_tuple_key_attrs, right_table_index_key_attrs,
                    left_table_join_column, right_table_join_column);
        break;
      }

      default:
        LOG_ERROR("Unsupported test type : %d", state.operator_type);
        break;
    }

  }
  // Experiment
  else {
    switch (state.experiment_type) {
      case EXPERIMENT_TYPE_ADAPT:
        RunAdaptExperiment();
        break;

      case EXPERIMENT_TYPE_QUERY:
        RunQueryExperiment();
        break;

      default:
        LOG_ERROR("Unsupported experiment_type : %d", state.experiment_type);
        break;
    }
  }
}

}  // namespace sdbench
}  // namespace benchmark
}  // namespace peloton

int main(int argc, char **argv) {
  peloton::benchmark::sdbench::ParseArguments(
      argc, argv, peloton::benchmark::sdbench::state);

  // seed generator
  srand(23);

  peloton::benchmark::sdbench::RunBenchmark();

  peloton::benchmark::sdbench::sdbench_table.release();

  // shutdown protocol buf library
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}
