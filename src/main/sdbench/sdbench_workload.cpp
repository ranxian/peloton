//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// sdbench_workload.cpp
//
// Identification: src/main/sdbench/sdbench_workload.cpp
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "brain/index_tuner.h"
#include "brain/layout_tuner.h"
#include "brain/sample.h"

#include "benchmark/sdbench/sdbench_loader.h"
#include "benchmark/sdbench/sdbench_workload.h"

#include "catalog/manager.h"
#include "catalog/schema.h"
#include "common/logger.h"
#include "common/macros.h"
#include "common/timer.h"
#include "common/types.h"
#include "common/value.h"
#include "common/value_factory.h"
#include "concurrency/transaction.h"
#include "concurrency/transaction_manager_factory.h"

#include "executor/abstract_executor.h"
#include "executor/aggregate_executor.h"
#include "executor/executor_context.h"
#include "executor/hybrid_scan_executor.h"
#include "executor/insert_executor.h"
#include "executor/logical_tile.h"
#include "executor/logical_tile_factory.h"
#include "executor/materialization_executor.h"
#include "executor/nested_loop_join_executor.h"
#include "executor/projection_executor.h"
#include "executor/seq_scan_executor.h"
#include "executor/update_executor.h"

#include "expression/abstract_expression.h"
#include "expression/comparison_expression.h"
#include "expression/conjunction_expression.h"
#include "expression/constant_value_expression.h"
#include "expression/expression_util.h"
#include "expression/operator_expression.h"
#include "expression/tuple_value_expression.h"

#include "planner/abstract_plan.h"
#include "planner/aggregate_plan.h"
#include "planner/hybrid_scan_plan.h"
#include "planner/insert_plan.h"
#include "planner/materialization_plan.h"
#include "planner/nested_loop_join_plan.h"
#include "planner/projection_plan.h"
#include "planner/seq_scan_plan.h"
#include "planner/update_plan.h"

#include "storage/data_table.h"
#include "storage/table_factory.h"
#include "storage/tile.h"
#include "storage/tile_group.h"
#include "storage/tile_group_header.h"

namespace peloton {
namespace benchmark {
namespace sdbench {

// Function definitions
static std::shared_ptr<index::Index> PickIndex(storage::DataTable *table,
                                               std::vector<oid_t> query_attrs);

static void AggregateQueryHelper(const std::vector<oid_t> &tuple_key_attrs,
                                 const std::vector<oid_t> &index_key_attrs);

static void JoinQueryHelper(
    const std::vector<oid_t> &left_table_tuple_key_attrs,
    const std::vector<oid_t> &left_table_index_key_attrs,
    const std::vector<oid_t> &right_table_tuple_key_attrs,
    const std::vector<oid_t> &right_table_index_key_attrs,
    const oid_t left_table_join_column, const oid_t right_table_join_column);

// Tuple id counter
oid_t sdbench_tuple_counter = -1000000;

std::vector<oid_t> column_counts = {50, 500};

// Index tuner
brain::IndexTuner &index_tuner = brain::IndexTuner::GetInstance();

// Layout tuner
brain::LayoutTuner &layout_tuner = brain::LayoutTuner::GetInstance();

static int GetLowerBound() {
  int tuple_count = state.scale_factor * state.tuples_per_tilegroup;
  int predicate_offset = 0.1 * tuple_count;

  LOG_TRACE("Tuple count : %d", tuple_count);

  int lower_bound = predicate_offset;
  return lower_bound;
}

static int GetUpperBound() {
  int tuple_count = state.scale_factor * state.tuples_per_tilegroup;
  int selected_tuple_count = state.selectivity * tuple_count;
  int predicate_offset = 0.1 * tuple_count;

  int upper_bound = predicate_offset + selected_tuple_count;
  return upper_bound;
}

static expression::AbstractExpression *CreateSimpleScanPredicate(
    oid_t key_attr, ExpressionType expression_type, oid_t constant) {
  // First, create tuple value expression.
  oid_t left_tuple_idx = 0;
  expression::AbstractExpression *tuple_value_expr_left =
      expression::ExpressionUtil::TupleValueFactory(VALUE_TYPE_INTEGER,
                                                    left_tuple_idx, key_attr);

  // Second, create constant value expression.
  Value constant_value_left = ValueFactory::GetIntegerValue(constant);

  expression::AbstractExpression *constant_value_expr_left =
      expression::ExpressionUtil::ConstantValueFactory(constant_value_left);

  // Finally, link them together using an greater than expression.
  expression::AbstractExpression *predicate =
      expression::ExpressionUtil::ComparisonFactory(
          expression_type, tuple_value_expr_left, constant_value_expr_left);

  return predicate;
}

/**
 * @brief Create the scan predicate given a set of attributes. The predicate
 * will be attr >= LOWER_BOUND AND attr < UPPER_BOUND.
 * LOWER_BOUND and UPPER_BOUND are determined by the selectivity config.
 */
static expression::AbstractExpression *CreateScanPredicate(
    std::vector<oid_t> key_attrs) {
  const int tuple_start_offset = GetLowerBound();
  const int tuple_end_offset = GetUpperBound();

  LOG_TRACE("Lower bound : %d", tuple_start_offset);
  LOG_TRACE("Upper bound : %d", tuple_end_offset);

  expression::AbstractExpression *predicate = nullptr;

  // Go over all key_attrs
  for (auto key_attr : key_attrs) {
    // ATTR >= LOWER_BOUND && < UPPER_BOUND

    auto left_predicate = CreateSimpleScanPredicate(
        key_attr, EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO,
        tuple_start_offset);

    auto right_predicate = CreateSimpleScanPredicate(
        key_attr, EXPRESSION_TYPE_COMPARE_LESSTHAN, tuple_end_offset);

    expression::AbstractExpression *attr_predicate =
        expression::ExpressionUtil::ConjunctionFactory(
            EXPRESSION_TYPE_CONJUNCTION_AND, left_predicate, right_predicate);

    // Build complex predicate
    if (predicate == nullptr) {
      predicate = attr_predicate;
    } else {
      // Join predicate with given attribute predicate
      predicate = expression::ExpressionUtil::ConjunctionFactory(
          EXPRESSION_TYPE_CONJUNCTION_AND, predicate, attr_predicate);
    }
  }

  return predicate;
}

static void CreateIndexScanPredicate(std::vector<oid_t> key_attrs,
                                     std::vector<oid_t> &key_column_ids,
                                     std::vector<ExpressionType> &expr_types,
                                     std::vector<Value> &values) {
  const int tuple_start_offset = GetLowerBound();
  const int tuple_end_offset = GetUpperBound();

  // Go over all key_attrs
  for (auto key_attr : key_attrs) {
    key_column_ids.push_back(key_attr);
    expr_types.push_back(
        ExpressionType::EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO);
    values.push_back(ValueFactory::GetIntegerValue(tuple_start_offset));

    key_column_ids.push_back(key_attr);
    expr_types.push_back(ExpressionType::EXPRESSION_TYPE_COMPARE_LESSTHAN);
    values.push_back(ValueFactory::GetIntegerValue(tuple_end_offset));
  }
}

/**
 * @brief Get the string for a list of oids.
 */
static inline std::string GetOidVectorString(const std::vector<oid_t> &oids) {
  std::string oid_str = "";
  for (oid_t o : oids) {
    oid_str += " " + std::to_string(o);
  }
  return oid_str;
}

/**
 * @brief Create a hybrid scan executor based on selected key columns.
 * @param tuple_key_attrs The columns which the seq scan predicate is on.
 * @param index_key_attrs The columns in the *index key tuple* which the index
 * scan predicate is on. It should match the corresponding columns in
 * \b tuple_key_columns.
 * @param column_ids Column ids to be added to the result tile after scan.
 * @return A hybrid scan executor based on the key columns.
 */
static std::shared_ptr<planner::HybridScanPlan> CreateHybridScanPlan(
    const std::vector<oid_t> &tuple_key_attrs,
    const std::vector<oid_t> &index_key_attrs,
    const std::vector<oid_t> &column_ids) {
  // Create and set up seq scan executor
  auto predicate = CreateScanPredicate(tuple_key_attrs);

  planner::IndexScanPlan::IndexScanDesc index_scan_desc;

  std::vector<oid_t> key_column_ids;
  std::vector<ExpressionType> expr_types;
  std::vector<Value> values;
  std::vector<expression::AbstractExpression *> runtime_keys;

  // Create index scan predicate
  CreateIndexScanPredicate(index_key_attrs, key_column_ids, expr_types, values);

  // Determine hybrid scan type
  auto hybrid_scan_type = HYBRID_SCAN_TYPE_SEQUENTIAL;

  // Pick index
  auto index = PickIndex(sdbench_table.get(), tuple_key_attrs);

  if (index != nullptr) {
    index_scan_desc = planner::IndexScanPlan::IndexScanDesc(
        index, key_column_ids, expr_types, values, runtime_keys);

    hybrid_scan_type = HYBRID_SCAN_TYPE_HYBRID;
  }

  LOG_TRACE("Hybrid scan type : %d", hybrid_scan_type);

  std::shared_ptr<planner::HybridScanPlan> hybrid_scan_node(
      new planner::HybridScanPlan(sdbench_table.get(), predicate, column_ids,
                                  index_scan_desc, hybrid_scan_type));

  return hybrid_scan_node;
}

const static std::string OUTPUT_FILE = "outputfile.summary";
std::ofstream out(OUTPUT_FILE);

oid_t query_itr;

double total_duration = 0;

UNUSED_ATTRIBUTE static void WriteOutput(double duration) {
  // Convert to ms
  duration *= 1000;

  auto index_count = index_tuner.GetIndexCount();

  // Write out output in verbose mode
  if (state.verbose == true) {
    LOG_INFO("----------------------------------------------------------");
    LOG_INFO("%d %d %.3lf %.3lf %u %.1lf %d %d %d %u :: %.1lf ms",
             state.index_usage_type, state.query_complexity_type,
             state.selectivity, state.projectivity, query_itr,
             state.write_ratio, state.scale_factor, state.attribute_count,
             state.tuples_per_tilegroup, index_count, duration);
  }

  out << state.index_usage_type << " ";
  out << state.query_complexity_type << " ";
  out << state.selectivity << " ";
  out << state.projectivity << " ";
  out << query_itr << " ";
  out << state.write_ratio << " ";
  out << state.scale_factor << " ";
  out << state.attribute_count << " ";
  out << state.tuples_per_tilegroup << " ";
  out << index_count << " ";
  out << std::fixed << std::setprecision(2) << duration << "\n";

  out.flush();
}

/**
 * @brief Map the accsessed columns to a access bitmap.
 * @details We should use the output of this method to construct a Sample for
 * layout tuning instead of passing in the accessed columns directly!!
 */
static std::vector<double> GetColumnsAccessed(
    const std::vector<oid_t> &column_ids) {
  std::vector<double> columns_accessed;
  std::map<oid_t, oid_t> columns_accessed_map;

  // Init map
  for (auto col : column_ids) columns_accessed_map[(int)col] = 1;

  for (int column_itr = 0; column_itr < state.attribute_count + 1;
       column_itr++) {
    auto location = columns_accessed_map.find(column_itr);
    auto end = columns_accessed_map.end();
    if (location != end)
      columns_accessed.push_back(1);
    else
      columns_accessed.push_back(0);
  }

  return columns_accessed;
}

/**
 * @brief Execute a set of executors and update access information.
 *
 * @param executors Executors to be executed.
 * @param index_columns_accessed Columns that are accessed by index scan, used
 * fpr index tuning.
 * @param tuple_columns_accessed Columns of the tuples that are accessed, used
 * for layout tuning.
 * @param selectivity The selectivity of the operation.
 */
static void ExecuteTest(std::vector<executor::AbstractExecutor *> &executors,
                        brain::SampleType sample_type,
                        std::vector<std::vector<double>> index_columns_accessed,
                        std::vector<std::vector<oid_t>> tuple_columns_accessed,
                        double selectivity) {
  Timer<> timer;

  bool status = false;

  // Increment query counter
  query_itr++;

  // Reset timer
  timer.Reset();
  timer.Start();

  // Run all the executors
  for (auto executor : executors) {
    status = executor->Init();
    if (status == false) {
      throw Exception("Init failed");
    }

    std::vector<std::unique_ptr<executor::LogicalTile>> result_tiles;

    while (executor->Execute() == true) {
      std::unique_ptr<executor::LogicalTile> result_tile(executor->GetOutput());
      result_tiles.emplace_back(result_tile.release());
    }

    // Execute stuff
    executor->Execute();
  }

  // Emit time
  timer.Stop();
  auto duration = timer.GetDuration();
  total_duration += duration;

  WriteOutput(duration);

  // Record index sample
  for (auto &index_columns : index_columns_accessed) {
    brain::Sample index_access_sample(index_columns,
                                      duration / index_columns_accessed.size(),
                                      sample_type, selectivity);
    sdbench_table->RecordIndexSample(index_access_sample);
  }

  // Record layout sample
  for (auto &tuple_columns : tuple_columns_accessed) {
    // Record layout sample
    brain::Sample tuple_access_bitmap(GetColumnsAccessed(tuple_columns),
                                      duration / tuple_columns_accessed.size());
    sdbench_table->RecordLayoutSample(tuple_access_bitmap);
  }
}

static std::shared_ptr<index::Index> PickIndex(storage::DataTable *table,
                                               std::vector<oid_t> query_attrs) {
  // Construct set
  std::set<oid_t> query_attrs_set(query_attrs.begin(), query_attrs.end());

  oid_t index_count = table->GetIndexCount();

  // Empty index pointer
  std::shared_ptr<index::Index> index;

  // Can't use indexes => return empty index
  if (state.index_usage_type == INDEX_USAGE_TYPE_NEVER) {
    return index;
  }

  // Go over all indices
  bool query_index_found = false;
  oid_t index_itr = 0;
  for (index_itr = 0; index_itr < index_count; index_itr++) {
    auto index_attrs = table->GetIndexAttrs(index_itr);

    auto index = table->GetIndex(index_itr);
    // Check if index exists
    if (index == nullptr) {
      continue;
    }

    // Some attribute did not match
    if (index_attrs != query_attrs_set) {
      continue;
    }

    // Can only use full indexes ?
    if (state.index_usage_type == INDEX_USAGE_TYPE_FULL) {
      auto indexed_tg_count = index->GetIndexedTileGroupOffset();
      auto table_tg_count = table->GetTileGroupCount();

      LOG_TRACE("Indexed TG Count : %lu", indexed_tg_count);
      LOG_TRACE("Table TG Count : %lu", table_tg_count);

      if (indexed_tg_count < table_tg_count) {
        continue;
      }
    }

    // Exact match
    query_index_found = true;
    break;

    // update index count
    index_count = table->GetIndexCount();
  }

  // Found index
  if (query_index_found == true) {
    LOG_TRACE("Found available Index");
    index = table->GetIndex(index_itr);
  } else {
    LOG_TRACE("Did not find available index");
  }

  return index;
}

static void RunSimpleQuery() {
  std::vector<oid_t> tuple_key_attrs;
  std::vector<oid_t> index_key_attrs;

  auto rand_sample = rand() % state.variability_threshold;
  if (rand_sample <= 5) {
    tuple_key_attrs = {1};
    index_key_attrs = {0};
  } else if (rand_sample <= 9) {
    tuple_key_attrs = {2};
    index_key_attrs = {0};
  } else if (rand_sample <= 11) {
    tuple_key_attrs = {3};
    index_key_attrs = {0};
  } else if (rand_sample <= 15) {
    tuple_key_attrs = {4};
    index_key_attrs = {0};
  } else if (rand_sample <= 17) {
    tuple_key_attrs = {5};
    index_key_attrs = {0};
  } else if (rand_sample <= 18) {
    tuple_key_attrs = {6};
    index_key_attrs = {0};
  } else if (rand_sample <= 19) {
    tuple_key_attrs = {7};
    index_key_attrs = {0};
  } else if (rand_sample <= 20) {
    tuple_key_attrs = {8};
    index_key_attrs = {0};
  } else if (rand_sample <= 21) {
    tuple_key_attrs = {9};
    index_key_attrs = {0};
  } else if (rand_sample <= 23) {
    tuple_key_attrs = {10};
    index_key_attrs = {0};
  } else {
    tuple_key_attrs = {11};
    index_key_attrs = {0};
  }

  // PHASE LENGTH
  for (oid_t txn_itr = 0; txn_itr < state.phase_length; txn_itr++) {
    AggregateQueryHelper(tuple_key_attrs, index_key_attrs);
  }
}

static void RunModerateQuery() {
  LOG_TRACE("Moderate Query");

  std::vector<oid_t> tuple_key_attrs;
  std::vector<oid_t> index_key_attrs;

  auto rand_sample = rand() % state.variability_threshold;
  if (rand_sample <= 5) {
    tuple_key_attrs = {3, 8, 9};
    index_key_attrs = {0, 1, 2};
  } else if (rand_sample <= 9) {
    tuple_key_attrs = {4, 9, 11};
    index_key_attrs = {0, 1, 2};
  } else if (rand_sample <= 11) {
    tuple_key_attrs = {1, 4, 7};
    index_key_attrs = {0, 1, 2};
  } else if (rand_sample <= 15) {
    tuple_key_attrs = {2, 3, 5};
    index_key_attrs = {0, 1, 2};
  } else if (rand_sample <= 17) {
    tuple_key_attrs = {5, 8, 9, 10};
    index_key_attrs = {0, 1, 2, 3};
  } else if (rand_sample <= 18) {
    tuple_key_attrs = {6, 10, 11, 12};
    index_key_attrs = {0, 1, 2, 3};
  } else if (rand_sample <= 19) {
    tuple_key_attrs = {7, 8, 10, 11};
    index_key_attrs = {0, 1, 2, 3};
  } else if (rand_sample <= 20) {
    tuple_key_attrs = {7, 8, 9, 11, 12, 13};
    index_key_attrs = {0, 1, 2, 3, 4, 5};
  } else if (rand_sample <= 21) {
    tuple_key_attrs = {4, 9, 10, 11, 12, 13};
    index_key_attrs = {0, 1, 2, 3, 4, 5};
  } else if (rand_sample <= 23) {
    tuple_key_attrs = {6, 7, 8, 9, 10, 11, 12};
    index_key_attrs = {0, 1, 2, 3, 4, 5, 6};
  } else {
    tuple_key_attrs = {1, 3, 5, 6, 7, 8, 11, 12};
    index_key_attrs = {0, 1, 2, 3, 4, 5, 6, 7};
  }

  LOG_TRACE("Moderate :: %s", GetOidVectorString(tuple_key_attrs).c_str());

  // PHASE LENGTH
  for (oid_t txn_itr = 0; txn_itr < state.phase_length; txn_itr++) {
    AggregateQueryHelper(tuple_key_attrs, index_key_attrs);
  }
}

/**
 * @brief Run complex query
 * @details 60% join test, 30% moderate query, 10% simple query
 */
static void RunComplexQuery() {
  LOG_TRACE("Complex Query");

  std::vector<oid_t> left_table_tuple_key_attrs;
  std::vector<oid_t> left_table_index_key_attrs;
  std::vector<oid_t> right_table_tuple_key_attrs;
  std::vector<oid_t> right_table_index_key_attrs;
  oid_t left_table_join_column;
  oid_t right_table_join_column;

  std::vector<oid_t> tuple_key_attrs;
  std::vector<oid_t> index_key_attrs;

  bool is_join_query = false;
  bool is_aggregate_query = false;

  // Assume there are 20 columns,
  // 10 for the left table, 10 for the right table
  auto rand_sample = rand() % state.variability_threshold;
  if (rand_sample <= 5) {
    left_table_tuple_key_attrs = {4};
    left_table_index_key_attrs = {0};
    right_table_tuple_key_attrs = {11};
    right_table_index_key_attrs = {0};
    left_table_join_column = 6;
    right_table_join_column = 13;
    is_join_query = true;
  } else if (rand_sample <= 9) {
    left_table_tuple_key_attrs = {5};
    left_table_index_key_attrs = {0};
    right_table_tuple_key_attrs = {12};
    right_table_index_key_attrs = {0};
    left_table_join_column = 6;
    right_table_join_column = 13;
    is_join_query = true;
  } else if (rand_sample <= 11) {
    left_table_tuple_key_attrs = {3, 4, 8};
    left_table_index_key_attrs = {0, 1, 2};
    right_table_tuple_key_attrs = {7, 10, 12};
    right_table_index_key_attrs = {0, 1, 2};
    left_table_join_column = 6;
    right_table_join_column = 13;
    is_join_query = true;
    is_aggregate_query = true;
  } else if (rand_sample <= 15) {
    left_table_tuple_key_attrs = {3, 4, 9};
    left_table_index_key_attrs = {0, 1, 2};
    right_table_tuple_key_attrs = {5, 8, 12};
    right_table_index_key_attrs = {0, 1, 2};
    left_table_join_column = 6;
    right_table_join_column = 13;
    is_join_query = true;
  } else if (rand_sample <= 17) {
    left_table_tuple_key_attrs = {1, 2, 5};
    left_table_index_key_attrs = {0, 1, 2};
    right_table_tuple_key_attrs = {7, 9, 11};
    right_table_index_key_attrs = {0, 1, 2};
    left_table_join_column = 6;
    right_table_join_column = 13;
    is_join_query = true;
  } else if (rand_sample <= 18) {
    left_table_tuple_key_attrs = {3, 4, 5, 8};
    left_table_index_key_attrs = {0, 1, 2, 3};
    right_table_tuple_key_attrs = {10, 11, 12, 15};
    right_table_index_key_attrs = {0, 1, 2, 3};
    left_table_join_column = 6;
    right_table_join_column = 13;
    is_join_query = true;
  } else if (rand_sample <= 19) {
    left_table_tuple_key_attrs = {3, 4, 5};
    left_table_index_key_attrs = {0, 1, 2};
    right_table_tuple_key_attrs = {10, 11, 12};
    right_table_index_key_attrs = {0, 1, 2};
    left_table_join_column = 6;
    right_table_join_column = 13;
    is_join_query = true;
  } else if (rand_sample <= 20) {
    left_table_tuple_key_attrs = {3};
    left_table_index_key_attrs = {0};
    right_table_tuple_key_attrs = {10};
    right_table_index_key_attrs = {0};
    left_table_join_column = 5;
    right_table_join_column = 12;
    is_join_query = true;
  } else if (rand_sample <= 21) {
    left_table_tuple_key_attrs = {5, 6, 7, 8};
    left_table_index_key_attrs = {0, 1, 2, 3};
    right_table_tuple_key_attrs = {12, 13};
    right_table_index_key_attrs = {0, 1};
    left_table_join_column = 4;
    right_table_join_column = 11;
    is_join_query = true;
  } else if (rand_sample <= 23) {
    left_table_tuple_key_attrs = {3, 5};
    left_table_index_key_attrs = {0, 1};
    right_table_tuple_key_attrs = {10, 12};
    right_table_index_key_attrs = {0, 1};
    left_table_join_column = 4;
    right_table_join_column = 11;
    is_join_query = true;
  } else {
    left_table_tuple_key_attrs = {3, 4};
    left_table_index_key_attrs = {0, 1};
    right_table_tuple_key_attrs = {10, 11};
    right_table_index_key_attrs = {0, 1};
    left_table_join_column = 5;
    right_table_join_column = 12;
    is_join_query = true;
  }

  if (is_join_query == true) {
    LOG_TRACE("Complex :: %s", GetOidVectorString(tuple_key_attrs).c_str());
  } else if (is_aggregate_query == true) {
    LOG_TRACE("Complex :: %s", GetOidVectorString(left_table_tuple_key_attrs +
                                                  right_table_tuple_key_attrs)
                                   .c_str());
  } else {
    LOG_ERROR("Invalid query \n");
    return;
  }

  // PHASE LENGTH
  for (oid_t txn_itr = 0; txn_itr < state.phase_length; txn_itr++) {
    // Invoke appropriate query
    if (is_join_query == true) {
      JoinQueryHelper(left_table_tuple_key_attrs, left_table_index_key_attrs,
                      right_table_tuple_key_attrs, right_table_index_key_attrs,
                      left_table_join_column, right_table_join_column);
    } else if (is_aggregate_query == true) {
      AggregateQueryHelper(tuple_key_attrs, index_key_attrs);
    } else {
      LOG_ERROR("Invalid query \n");
      return;
    }
  }
}

static void JoinQueryHelper(
    const std::vector<oid_t> &left_table_tuple_key_attrs,
    const std::vector<oid_t> &left_table_index_key_attrs,
    const std::vector<oid_t> &right_table_tuple_key_attrs,
    const std::vector<oid_t> &right_table_index_key_attrs,
    const oid_t left_table_join_column, const oid_t right_table_join_column) {
  LOG_TRACE("Run join query on left table: %s and right table: %s",
            GetOidVectorString(left_table_tuple_key_attrs).c_str(),
            GetOidVectorString(right_table_tuple_key_attrs).c_str());
  const bool is_inlined = true;
  auto &txn_manager = concurrency::TransactionManagerFactory::GetInstance();

  auto txn = txn_manager.BeginTransaction();

  /////////////////////////////////////////////////////////
  // SEQ SCAN + PREDICATE
  /////////////////////////////////////////////////////////

  std::unique_ptr<executor::ExecutorContext> context(
      new executor::ExecutorContext(txn));

  // Column ids to be added to logical tile after scan.
  // Left half of the columns are considered left table,
  // right half of the columns are considered right table.
  std::vector<oid_t> column_ids;
  oid_t column_count = state.attribute_count;

  for (oid_t col_itr = 0; col_itr < column_count; col_itr++) {
    column_ids.push_back(sdbench_column_ids[col_itr]);
  }

  // Create and set up seq scan executor
  auto left_table_scan_node = CreateHybridScanPlan(
      left_table_tuple_key_attrs, left_table_index_key_attrs, column_ids);
  auto right_table_scan_node = CreateHybridScanPlan(
      right_table_tuple_key_attrs, right_table_index_key_attrs, column_ids);

  executor::HybridScanExecutor left_table_hybrid_scan_executor(
      left_table_scan_node.get(), context.get());
  executor::HybridScanExecutor right_table_hybrid_scan_executor(
      right_table_scan_node.get(), context.get());

  /////////////////////////////////////////////////////////
  // JOIN EXECUTOR
  /////////////////////////////////////////////////////////

  auto join_type = JOIN_TYPE_INNER;

  // Create join predicate
  expression::TupleValueExpression *left_table_attr =
      new expression::TupleValueExpression(VALUE_TYPE_INTEGER, 0,
                                           left_table_join_column);
  expression::TupleValueExpression *right_table_attr =
      new expression::TupleValueExpression(VALUE_TYPE_INTEGER, 1,
                                           right_table_join_column);

  std::unique_ptr<expression::ComparisonExpression<expression::CmpLt>>
      join_predicate(new expression::ComparisonExpression<expression::CmpLt>(
          EXPRESSION_TYPE_COMPARE_LESSTHAN, left_table_attr, right_table_attr));

  std::unique_ptr<const planner::ProjectInfo> project_info(nullptr);
  std::shared_ptr<const catalog::Schema> schema(nullptr);

  planner::NestedLoopJoinPlan nested_loop_join_node(
      join_type, std::move(join_predicate), std::move(project_info), schema);

  // Run the nested loop join executor
  executor::NestedLoopJoinExecutor nested_loop_join_executor(
      &nested_loop_join_node, nullptr);

  // Construct the executor tree
  nested_loop_join_executor.AddChild(&left_table_hybrid_scan_executor);
  nested_loop_join_executor.AddChild(&right_table_hybrid_scan_executor);

  /////////////////////////////////////////////////////////
  // MATERIALIZE
  /////////////////////////////////////////////////////////

  // Create and set up materialization executor
  // FIXME: this will always retreive all columns, projectivity is ignored
  std::vector<catalog::Column> output_columns;
  std::unordered_map<oid_t, oid_t> old_to_new_cols;
  oid_t join_column_count = column_count * 2;
  for (oid_t col_itr = 0; col_itr < join_column_count; col_itr++) {
    auto column =
        catalog::Column(VALUE_TYPE_INTEGER, GetTypeSize(VALUE_TYPE_INTEGER),
                        "" + std::to_string(col_itr), is_inlined);
    output_columns.push_back(column);

    old_to_new_cols[col_itr] = col_itr;
  }

  std::shared_ptr<const catalog::Schema> output_schema(
      new catalog::Schema(output_columns));
  bool physify_flag = true;  // is going to create a physical tile
  planner::MaterializationPlan mat_node(old_to_new_cols, output_schema,
                                        physify_flag);

  executor::MaterializationExecutor mat_executor(&mat_node, nullptr);
  mat_executor.AddChild(&nested_loop_join_executor);

  /////////////////////////////////////////////////////////
  // EXECUTE
  /////////////////////////////////////////////////////////

  std::vector<executor::AbstractExecutor *> executors;
  executors.push_back(&mat_executor);

  /////////////////////////////////////////////////////////
  // COLLECT STATS
  /////////////////////////////////////////////////////////

  std::vector<double> left_table_index_columns_accessed(
      left_table_tuple_key_attrs.begin(), left_table_tuple_key_attrs.end());
  std::vector<double> right_table_index_columns_accessed(
      right_table_tuple_key_attrs.begin(), right_table_tuple_key_attrs.end());

  // Prepare tuple columns accessed
  auto left_table_tuple_columns_accessed = left_table_tuple_key_attrs;
  auto right_table_tuple_columns_accessed = right_table_tuple_key_attrs;
  left_table_tuple_columns_accessed.push_back(left_table_join_column);
  right_table_tuple_columns_accessed.push_back(right_table_join_column);

  auto selectivity = state.selectivity;

  ExecuteTest(
      executors, brain::SAMPLE_TYPE_ACCESS,
      {left_table_index_columns_accessed, right_table_index_columns_accessed},
      {left_table_tuple_columns_accessed, right_table_tuple_columns_accessed},
      selectivity);

  txn_manager.CommitTransaction();
}

static void AggregateQueryHelper(const std::vector<oid_t> &tuple_key_attrs,
                                 const std::vector<oid_t> &index_key_attrs) {
  if (state.verbose) {
    LOG_INFO("Run aggregate query on %s ",
             GetOidVectorString(tuple_key_attrs).c_str());
  }

  const bool is_inlined = true;
  auto &txn_manager = concurrency::TransactionManagerFactory::GetInstance();

  auto txn = txn_manager.BeginTransaction();

  /////////////////////////////////////////////////////////
  // SEQ SCAN + PREDICATE
  /////////////////////////////////////////////////////////

  // Column ids to be added to logical tile after scan.
  // We need all columns because projection can require any column
  std::vector<oid_t> column_ids;
  oid_t column_count = state.attribute_count;

  column_ids.push_back(0);
  for (oid_t col_itr = 0; col_itr < column_count; col_itr++) {
    column_ids.push_back(sdbench_column_ids[col_itr]);
  }

  std::unique_ptr<executor::ExecutorContext> context(
      new executor::ExecutorContext(txn));

  auto hybrid_scan_node =
      CreateHybridScanPlan(tuple_key_attrs, index_key_attrs, column_ids);
  executor::HybridScanExecutor hybrid_scan_executor(hybrid_scan_node.get(),
                                                    context.get());

  /////////////////////////////////////////////////////////
  // AGGREGATION
  /////////////////////////////////////////////////////////

  // Resize column ids to contain only columns
  // over which we compute aggregates
  column_count = state.projectivity * state.attribute_count;
  column_ids.resize(column_count);
  if (state.verbose) {
    LOG_INFO("Aggregate on: %s", GetOidVectorString(column_ids).c_str());
  }

  // (1-5) Setup plan node

  // 1) Set up group-by columns
  std::vector<oid_t> group_by_columns;

  // 2) Set up project info
  DirectMapList direct_map_list;
  oid_t col_itr = 0;
  oid_t tuple_idx = 1;  // tuple2
  for (col_itr = 0; col_itr < column_count; col_itr++) {
    direct_map_list.push_back({col_itr, {tuple_idx, col_itr}});
    col_itr++;
  }

  std::unique_ptr<const planner::ProjectInfo> proj_info(
      new planner::ProjectInfo(TargetList(), std::move(direct_map_list)));

  // 3) Set up aggregates
  std::vector<planner::AggregatePlan::AggTerm> agg_terms;
  for (auto column_id : column_ids) {
    planner::AggregatePlan::AggTerm max_column_agg(
        EXPRESSION_TYPE_AGGREGATE_MAX,
        expression::ExpressionUtil::TupleValueFactory(VALUE_TYPE_INTEGER, 0,
                                                      column_id),
        false);
    agg_terms.push_back(max_column_agg);
  }

  // 4) Set up predicate (empty)
  std::unique_ptr<const expression::AbstractExpression> aggregate_predicate(
      nullptr);

  // 5) Create output table schema
  auto data_table_schema = sdbench_table->GetSchema();
  std::vector<catalog::Column> columns;
  for (auto column_id : column_ids) {
    columns.push_back(data_table_schema->GetColumn(column_id));
  }

  std::shared_ptr<const catalog::Schema> output_table_schema(
      new catalog::Schema(columns));

  // OK) Create the plan node
  planner::AggregatePlan aggregation_node(
      std::move(proj_info), std::move(aggregate_predicate),
      std::move(agg_terms), std::move(group_by_columns), output_table_schema,
      AGGREGATE_TYPE_PLAIN);

  executor::AggregateExecutor aggregation_executor(&aggregation_node,
                                                   context.get());

  aggregation_executor.AddChild(&hybrid_scan_executor);

  /////////////////////////////////////////////////////////
  // MATERIALIZE
  /////////////////////////////////////////////////////////

  // Create and set up materialization executor
  std::vector<catalog::Column> output_columns;
  std::unordered_map<oid_t, oid_t> old_to_new_cols;
  col_itr = 0;
  for (auto column_id : column_ids) {
    auto column =
        catalog::Column(VALUE_TYPE_INTEGER, GetTypeSize(VALUE_TYPE_INTEGER),
                        std::to_string(column_id), is_inlined);
    output_columns.push_back(column);

    old_to_new_cols[col_itr] = col_itr;
    col_itr++;
  }

  std::shared_ptr<const catalog::Schema> output_schema(
      new catalog::Schema(output_columns));
  bool physify_flag = true;  // is going to create a physical tile
  planner::MaterializationPlan mat_node(old_to_new_cols, output_schema,
                                        physify_flag);

  executor::MaterializationExecutor mat_executor(&mat_node, nullptr);
  mat_executor.AddChild(&aggregation_executor);

  /////////////////////////////////////////////////////////
  // EXECUTE
  /////////////////////////////////////////////////////////

  std::vector<executor::AbstractExecutor *> executors;
  executors.push_back(&mat_executor);

  /////////////////////////////////////////////////////////
  // COLLECT STATS
  /////////////////////////////////////////////////////////
  std::vector<double> index_columns_accessed(tuple_key_attrs.begin(),
                                             tuple_key_attrs.end());
  auto selectivity = state.selectivity;

  // Only record the access to projected columns
  auto tuple_columns_accessed = column_ids;

  ExecuteTest(executors, brain::SAMPLE_TYPE_ACCESS, {index_columns_accessed},
              {tuple_columns_accessed}, selectivity);

  txn_manager.CommitTransaction();
}

/**
 * @brief Run write transactions
 *
 * @param tuple_key_attrs Tuple attributes to query on.
 * @param index_key_attrs Index attributes to query on.
 * @param update_attrs Columns to be updated. The value value of each attribute
 * in update_attrs will be updated to -v, where v is the original value, and -v
 * is minus original value.
 */
static void UpdateHelper(const std::vector<oid_t> &tuple_key_attrs,
                         const std::vector<oid_t> &index_key_attrs,
                         const std::vector<oid_t> &update_attrs) {
  auto &txn_manager = concurrency::TransactionManagerFactory::GetInstance();
  auto txn = txn_manager.BeginTransaction();

  //////////////////////////////////////////
  // SCAN + PREDICATE
  //////////////////////////////////////////

  std::vector<oid_t> column_ids;
  oid_t column_count = state.attribute_count;

  column_ids.push_back(0);
  for (oid_t col_itr = 0; col_itr < column_count; col_itr++) {
    column_ids.push_back(col_itr);
  }

  std::unique_ptr<executor::ExecutorContext> context(
      new executor::ExecutorContext(txn));

  auto hybrid_scan_node =
      CreateHybridScanPlan(tuple_key_attrs, index_key_attrs, column_ids);
  executor::HybridScanExecutor hybrid_scan_executor(hybrid_scan_node.get(),
                                                    context.get());

  //////////////////////////////////////////
  // UPDATE
  //////////////////////////////////////////

  // Update the value of each attribute in update_attrs to -v, where v is the
  // original value, and -v is minus original value.

  std::vector<Value> values;
  TargetList target_list;
  DirectMapList direct_map_list;

  target_list.clear();
  direct_map_list.clear();

  // Build target_list: -value for update_attrs
  for (oid_t update_attr : update_attrs) {
    auto tuple_value_expression = new expression::TupleValueExpression(
        VALUE_TYPE_INTEGER, 0, update_attr);
    auto minus_value_expression =
        new expression::OperatorUnaryMinusExpression(tuple_value_expression);
    target_list.emplace_back(update_attr, minus_value_expression);
  }

  // Build direct_map_list: value unchanged for other attributes
  oid_t update_attr_itr = 0;
  for (oid_t col_itr = 0; col_itr < column_count; col_itr++) {
    // Skip the updated column
    if (update_attr_itr > update_attrs.size() ||
        col_itr != update_attrs[update_attr_itr]) {
      direct_map_list.emplace_back(col_itr,
                                   std::pair<oid_t, oid_t>(0, col_itr));
    } else {
      update_attr_itr++;
    }
  }

  std::unique_ptr<const planner::ProjectInfo> project_info(
      new planner::ProjectInfo(std::move(target_list),
                               std::move(direct_map_list)));
  planner::UpdatePlan update_node(sdbench_table.get(), std::move(project_info));

  executor::UpdateExecutor update_executor(&update_node, context.get());
  update_executor.AddChild(&hybrid_scan_executor);

  /////////////////////////////////////////////////////////
  // EXECUTE
  /////////////////////////////////////////////////////////

  std::vector<executor::AbstractExecutor *> executors;
  executors.push_back(&update_executor);

  /////////////////////////////////////////////////////////
  // COLLECT STATS
  /////////////////////////////////////////////////////////
  std::vector<double> index_columns_accessed(tuple_key_attrs.begin(),
                                             tuple_key_attrs.end());
  auto selectivity = state.selectivity;

  auto tuple_columns_accessed = tuple_key_attrs;
  for (oid_t update_attr : update_attrs) {
    tuple_columns_accessed.push_back(update_attr);
  }

  ExecuteTest(executors, brain::SAMPLE_TYPE_ACCESS, {index_columns_accessed},
              {tuple_columns_accessed}, selectivity);

  txn_manager.CommitTransaction();
}

static void RunSimpleUpdate() {
  std::vector<oid_t> tuple_key_attrs;
  std::vector<oid_t> index_key_attrs;
  std::vector<oid_t> update_attrs;

  update_attrs = {15, 16, 17, 18, 19};

  auto rand_sample = rand() % state.variability_threshold;
  if (rand_sample <= 5) {
    tuple_key_attrs = {1};
    index_key_attrs = {0};
  } else if (rand_sample <= 9) {
    tuple_key_attrs = {2};
    index_key_attrs = {0};
  } else if (rand_sample <= 11) {
    tuple_key_attrs = {3};
    index_key_attrs = {0};
  } else if (rand_sample <= 15) {
    tuple_key_attrs = {4};
    index_key_attrs = {0};
  } else if (rand_sample <= 17) {
    tuple_key_attrs = {5};
    index_key_attrs = {0};
  } else if (rand_sample <= 18) {
    tuple_key_attrs = {6};
    index_key_attrs = {0};
  } else if (rand_sample <= 19) {
    tuple_key_attrs = {7};
    index_key_attrs = {0};
  } else if (rand_sample <= 20) {
    tuple_key_attrs = {8};
    index_key_attrs = {0};
  } else if (rand_sample <= 21) {
    tuple_key_attrs = {9};
    index_key_attrs = {0};
  } else if (rand_sample <= 23) {
    tuple_key_attrs = {10};
    index_key_attrs = {0};
  } else {
    tuple_key_attrs = {11};
    index_key_attrs = {0};
  }

  UNUSED_ATTRIBUTE std::stringstream os;
  os << "Simple Update :: ";
  for (auto tuple_key_attr : tuple_key_attrs) {
    os << tuple_key_attr << " ";
  }
  if (state.verbose) {
    LOG_INFO("%s", os.str().c_str());
  }

  // PHASE LENGTH
  for (oid_t txn_itr = 0; txn_itr < state.phase_length; txn_itr++) {
    UpdateHelper(tuple_key_attrs, index_key_attrs, update_attrs);
  }
}

static void RunComplexUpdate() {
  std::vector<oid_t> tuple_key_attrs;
  std::vector<oid_t> index_key_attrs;
  std::vector<oid_t> update_attrs;

  update_attrs = {15, 16, 17, 18, 19};

  auto rand_sample = rand() % state.variability_threshold;
  if (rand_sample <= 5) {
    tuple_key_attrs = {3, 8, 9};
    index_key_attrs = {0, 1, 2};
  } else if (rand_sample <= 9) {
    tuple_key_attrs = {4, 9, 11};
    index_key_attrs = {0, 1, 2};
  } else if (rand_sample <= 11) {
    tuple_key_attrs = {1, 4, 7};
    index_key_attrs = {0, 1, 2};
  } else if (rand_sample <= 15) {
    tuple_key_attrs = {2, 3, 5};
    index_key_attrs = {0, 1, 2};
  } else if (rand_sample <= 17) {
    tuple_key_attrs = {5, 8, 9, 10};
    index_key_attrs = {0, 1, 2, 3};
  } else if (rand_sample <= 18) {
    tuple_key_attrs = {6, 10, 11, 12};
    index_key_attrs = {0, 1, 2, 3};
  } else if (rand_sample <= 19) {
    tuple_key_attrs = {7, 8, 10, 11};
    index_key_attrs = {0, 1, 2, 3};
  } else if (rand_sample <= 20) {
    tuple_key_attrs = {7, 8, 9, 11, 12, 13};
    index_key_attrs = {0, 1, 2, 3, 4, 5};
  } else if (rand_sample <= 21) {
    tuple_key_attrs = {4, 9, 10, 11, 12, 13};
    index_key_attrs = {0, 1, 2, 3, 4, 5};
  } else if (rand_sample <= 23) {
    tuple_key_attrs = {6, 7, 8, 9, 10, 11, 12};
    index_key_attrs = {0, 1, 2, 3, 4, 5, 6};
  } else {
    tuple_key_attrs = {1, 3, 5, 6, 7, 8, 11, 12};
    index_key_attrs = {0, 1, 2, 3, 4, 5, 6, 7};
  }

  UNUSED_ATTRIBUTE std::stringstream os;
  os << "Complex Update :: ";
  for (auto tuple_key_attr : tuple_key_attrs) {
    os << tuple_key_attr << " ";
  }
  LOG_TRACE("%s", os.str().c_str());

  // PHASE LENGTH
  for (oid_t txn_itr = 0; txn_itr < state.phase_length; txn_itr++) {
    UpdateHelper(tuple_key_attrs, index_key_attrs, update_attrs);
  }
}

/**
 * @brief Run query depending on query type
 */
static void RunQuery() {
  LOG_TRACE("Run query");
  switch (state.query_complexity_type) {
    case QUERY_COMPLEXITY_TYPE_SIMPLE:
      RunSimpleQuery();
      break;
    case QUERY_COMPLEXITY_TYPE_MODERATE:
      RunModerateQuery();
      break;
    case QUERY_COMPLEXITY_TYPE_COMPLEX:
      RunComplexQuery();
      break;
    default:
      break;
  }
}

/**
 * @brief Run write txn depending on write type
 */
static void RunWrite() {
  LOG_TRACE("Run write");
  switch (state.write_complexity_type) {
    case WRITE_COMPLEXITY_TYPE_SIMPLE:
      RunSimpleUpdate();
      break;
    case WRITE_COMPLEXITY_TYPE_COMPLEX:
      RunComplexUpdate();
      break;
    default:
      break;
  }
}

/**
 * @brief A data structure to hold index information of a table.
 */
struct IndexSummary {
  // Index oids
  std::vector<oid_t> index_oids;

  // Index has complete built?
  bool completed;
};

static int index_unchanged_phase_count = 0;

/**
 * @brief Check if index scheme has converged.
 * Determine by looking at how many times index has not been changed.
 *
 * @return true if the index configuration has converged. False otherwise.
 */
static bool HasIndexConfigurationConverged() {
  static IndexSummary prev_index_summary;
  // If the index configuration stays the same
  // for "INDEX_CONVERGE_THRESHOLD" continuous phases,
  // then it's considered as converged.

  IndexSummary index_summary;
  index_summary.completed = true;

  // Get index summary
  oid_t index_count = sdbench_table->GetIndexCount();
  auto table_tile_group_count = sdbench_table->GetTileGroupCount();
  for (oid_t index_itr = 0; index_itr < index_count; index_itr++) {
    // Get index
    auto index = sdbench_table->GetIndex(index_itr);
    if (index == nullptr) {
      continue;
    }

    auto indexed_tile_group_offset = index->GetIndexedTileGroupOffset();

    // Get percentage completion
    double fraction = 0.0;
    if (table_tile_group_count != 0) {
      fraction =
          (double)indexed_tile_group_offset / (double)table_tile_group_count;
      fraction *= 100;
    }

    if (fraction < 0) {
      index_summary.completed = false;
    }

    // Get index columns
    index_summary.index_oids.push_back(index->GetOid());
  }

  if (index_summary.completed == false) {
    prev_index_summary = index_summary;
    index_unchanged_phase_count = 0;
    return false;
  }

  // Check if the index summary is the same
  bool identical = true;
  if (index_summary.index_oids.size() == prev_index_summary.index_oids.size()) {
    for (size_t i = 0; i < index_summary.index_oids.size(); i++) {
      if (index_summary.index_oids[i] != prev_index_summary.index_oids[i]) {
        identical = false;
        break;
      }
    }
  } else {
    identical = false;
  }

  // Update index unchanged phase count
  if (identical) {
    index_unchanged_phase_count += 1;
  } else {
    index_unchanged_phase_count = 0;
  }

  prev_index_summary = index_summary;

  // Check threshold # of phases
  int convergence_phase_threshold =
      state.convergence_query_threshold / state.phase_length;
  if (index_unchanged_phase_count >= convergence_phase_threshold) {
    return true;
  }

  return false;
}

void RunSDBenchTest() {
  // Setup index tuner
  index_tuner.SetSampleCountThreshold(state.sample_count_threshold);
  index_tuner.SetMaxTileGroupsIndexed(state.max_tile_groups_indexed);
  index_tuner.SetIndexUtilityThreshold(state.index_utility_threshold);
  index_tuner.SetIndexCountThreshold(state.index_count_threshold);
  index_tuner.SetWriteRatioThreshold(state.write_ratio_threshold);

  std::thread index_builder;

  // Generate sequence
  GenerateSequence(state.attribute_count);

  CreateAndLoadTable((LayoutType)state.layout_mode);

  double write_ratio = state.write_ratio;
  double phase_count = state.total_ops / state.phase_length;

  // Convergence test
  const double CONVERGENCE_PHASE_COUNT_THRESHOLD = 10000;
  if (state.convergence == true) {
    phase_count = CONVERGENCE_PHASE_COUNT_THRESHOLD;
  }

  if (phase_count > CONVERGENCE_PHASE_COUNT_THRESHOLD) {
    LOG_INFO("Too many phases, current phase count threshold is %.0lf",
             CONVERGENCE_PHASE_COUNT_THRESHOLD);
    exit(EXIT_FAILURE);
  }

  // Reset total duration
  total_duration = 0;

  // Reset query counter
  query_itr = 0;

  Timer<> index_unchanged_timer;

  // Start index tuner
  if (state.index_usage_type != INDEX_USAGE_TYPE_NEVER) {
    index_tuner.AddTable(sdbench_table.get());

    // Start after adding tables
    index_tuner.Start();
  }

  // Start layout tuner
  if (state.layout_mode == LAYOUT_TYPE_HYBRID) {
    layout_tuner.AddTable(sdbench_table.get());

    // Start layout tuner
    layout_tuner.Start();
  }

  // seed generator
  srand(generator_seed);

  for (oid_t phase_itr = 0; phase_itr < phase_count; phase_itr++) {
    double rand_sample = (double)rand() / RAND_MAX;

    // Do insert
    if (rand_sample < write_ratio) {
      RunWrite();
    }
    // Do read
    else {
      RunQuery();
    }

    // Check index convergence
    if (state.convergence == true) {
      bool converged = HasIndexConfigurationConverged();
      if (converged == true) {
        break;
      }
    }
  }

  // Stop index tuner
  if (state.index_usage_type != INDEX_USAGE_TYPE_NEVER) {
    index_tuner.Stop();
    index_tuner.ClearTables();
  }

  if (state.layout_mode == LAYOUT_TYPE_HYBRID) {
    layout_tuner.Stop();
    layout_tuner.ClearTables();
  }

  // Drop Indexes
  DropIndexes();

  // Reset
  query_itr = 0;

  LOG_INFO("Duration : %.2lf", total_duration);

  out.close();
}

}  // namespace sdbench
}  // namespace benchmark
}  // namespace peloton
