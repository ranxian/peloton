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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <ctime>
#include <thread>
#include <algorithm>
#include <chrono>

#include "brain/sample.h"
#include "brain/layout_tuner.h"
#include "brain/index_tuner.h"

#include "benchmark/sdbench/sdbench_workload.h"
#include "benchmark/sdbench/sdbench_loader.h"

#include "catalog/manager.h"
#include "catalog/schema.h"
#include "common/types.h"
#include "common/value.h"
#include "common/value_factory.h"
#include "common/logger.h"
#include "common/timer.h"
#include "common/macros.h"
#include "concurrency/transaction.h"
#include "concurrency/transaction_manager_factory.h"

#include "executor/executor_context.h"
#include "executor/abstract_executor.h"
#include "executor/aggregate_executor.h"
#include "executor/seq_scan_executor.h"
#include "executor/logical_tile.h"
#include "executor/logical_tile_factory.h"
#include "executor/materialization_executor.h"
#include "executor/projection_executor.h"
#include "executor/insert_executor.h"
#include "executor/update_executor.h"
#include "executor/nested_loop_join_executor.h"
#include "executor/hybrid_scan_executor.h"

#include "expression/abstract_expression.h"
#include "expression/constant_value_expression.h"
#include "expression/tuple_value_expression.h"
#include "expression/comparison_expression.h"
#include "expression/conjunction_expression.h"
#include "expression/expression_util.h"

#include "planner/abstract_plan.h"
#include "planner/aggregate_plan.h"
#include "planner/materialization_plan.h"
#include "planner/seq_scan_plan.h"
#include "planner/insert_plan.h"
#include "planner/update_plan.h"
#include "planner/projection_plan.h"
#include "planner/nested_loop_join_plan.h"
#include "planner/hybrid_scan_plan.h"

#include "storage/tile.h"
#include "storage/tile_group.h"
#include "storage/tile_group_header.h"
#include "storage/data_table.h"
#include "storage/table_factory.h"

namespace peloton {
namespace benchmark {
namespace sdbench {

// Function definitions
std::shared_ptr<index::Index> PickIndex(storage::DataTable *table,
                                        std::vector<oid_t> query_attrs);

// Tuple id counter
oid_t sdbench_tuple_counter = -1000000;

std::vector<oid_t> column_counts = {50, 500};

static int GetLowerBound() {
  int tuple_count = state.scale_factor * state.tuples_per_tilegroup;
  int predicate_offset = 0.1 * tuple_count;

  LOG_INFO("Tuple count : %d", tuple_count);

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

expression::AbstractExpression *CreateSimpleScanPredicate(
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
expression::AbstractExpression *CreateScanPredicate(
    std::vector<oid_t> key_attrs) {
  const int tuple_start_offset = GetLowerBound();
  const int tuple_end_offset = GetUpperBound();

  LOG_INFO("Lower bound : %d", tuple_start_offset);
  LOG_INFO("Upper bound : %d", tuple_end_offset);

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

void CreateIndexScanPredicate(std::vector<oid_t> key_attrs,
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
 * @brief Create a hybrid scan executor based on selected key columns.
 * @param tuple_key_attrs The columns which the seq scan predicate is on.
 * @param index_key_attrs The columns in the *index key tuple* which the index
 * scan predicate is on. It should match the corresponding columns in
 * \b tuple_key_columns.
 * @param column_ids Column ids to added to the result tile after scan.
 * @return A hybrid scan executor based on the key columns.
 */
std::shared_ptr<planner::HybridScanPlan> CreateHybridScanPlan(
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

  LOG_INFO("Hybrid scan type : %d", hybrid_scan_type);

  std::shared_ptr<planner::HybridScanPlan> hybrid_scan_node(
      new planner::HybridScanPlan(sdbench_table.get(), predicate, column_ids,
                                  index_scan_desc, hybrid_scan_type));

  return hybrid_scan_node;
}

std::ofstream out("outputfile.summary");

oid_t query_itr;

double total_duration = 0;

UNUSED_ATTRIBUTE static void WriteOutput(double duration) {
  // Convert to ms
  duration *= 1000;

  LOG_INFO("----------------------------------------------------------");
  LOG_INFO("%d %d %.3lf %.3lf %u %.1lf %d %d %d :: %.1lf ms", state.layout_mode,
           state.operator_type, state.selectivity, state.projectivity,
           query_itr, state.write_ratio, state.scale_factor, state.column_count,
           state.tuples_per_tilegroup, duration);

  out << state.layout_mode << " ";
  out << state.operator_type << " ";
  out << state.selectivity << " ";
  out << state.projectivity << " ";
  out << query_itr << " ";
  out << state.write_ratio << " ";
  out << state.scale_factor << " ";
  out << state.column_count << " ";
  out << state.tuples_per_tilegroup << " ";
  out << duration << "\n";

  out.flush();
}

static void ExecuteTest(std::vector<executor::AbstractExecutor *> &executors,
                        brain::SampleType sample_type,
                        std::vector<std::vector<double>> index_columns_accessed,
                        double selectivity) {
  Timer<> timer;

  auto txn_count = state.phase_length;
  bool status = false;

  // Run these many transactions
  for (oid_t txn_itr = 0; txn_itr < txn_count; txn_itr++) {
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
        std::unique_ptr<executor::LogicalTile> result_tile(
            executor->GetOutput());
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

    // Construct sample
    for (auto &index_columns : index_columns_accessed) {
      brain::Sample index_sample(index_columns,
                                 duration / index_columns_accessed.size(),
                                 sample_type, selectivity);

      // Record sample
      sdbench_table->RecordIndexSample(index_sample);
    }
  }
}

std::vector<double> GetColumnsAccessed(const std::vector<oid_t> &column_ids) {
  std::vector<double> columns_accessed;
  std::map<oid_t, oid_t> columns_accessed_map;

  // Init map
  for (auto col : column_ids) columns_accessed_map[col] = 1;

  for (int column_itr = 0; column_itr < state.column_count; column_itr++) {
    auto location = columns_accessed_map.find(column_itr);
    auto end = columns_accessed_map.end();
    if (location != end)
      columns_accessed.push_back(1);
    else
      columns_accessed.push_back(0);
  }

  return columns_accessed;
}

std::shared_ptr<index::Index> PickIndex(storage::DataTable *table,
                                        std::vector<oid_t> query_attrs) {
  // Construct set
  std::set<oid_t> query_attrs_set(query_attrs.begin(), query_attrs.end());

  oid_t index_count = table->GetIndexCount();

  // Go over all indices
  bool query_index_found = false;
  oid_t index_itr = 0;
  for (index_itr = 0; index_itr < index_count; index_itr++) {
    auto index_attrs = table->GetIndexAttrs(index_itr);

    auto index = table->GetIndex(index_itr);
    UNUSED_ATTRIBUTE auto index_metadata = index->GetMetadata();
    LOG_INFO("Available Index :: %s", index_metadata->GetInfo().c_str());

    // Some attribute did not match
    if (index_attrs != query_attrs_set) {
      continue;
    }

    // Fully constructed or not ?
    if (state.only_use_full_indexes == true) {
      auto indexed_tg_count = index->GetIndexedTileGroupOffset();
      auto table_tg_count = table->GetTileGroupCount();

      LOG_INFO("Indexed TG Count : %lu", indexed_tg_count);
      LOG_INFO("Table TG Count : %lu", table_tg_count);

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

  std::shared_ptr<index::Index> index;

  // Found index
  if (query_index_found == true) {
    LOG_INFO("Found available Index");
    index = table->GetIndex(index_itr);
  } else {
    LOG_INFO("Did not find available index");
  }

  return index;
}

void RunSimpleQuery() {
  std::vector<oid_t> tuple_key_attrs;
  std::vector<oid_t> index_key_attrs;

  auto rand_sample = rand() % 10;
  if (rand_sample <= 3) {
    tuple_key_attrs = {4};
    index_key_attrs = {0};
  } else if (rand_sample <= 6) {
    tuple_key_attrs = {3};
    index_key_attrs = {0};
  } else {
    tuple_key_attrs = {2};
    index_key_attrs = {0};
  }

  UNUSED_ATTRIBUTE std::stringstream os;
  os << "Direct :: ";
  for (auto tuple_key_attr : tuple_key_attrs) {
    os << tuple_key_attr << " ";
  }
  LOG_INFO("%s", os.str().c_str());

  RunQuery(tuple_key_attrs, index_key_attrs);
}

void RunModerateQuery() {
  LOG_INFO("Moderate Query");

  std::vector<oid_t> tuple_key_attrs;
  std::vector<oid_t> index_key_attrs;

  auto rand_sample = rand() % 10;

  if (rand_sample <= 3) {
    tuple_key_attrs = {3, 4};
    index_key_attrs = {0, 1};
  } else if (rand_sample <= 6) {
    tuple_key_attrs = {3, 6};
    index_key_attrs = {0, 1};
  } else {
    tuple_key_attrs = {2};
    index_key_attrs = {0};
  }

  RunQuery(tuple_key_attrs, index_key_attrs);
}

/**
 * @brief Run complex query
 * @details 60% join test, 30% moderate query, 10% simple query
 */
void RunComplexQeury() {
  LOG_INFO("Complex Query");

  auto rand_sample = rand() % 10;

  // Assume there are 20 columns, 10 for the left table, 10 for the right table
  if (rand_sample <= 2) {
    RunJoinTest({3, 4}, {0, 1}, {10, 11}, {0, 1}, 5, 12);
  } else if (rand_sample <= 5) {
    RunJoinTest({3, 6}, {0, 1}, {10, 14}, {0, 1}, 4, 13);
  } else if (rand_sample <= 8) {
    RunQuery({3, 4}, {0, 1});
  } else {
    RunQuery({2}, {0});
  }
}

void RunQuery(const std::vector<oid_t> &tuple_key_attrs,
              const std::vector<oid_t> &index_key_attrs) {
  const bool is_inlined = true;
  auto &txn_manager = concurrency::TransactionManagerFactory::GetInstance();

  auto txn = txn_manager.BeginTransaction();

  /////////////////////////////////////////////////////////
  // SEQ SCAN + PREDICATE
  /////////////////////////////////////////////////////////

  // Column ids to be added to logical tile after scan.
  // We need all columns because projection can require any column
  std::vector<oid_t> column_ids;
  oid_t column_count = state.column_count;

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
  column_count = state.projectivity * state.column_count;
  column_ids.resize(column_count);

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

  ExecuteTest(executors, brain::SAMPLE_TYPE_ACCESS, {index_columns_accessed},
              selectivity);

  txn_manager.CommitTransaction();
}

void RunInsertTest() {
  auto &txn_manager = concurrency::TransactionManagerFactory::GetInstance();

  auto txn = txn_manager.BeginTransaction();

  /////////////////////////////////////////////////////////
  // INSERT
  /////////////////////////////////////////////////////////

  std::unique_ptr<executor::ExecutorContext> context(
      new executor::ExecutorContext(txn));

  std::vector<Value> values;
  Value insert_val = ValueFactory::GetIntegerValue(++sdbench_tuple_counter);
  TargetList target_list;
  DirectMapList direct_map_list;
  std::vector<oid_t> column_ids;

  target_list.clear();
  direct_map_list.clear();

  for (auto col_id = 0; col_id <= state.column_count; col_id++) {
    auto expression =
        expression::ExpressionUtil::ConstantValueFactory(insert_val);
    target_list.emplace_back(col_id, expression);
    column_ids.push_back(col_id);
  }

  std::unique_ptr<const planner::ProjectInfo> project_info(
      new planner::ProjectInfo(std::move(target_list),
                               std::move(direct_map_list)));

  auto orig_tuple_count = state.scale_factor * state.tuples_per_tilegroup;
  auto bulk_insert_count = state.write_ratio * orig_tuple_count;

  LOG_INFO("Bulk insert count : %lf", bulk_insert_count);
  planner::InsertPlan insert_node(sdbench_table.get(), std::move(project_info),
                                  bulk_insert_count);
  executor::InsertExecutor insert_executor(&insert_node, context.get());

  /////////////////////////////////////////////////////////
  // EXECUTE
  /////////////////////////////////////////////////////////

  std::vector<executor::AbstractExecutor *> executors;
  executors.push_back(&insert_executor);

  /////////////////////////////////////////////////////////
  // COLLECT STATS
  /////////////////////////////////////////////////////////
  std::vector<double> index_columns_accessed;
  double selectivity = 0;

  ExecuteTest(executors, brain::SAMPLE_TYPE_UPDATE, {index_columns_accessed},
              selectivity);

  txn_manager.CommitTransaction();
}

void RunJoinTest(const std::vector<oid_t> &left_table_tuple_key_attrs,
                 const std::vector<oid_t> &left_table_index_key_attrs,
                 const std::vector<oid_t> &right_table_tuple_key_attrs,
                 const std::vector<oid_t> &right_table_index_key_attrs,
                 const oid_t left_table_join_column,
                 const oid_t right_table_join_column) {
  const bool is_inlined = true;
  auto &txn_manager = concurrency::TransactionManagerFactory::GetInstance();

  auto txn = txn_manager.BeginTransaction();

  /////////////////////////////////////////////////////////
  // SEQ SCAN + PREDICATE
  /////////////////////////////////////////////////////////

  std::unique_ptr<executor::ExecutorContext> context(
      new executor::ExecutorContext(txn));

  // Column ids to be added to logical tile after scan.
  // Left half of the columns are considered left table, right half of the
  // columns are considered right table.
  std::vector<oid_t> column_ids;
  oid_t column_count = state.column_count;

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
  std::unique_ptr<expression::TupleValueExpression> left_table_attr(
      new expression::TupleValueExpression(VALUE_TYPE_INTEGER, 0,
                                           left_table_join_column));
  std::unique_ptr<expression::TupleValueExpression> right_table_attr(
      new expression::TupleValueExpression(VALUE_TYPE_INTEGER, 1,
                                           right_table_join_column));

  std::unique_ptr<expression::ComparisonExpression<expression::CmpLt>>
      join_predicate(new expression::ComparisonExpression<expression::CmpLt>(
          EXPRESSION_TYPE_COMPARE_LESSTHAN, left_table_attr.get(),
          right_table_attr.get()));

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

  auto selectivity = state.selectivity;

  ExecuteTest(
      executors, brain::SAMPLE_TYPE_ACCESS,
      {left_table_index_columns_accessed, right_table_index_columns_accessed},
      selectivity);

  txn_manager.CommitTransaction();
}

static void RunAdaptTest() {
  double write_ratio = state.write_ratio;
  double repeat_count = state.total_ops / state.phase_length;

  total_duration = 0;

  state.operator_type = OPERATOR_TYPE_DIRECT;

  for (oid_t repeat_itr = 0; repeat_itr < repeat_count; repeat_itr++) {
    double rand_sample = (double)rand() / RAND_MAX;

    if (rand_sample < write_ratio) {
      // Do insert
      LOG_INFO("Do insert");
      RunInsertTest();
    } else {
      // Do read
      LOG_INFO("Do read");
      RunModerateQuery();
    }
  }

  LOG_INFO("Total Duration : %.2lf", total_duration);
}

static void RunQueryTest() {
  double repeat_count = state.total_ops / state.phase_length;

  total_duration = 0;

  for (oid_t repeat_itr = 0; repeat_itr < repeat_count; repeat_itr++) {
    double rand_sample = (double)rand() / RAND_MAX;

    // Uniform distribution of simple, moderate and complex query
    if (rand_sample < 0.33) {
      RunSimpleQuery();
    } else if (rand_sample < 0.66) {
      RunModerateQuery();
    } else {
      RunComplexQeury();
    }
  }

  LOG_INFO("Total Duration : %.2lf", total_duration);
}

std::vector<std::size_t> phase_lengths = {40};

void RunAdaptExperiment() {
  // Setup layout tuner
  auto &index_tuner = brain::IndexTuner::GetInstance();
  std::thread index_builder;

  state.projectivity = 1.0;
  state.selectivity = 0.001;
  state.column_count = 10;
  state.layout_mode = LAYOUT_TYPE_ROW;
  state.adapt_layout = true;

  state.total_ops = 400;

  peloton_layout_mode = state.layout_mode;

  // Generate sequence
  GenerateSequence(state.column_count);

  CreateAndLoadTable((LayoutType)peloton_layout_mode);

  for (auto phase_length : phase_lengths) {
    // Set phase length
    state.phase_length = phase_length;
    LOG_INFO("Phase Length: %lu", state.phase_length);

    // Reset query counter
    query_itr = 0;

    // Start index tuner
    index_tuner.Start();
    index_tuner.AddTable(sdbench_table.get());

    // Run adapt test
    RunAdaptTest();

    // Stop index tuner
    index_tuner.Stop();
    index_tuner.ClearTables();

    // Drop Indexes
    DropIndexes();
  }

  // Reset
  state.adapt_layout = false;
  query_itr = 0;

  out.close();
}

void RunQueryExperiment() {
  LOG_INFO("Run query experiment");
  auto &index_tuner = brain::IndexTuner::GetInstance();
  std::thread index_builder;

  state.projectivity = 1.0;
  state.selectivity = 0.001;
  state.column_count = 10;
  state.layout_mode = LAYOUT_TYPE_ROW;
  state.adapt_layout = true;

  state.total_ops = 3000;
  state.phase_length = 300;

  peloton_layout_mode = state.layout_mode;

  // Generate sequence
  GenerateSequence(state.column_count);

  CreateAndLoadTable((LayoutType)peloton_layout_mode);

  LOG_INFO("Phase length: %lu", state.phase_length);

  // Reset query counter
  query_itr = 0;

  // Start index tuner
  index_tuner.Start();
  index_tuner.AddTable(sdbench_table.get());

  // Run query test
  RunQueryTest();

  index_tuner.Stop();
  index_tuner.ClearTables();

  DropIndexes();

  state.adapt_layout = false;
  query_itr = 0;

  out.close();
}

}  // namespace sdbench
}  // namespace benchmark
}  // namespace peloton
