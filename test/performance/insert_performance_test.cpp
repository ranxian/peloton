//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// loader_test.cpp
//
// Identification: test/performance/insert_performance_test.cpp
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//


#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <atomic>

#include "common/harness.h"

#include "catalog/schema.h"
#include "common/value_factory.h"
#include "common/pool.h"
#include "common/timer.h"
#include "concurrency/transaction_manager_factory.h"

#include "executor/executor_context.h"
#include "executor/insert_executor.h"
#include "executor/logical_tile_factory.h"
#include "expression/expression_util.h"
#include "expression/tuple_value_expression.h"
#include "expression/comparison_expression.h"
#include "expression/abstract_expression.h"
#include "storage/tile.h"
#include "storage/tile_group.h"
#include "storage/table_factory.h"

#include "executor/executor_tests_util.h"
#include "executor/mock_executor.h"

#include "planner/insert_plan.h"

using ::testing::NotNull;
using ::testing::Return;

namespace peloton {
namespace test {

//===--------------------------------------------------------------------===//
// Insert Tests
//===--------------------------------------------------------------------===//

class InsertTests : public PelotonTest {};

std::atomic<int> loader_tuple_id;

//===------------------------------===//
// Utility
//===------------------------------===//

void InsertTuple(storage::DataTable *table, VarlenPool *pool,
                 oid_t tilegroup_count_per_loader,
                 UNUSED_ATTRIBUTE uint64_t thread_itr) {
  auto &txn_manager = concurrency::TransactionManagerFactory::GetInstance();

  oid_t tuple_count = tilegroup_count_per_loader * TEST_TUPLES_PER_TILEGROUP;

  // Start a txn for each insert
  auto txn = txn_manager.BeginTransaction();
  std::unique_ptr<storage::Tuple> tuple(
      ExecutorTestsUtil::GetTuple(table, ++loader_tuple_id, pool));

  std::unique_ptr<executor::ExecutorContext> context(
      new executor::ExecutorContext(txn));

  planner::InsertPlan node(table, std::move(tuple));

  // Insert the desired # of tuples
  for (oid_t tuple_itr = 0; tuple_itr < tuple_count; tuple_itr++) {
    executor::InsertExecutor executor(&node, context.get());
    executor.Execute();
  }

  txn_manager.CommitTransaction();
}

TEST_F(InsertTests, LoadingTest) {
  // We are going to simply load tile groups concurrently in this test
  // WARNING: This test may potentially run for a long time if
  // TEST_TUPLES_PER_TILEGROUP is large, consider rewrite the test or hard
  // code the number of tuples per tile group in this test
  oid_t tuples_per_tilegroup = TEST_TUPLES_PER_TILEGROUP;
  bool build_indexes = false;

  // Control the scale
  oid_t loader_threads_count = 1;
  oid_t tilegroup_count_per_loader = 1;

  // Each tuple size ~40 B.
  oid_t tuple_size = 41;

  std::unique_ptr<storage::DataTable> data_table(
      ExecutorTestsUtil::CreateTable(tuples_per_tilegroup, build_indexes));

  auto testing_pool = TestingHarness::GetInstance().GetTestingPool();

  Timer<> timer;

  timer.Start();

  LaunchParallelTest(loader_threads_count, InsertTuple, data_table.get(),
                     testing_pool, tilegroup_count_per_loader);

  timer.Stop();
  auto duration = timer.GetDuration();

  LOG_INFO("Duration: %.2lf", duration);

  //EXPECT_LE(duration, 0.2);

  auto expected_tile_group_count =
      loader_threads_count * tilegroup_count_per_loader + 1;
  auto bytes_to_megabytes_converter = (1024 * 1024);

  EXPECT_EQ(data_table->GetTileGroupCount(), expected_tile_group_count);

  LOG_INFO("Dataset size : %u MB \n",
           (expected_tile_group_count * tuples_per_tilegroup * tuple_size) /
               bytes_to_megabytes_converter);
}

}  // namespace test
}  // namespace peloton
