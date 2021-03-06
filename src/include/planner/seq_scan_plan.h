//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// seq_scan_plan.h
//
// Identification: src/include/planner/seq_scan_plan.h
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "abstract_scan_plan.h"
#include "common/types.h"
#include "common/serializer.h"
#include "expression/abstract_expression.h"
#include "common/logger.h"

namespace peloton {

namespace parser {
struct SelectStatement;
}
namespace storage {
class DataTable;
}

namespace planner {

class SeqScanPlan : public AbstractScan {
 public:
  SeqScanPlan(const SeqScanPlan &) = delete;
  SeqScanPlan &operator=(const SeqScanPlan &) = delete;
  SeqScanPlan(SeqScanPlan &&) = delete;
  SeqScanPlan &operator=(SeqScanPlan &&) = delete;

  SeqScanPlan(storage::DataTable *table,
              expression::AbstractExpression *predicate,
              const std::vector<oid_t> &column_ids)
      : AbstractScan(table, predicate, column_ids) {
	  LOG_INFO("Creating a Sequential Scan Plan");
	  target_table_ = table;
	  where_ = predicate;
	  if(predicate != nullptr)
		  where_with_params_ = predicate->Copy();
  }

  SeqScanPlan(parser::SelectStatement *select_node);

  SeqScanPlan() : AbstractScan() { }

  inline PlanNodeType GetPlanNodeType() const { return PLAN_NODE_TYPE_SEQSCAN; }

  const std::string GetInfo() const { return "SeqScan"; }

  void SetParameterValues(std::vector<Value>* values);

  //===--------------------------------------------------------------------===//
  // Serialization/Deserialization
  //===--------------------------------------------------------------------===//
  bool SerializeTo(SerializeOutput &output);
  bool DeserializeFrom(SerializeInputBE &input);

  /* For init SerializeOutput */
  int SerializeSize();

  oid_t GetColumnID(std::string col_name);

  std::unique_ptr<AbstractPlan> Copy() const {
    AbstractPlan *new_plan = new SeqScanPlan(
        this->GetTable(), this->GetPredicate()->Copy(), this->GetColumnIds());
    return std::unique_ptr<AbstractPlan>(new_plan);
  }

 private:
  // Target Table
  storage::DataTable *target_table_ = nullptr;
  // The Where condition
  expression::AbstractExpression *where_ = nullptr;
  // The Where condition with parameter value expression
  expression::AbstractExpression *where_with_params_ = nullptr;
};

}  // namespace planner
}  // namespace peloton
