//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// drop_plan.cpp
//
// Identification: src/planner/drop_plan.cpp
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "planner/update_plan.h"

#include "planner/project_info.h"
#include "common/types.h"

#include "storage/data_table.h"
#include "catalog/bootstrapper.h"
#include "parser/statement_update.h"
#include "parser/table_ref.h"
#include "expression/expression_util.h"

namespace peloton {
namespace planner {

UpdatePlan::UpdatePlan(storage::DataTable *table,
                       std::unique_ptr<const planner::ProjectInfo> project_info)
    : target_table_(table),
      project_info_(std::move(project_info)),
      updates_(NULL),
      where_(NULL) {
	LOG_TRACE("Creating an Update Plan");
}

UpdatePlan::UpdatePlan(parser::UpdateStatement *parse_tree) {
  LOG_TRACE("Creating an Update Plan");
  updates_ = new std::vector<parser::UpdateClause *>();
  auto t_ref = parse_tree->table;
  table_name = std::string(t_ref->name);
  target_table_ = catalog::Bootstrapper::global_catalog->GetTableFromDatabase(
      DEFAULT_DB_NAME, table_name);

  for(auto update_clause : *parse_tree->updates) {
	  updates_->push_back(update_clause->Copy());
  }
  TargetList tlist;
  DirectMapList dmlist;
  oid_t col_id;
  auto schema = target_table_->GetSchema();

  std::vector<oid_t> columns;
  for (auto update : *updates_) {
    // get oid_t of the column and push it to the vector;
    col_id = schema->GetColumnID(std::string(update->column));
    columns.push_back(col_id);
    tlist.emplace_back(col_id, update->value->Copy());
  }

  for (uint i = 0; i < schema->GetColumns().size(); i++) {
    if (schema->GetColumns()[i].column_name !=
        schema->GetColumns()[col_id].column_name)
      dmlist.emplace_back(i, std::pair<oid_t, oid_t>(0, i));
  }

  std::unique_ptr<const planner::ProjectInfo> project_info(
      new planner::ProjectInfo(std::move(tlist), std::move(dmlist)));
  project_info_ = std::move(project_info);

  where_ = parse_tree->where->Copy();
  ReplaceColumnExpressions(target_table_->GetSchema(), where_);

  std::unique_ptr<planner::SeqScanPlan> seq_scan_node(
      new planner::SeqScanPlan(target_table_, where_, columns));
  AddChild(std::move(seq_scan_node));
}

void UpdatePlan::SetParameterValues(std::vector<Value> *values) {
  LOG_TRACE("Setting parameter values in Update");
  // First update project_info_ target list
  // Create new project_info_
  TargetList tlist;
  DirectMapList dmlist;
  oid_t col_id;
  auto schema = target_table_->GetSchema();

  std::vector<oid_t> columns;
  for (auto update : *updates_) {
    // get oid_t of the column and push it to the vector;
    col_id = schema->GetColumnID(std::string(update->column));
    columns.push_back(col_id);
    tlist.emplace_back(col_id, update->value->Copy());
  }

  for (uint i = 0; i < schema->GetColumns().size(); i++) {
    if (schema->GetColumns()[i].column_name !=
        schema->GetColumns()[col_id].column_name)
      dmlist.emplace_back(i, std::pair<oid_t, oid_t>(0, i));
  }

  auto new_proj_info = new planner::ProjectInfo(std::move(tlist), std::move(dmlist));
  new_proj_info->transformParameterToConstantValueExpression(values, target_table_->GetSchema());
  project_info_.reset(new_proj_info);

  LOG_TRACE("Setting values for parameters in where");
  auto &children = GetChildren();
  // One sequential scan
  children[0]->SetParameterValues(values);
}

}  // namespace planner
}  // namespace peloton
