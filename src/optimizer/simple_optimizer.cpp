//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// simple_optimizer.cpp
//
// Identification: src/optimizer/simple_optimizer.cpp
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "optimizer/simple_optimizer.h"

#include "parser/abstract_parse.h"
#include "parser/insert_parse.h"

#include "planner/abstract_plan.h"
#include "planner/drop_plan.h"
#include "planner/insert_plan.h"
#include "planner/seq_scan_plan.h"
#include "planner/index_scan_plan.h"
#include "planner/create_plan.h"
#include "planner/delete_plan.h"
#include "planner/update_plan.h"
#include "planner/aggregate_plan.h"
#include "parser/abstract_parse.h"
#include "parser/drop_parse.h"
#include "parser/create_parse.h"
#include "catalog/schema.h"
#include "expression/abstract_expression.h"
#include "expression/parser_expression.h"
#include "expression/expression_util.h"
#include "expression/constant_value_expression.h"
#include "parser/sql_statement.h"
#include "parser/statements.h"
#include "catalog/bootstrapper.h"
#include "storage/data_table.h"

#include "common/logger.h"
#include "common/value_factory.h"

#include <memory>

namespace peloton {
namespace planner {
class AbstractPlan;
}
namespace optimizer {

SimpleOptimizer::SimpleOptimizer() {};

SimpleOptimizer::~SimpleOptimizer() {};

std::shared_ptr<planner::AbstractPlan> SimpleOptimizer::BuildPelotonPlanTree(
    const std::unique_ptr<parser::SQLStatement>& parse_tree) {

  std::shared_ptr<planner::AbstractPlan> plan_tree;

  // Base Case
  if (parse_tree.get() == nullptr) return plan_tree;

  std::unique_ptr<planner::AbstractPlan> child_plan = nullptr;

  // One to one Mapping
  auto parse_item_node_type = parse_tree->GetType();

  switch (parse_item_node_type) {
    case STATEMENT_TYPE_DROP: {
      LOG_TRACE("Adding Drop plan...");
      std::unique_ptr<planner::AbstractPlan> child_DropPlan(
          new planner::DropPlan((parser::DropStatement*)parse_tree.get()));
      child_plan = std::move(child_DropPlan);
    } break;

    case STATEMENT_TYPE_CREATE: {
      LOG_TRACE("Adding Create plan...");
      std::unique_ptr<planner::AbstractPlan> child_CreatePlan(
          new planner::CreatePlan((parser::CreateStatement*)parse_tree.get()));
      child_plan = std::move(child_CreatePlan);
    } break;

    case STATEMENT_TYPE_SELECT: {
      
      LOG_TRACE("Processing SELECT...");
      auto select_stmt = (parser::SelectStatement*)parse_tree.get();
      LOG_TRACE("SELECT Info: %s", select_stmt->GetInfo().c_str());
      auto agg_type = AGGREGATE_TYPE_PLAIN;  // default aggregator
      std::vector<oid_t> group_by_columns;
      auto group_by = select_stmt->group_by;
      expression::AbstractExpression* having = nullptr;
      if (select_stmt->from_table->list != NULL) {
        LOG_TRACE("have join condition? %d",
                 select_stmt->from_table->join != NULL);
        LOG_TRACE("have sub select statement? %d",
                 select_stmt->from_table->select != NULL);
      }

      auto target_table =
          catalog::Bootstrapper::global_catalog->GetTableFromDatabase(
              DEFAULT_DB_NAME, select_stmt->from_table->name);
      LOG_TRACE("FOUND TABLE!");

      // Preparing the group by columns
      if (group_by != NULL) {
        LOG_TRACE("Found GROUP BY");
        for (auto elem : *group_by->columns) {
          std::string col_name(elem->GetName());
          auto column_id = target_table->GetSchema()->GetColumnID(col_name);
          group_by_columns.push_back(column_id);
        }
        // Having Expression needs to be prepared
        // Currently it's mostly ParserExpression
        // Needs to be prepared
        having = group_by->having;
      }

      // Check if there are any aggregate functions
      bool func_flag = false;
      for (auto expr : *select_stmt->getSelectList()) {
        if (expr->GetExpressionType() == EXPRESSION_TYPE_FUNCTION_REF) {
          LOG_TRACE("Query has aggregate functions");
          func_flag = true;
          break;
        }
      }

      // If there is no aggregate functions, just do a sequential scan
      if (!func_flag && group_by_columns.size() == 0) {
        LOG_TRACE("No aggregate functions found.");
        auto child_SelectPlan = CreateScanPlan(target_table, select_stmt);
        child_plan = std::move(child_SelectPlan);
      }
      // Else, do aggregations on top of scan
      else {
        // Create sequential scan plan
        target_table =
            catalog::Bootstrapper::global_catalog->GetTableFromDatabase(
                DEFAULT_DB_NAME, select_stmt->from_table->name);
        auto scan_node = CreateScanPlan(target_table, select_stmt);

        // Prepare aggregate plan
        std::vector<catalog::Column> output_schema_columns;
        std::vector<planner::AggregatePlan::AggTerm> agg_terms;
        DirectMapList direct_map_list = {};
        oid_t new_col_id = 0;
        oid_t agg_id = 0;
        int col_cntr_id = 0;
        for (auto expr : *select_stmt->getSelectList()) {
          LOG_TRACE("Expression %d type in Select: %s", index++,
                   ExpressionTypeToString(expr->GetExpressionType()).c_str());
          
          // If an aggregate function is found
          if (expr->GetExpressionType() == EXPRESSION_TYPE_FUNCTION_REF) {
            auto func_expr = (expression::ParserExpression*)expr;
            LOG_TRACE("Expression type in Function Expression: %s",
                     ExpressionTypeToString(
                         func_expr->expr->GetExpressionType()).c_str());
            LOG_TRACE("Distinct flag: %d", func_expr->distinct);
          
            // Count a column expression
            if (func_expr->expr->GetExpressionType() ==
                EXPRESSION_TYPE_COLUMN_REF) {

              LOG_TRACE("Function name: %s", func_expr->GetName());
              LOG_TRACE(
                  "Aggregate type: %s",
                  ExpressionTypeToString(ParserExpressionNameToExpressionType(
                                             func_expr->GetName())).c_str());
              planner::AggregatePlan::AggTerm agg_term(
                  ParserExpressionNameToExpressionType(func_expr->GetName()),
                  expression::ExpressionUtil::ConvertToTupleValueExpression(
                      target_table->GetSchema(), func_expr->expr->GetName()),
                  func_expr->distinct);
              agg_terms.push_back(agg_term);

              std::pair<oid_t, oid_t> inner_pair = std::make_pair(1, agg_id);
              std::pair<oid_t, std::pair<oid_t, oid_t>> outer_pair =
                  std::make_pair(new_col_id, inner_pair);
              direct_map_list.emplace_back(outer_pair);
              LOG_TRACE("Direct map list: (%d, (%d, %d))", outer_pair.first,
                       outer_pair.second.first, outer_pair.second.second);

              // If aggregate type is average the value type should be double
              if (ParserExpressionNameToExpressionType(func_expr->GetName()) ==
                  EXPRESSION_TYPE_AGGREGATE_AVG) {

                auto column = catalog::Column(
                    VALUE_TYPE_DOUBLE, GetTypeSize(VALUE_TYPE_DOUBLE),
                    "COL_" + std::to_string(col_cntr_id++),  // COL_A should be
                                                             // used only when
                                                             // there is no AS
                    true);

                output_schema_columns.push_back(column);
              }
          
              // Else it is the same as the column type
              else {
                oid_t old_col_id = target_table->GetSchema()->GetColumnID(
                    func_expr->expr->GetName());
                auto table_column =
                    target_table->GetSchema()->GetColumn(old_col_id);

                auto column = catalog::Column(
                    table_column.GetType(), GetTypeSize(table_column.GetType()),
                    "COL_" + std::to_string(col_cntr_id++),  // COL_A should be
                                                             // used only when
                                                             // there is no AS
                    true);

                output_schema_columns.push_back(column);
              }

            }
          
            // Check for COUNT STAR Expression
            else if (func_expr->expr->GetExpressionType() ==
                     EXPRESSION_TYPE_STAR) {
              LOG_TRACE("Creating an aggregate plan");
              planner::AggregatePlan::AggTerm agg_term(
                  EXPRESSION_TYPE_AGGREGATE_COUNT_STAR,
                  nullptr,  // No predicate for star expression. Nothing to
                            // evaluate
                  func_expr->distinct);
              agg_terms.push_back(agg_term);

              std::pair<oid_t, oid_t> inner_pair = std::make_pair(1, agg_id);
              std::pair<oid_t, std::pair<oid_t, oid_t>> outer_pair =
                  std::make_pair(new_col_id, inner_pair);
              direct_map_list.emplace_back(outer_pair);
              LOG_TRACE("Direct map list: (%d, (%d, %d))", outer_pair.first,
                       outer_pair.second.first, outer_pair.second.second);

              auto column = catalog::Column(
                  VALUE_TYPE_INTEGER, GetTypeSize(VALUE_TYPE_INTEGER),
                  "COL_" + std::to_string(col_cntr_id++),  // COL_A should be
                                                           // used only when
                                                           // there is no AS
                  true);

              output_schema_columns.push_back(column);
            } else {
              LOG_TRACE("Unrecognized type in function expression!");
            }
            ++agg_id;
          }
          
          // Column name
          else {

            // There are columns in the query
            agg_type = AGGREGATE_TYPE_HASH;
            std::string col_name(expr->GetName());
            oid_t old_col_id = target_table->GetSchema()->GetColumnID(col_name);

            std::pair<oid_t, oid_t> inner_pair = std::make_pair(0, old_col_id);
            std::pair<oid_t, std::pair<oid_t, oid_t>> outer_pair =
                std::make_pair(new_col_id, inner_pair);
            direct_map_list.emplace_back(outer_pair);
            LOG_TRACE("Direct map list: (%d, (%d, %d))", outer_pair.first,
                     outer_pair.second.first, outer_pair.second.second);

            auto table_column =
                target_table->GetSchema()->GetColumn(old_col_id);

            auto column = catalog::Column(
                table_column.GetType(), GetTypeSize(table_column.GetType()),
                "COL_" + std::to_string(col_cntr_id++),  // COL_A should be used
                                                         // only when there is
                                                         // no AS
                true);

            output_schema_columns.push_back(
                target_table->GetSchema()->GetColumn(old_col_id));
          }
          ++new_col_id;
        }
        LOG_TRACE("Creating a ProjectInfo");
        std::unique_ptr<const planner::ProjectInfo> proj_info(
            new planner::ProjectInfo(TargetList(), std::move(direct_map_list)));

        std::unique_ptr<const expression::AbstractExpression> predicate(having);
        std::shared_ptr<const catalog::Schema> output_table_schema(
            new catalog::Schema(output_schema_columns));
        LOG_TRACE("Output Schema Info: %s",
                 output_table_schema.get()->GetInfo().c_str());

        std::unique_ptr<planner::AggregatePlan> child_agg_plan(
            new planner::AggregatePlan(
                std::move(proj_info), std::move(predicate),
                std::move(agg_terms), std::move(group_by_columns),
                output_table_schema, agg_type));

        child_agg_plan->AddChild(std::move(scan_node));
        child_plan = std::move(child_agg_plan);
      }

    } break;

    case STATEMENT_TYPE_INSERT: {
      LOG_TRACE("Adding Insert plan...");
      std::unique_ptr<planner::AbstractPlan> child_InsertPlan(
          new planner::InsertPlan((parser::InsertStatement*)parse_tree.get()));
      child_plan = std::move(child_InsertPlan);
    } break;

    case STATEMENT_TYPE_DELETE: {
      LOG_TRACE("Adding Delete plan...");
      std::unique_ptr<planner::AbstractPlan> child_DeletePlan(
          new planner::DeletePlan((parser::DeleteStatement*)parse_tree.get()));
      child_plan = std::move(child_DeletePlan);
    } break;

    case STATEMENT_TYPE_UPDATE: {
      LOG_TRACE("Adding Update plan...");
      std::unique_ptr<planner::AbstractPlan> child_InsertPlan(
          new planner::UpdatePlan((parser::UpdateStatement*)parse_tree.get()));
      child_plan = std::move(child_InsertPlan);
    } break;

    default:
      LOG_TRACE("Unsupported Parse Node Type");
  }

  if (child_plan != nullptr) {
    if (plan_tree != nullptr) {
      plan_tree->AddChild(std::move(child_plan));
    } else {
      plan_tree = std::move(child_plan);
    }
  }

  // Recurse will be modified later
  /*auto &children = parse_tree->GetChildren();
   for (auto &child : children) {
   std::shared_ptr<planner::AbstractPlan> child_parse =
   std::move(BuildPlanTree(child));
   child_plan = std::move(child_parse);
   }*/
  return plan_tree;
}

std::unique_ptr<planner::AbstractScan> SimpleOptimizer::CreateScanPlan(
    storage::DataTable* target_table, parser::SelectStatement* select_stmt) {

  bool index_searchable = false;
  int index_id = 0;

  // column predicates passing to the index
  std::vector<oid_t> key_column_ids;
  std::vector<ExpressionType> expr_types;
  std::vector<Value> values;

  // column predicates between the tuple value and the constant in the where
  // clause
  std::vector<oid_t> predicate_column_ids = {};
  std::vector<ExpressionType> predicate_expr_types;
  std::vector<Value> predicate_values;

  if (select_stmt->where_clause != NULL) {
    index_searchable = true;

    LOG_TRACE("Getting predicate columns");
    GetPredicateColumns(target_table->GetSchema(), select_stmt->where_clause,
                        predicate_column_ids, predicate_expr_types,
                        predicate_values, index_searchable);
    LOG_TRACE("Finished Getting predicate columns");

    if (index_searchable == true) {
      index_searchable = false;

      // Loop through the indexes to find to most proper one (if any)
      int max_columns = 0;
      int index_index = 0;
      for (auto& column_set : target_table->GetIndexColumns()) {
        int matched_columns = 0;
        for (auto column_id : predicate_column_ids)
          if (column_set.find(column_id) != column_set.end()) matched_columns++;
        if (matched_columns > max_columns) {
          index_searchable = true;
          index_id = index_index;
        }
        index_index++;
      }
    }
  }

  // index_searchable = false;
  // using the index scan causes an error:
  //Exception Type :: Mismatch Type
  //Message :: Type VARCHAR does not match with BIGINTType VARCHAR can't be cast as BIGINT...
  //terminate called after throwing an instance of 'peloton::TypeMismatchException'
  //what():  Type VARCHAR does not match with BIGINTType VARCHAR can't be cast as BIGINT...

  if (!index_searchable) {
//  if (true) {
    // Create sequential scan plan
    LOG_TRACE("Creating a sequential scan plan");
    std::unique_ptr<planner::SeqScanPlan> child_SelectPlan(
        new planner::SeqScanPlan(select_stmt));
    LOG_TRACE("Sequential scan plan created");
    return std::move(child_SelectPlan);
  }

  // Create index scan plan
  LOG_TRACE("Creating a index scan plan");
  auto index = target_table->GetIndex(index_id);
  std::vector<expression::AbstractExpression*> runtime_keys;

  auto index_columns = target_table->GetIndexColumns()[index_id];
  int column_idx = 0;
  for (auto column_id : predicate_column_ids) {
    if (index_columns.find(column_id) != index_columns.end()) {
      key_column_ids.push_back(column_id);
      expr_types.push_back(predicate_expr_types[column_idx]);
      values.push_back(predicate_values[column_idx]);
      LOG_TRACE("Adding for IndexScanDesc: id(%d), expr(%s), values(%s)",
               column_id, ExpressionTypeToString(*expr_types.rbegin()).c_str(),
               values.rbegin()->GetInfo().c_str());
    }
    column_idx++;
  }

  // Create index scan desc
  planner::IndexScanPlan::IndexScanDesc index_scan_desc(
      index, key_column_ids, expr_types, values, runtime_keys);

  std::vector<oid_t> column_ids = {};
  bool function_found = false;
  for (auto elem : *select_stmt->select_list) {
    if (elem->GetExpressionType() == EXPRESSION_TYPE_FUNCTION_REF) {
      function_found = true;
      break;
    }
  }
  // Pass all columns
  if (function_found || select_stmt->select_list->at(0)->GetExpressionType() ==
                            EXPRESSION_TYPE_STAR) {
    auto column_count = target_table->GetSchema()->GetColumnCount();
    for (uint i = 0; i < column_count; i++) column_ids.push_back(i);
  }
  // Pass columns in select_list
  else {
    for (auto col : *select_stmt->select_list) {
      LOG_TRACE("ExpressionType: %s",
               ExpressionTypeToString(col->GetExpressionType()).c_str());
      column_ids.push_back(
          target_table->GetSchema()->GetColumnID(col->GetName()));
    }
  }
  LOG_TRACE("Index scan column size: %ld\n", column_ids.size());

  // Create plan node.
  std::unique_ptr<planner::IndexScanPlan> node(new planner::IndexScanPlan(
      target_table, select_stmt->where_clause, column_ids, index_scan_desc));
  LOG_TRACE("Index scan plan created");

  return std::move(node);
}

/**
 * This function replaces all COLUMN_REF expressions with TupleValue
 * expressions
 */
void SimpleOptimizer::GetPredicateColumns(
    catalog::Schema* schema, expression::AbstractExpression* expression,
    std::vector<oid_t>& column_ids, std::vector<ExpressionType>& expr_types,
    std::vector<Value>& values, bool& index_searchable) {

  // For now, all conjunctions should be AND when using index scan.
  if (expression->GetExpressionType() == EXPRESSION_TYPE_CONJUNCTION_OR)
    index_searchable = false;

  LOG_TRACE("Expression Type --> %s",
           ExpressionTypeToString(expression->GetExpressionType()).c_str());
  LOG_TRACE("Left Type --> %s",
           ExpressionTypeToString(expression->GetLeft()->GetExpressionType())
               .c_str());
  LOG_TRACE("Right Type --> %s",
           ExpressionTypeToString(expression->GetRight()->GetExpressionType())
               .c_str());

  // We're only supporting comparing a column_ref to a constant/parameter for
  // index scan right now
  if (expression->GetLeft()->GetExpressionType() ==
      EXPRESSION_TYPE_COLUMN_REF) {
    auto right_type = expression->GetRight()->GetExpressionType();
    if (right_type == EXPRESSION_TYPE_VALUE_CONSTANT ||
        right_type == EXPRESSION_TYPE_VALUE_PARAMETER) {
      auto expr = expression->GetLeft();
      std::string col_name(expr->GetName());
      LOG_TRACE("Column name: %s", col_name.c_str());
      auto column_id = schema->GetColumnID(col_name);
      column_ids.push_back(column_id);
      expr_types.push_back(expression->GetExpressionType());
      if (right_type == EXPRESSION_TYPE_VALUE_CONSTANT) {
        values.push_back(reinterpret_cast<expression::ConstantValueExpression*>(
            expression->GetModifiableRight())->getValue());
        LOG_TRACE("Value Type: %d",
                 reinterpret_cast<expression::ConstantValueExpression*>(
                     expression->GetModifiableRight())
                     ->getValue()
                     .GetValueType());
      } else
        values.push_back(std::move(ValueFactory::GetBindingOnlyIntegerValue(
            reinterpret_cast<expression::ParameterValueExpression*>(
                expression->GetModifiableRight())->GetValueIdx())));
    }
  } else if (expression->GetRight()->GetExpressionType() ==
             EXPRESSION_TYPE_COLUMN_REF) {
    auto left_type = expression->GetLeft()->GetExpressionType();
    if (left_type == EXPRESSION_TYPE_VALUE_CONSTANT ||
        left_type == EXPRESSION_TYPE_VALUE_PARAMETER) {
      auto expr = expression->GetRight();
      std::string col_name(expr->GetName());
      LOG_TRACE("Column name: %s", col_name.c_str());
      auto column_id = schema->GetColumnID(col_name);
      column_ids.push_back(column_id);
      expr_types.push_back(expression->GetExpressionType());
      if (left_type == EXPRESSION_TYPE_VALUE_CONSTANT) {
        values.push_back(reinterpret_cast<expression::ConstantValueExpression*>(
            expression->GetModifiableRight())->getValue());
        LOG_TRACE("Value Type: %d",
                 reinterpret_cast<expression::ConstantValueExpression*>(
                     expression->GetModifiableLeft())
                     ->getValue()
                     .GetValueType());
      } else
        values.push_back(std::move(ValueFactory::GetBindingOnlyIntegerValue(
            reinterpret_cast<expression::ParameterValueExpression*>(
                expression->GetModifiableLeft())->GetValueIdx())));
    }
  } else {
    GetPredicateColumns(schema, expression->GetModifiableLeft(), column_ids,
                        expr_types, values, index_searchable);
    GetPredicateColumns(schema, expression->GetModifiableRight(), column_ids,
                        expr_types, values, index_searchable);
  }
}

}  // namespace optimizer
}  // namespace peloton
