/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-26 17:26:36
 */
#include <glog/logging.h>

#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "factory/layer_factory.hpp"
#include "ir.h"
#include "runtime_op.hpp"
#include "runtime_operand.hpp"

// runtimegraph -> pnnx::graph
// runtimeOperator -> pnnx::operator
// runtimeOperand -> pnnx::operand
// runtimeParameter -> pnnx::Params
// runtimeAttrs -> pnnx::attrs
// to initialize the runtimegraph one operator at a time
// int load_result = this->graph_->load(param_path_, bin_path_);
// iterate pnnx::operators to initialize RuntimeOperator
// 1. to initialize RuntimeOperator::runtime_operator->input_operands based on
// the inputs of pnnx operator
// 2 in the same way, to initialize
// RuntimeOperator::runtime_operator->output_operands based on the outputs pnnx
// operator
// 3. to initialize runtimeParamter based on pnnx::attr
// 4. to initialize runtimeAttr based on pnnx::attr
// after 1, 2, 3, 4, runtimeParamter, runtimeAttr, output_operands,
// inputs_operand are stored in a runtime_operator
// 5. store the runtime_operator

namespace YAInfer {
// computation graph, a data flow graph, consisted of computation nodes and
// nodes
class RuntimeGraph {
 public:
  /**
   * @description: initialization of the computation graph
   * @return whether the initialization is successful
   */
  bool Init();
  /**
   * @description: initialize the computation graph
   * @param {string} param_path
   * @param {string} bin_path
   */
  RuntimeGraph(std::string param_path, std::string bin_path);

  /**
   * @description: set weights files
   * @param {string} &bin_path
   * @return
   */
  void set_bin_path(const std::string &bin_path);

  /**
   * @description: set param file，which describe the structure of computation
   * graph
   * @param {string} &param_path
   * @return
   */
  void set_param_path(const std::string &param_path);

  /**
   * @description: return the param file
   * @return {*}
   */
  const std::string &param_path() const;

  /**
   * @description: return the param file
   * @return {*}
   */
  const std::string &bin_path() const;

  /**
   * @description: return Operators
   * @return {*}
   */
  const std::vector<std::shared_ptr<RuntimeOperator>> operators() const;

 private:
  /**
   * @description: initialize the input operators of YAInfer computation graph
   * @param inputs the input operands of pnnx
   * @param runtime_operator a node of computation graph
   * @return {*}
   */
  static void InitInputOperators(
      const std::vector<pnnx::Operand *> &inputs,
      const std::shared_ptr<RuntimeOperator> &runtime_operator);
  /**
   * @description: initialize the output operators of YAInfer computation graph
   * @param outputs  the output operands of pnnx
   * @param runtime_operator a node of computation graph
   * @return {*}
   */
  static void InitOutputOperators(
      const std::vector<pnnx::Operand *> &outputs,
      const std::shared_ptr<RuntimeOperator> &runtime_operator);
  /**
   * @description: initialize attrs of nodes in the computation graph
   * @param attrs attrs of nodes of pnnx
   * @param runtime_operator a node of computation graph
   * @return {*}
   */
  static void InitGraphAttrs(
      const std::map<std::string, pnnx::Attribute> &attrs,
      const std::shared_ptr<RuntimeOperator> &runtime_operator);
  /**
   * @description: initialize params of nodes in the computation graph
   * @param params params of nodes of pnnx
   * @param runtime_operator a node of computation graph
   * @return {*}
   */
  static void InitGraphParams(
      const std::map<std::string, pnnx::Parameter> &params,
      const std::shared_ptr<RuntimeOperator> &runtime_operator);

 private:
  enum class GraphState {
    NeedInit = -2,
    NeedBuild = -1,
    Complete = 0,
  };

  GraphState graph_state_ = GraphState::NeedInit;
  std::string input_name_;   // names of input nodes of computation graph
  std::string output_name_;  // names of output nodes of computation graph
  std::string param_path_;   // params file of computation graph
  std::string bin_path_;     // weights file of computation graph
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      input_operators_maps_;  // store input nodes
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      output_operators_maps_;  // store output nodes
  std::vector<std::shared_ptr<RuntimeOperator>>
      operators_;                       // operators computation nodes
  std::unique_ptr<pnnx::Graph> graph_;  // the graph of pnnx
};
}  // namespace YAInfer