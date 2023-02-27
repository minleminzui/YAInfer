/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-27 10:52:13
 */
#pragma once
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "factory/layer_factory.hpp"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"

namespace YAInfer {
class Layer;
struct RuntimeOperator {
  int32_t meet_num =
      0;  // calculate the number of being accessed by the connected nodes
  ~RuntimeOperator() {
    for (const auto &param : this->params) {
      delete param.second;
    }
  }

  std::string name;              // name of computation node
  std::string type;              // type of computation node
  std::shared_ptr<Layer> layer;  // the corresponding Layer of this node

  std::vector<std::string>
      output_names;  // name of the output node of this node
  std::shared_ptr<RuntimeOperand> output_operands;  // the output operand

  std::map<std::string, std::shared_ptr<RuntimeOperand>>
      input_operands;  // the input operand
  std::vector<std::shared_ptr<RuntimeOperand>>
      input_operands_seq;  // the input operand, sequential sequence
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      output_operators;  // the name of the output node corresponds to the node

  std::map<std::string, RuntimeParameter *> params;  // params of operators
  std::map<std::string, std::shared_ptr<RuntimeAttribute>>
      attribute;  // attributes of operator, i.e. weights of nodes
};
}  // namespace YAInfer
