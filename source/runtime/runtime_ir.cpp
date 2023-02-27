/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-27 15:06:22
 */
#include "runtime/runtime_ir.hpp"

#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <utility>

#include "factory/layer_factory.hpp"

namespace YAInfer {
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

void RuntimeGraph::set_bin_path(const std::string &bin_path) {
  this->bin_path_ = bin_path;
}

void RuntimeGraph::set_param_path(const std::string &param_path) {
  this->param_path_ = param_path;
}

const std::string &RuntimeGraph::param_path() const {
  return this->param_path_;
}
const std::string &RuntimeGraph::bin_path() const { return this->bin_path_; }

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  // load the pnnx graph based on the param file and the bin file
  int load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Load param path and bin path error: " << param_path_ << " "
               << bin_path_;
    return false;
  }

  std::vector<pnnx::Operator *> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  this->operators_.clear();
  // assign 'std::shared_ptr<RuntimeOperator> runtime_operator' based on 'const
  // pnnx::Operator *op'
  for (const pnnx::Operator *op : operators) {
    if (!op) {
      LOG(ERROR) << "Meet the empty node";
      continue;
    } else {
      // init a empty RuntimeOperator
      std::shared_ptr<RuntimeOperator> runtime_operator =
          std::make_shared<RuntimeOperator>();
      // init the operator
      runtime_operator->name = op->name;
      runtime_operator->type = op->type;

      // init the input of the operator, assign 'operator' with 'runtimeoperand'
      // as the input based on 'pnnx::operand'
      const std::vector<pnnx::Operand *> &inputs = op->inputs;
      if (!inputs.empty()) {
        InitInputOperators(inputs, runtime_operator);
      }

      const std::vector<pnnx::Operand *> &outputs = op->outputs;
      if (!outputs.empty()) {
        InitOutputOperators(outputs, runtime_operator);
      }

      const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;
      if (!attrs.empty()) {
        InitGraphAttrs(attrs, runtime_operator);
      }

      const std::map<std::string, pnnx::Parameter> &params = op->params;
      if (!params.empty()) {
        InitGraphParams(params, runtime_operator);
      }

      this->operators_.push_back(runtime_operator);
    }
  }
  graph_state_ = GraphState::NeedBuild;
  return true;
}

void RuntimeGraph::InitInputOperators(
    const std::vector<pnnx::Operand *> &inputs,
    const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const pnnx::Operand *input : inputs) {
    if (!input) {
      continue;
    }
    const pnnx::Operator *producer = input->producer;
    // what is difference between 'names of operators' and 'names of operands'?
    std::shared_ptr<RuntimeOperand> runtime_operand =
        std::make_shared<RuntimeOperand>();
    runtime_operand->name = producer->name;
    runtime_operand->shapes = input->shape;

    switch (input->type) {
      case 0: {
        runtime_operand->type = RuntimeDataType::kTypeFloat32;
        break;
      }
      case 1: {
        runtime_operand->type = RuntimeDataType::kTypeUnknown;
        break;
      }
      default: {
        LOG(FATAL) << "Unknown input operand type: " << input->type;
      }
    }
    runtime_operator->input_operands.insert({producer->name, runtime_operand});
    runtime_operator->input_operands_seq.push_back(runtime_operand);
  }
}

void RuntimeGraph::InitOutputOperators(
    const std::vector<pnnx::Operand *> &outputs,
    const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const pnnx::Operand *output : outputs) {
    if (!output) {
      continue;
    }
    // what does this inner for-loop means?
    const auto &consumers = output->consumers;
    for (const auto &c : consumers) {
      runtime_operator->output_names.push_back(c->name);
    }
  }
}

void RuntimeGraph::InitGraphParams(
    const std::map<std::string, pnnx::Parameter> &params,
    const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const auto &pair : params) {
    const std::string &name = pair.first;
    const pnnx::Parameter &parameter = pair.second;
    const int type = parameter.type;

    // according to the incoming pnnx::params, to iterate them to get initilize
    // to speicfied RuntimeParameter put RuntimeParameter into RuntimeOperator
    switch (type) {
      case int(RuntimeParameterType::kParameterUnknown): {
        RuntimeParameter *runtime_parameter = new RuntimeParameter;
        // store the parameter name and the specified class instance
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      case int(RuntimeParameterType::kParameterBool): {
        RuntimeParameterBool *runtime_parameter = new RuntimeParameterBool;
        runtime_parameter->value = parameter.b;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      case int(RuntimeParameterType::kParameterInt): {
        RuntimeParameterFloat *runtime_parameter = new RuntimeParameterFloat;
        runtime_parameter->value = parameter.f;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterString): {
        RuntimeParameterString *runtime_parameter = new RuntimeParameterString;
        runtime_parameter->value = parameter.s;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterIntArray): {
        RuntimeParameterIntArray *runtime_parameter =
            new RuntimeParameterIntArray;
        runtime_parameter->value = parameter.ai;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterFloatArray): {
        RuntimeParameterFloatArray *runtime_parameter =
            new RuntimeParameterFloatArray;
        runtime_parameter->value = parameter.af;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterStringArray): {
        RuntimeParameterStringArray *runtime_parameter =
            new RuntimeParameterStringArray;
        runtime_parameter->value = parameter.as;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown parameter type";
      }
    }
  }
}

void RuntimeGraph::InitGraphAttrs(
    const std::map<std::string, pnnx::Attribute> &attrs,
    const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const auto &pair : attrs) {
    const std::string &name = pair.first;
    const pnnx::Attribute &attr = pair.second;
    switch (attr.type) {
      case 1: {
        std::shared_ptr<RuntimeAttribute> runtime_attribute =
            std::make_shared<RuntimeAttribute>();
        runtime_attribute->type = RuntimeDataType::kTypeFloat32;
        runtime_attribute->weight_data = attr.data;
        runtime_attribute->shape = attr.shape;
        runtime_operator->attribute.insert({name, runtime_attribute});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown attribute type";
      }
    }
  }
}

const std::vector<std::shared_ptr<RuntimeOperator>> RuntimeGraph::operators()
    const {
  return this->operators_;
}
}  // namespace YAInfer