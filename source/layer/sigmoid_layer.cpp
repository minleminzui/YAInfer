/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-25 14:31:02
 */
#include "layer/sigmoid_layer.hpp"

#include <glog/logging.h>

#include <armadillo>

#include "factory/layer_factory.hpp"
#include "ops/relu_op.hpp"

namespace YAInfer {
SigmoidLayer::SigmoidLayer(const std::shared_ptr<Operator> &op)
    : Layer("Sigmoid") {
  CHECK(op->op_type_ == OpType::kOperatorSigmoid)
      << "Operator has a wrong type: " << int(op->op_type_);
  SigmoidOperator *sigmoid_op = dynamic_cast<SigmoidOperator *>(op.get());

  CHECK(sigmoid_op != nullptr) << "dynamic_cast failed!";

  this->op_ = std::make_unique<SigmoidOperator>();
}

void SigmoidLayer::Forwards(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorSigmoid);
  CHECK(!inputs.empty());

  const uint32_t batch_size = inputs.size();
  for (int i = 0; i < batch_size; ++i) {
    CHECK(!inputs.at(i)->empty());
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    const std::shared_ptr<Tensor<float>> output_data = input_data->Clone();

    output_data->data().transform(
        [&](float value) { return 1 / (1 + std::exp(-value)); });

    outputs.push_back(output_data);
  };
}

std::shared_ptr<Layer> SigmoidLayer::CreateInstance(
    const std::shared_ptr<Operator> &op) {
  std::shared_ptr<Layer> sigmoid_layer = std::make_shared<SigmoidLayer>(op);
  return sigmoid_layer;
}

LayerRegistererWrapper kSigmoidLayer(OpType::kOperatorSigmoid,
                                     SigmoidLayer::CreateInstance);
}  // namespace YAInfer