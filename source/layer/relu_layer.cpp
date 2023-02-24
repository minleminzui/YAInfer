/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 18:29:22
 */
#include "layer/relu_layer.hpp"

#include <glog/logging.h>

#include "factory/layer_factory.hpp"
#include "layer/relu_layer.hpp"
#include "ops/relu_op.hpp"

namespace YAInfer {
ReluLayer::ReluLayer(const std::shared_ptr<Operator> &op) : Layer("Relu") {
  CHECK(op->op_type_ == OpType::kOperatorRelu)
      << "Operator has a wrong type: " << int(op->op_type_);
  // use dynamic_cast to check the type of Operator
  ReluOperator *relu_op = dynamic_cast<ReluOperator *>(op.get());

  CHECK(relu_op != nullptr) << "Relu operator is empty";
  // a op example corresponds to a layer. relu op corresponds to relu layer
  this->op_ = std::make_unique<ReluOperator>(relu_op->get_thresh());
}

void ReluLayer::Forwards(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  // realize the separation of attributes storage and operation process
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorRelu);

  const uint32_t batch_size = inputs.size();
  // a batchsize number of tensor need to relu
  for (int i = 0; i < batch_size; ++i) {
    CHECK(!inputs.at(i)->empty());
    const std::shared_ptr<Tensor<float>> &input_data =
        inputs.at(i);  // get a tensor from the batch

    // fcube.transform
    input_data->data().transform([&](float value) {
      if (value >= this->op_->get_thresh()) {
        return value;
      } else {
        return 0.f;
      }
    });

    outputs.push_back(input_data);
  }
}

std::shared_ptr<Layer> ReluLayer::CreateInstance(
    const std::shared_ptr<Operator> &op) {
  std::shared_ptr<Layer> relu_layer = std::make_shared<ReluLayer>(op);
  return relu_layer;
}

LayerRegistererWrapper kRelulayer(OpType::kOperatorRelu,
                                  ReluLayer::CreateInstance);
}  // namespace YAInfer