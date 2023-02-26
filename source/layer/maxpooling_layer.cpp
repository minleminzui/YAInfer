/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-26 14:24:48
 */
#include "layer/maxpooling_layer.hpp"

#include <glog/logging.h>

#include "factory/layer_factory.hpp"

namespace YAInfer {
MaxPoolingLayer::MaxPoolingLayer(const std::shared_ptr<Operator> &op)
    : Layer("maxpooling") {
  CHECK(op->op_type_ == OpType::kOperatorMaxPooling)
      << "Opeartor has a wrong type: " << int(op->op_type_);

  MaxPoolingOp *max_pooling_op = dynamic_cast<MaxPoolingOp *>(op.get());

  CHECK(max_pooling_op != nullptr) << "MaxPooling operator is empty";

  this->op_ = std::make_unique<MaxPoolingOp>(*max_pooling_op);
}

void MaxPoolingLayer::Forwards(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorMaxPooling);
  CHECK(!inputs.empty());
  // Same padding?
  const uint32_t padding_h = this->op_->padding_height();
  const uint32_t padding_w = this->op_->padding_width();
  const uint32_t kernel_h = this->op_->pooling_height();
  const uint32_t kernel_w = this->op_->pooling_width();
  const uint32_t stride_h = this->op_->stride_height();
  const uint32_t stride_w = this->op_->stride_width();

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i)->Clone();
    input_data->Padding({padding_h, padding_h, padding_w, padding_w},
                        std::numeric_limits<float>::lowest());
    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_c = input_data->channels();
    const uint32_t output_c = input_c;

    const uint32_t output_h =
        uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
    const uint32_t output_w =
        uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));

    std::shared_ptr<Tensor<float>> output_data =
        std::make_shared<Tensor<float>>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &input_channel = input_data->at(ic);
      arma::fmat &output_channel = output_data->at(ic);

      for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
          const arma::fmat &region =
              input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
          output_channel.at(int(r / stride_h), int(c / stride_w)) =
              region.max();
        }
      }
    }
    outputs.push_back(output_data);
  }
}

std::shared_ptr<Layer> MaxPoolingLayer::CreateInstance(
    const std::shared_ptr<Operator> &op) {
  CHECK(op->op_type_ == OpType::kOperatorMaxPooling);
  std::shared_ptr<Layer> max_layer = std::make_shared<MaxPoolingLayer>(op);
  return max_layer;
}

LayerRegistererWrapper kMaxPoolingLayer(OpType::kOperatorMaxPooling,
                                        MaxPoolingLayer::CreateInstance);
}  // namespace YAInfer