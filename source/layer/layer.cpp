/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 18:06:19
 */
#include "layer/layer.hpp"

#include <glog/logging.h>

namespace YAInfer {
Layer::Layer(const std::string &layer_name) : layer_name_(layer_name) {}

void Layer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &input,
                     std::vector<std::shared_ptr<Tensor<float>>> &output) {
  LOG(FATAL) << "The layer" << this->layer_name_ << " not impletment yet!";
}
}  // namespace YAInfer
