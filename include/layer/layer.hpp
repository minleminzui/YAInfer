/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 14:27:36
 */
#pragma once

#include <string>

#include "data/tensor.hpp"

namespace YAInfer {
class Layer {
 public:
  explicit Layer(const std::string &layer_name);

  virtual void Forwards(
      const std::vector<std::shared_ptr<Tensor<float>>> &input,
      std::vector<std::shared_ptr<Tensor<float>>> &output);
  // in relyLayer, we assume inputs are x, then outputs are y = x if x >= thresh
  virtual ~Layer() = default;

 private:
  std::string layer_name_;  // relu layer "relu"
};
}  // namespace YAInfer