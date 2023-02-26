/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-26 13:35:53
 */
#pragma once
#include "layer.hpp"
#include "ops/maxpooling_op.hpp"
namespace YAInfer {
class MaxPoolingLayer : public Layer {
 public:
  explicit MaxPoolingLayer(const std::shared_ptr<Operator> &op);

  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static std::shared_ptr<Layer> CreateInstance(
      const std::shared_ptr<Operator> &op);

 private:
  std::unique_ptr<MaxPoolingOp> op_;
};
}  // namespace YAInfer