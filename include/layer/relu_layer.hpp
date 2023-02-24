/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 15:46:41
 */
#pragma once
#include "layer.hpp"
#include "ops/relu_op.hpp"

namespace YAInfer {
class ReluLayer : public Layer {
 public:
  ~ReluLayer() override = default;

  // transferring the thresh in relu_op to relu layer, we will use it in
  // computing
  explicit ReluLayer(const std::shared_ptr<Operator> &op);

  // the specific function to perform relu
  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &output) override;

  static std::shared_ptr<Layer> CreateInstance(
      const std::shared_ptr<Operator> &op);

 private:
  std::unique_ptr<ReluOperator> op_;
};
}  // namespace YAInfer