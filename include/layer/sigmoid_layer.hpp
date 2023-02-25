/*
 * @Author: yitong 2969413251@qq.com
 * @Date: 2023-02-24 22:45:27
 */
#pragma once
#include "layer.hpp"
#include "ops/sigmoid_op.hpp"

namespace YAInfer {
class SigmoidLayer : public Layer {
 public:
  ~SigmoidLayer() override = default;

  explicit SigmoidLayer(const std::shared_ptr<Operator> &op);

  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
  static std::shared_ptr<Layer> CreateInstance(
      const std::shared_ptr<Operator> &op);

 private:
  std::unique_ptr<SigmoidOperator> op_;
};
}  // namespace YAInfer